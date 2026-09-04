"""CLI: build the unified LabMiM station database from the raw ``.dat`` archive.

Produces three artifacts from one pass over the archive, and **verifies the
merge against the audited row counts** before writing anything:

``station_5min_raw``
    Every 5-minute sample, merged from the explicit manifest in
    :mod:`micrometeorology.sensors.archive` with only the three clock repairs
    applied. Values as the logger wrote them, sentinels included.

``station_5min_qc``
    The same grid after sentinel masking, physical gates, instrument
    calibrations and era-to-era column unification — the frame the hourly means
    are computed from, written out so the chain can be audited at its midpoint.

``station_hourly``
    Hourly aggregation of the QC frame: means, sums for the tipping bucket, and
    vector means for wind direction weighted by the paired speed.

Nothing under ``data/`` is ever written; the clock repairs land in a scratch
directory.

Examples
--------
Build the whole database and fail if the merge does not reproduce the audit::

    labmim-archive -d data -o output/archive --strict

Write CSV instead of Parquet (much larger; Parquet keeps dtypes and is ~20x
smaller for this frame)::

    labmim-archive -d data -o output/archive --format csv
"""

import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from labmim_core.atomic import atomic_write, atomic_write_strict_json
from micrometeorology.common.config import get_settings
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.archive import (
    DIFFUSE_RATIO_LIMIT,
    LENTA_MANIFEST,
    NET_RADIATION_COMPONENTS,
    NIGHT_CORRUPTION_CHANNELS,
    NIGHT_CORRUPTION_FLUX_WM2,
    NOCTURNAL_SHORTWAVE_CHANNELS,
    RAIN_MANIFEST,
    STATUS_COLUMNS,
    UNGATED_RADIATION_TWINS,
    ArchiveReport,
    blocked_gauge_runs,
    build_five_minute_frame,
    close_net_radiation,
    close_nocturnal_net_radiation,
    mask_impossible_shortwave,
    mask_night_corrupted_days,
    mask_nocturnal_shortwave,
    mask_sentinels,
    months_never_reaching_saturation,
    night_corrupted_days,
    nocturnal_offset_statistics,
    station_elevation_deg,
    unquantised_rain_samples,
    unshaded_diffuse_days,
    verify_frame,
    verify_window,
)
from micrometeorology.sensors.calibration import (
    SHADE_RING_FACTOR_FILE,
    apply_calibrations,
    apply_shade_ring_correction,
    load_calibrations,
    load_sensor_switches,
    load_shade_ring_factors,
    resolve_mapping_windows,
    uncalibrated_mapping_windows,
    unify_sensor_columns,
)
from micrometeorology.sensors.ingestion import (
    apply_physical_limits,
    merge_dat_files,
    values_outside_declared_limits,
)
from micrometeorology.sensors.quality import mask_persistent_runs, mask_step_excursions

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)


class OutputFormat(StrEnum):
    """Encoding of the three published artifacts."""

    parquet = "parquet"
    csv = "csv"


def _write(frame: pd.DataFrame, base: Path, output_format: OutputFormat) -> Path:
    """Write one artifact, defaulting to Parquet.

    The 5-minute frame is ~988k rows by 94 columns: ~1 GB as CSV, with dtypes
    guessed back on read. Parquet keeps the dtypes and the index; CSV stays
    available because a spreadsheet cannot open Parquet.
    """
    if output_format == "csv":
        path = base.with_suffix(".csv")
        atomic_write(path, lambda tmp: frame.to_csv(tmp))
    else:
        path = base.with_suffix(".parquet")
        atomic_write(path, lambda tmp: frame.to_parquet(tmp))
    size_mb = path.stat().st_size / 1024 / 1024
    typer.echo(
        f"  [ok] {path.name}  {len(frame):,} linhas x {len(frame.columns)} colunas  {size_mb:.1f} MB"
    )
    return path


def _largest(counts: dict[str, int], limit: int = 8) -> list[tuple[str, int]]:
    """The *limit* columns with the highest counts, largest first."""
    return sorted(counts.items(), key=lambda item: -item[1])[:limit]


def _echo_report(report: ArchiveReport) -> None:
    status = "OK" if report.ok else "FALHOU"
    typer.echo(
        f"  {report.kind:6s} {report.rows:,} linhas (auditoria: {report.expected_rows:,})"
        f" x {report.columns} colunas | {report.first} .. {report.last}"
        f" | duplicadas={report.duplicated} monotonico={report.monotonic} -> {status}"
    )
    for problem in report.problems:
        typer.echo(f"     ! {problem}")


@app.command()
def run(
    data_dir: Annotated[
        Path, typer.Option("-d", "--data", help="Archive root (never written to).", exists=True)
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output", help="Directory for the built database.")
    ],
    staging_dir: Annotated[
        Path | None,
        typer.Option(
            "--staging",
            help="Scratch dir for clock-repaired copies. Default: `<output>/_staged`.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat, typer.Option("--format", help="`parquet` (default) or `csv`.")
    ] = OutputFormat.parquet,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit non-zero if the merge does not reproduce the audited row counts.",
        ),
    ] = False,
    source_files: Annotated[
        list[Path] | None,
        typer.Option(
            "--source",
            help=(
                "Build the window from these .dat files instead of the historical "
                "manifest. Repeatable. For the rolling monitoring window, where "
                "re-merging ten years to publish seven days is the wrong unit of work."
            ),
        ),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Merge, verify and aggregate the station archive into one database.

    The verification is the point: a manifest entry silently removed, a staging
    repair that stops matching its file, or a reader change that eats a header
    row all shorten the record, and none of them raise on their own. Pass
    `--strict` in any automated run.

    ``--source`` cannot be checked against the audited row counts — it is a few
    days of what the logger is writing now, not the history — so it is checked
    against the invariants that hold for any window instead
    (:func:`~micrometeorology.sensors.archive.verify_window`). Without that,
    the operational mode was the one mode where `--strict` verified nothing.
    """
    setup_logging(log_level)
    settings = get_settings()

    out = ensure_dir(output_dir)
    staging = staging_dir or (out / "_staged")

    typer.echo("Merge do acervo (manifesto explicito):")
    if source_files:
        # The logger's live tables, read as they are: no manifest, because the
        # manifest is the HISTORY — every entry unique coverage or a documented
        # repair, and it fails hard on a missing one. A rolling window is a
        # different question, asked of the files the datalogger is writing now.
        lenta_paths = [path for path in source_files if "lenta" in path.name.lower()]
        rain_paths = [path for path in source_files if "rain" in path.name.lower()]
        unmatched = [
            path for path in source_files if path not in lenta_paths and path not in rain_paths
        ]
        if unmatched:
            raise typer.BadParameter(
                "cada --source precisa ter 'lenta' ou 'rain' no nome para ser atribuido a "
                f"uma tabela; sem tabela: {', '.join(path.name for path in unmatched)}",
                param_hint="--source",
            )
        lenta = merge_dat_files(lenta_paths, text_columns=list(STATUS_COLUMNS))
        rain = (
            merge_dat_files(rain_paths, text_columns=list(STATUS_COLUMNS))
            if rain_paths
            else pd.DataFrame(index=lenta.index[:0])
        )
        typer.echo(
            f"Janela a partir de {len(lenta_paths)} arquivo(s) lenta e {len(rain_paths)} "
            f"rain do registrador (sem manifesto, sem verificacao contra a auditoria do acervo)"
        )
        typer.echo("\nVerificacao de forma da janela:")
        reports = [verify_window(lenta, "lenta")]
        if rain_paths:
            reports.append(verify_window(rain, "rain"))
        for report in reports:
            _echo_report(report)
    else:
        lenta = build_five_minute_frame(LENTA_MANIFEST, data_dir, staging / "lenta")
        rain = build_five_minute_frame(RAIN_MANIFEST, data_dir, staging / "rain")

        typer.echo("\nVerificacao contra a auditoria do acervo:")
        reports = [verify_frame(lenta, "lenta"), verify_frame(rain, "rain")]
        for report in reports:
            _echo_report(report)

    # The rain logger is a separate table on the same 5-minute grid: JOINED, not
    # concatenated, which would double the index.
    rain_columns = [column for column in rain.columns if column not in lenta.columns]
    raw = lenta.join(rain[rain_columns], how="outer")
    period = {"start": str(raw.index.min()), "end": str(raw.index.max())}
    five_minute_rows = len(raw)
    typer.echo(f"\nBase unificada de 5 min: {len(raw):,} linhas x {len(raw.columns)} colunas")

    failed = [problem for report in reports for problem in report.problems]
    if failed:
        typer.echo(f"\n! {len(failed)} divergencia(s) em relacao a auditoria do acervo")
        if strict:
            raise typer.Exit(code=1)
    elif source_files:
        typer.echo("\n>> Janela bem formada: indice crescente, sem carimbo repetido")
    else:
        typer.echo("\n>> Merge confere com a auditoria: nenhuma linha perdida ou duplicada")

    # --strict sai UMA vez, no fim, depois de todos os artefatos e do manifesto.
    # Sair no meio deixava em disco só station_5min_raw, sem o frame com QC, sem o
    # horário e sem o relatório — justamente o que o operador precisa para
    # diagnosticar a reprovação.
    blocking: list[str] = []

    typer.echo("\nArtefatos:")
    _write(raw, out / "station_5min_raw", output_format)

    # Aliased, not copied: every stage below mutates in place and returns the same
    # frame, and the three scalars the manifest needs were captured above.
    qc = raw
    qc, sentinels_removed = mask_sentinels(qc)
    typer.echo(
        f"\nSentinelas mascaradas: {sum(sentinels_removed.values()):,} amostras "
        f"em {len(sentinels_removed)} colunas"
    )
    for column, count in _largest(sentinels_removed):
        typer.echo(f"  {column:20s} {count:,}")

    # INVALID_WINDOWS is hand-curated and goes stale silently when the shade ring
    # comes off: the column keeps its name while publishing global irradiance as
    # diffuse. Re-running the criterion reports whatever the table misses.
    unshaded = unshaded_diffuse_days(qc)
    if unshaded:
        typer.echo(
            f"\n  ! {len(unshaded)} dia(s) com difusa/global de céu limpo acima de "
            f"{DIFFUSE_RATIO_LIMIT:.2f} fora de INVALID_WINDOWS (anel de sombreamento suspeito):"
        )
        for day, ratio in unshaded[:8]:
            typer.echo(f"    {day}  razao {ratio:.2f}")
        blocking.append(
            f"{len(unshaded)} dia(s) de difusa sem sombreamento fora de INVALID_WINDOWS"
        )

    # A blocked funnel and a dry spell are the same run of zeros to every gate
    # here; only the length parts them, and the curated window that covers the
    # one known episode ages silently the next time it clogs.
    blocked = blocked_gauge_runs(qc)
    for column, first, last, days in blocked:
        typer.echo(f"\n  ! {column} sem chuva por {days} dias ({first} a {last}): funil suspeito")
    unquantised = unquantised_rain_samples(qc)
    for column, count in unquantised.items():
        typer.echo(f"  ! {column}: {count} total(is) fora da grade da bascula")

    # Every stage below is conditional; pre-declared so the report can say "this
    # stage removed nothing" rather than omit the key.
    limits_fired: dict[str, int] = {}
    limits_absent_columns: list[str] = []
    gaps: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    net_gained = net_dropped = 0
    step_excursions_removed = persistence_runs_removed = 0
    shade_ring_corrected: dict[str, int] = {}
    invalidated: dict[str, int] = {}
    outside_after_calibration: dict[str, int] = {}

    if settings.sensor_limits:
        # apply_physical_limits skips a limit naming an absent column in silence,
        # so counting which gates actually FIRED is what proves the rule ran.
        before = qc.notna().sum()
        limits_absent_columns = [
            limit.column for limit in settings.sensor_limits if limit.column not in qc.columns
        ]
        qc = apply_physical_limits(qc, settings.sensor_limits)
        cut = before - qc.notna().sum()
        limits_fired = {str(column): int(count) for column, count in cut.items() if int(count) > 0}
        typer.echo(
            f"\nLimites fisicos: {len(settings.sensor_limits)} declarados, "
            f"{len(limits_fired)} dispararam, {sum(limits_fired.values()):,} amostras removidas"
        )
        for column, count in _largest(limits_fired):
            typer.echo(f"  {column:20s} {count:,}")
        if limits_absent_columns:
            typer.echo(
                f"  ! {len(limits_absent_columns)} limite(s) nomeiam coluna ausente: "
                f"{', '.join(limits_absent_columns[:6])}"
            )
            blocking.append(f"{len(limits_absent_columns)} limite(s) nomeiam coluna ausente")
    else:
        # The gate is fail-open by construction: no declared limit means no
        # sample is ever refused, and the run publishes an ungated archive with
        # exit 0. Silence there is indistinguishable from "every sample was
        # inside its bounds".
        typer.echo("\n  ! nenhum limite fisico declarado: o portao de faixa nao rodou")
        blocking.append("nenhum limite fisico declarado")
    calibrations_path = settings.configs_dir / "calibrations.yaml"
    sources: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    if calibrations_path.is_file():
        calibrations = load_calibrations(calibrations_path)
        before_calibration = qc.notna().sum()
        qc = apply_calibrations(qc, calibrations)
        # A `factor: null` record NaNs its whole window, which is a removal like
        # any other and belongs in the tally.
        invalidated = {
            str(column): int(count)
            for column, count in (before_calibration - qc.notna().sum()).items()
            if int(count) > 0
        }
        switches = load_sensor_switches(calibrations_path)
        # Resolved once, before the copy: the windows are read off the RAW columns
        # and the frame's own bounds, neither of which unification changes.
        sources = resolve_mapping_windows(
            qc, switches, (*NIGHT_CORRUPTION_CHANNELS, *NOCTURNAL_SHORTWAVE_CHANNELS)
        )
        qc, shade_ring_corrected = apply_shade_ring_correction(
            qc,
            load_shade_ring_factors(data_dir / SHADE_RING_FACTOR_FILE),
            sources.get("Sw_dif", ()),
        )
        if shade_ring_corrected:
            typer.echo(
                f"\nCorrecao do anel de sombreamento aplicada a "
                f"{sum(shade_ring_corrected.values()):,} amostras de difusa "
                f"em {len(shade_ring_corrected)} coluna(s)"
            )

        # Statistical QC runs HERE, after calibration and before unification.
        # After calibration because the thresholds are in physical units, the same
        # reason the second apply_physical_limits pass exists; before unification
        # because that step COPIES, so masking the raw alias reaches the unified
        # channel for free. Per raw alias also means per instrument, so no era
        # boundary can manufacture a step across a sensor swap.
        qc, step_excursions_removed = mask_step_excursions(qc, settings.sensor_step_limits)
        qc, persistence_runs_removed = mask_persistent_runs(qc, settings.sensor_persistence_limits)

        # The gate runs twice: the first pass on the RAW signal keeps the
        # calibration from scaling a never-physical value, the second because a
        # value sitting AT the boundary crosses it once scaled — as the Eppley
        # PSP factor declared in this same config does.
        # Before unify_sensor_columns: sensor_limits names raw columns only, so a
        # sample rejected after the copy stays published under the unified name.
        outside_after_calibration = values_outside_declared_limits(qc, settings.sensor_limits)
        if outside_after_calibration:
            qc = apply_physical_limits(qc, settings.sensor_limits)
            typer.echo(
                f"\nLimites reaplicados apos calibracao: "
                f"{sum(outside_after_calibration.values()):,} amostra(s) "
                f"em {len(outside_after_calibration)} coluna(s) cruzaram o portao ao serem escaladas"
            )
            for column, count in _largest(outside_after_calibration):
                typer.echo(f"  {column:22s} {count:,}")

        # Without this the sensor_switches block parses and does nothing, and
        # every era-spanning variable has to be reassembled by hand downstream.
        qc = unify_sensor_columns(qc, switches)
        # Unification COPIES: each unified channel keeps a raw twin under the
        # logger's own name, and the masks below must reach both or the artifact
        # publishes the rejected value under the other name.

        qc, net_gained, net_dropped = close_net_radiation(qc)
        if net_gained or net_dropped:
            typer.echo(
                f"\nSaldo recomposto dos quatro componentes: +{net_gained:,} amostras "
                f"(componentes sem saldo do registrador), -{net_dropped:,} (componente ausente)"
            )
        # A window where one component is dead leaves Net_CNR1 entirely absent,
        # and export_monitoring OMITS an all-null series by design: the balance
        # chart then disappears from the published page with no error anywhere.
        # The gate is emptiness, not a share: the historical build legitimately
        # drops the pre-net-channel era, and no fraction of it is a magic number.
        # It cannot key on net_dropped: the logger computes ITS net from the same
        # four channels, so a dead component empties that column too and the
        # recomposition has nothing to drop (dropped counts only the 125
        # historical rows where a net existed without its components).
        components_present = [
            column
            for column in NET_RADIATION_COMPONENTS
            if column in qc.columns and qc[column].notna().any()
        ]
        if components_present and "Net_CNR1" in qc.columns and not qc["Net_CNR1"].notna().any():
            blocking.append(
                "saldo recomposto ficou inteiramente ausente enquanto "
                f"{', '.join(components_present)} carrega(m) dado: um componente do balanco "
                "esta morto nesta janela"
            )

        # Reported, never fatal: the sensitivity is a laboratory decision, and
        # failing the build would trade a scaling error for no record.
        gaps = uncalibrated_mapping_windows(qc, calibrations, switches)
        if gaps:
            typer.echo(
                f"\n  ! {len(gaps)} janela(s) alimentam uma serie unificada sem nenhuma "
                "calibracao declarada (degrau artificial na fronteira):"
            )
            for unified_name, column, start, end in gaps[:8]:
                typer.echo(f"    {unified_name:9s} <- {column:16s} {start.date()} .. {end.date()}")
    else:
        # Without the file no instrument factor is applied and no era-spanning
        # column is unified: the archive publishes raw logger counts under the
        # names the unified channels would have had, and every consumer reads
        # them as calibrated. Exit 0 there says the build is sound.
        typer.echo(
            f"  ! sem calibracoes em {calibrations_path}: exportando sem correcao nem unificacao"
        )
        blocking.append(f"sem calibracoes em {calibrations_path}")

    # After unification: the 2017 episodes exist only in the unified column.
    # Radiation in deep night is a slipped clock, not a sky, and the flat gates
    # of default.yaml pass 1313 W/m2 at 04h — so the day is removed whole from
    # the two channels it invalidates.
    # Computed once for the four stages below: a second call is a second
    # definition of "night", free to drift from the first without failing.
    elevation = station_elevation_deg(pd.DatetimeIndex(qc.index))
    corrupted = night_corrupted_days(qc, elevation_deg=elevation)
    qc, night_masked = mask_night_corrupted_days(qc, corrupted, sources)
    # The order of the four stages below is load-bearing and every way of getting
    # it wrong fails silently; test_the_pipeline_calls_its_radiation_stages_in_the
    # _load_bearing_order pins it and docs/quality-control.md explains it.
    offsets = nocturnal_offset_statistics(qc, elevation_deg=elevation)
    qc, impossible = mask_impossible_shortwave(qc, sources)
    qc, nocturnal = mask_nocturnal_shortwave(qc, sources, elevation_deg=elevation)
    qc, nocturnal_net = close_nocturnal_net_radiation(qc, elevation_deg=elevation)
    if nocturnal_net:
        typer.echo(f"Saldo noturno recomposto so da onda longa: {nocturnal_net:,} amostras")
    typer.echo(
        f"\nDias com carimbo de tempo corrompido: {len(corrupted)} "
        f"({sum(night_masked.values()):,} amostras mascaradas em {len(night_masked)} colunas)"
    )
    if impossible:
        typer.echo(
            f"Irradiancia impossivel para a posicao do sol (limite BSRN): "
            f"{sum(impossible.values()):,} amostras em {len(impossible)} colunas"
        )
    if nocturnal:
        typer.echo(
            f"Onda curta com o sol abaixo do horizonte (offset termico, nao fluxo): "
            f"{sum(nocturnal.values()):,} amostras em {len(nocturnal)} colunas"
        )
    for column, statistic in offsets.items():
        if statistic.median_wm2 is not None:
            typer.echo(
                f"  offset noturno {column:8s} mediana {statistic.median_wm2:+7.3f} W/m2 "
                f"({statistic.night_samples:,} amostras)"
            )
    alarms = [
        (column, month, median)
        for column, statistic in offsets.items()
        for month, median in statistic.drift_alarms
    ]
    # Reportados e arquivados em nocturnal_offset_monitor, nunca fatais: são
    # episódios datados do próprio registro, permanentes, e reprovar a construção
    # por eles faria --strict nunca passar — o portão que grita sempre não verifica.
    for column, month, median in alarms:
        typer.echo(f"  ! deriva de offset em {column} ({month}): mediana {median:+.3f} W/m2")
    for day, count in sorted(corrupted, key=lambda item: -item[1])[:8]:
        typer.echo(
            f"  {day}  {count} amostra(s) acima de {NIGHT_CORRUPTION_FLUX_WM2:.0f} W/m2 de madrugada"
        )

    # AFTER unification, deliberately: the one channel this matters most for is
    # the unified ``ur``, which does not exist before it. The fault family is the
    # one every other gate here is blind to by construction — a hygrometer reading
    # a steady offset low never repeats, never steps and never leaves its range,
    # so saturation is the only external anchor that reaches it.
    unsaturated = months_never_reaching_saturation(qc)
    for column, month, peak in unsaturated[-6:]:
        typer.echo(f"  ! {column} nunca passou de {peak:.1f} %UR em {month}: vies seco suspeito")

    _write(qc, out / "station_5min_qc", output_format)

    # The logger's quality flag is text, which no hourly mean can carry; the
    # fraction of samples reading OK keeps it in the hourly frame. It must be
    # ``float64``: ``aggregate_to_hourly`` keeps only
    # ``select_dtypes(include="number")`` columns, which matches neither bool nor
    # the object dtype a null introduces. ``qc_flag`` is the unified spelling.
    # An hourly mean of a monotonic counter is not a record number.
    discarded = [
        column
        for column in qc.columns
        if column in {"RECORD", *UNGATED_RADIATION_TWINS}
        or str(column).startswith("rtime")
        or str(column).endswith("_mv_Avg")
    ]
    hourly_input = qc.drop(columns=discarded)
    for column in (*STATUS_COLUMNS, "qc_flag"):
        if column in hourly_input.columns:
            flag = hourly_input.pop(column)
            hourly_input[f"{column}_ok_fraction"] = (
                flag.eq("OK").astype("float64").mask(flag.isna())
            )

    hourly = aggregate_to_hourly(
        hourly_input,
        min_samples=settings.sensor_min_samples_per_hour,
        sum_columns=settings.sensor_sum_columns,
        wind_dir_columns=settings.sensor_wind_dir_columns,
        wind_speed_column_map=settings.sensor_wind_speed_column_map,
    )
    _write(hourly, out / "station_hourly", output_format)

    manifest = {
        "format": "labmim-station-archive-v1",
        "period": period,
        "five_minute_rows": five_minute_rows,
        "hourly_rows": len(hourly),
        "min_samples_per_hour": settings.sensor_min_samples_per_hour,
        "verification": [
            {
                "kind": report.kind,
                "rows": report.rows,
                "expected_rows": report.expected_rows,
                "columns": report.columns,
                "duplicated": report.duplicated,
                "monotonic": report.monotonic,
                "problems": list(report.problems),
            }
            for report in reports
        ],
        # Every stage that removes a sample reports here. The tallies close the
        # raw-to-QC accounting NET of what unification and net recomposition ADD:
        # raw + copied cells + recomposed - removed = qc, residue zero. The naive
        # raw-minus-qc delta is negative, because unify_sensor_columns copies about
        # 12.9 million cells into the canonical channels.
        "sentinels_removed": sentinels_removed,
        "physical_limits_removed": limits_fired,
        "physical_limits_absent_columns": limits_absent_columns,
        "physical_limits_after_calibration": outside_after_calibration,
        "calibration_invalidated": invalidated,
        "shade_ring_corrected": shade_ring_corrected,
        "blocked_gauge_runs": [
            {"column": column, "first": first, "last": last, "days": days}
            for column, first, last, days in blocked
        ],
        "unquantised_rain_samples": unquantised,
        "months_never_reaching_saturation": [
            {"column": column, "month": month, "peak": peak} for column, month, peak in unsaturated
        ],
        "step_excursions_removed": step_excursions_removed,
        "persistence_runs_removed": persistence_runs_removed,
        "impossible_shortwave_removed": impossible,
        "nocturnal_shortwave_masked": nocturnal,
        "nocturnal_offset_monitor": {
            column: statistic.as_report() for column, statistic in offsets.items()
        },
        # Dated, so the episode stays auditable after the samples are gone.
        "timestamp_corrupted_days": [day for day, _count in corrupted],
        "timestamp_corruption_masked": night_masked,
        "uncalibrated_mapping_windows": [
            {"unified": unified, "column": column, "start": str(start), "end": str(end)}
            for unified, column, start, end in gaps
        ],
        "net_radiation_recomposed": {
            "gained": net_gained,
            "dropped": net_dropped,
            "nocturnal_longwave_only": nocturnal_net,
        },
    }
    report_path = out / "archive_report.json"
    atomic_write_strict_json(report_path, manifest)
    typer.echo(f"  [ok] {report_path.name}")

    if blocking:
        typer.echo(f"\n! {len(blocking)} verificacao(oes) reprovaram:")
        for problem in blocking:
            typer.echo(f"    {problem}")
        if strict:
            raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-archive``)."""
    app()


if __name__ == "__main__":
    main()
