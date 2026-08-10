"""CLI: build the unified LabMiM station database from the raw ``.dat`` archive.

Produces three artifacts from one pass over the archive, and **verifies the
merge against the audited row counts** before writing anything:

``station_5min_raw``
    Every 5-minute sample the station ever recorded, merged from the explicit
    manifest in :mod:`micrometeorology.sensors.archive`, with only the three
    clock repairs applied. Values exactly as the logger wrote them, sentinels
    included — this is the archive, not an interpretation of it.

``station_5min_qc``
    The same grid after sentinel masking, physical gates, instrument
    calibrations and era-to-era column unification. This is the frame the
    hourly means are computed from, written out so the chain from raw sample to
    published statistic can be audited at its midpoint.

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

import json
import logging
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from micrometeorology.common.config import get_settings
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.archive import (
    DIFFUSE_RATIO_LIMIT,
    LENTA_MANIFEST,
    NIGHT_CORRUPTION_CHANNELS,
    NIGHT_CORRUPTION_FLUX_WM2,
    RAIN_MANIFEST,
    STATUS_COLUMNS,
    ArchiveReport,
    build_five_minute_frame,
    close_net_radiation,
    mask_impossible_shortwave,
    mask_night_corrupted_days,
    mask_sentinels,
    night_corrupted_days,
    unshaded_diffuse_days,
    verify_frame,
)
from micrometeorology.sensors.calibration import (
    apply_calibrations,
    load_calibrations,
    load_sensor_switches,
    resolve_mapping_windows,
    uncalibrated_mapping_windows,
    unify_sensor_columns,
)
from micrometeorology.sensors.ingestion import (
    apply_physical_limits,
    values_outside_declared_limits,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)


def _write(frame: pd.DataFrame, base: Path, output_format: str) -> Path:
    """Write one artifact, defaulting to Parquet.

    The 5-minute frame is ~988k rows by 94 columns: as CSV that is roughly a
    gigabyte of text whose dtypes have to be guessed back on read, while Parquet
    keeps the dtypes and the index and is an order of magnitude smaller. CSV
    stays available because a spreadsheet cannot open Parquet.
    """
    if output_format == "csv":
        path = base.with_suffix(".csv")
        frame.to_csv(path, float_format="%.6g")
    else:
        path = base.with_suffix(".parquet")
        frame.to_parquet(path)
    size_mb = path.stat().st_size / 1024 / 1024
    typer.echo(
        f"  [ok] {path.name}  {len(frame):,} linhas x {len(frame.columns)} colunas  {size_mb:.1f} MB"
    )
    return path


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
            "--staging", help="Scratch dir for clock-repaired copies. Default: <output>/_staged."
        ),
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", help="`parquet` (default) or `csv`.")
    ] = "parquet",
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit non-zero if the merge does not reproduce the audited row counts.",
        ),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Merge, verify and aggregate the station archive into one database.

    The verification is the point: a manifest entry silently removed, a staging
    repair that stops matching its file, or a reader change that eats a header
    row all shorten the record, and none of them raise on their own. Pass
    `--strict` in any automated run.
    """
    setup_logging(log_level)
    settings = get_settings()
    if output_format not in {"parquet", "csv"}:
        raise typer.BadParameter("--format must be 'parquet' or 'csv'")

    out = ensure_dir(output_dir)
    staging = staging_dir or (out / "_staged")

    typer.echo("Merge do acervo (manifesto explicito):")
    lenta = build_five_minute_frame(LENTA_MANIFEST, data_dir, Path(staging) / "lenta")
    rain = build_five_minute_frame(RAIN_MANIFEST, data_dir, Path(staging) / "rain")

    typer.echo("\nVerificacao contra a auditoria do acervo:")
    reports = [verify_frame(lenta, "lenta"), verify_frame(rain, "rain")]
    for report in reports:
        _echo_report(report)

    # The rain logger is a separate table on the same 5-minute grid, so it is
    # JOINED rather than concatenated: a concat would interleave rows whose
    # column sets barely overlap and double the index.
    rain_columns = [column for column in rain.columns if column not in lenta.columns]
    raw = lenta.join(rain[rain_columns], how="outer")
    typer.echo(f"\nBase unificada de 5 min: {len(raw):,} linhas x {len(raw.columns)} colunas")

    typer.echo("\nArtefatos:")
    _write(raw, out / "station_5min_raw", output_format)

    qc = raw.copy()
    qc, sentinels_removed = mask_sentinels(qc)
    typer.echo(
        f"\nSentinelas mascaradas: {sum(sentinels_removed.values()):,} amostras "
        f"em {len(sentinels_removed)} colunas"
    )
    for column, count in sorted(sentinels_removed.items(), key=lambda item: -item[1])[:8]:
        typer.echo(f"  {column:20s} {count:,}")

    # INVALID_WINDOWS is hand-curated, so it goes stale the moment the shade ring
    # comes off — and silently, because the column keeps its name while
    # publishing global irradiance as diffuse. Re-running the detection criterion
    # on the masked frame reports whatever the table misses.
    unshaded = unshaded_diffuse_days(qc)
    if unshaded:
        typer.echo(
            f"\n  ! {len(unshaded)} dia(s) com difusa/global de céu limpo acima de "
            f"{DIFFUSE_RATIO_LIMIT:.2f} fora de INVALID_WINDOWS (anel de sombreamento suspeito):"
        )
        for day, ratio in unshaded[:8]:
            typer.echo(f"    {day}  razao {ratio:.2f}")
        if strict:
            raise typer.Exit(code=1)

    # Declared before the branches that fill them: every one is conditional, and
    # the report has to be able to say "this stage removed nothing" rather than
    # omit the key.
    limits_fired: dict[str, int] = {}
    limits_absent_columns: list[str] = []
    gaps: list[tuple[str, str, pd.Timestamp, pd.Timestamp]] = []
    net_gained = net_dropped = 0
    invalidated: dict[str, int] = {}

    if settings.sensor_limits:
        # Report which gates actually FIRED before applying them: a limit naming
        # a column the frame does not carry is skipped in silence by
        # apply_physical_limits, so counting here is what turns "the YAML parses"
        # into "the rule ran".
        before = qc.notna().sum()
        limits_absent_columns = [
            limit["column"] for limit in settings.sensor_limits if limit["column"] not in qc.columns
        ]
        qc = apply_physical_limits(qc, settings.sensor_limits)
        cut = before - qc.notna().sum()
        limits_fired = {str(column): int(count) for column, count in cut.items() if int(count) > 0}
        typer.echo(
            f"\nLimites fisicos: {len(settings.sensor_limits)} declarados, "
            f"{len(limits_fired)} dispararam, {sum(limits_fired.values()):,} amostras removidas"
        )
        for column, count in sorted(limits_fired.items(), key=lambda item: -item[1])[:8]:
            typer.echo(f"  {column:20s} {count:,}")
        if limits_absent_columns:
            typer.echo(
                f"  ! {len(limits_absent_columns)} limite(s) nomeiam coluna ausente: "
                f"{', '.join(limits_absent_columns[:6])}"
            )
            if strict:
                raise typer.Exit(code=1)
    calibrations_path = settings.configs_dir / "calibrations.yaml"
    sources: dict[str, list[tuple[str, pd.Timestamp, pd.Timestamp]]] = {}
    if calibrations_path.is_file():
        before_calibration = qc.notna().sum()
        qc = apply_calibrations(qc, load_calibrations(calibrations_path))
        # A `factor: null` record NaNs its whole window, which is a removal like
        # any other and belongs in the tally.
        invalidated = {
            str(column): int(count)
            for column, count in (before_calibration - qc.notna().sum()).items()
            if int(count) > 0
        }
        # Without this the sensor_switches block parses and does nothing, and
        # every era-spanning variable has to be reassembled by hand downstream.
        switches = load_sensor_switches(calibrations_path)
        qc = unify_sensor_columns(qc, switches)
        # Unification COPIES, so each unified channel now has a raw twin holding
        # the same values under the logger's own name. The masks below have to
        # reach both, or the artifact publishes the rejected value under the
        # other name and the two disagree inside one file.
        sources = resolve_mapping_windows(qc, switches, NIGHT_CORRUPTION_CHANNELS)

        qc, net_gained, net_dropped = close_net_radiation(qc)
        if net_gained or net_dropped:
            typer.echo(
                f"\nSaldo recomposto dos quatro componentes: +{net_gained:,} amostras "
                f"(componentes sem saldo do registrador), -{net_dropped:,} (componente ausente)"
            )

        # Reported, never fatal: which sensitivity applied is a laboratory
        # decision, and refusing to build over it would trade a scaling error
        # for no record.
        gaps = uncalibrated_mapping_windows(qc, load_calibrations(calibrations_path), switches)
        if gaps:
            typer.echo(
                f"\n  ! {len(gaps)} janela(s) alimentam uma serie unificada sem nenhuma "
                "calibracao declarada (degrau artificial na fronteira):"
            )
            for unified_name, column, start, end in gaps[:8]:
                typer.echo(f"    {unified_name:9s} <- {column:16s} {start.date()} .. {end.date()}")
    else:
        typer.echo(
            f"  ! sem calibracoes em {calibrations_path}: exportando sem correcao nem unificacao"
        )

    # After the unification, because the 2017 episodes only exist in the unified
    # column. Radiation recorded in deep night is a clock that slipped, not a
    # sky: the day is removed whole from the two channels it invalidates, since
    # the flat gates of default.yaml pass 1313 W/m2 at 04h without blinking and
    # every statistic downstream then reports the fault as climate.
    corrupted = night_corrupted_days(qc)
    qc, night_masked = mask_night_corrupted_days(qc, corrupted, sources)
    qc, impossible = mask_impossible_shortwave(qc, sources)
    typer.echo(
        f"\nDias com carimbo de tempo corrompido: {len(corrupted)} "
        f"({sum(night_masked.values()):,} amostras mascaradas em {len(night_masked)} colunas)"
    )
    if impossible:
        typer.echo(
            f"Irradiancia impossivel para a posicao do sol (limite BSRN): "
            f"{sum(impossible.values()):,} amostras em {len(impossible)} colunas"
        )
    for day, count in sorted(corrupted, key=lambda item: -item[1])[:8]:
        typer.echo(
            f"  {day}  {count} amostra(s) acima de {NIGHT_CORRUPTION_FLUX_WM2:.0f} W/m2 de madrugada"
        )

    # The gate runs twice. The first pass runs on the RAW signal, which is what
    # protects the calibration from multiplying a value that was never physical;
    # the second is what makes the published column obey its own declared range,
    # because a value sitting AT the boundary crosses it once scaled — the
    # Eppley PSP factor this same config declares does exactly that.
    outside_after_calibration = values_outside_declared_limits(qc, settings.sensor_limits)
    if outside_after_calibration:
        qc = apply_physical_limits(qc, settings.sensor_limits)
        typer.echo(
            f"\nLimites reaplicados apos calibracao: "
            f"{sum(outside_after_calibration.values()):,} amostra(s) "
            f"em {len(outside_after_calibration)} coluna(s) cruzaram o portao ao serem escaladas"
        )
        for column, count in sorted(outside_after_calibration.items(), key=lambda item: -item[1])[
            :8
        ]:
            typer.echo(f"  {column:22s} {count:,}")

    _write(qc, out / "station_5min_qc", output_format)

    # The logger's quality flag is text, which no hourly mean can carry. Turning
    # it into the fraction of samples reading OK keeps the information in the
    # hourly frame instead of dropping the only per-row flag the record has.
    # ``astype("float64")`` is what makes the fraction survive:
    # ``aggregate_to_hourly`` keeps only ``select_dtypes(include="number")``
    # columns, which matches neither bool nor the object dtype ``.where``
    # produces once a null appears. ``qc_flag`` is the unified spelling of the
    # same information and goes the same way.
    hourly_input = qc.copy()
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
        "period": {"start": str(raw.index.min()), "end": str(raw.index.max())},
        "five_minute_rows": len(raw),
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
        # Every stage that removes a sample reports here, so these tallies sum
        # to the raw-to-QC delta rather than living only in a console log.
        "sentinels_removed": sentinels_removed,
        "physical_limits_removed": limits_fired,
        "physical_limits_absent_columns": limits_absent_columns,
        "physical_limits_after_calibration": outside_after_calibration,
        "calibration_invalidated": invalidated,
        "impossible_shortwave_removed": impossible,
        # Dated, so the episode stays auditable after the samples are gone.
        "timestamp_corrupted_days": [day for day, _count in corrupted],
        "timestamp_corruption_masked": night_masked,
        "uncalibrated_mapping_windows": [
            {"unified": unified, "column": column, "start": str(start), "end": str(end)}
            for unified, column, start, end in gaps
        ],
        "net_radiation_recomposed": {"gained": net_gained, "dropped": net_dropped},
    }
    report_path = out / "archive_report.json"
    report_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(f"  [ok] {report_path.name}")

    failed = [problem for report in reports for problem in report.problems]
    if failed:
        typer.echo(f"\n! {len(failed)} divergencia(s) em relacao a auditoria do acervo")
        if strict:
            raise typer.Exit(code=1)
    else:
        typer.echo("\n>> Merge confere com a auditoria: nenhuma linha perdida ou duplicada")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-archive``)."""
    app()


if __name__ == "__main__":
    main()
