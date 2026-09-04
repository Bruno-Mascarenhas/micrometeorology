"""CLI: publish the rolling-window payload the interactive monitoring page reads.

Writes ``labmim-monitoring-v1`` JSON: one document carrying, for every chart,
the raw station samples, their hourly means and the WRF series where one exists.
The page draws whatever layers the document contains.

This is the **interactive** producer. ``labmim-site-graphs``
(:mod:`micrometeorology.cli.plot_station_graphs`) still writes the nine
fixed-name PNGs in the same three-layer style, because those go into papers.

Sizing, measured on the reference week: raw ~110 kB, hourly ~9 kB, WRF ~7 kB —
~133 kB in all, about a third of the ~380 kB the nine PNGs cost the visitor.

Like the climatology artifacts, the output is **not** committed to the site
repository: it derives from the laboratory's own sensor archive, so it is
gitignored there and attached by the hourly deploy.

Examples
--------
Publish the last seven days straight into a checkout of the site::

    labmim-monitoring -i output/archive -o ../site-labmim/site/Monitoramento \\
        -w data/series/labmim_series_operacional.dat

Reproduce the reference test window (station and model both essentially
complete)::

    labmim-monitoring -i output/archive -o out/ -w data/series/labmim_series_operacional.dat \\
        --end 2022-07-08
"""

import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from micrometeorology.common.git import short_commit
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.site_json import rounded_list, write_json
from micrometeorology.common.timeparse import parse_naive_timestamp
from micrometeorology.sensors.monitoring import MONITORING_CHARTS, resolve_wrf_column

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

PAYLOAD_FORMAT = "labmim-monitoring-v1"
PAYLOAD_FILENAME = "monitoring.json"

# Rolling window the page shows. Seven days is the legacy contract of
# labmim-site-graphs, kept so the two products stay comparable.
DEFAULT_DAYS = 7

# Nominal cadence of each layer, declared rather than inferred: the axis is
# published as an origin plus a step, so a layer has to sit on a complete grid
# before it is serialised. These are the grids ``labmim-archive`` writes —
# ``station_5min_qc`` at five minutes, ``station_hourly`` and the WRF series hourly.
RAW_CADENCE = "5min"
HOURLY_CADENCE = "h"

# Decimals per unit, matching what the page renders (``unitDigits``) so the CSV
# download never carries a digit its chart does not show. Pressure keeps one to
# stay readable at ~1013 hPa; precipitation needs three so the 0.254 mm
# tipping-bucket quantum survives the round trip; irradiance is integer because a
# tenth of a W/m2 is noise.
_DECIMALS = {
    "°C": 1,
    "%": 1,
    "hPa": 1,
    "m/s": 1,
    "°": 1,
    "mm": 3,
    "W/m²": 0,
}

STATION_NAME = "Estação Micrometeorológica LabMiM"


def _round(values: pd.Series, decimals: int) -> list[float | int | None]:
    """Serialise a series, mapping every non-finite sample to ``null``.

    NaN is not valid JSON and the writer refuses it outright; ``null`` is also
    the cheaper encoding.

    A zero-decimal unit is emitted as an ``int``, not a rounded float: ``round(x,
    0)`` returns ``489.0``, which JSON writes as ``"489.0"`` — LONGER than the
    ``"489.5"`` it was meant to shorten — and turns the pyranometer's small
    negative night offsets into ``-0.0``. Mixing int and float across charts is
    safe: the page reads either as a number.
    """
    if decimals <= 0:
        return [
            None if not np.isfinite(value) else round(float(value))
            for value in values.to_numpy(dtype=float)
        ]
    return rounded_list(values.to_numpy(dtype=float).tolist(), decimals)


def _regular(
    frame: pd.DataFrame, index: pd.DatetimeIndex, cadence: str
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Put a layer back on its complete cadence grid before it is serialised.

    The axis below is an origin plus a step, so the page rebuilds every abscissa
    as ``start + i * step``. Neither frame that reaches it is gapless — the logger
    stops (247 non-5-minute steps over the record, the longest 17 days) and
    ``read_wrf_series`` drops the spin-up hour of every day — so serialising the
    rows as they come would slide every sample after a hole earlier than it
    happened. Reindexing turns each hole into a row of ``null`` instead.
    """
    if frame.empty:
        return frame, index
    # Before `reindex`, which raises a raw pandas ValueError on a duplicated
    # label — "cannot reindex on an axis with duplicate labels" — that reaches
    # the operator as a traceback with no mention of the file or the cadence.
    # The axis guard below cannot see it either: it reads the deltas, and a
    # repeated stamp is a delta of zero among many, not a second cadence.
    duplicated = index[index.duplicated()]
    if len(duplicated):
        raise ValueError(
            f"{len(duplicated)} duplicated timestamp(s) on the {cadence} layer "
            f"(first: {duplicated[0]}); the merge upstream must resolve them"
        )
    grid = pd.date_range(index.min(), index.max(), freq=cadence, name=index.name)
    off_grid = index.difference(grid)
    if not off_grid.empty:
        raise ValueError(
            f"{len(off_grid)} sample(s) do not sit on the {cadence} grid "
            f"(first: {off_grid[0]}); reindexing would silently drop them"
        )
    return frame.reindex(grid), grid


def _axis(index: pd.DatetimeIndex) -> dict[str, object]:
    """Publish the time axis as start + step instead of one stamp per sample.

    2,016 ISO-8601 strings cost about 50 kB on their own; a regular grid needs
    only its origin and its cadence, with gaps carried as ``null`` values. That
    holds only while the index really is a complete grid, so an irregular one
    raises here rather than being drawn at the wrong times.
    """
    deltas = (index[1:] - index[:-1]).unique() if len(index) > 1 else pd.TimedeltaIndex([])
    if len(deltas) > 1:
        raise ValueError(
            f"a monitoring axis needs one cadence, got {len(deltas)}: "
            f"{', '.join(str(delta) for delta in sorted(deltas)[:4])}"
        )
    step = round(deltas[0].total_seconds() / 60) if len(deltas) else 0
    return {"start": index[0].isoformat(sep=" "), "step_minutes": step, "count": len(index)}


def _layer(
    frame: pd.DataFrame, columns: dict[str, str], decimals: int, cadence: str
) -> dict[str, object] | None:
    """One layer of one chart: its time axis plus a value array per series.

    A series present but all-null over the window is OMITTED rather than
    published as an array of ``null``: the page treats a missing
    ``layer.series[id]`` as "nothing for this series", while an all-null array
    passes that guard and legends a line the reader cannot see.

    Live case: the Gill thermohygrometer railed in December 2025 and its readings
    are masked as sentinels, so the default operational window (no ``--end``)
    carries no air temperature and no humidity at all.
    """
    present = {sid: column for sid, column in columns.items() if column in frame.columns}
    if frame.empty or not present:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"a monitoring layer needs a time index, got {type(index).__name__}")
    gridded, grid = _regular(frame, index, cadence)
    values = {
        sid: _round(gridded[column], decimals)
        for sid, column in present.items()
        if gridded[column].notna().any()
    }
    if not values:
        return None
    return {"axis": _axis(grid), "series": values}


@app.command()
def run(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i", "--input", help="Directory holding the labmim-archive parquet files.", exists=True
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output", help="Site's dataset.paths.monitoring directory.")
    ],
    wrf_path: Annotated[
        Path | None, typer.Option("-w", "--wrf", help="series_operacional.dat for the model layer.")
    ] = None,
    days: Annotated[
        int, typer.Option("--days", min=1, help="Length of the rolling window.")
    ] = DEFAULT_DAYS,
    end: Annotated[
        str | None,
        typer.Option("--end", help="End of the window (YYYY-MM-DD). Default: the newest sample."),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Publish the rolling monitoring window as one JSON document.

    The WRF layer is resolved per series against the columns
    `series_operacional.dat` actually carries, so a variable the extraction
    gains later starts appearing with no change here.
    """
    setup_logging(log_level)
    version = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    raw = pd.read_parquet(Path(input_dir) / "station_5min_qc.parquet")
    hourly = pd.read_parquet(Path(input_dir) / "station_hourly.parquet")

    # The model is read BEFORE the window is fixed, because it may run ahead of
    # the station: `series_operacional.dat` grows forward day by day, so an hour
    # carrying a WRF value and no observation is the normal state, and anchoring
    # the window's end on the newest SAMPLE would clip exactly the part of the
    # model worth looking at.
    model_full = pd.DataFrame()
    if wrf_path is not None:
        from micrometeorology.wrf.operational_record import read_wrf_series

        model_full = read_wrf_series(
            wrf_path,
            consumes=[name for chart in MONITORING_CHARTS for s in chart.series for name in s.wrf],
        )

    # `first` still anchors on the station, so the reader keeps the same seven
    # days of record behind them; only the END reaches forward. `--end` is an
    # explicit instruction and overrides both.
    station_last = pd.Timestamp(raw.index.max())
    if end is not None:
        try:
            last = parse_naive_timestamp(end, "%Y-%m-%d")
        except ValueError as exc:
            raise typer.BadParameter(f"--end must be a YYYY-MM-DD date (got {end!r})") from exc
        first = last - pd.Timedelta(days=days)
    else:
        first = station_last - pd.Timedelta(days=days)
        last = station_last
        if not model_full.empty:
            last = max(last, pd.Timestamp(model_full.index.max()))
    # Raw samples are instantaneous, so both ends are kept. An hourly mean
    # labelled T covers [T, T+1h), so the aggregated layers are right-open —
    # and bounded by the STATION's last sample, not by `last`, which is the
    # model's end whenever the model runs ahead. Trimming against `last` would
    # publish the station's trailing PARTIAL hour as a complete hourly mean.
    raw = raw.loc[first:last]
    hourly = hourly.loc[first : min(last, station_last) - pd.Timedelta(hours=1)]
    # Measured on the SLICED frame, not on the archive: with an explicit `--end`
    # the newest sample in the file can be years after the window, and an anchor
    # outside the window is worse than none.
    station_end = pd.Timestamp(raw.index.max()) if len(raw) else last
    typer.echo(f"Janela: {first} .. {last}  ({len(raw):,} amostras brutas, {len(hourly)} horas)")

    model = pd.DataFrame()
    if not model_full.empty:
        model = model_full.loc[first : last - pd.Timedelta(hours=1)]
        ahead = pd.Timestamp(model_full.index.max()) - station_last
        ahead_note = f", {ahead} à frente da estação" if ahead > pd.Timedelta(0) else ""
        typer.echo(
            f"WRF: {len(model)} horas na janela, "
            f"{len(model.columns)} colunas disponiveis{ahead_note}"
        )

    charts: list[dict[str, object]] = []
    for chart in MONITORING_CHARTS:
        station_columns = {series.id: series.station for series in chart.series}
        if chart.unit not in _DECIMALS:
            # A unit with no entry would silently take two decimals, so a new
            # chart publishes a precision nobody chose — and for W/m2 that is
            # three characters per sample across the whole payload.
            raise typer.BadParameter(
                f"chart {chart.id!r} declares unit {chart.unit!r}, which has no entry in "
                "_DECIMALS; add the precision this unit publishes at"
            )
        decimals = _DECIMALS[chart.unit]

        wrf_columns: dict[str, str] = {}
        missing: list[str] = []
        for series in chart.series:
            column = resolve_wrf_column(series, model.columns) if not model.empty else None
            # A column PRESENT in the header but empty over this window is the
            # designed signal that the extraction stopped writing the variable
            # (export_operational_series empties one column rather than shifting
            # every column after it). Read as "resolved", it produced neither the
            # line — `_layer` drops an all-null series — nor the pending note,
            # which is the one case the note exists for.
            if column and model[column].notna().any():
                wrf_columns[series.id] = column
            # Only when a model was actually loaded: `wrf_pending` claims to the
            # reader that "the extraction does not write this variable yet", so a
            # forgotten `-w` must not publish that false claim for every variable
            # the extraction does write.
            elif series.wrf and not model.empty:
                missing.append(series.id)

        layers: dict[str, dict[str, object] | None] = {
            "raw": _layer(raw, station_columns, decimals, RAW_CADENCE),
            "hourly": _layer(hourly, station_columns, decimals, HOURLY_CADENCE),
            "wrf": _layer(model, wrf_columns, decimals, HOURLY_CADENCE) if wrf_columns else None,
        }
        payload_chart: dict[str, object] = {
            "id": chart.id,
            "title": chart.title,
            "unit": chart.unit,
            "kind": chart.kind,
            "y_limits": list(chart.y_limits) if chart.y_limits else None,
            "caveats": list(chart.caveats),
            "series": [
                {"id": s.id, "label": s.label, "hue": s.hue, "direction": s.direction}
                for s in chart.series
            ],
            "layers": layers,
            # Which model columns were looked for and not found. The page prints
            # this instead of showing a legend that is silently one entry short.
            "wrf_pending": {s.id: list(s.wrf) for s in chart.series if s.id in missing},
        }
        charts.append(payload_chart)

        drawn = [name for name, value in layers.items() if value]
        pending_note = f" (WRF ausente para {', '.join(missing)})" if missing else ""
        if drawn:
            typer.echo(f"  [ok] {chart.id:18s} camadas: {'+'.join(drawn)}{pending_note}")
        else:
            # Every layer empty is a real operational state, not a bug: the Gill
            # thermohygrometer railed in December 2025 and its readings are
            # masked, so temperature and humidity have no data in any recent window.
            typer.echo(f"  [--] {chart.id:18s} SEM DADO na janela — publicado vazio")

    payload = {
        "format": PAYLOAD_FORMAT,
        "version": version,
        "generated_utc": version,
        "commit": short_commit(),
        "station": {"name": STATION_NAME, "timezone": "America/Bahia"},
        # `end` is the end of what this document CARRIES, which can be later than
        # the newest observation. `station_end` sits beside it because the page
        # derives its visible window as `end - <selected days>`: it must anchor
        # the START on `station_end` and the MAXIMUM on `end`, or a model running
        # three days ahead pushes three days of observations out of the view.
        "window": {
            "start": str(first),
            "end": str(last),
            "days": days,
            "station_end": str(station_end),
        },
        "charts": charts,
    }
    path = write_json(Path(output_dir) / PAYLOAD_FILENAME, payload)
    typer.echo(f"\n>> {path} ({path.stat().st_size / 1024:.1f} kB)")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-monitoring``)."""
    app()


if __name__ == "__main__":
    main()
