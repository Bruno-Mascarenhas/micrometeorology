"""CLI: publish the rolling-window payload the interactive monitoring page reads.

Writes ``labmim-monitoring-v1`` JSON: one document carrying, for every chart,
the raw station samples, their hourly means and the WRF series where one exists.
The page draws whatever layers the document contains.

This is the **interactive** producer. The static PNGs stay: ``labmim-site-graphs``
(:mod:`micrometeorology.cli.plot_station_graphs`) keeps writing the nine
fixed-name images in the same three-layer style, because those are what goes
into papers.

Sizing, measured on the reference week: the raw layer is ~110 kB, hourly ~9 kB
and WRF ~7 kB, so the whole document is ~133 kB — about a third of the ~380 kB
the nine PNGs cost the visitor, and it arrives as numbers a reader can hover,
toggle and download.

Like the climatology artifacts, the output is **not** committed to the site
repository: it derives from the laboratory's own sensor archive, so it is
gitignored there and attached by the hourly deploy.

Examples
--------
Publish the last seven days straight into a checkout of the site::

    labmim-monitoring -i output/archive -o ../site-labmim/site/Monitoramento \\
        -w data/series_operacional.dat

Reproduce the reference test window (the one week where the station and the
model are both essentially complete)::

    labmim-monitoring -i output/archive -o out/ -w data/series_operacional.dat \\
        --end 2022-07-08
"""

import logging
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from micrometeorology.common.git import run_git, source_root
from micrometeorology.common.logging import setup_logging
from micrometeorology.sensors.monitoring import MONITORING_CHARTS, resolve_wrf_column
from micrometeorology.stats.climatology_export import write_json

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

PAYLOAD_FORMAT = "labmim-monitoring-v1"
PAYLOAD_FILENAME = "monitoring.json"

# Rolling window the page shows. Seven days is the legacy contract of
# labmim-site-graphs, kept so the two products stay comparable.
DEFAULT_DAYS = 7

# Nominal cadence of each layer, declared rather than inferred. The axis is
# published as an origin plus a step, so a layer has to sit on a complete grid
# before it is serialised; these are the grids ``labmim-archive`` writes
# (``station_5min_qc`` at five minutes, ``station_hourly`` and the WRF series
# hourly).
RAW_CADENCE = "5min"
HOURLY_CADENCE = "h"

# Decimals per unit, matching what the page renders (``unitDigits``) so the CSV
# download never carries a digit the chart it came from does not show. Pressure
# needs one to stay readable at ~1013 hPa; precipitation needs three so the
# 0.254 mm tipping-bucket quantum survives the round trip; irradiance is integer
# because a tenth of a W/m2 is noise.
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

    ``null`` is both correct and cheap here: the writer refuses NaN outright
    (it is not valid JSON), and a gap costs fewer bytes than a real value would.

    A zero-decimal unit is emitted as an ``int``, not a rounded float: ``round(x,
    0)`` returns ``489.0``, which JSON writes as ``"489.0"`` — LONGER than the
    ``"489.5"`` it was meant to shorten, and it turns the pyranometer's small
    negative night offsets into ``-0.0``. Both sides read the value as a number
    either way, so the encoding is free to be the short one.
    """
    if decimals <= 0:
        return [
            None if not np.isfinite(value) else round(float(value))
            for value in values.to_numpy(dtype=float)
        ]
    return [
        None if not np.isfinite(value) else round(float(value), decimals)
        for value in values.to_numpy(dtype=float)
    ]


def _regular(
    frame: pd.DataFrame, index: pd.DatetimeIndex, cadence: str
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Put a layer back on its complete cadence grid before it is serialised.

    The axis below is an origin plus a step, so the page rebuilds every abscissa
    as ``start + i * step``. Neither frame that reaches it is gapless: the logger
    stops (247 non-5-minute steps over the record, the longest 17 days) and
    ``read_wrf_series`` drops the spin-up hour of every day. Serialising the rows
    as they come would therefore slide every sample after a hole earlier than it
    happened — one hour per day for the model layer, the whole outage length for
    the raw one — under an hourly line that stays correctly placed.

    Reindexing turns each hole into a row whose values are ``null``, which is
    what both the encoding and the page already promise.
    """
    if frame.empty:
        return frame, index
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
    only its origin and its cadence. Gaps stay visible because the values carry
    ``null`` at the missing positions — which holds only while the index really
    is a complete grid, so an irregular one raises here rather than being
    re-encoded into a page that would draw it at the wrong times.
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

    A series whose column is present but holds no value over the window is
    OMITTED rather than published as an array of ``null``. The page reads
    ``layer.series[id]`` and treats an absent key as "this layer has nothing for
    this series"; an all-null array passes that guard, builds a dataset, and puts
    an entry in the legend for a line the reader cannot see.

    Live case: the Gill thermohygrometer railed in December 2025 and its readings
    are masked as sentinels, so the DEFAULT operational window — the one the
    hourly deploy runs, with no ``--end`` — carries no air temperature and no
    humidity at all.
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
        int, typer.Option("--days", help="Length of the rolling window.")
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

    # The model is read BEFORE the window is fixed, because it is allowed to run
    # ahead of the station: the operational `series_operacional.dat` grows
    # forward day by day, so an hour with a WRF value and no observation is the
    # normal state, and anchoring the window's end on the newest SAMPLE would
    # clip exactly the part of the model worth looking at.
    model_full = pd.DataFrame()
    if wrf_path is not None:
        from micrometeorology.cli.export_climatology import read_wrf_series

        model_full = read_wrf_series(wrf_path)

    # `first` still anchors on the station, so the reader keeps the same seven
    # days of record behind them; only the END reaches forward. `--end` is an
    # explicit instruction and overrides both.
    station_last = pd.Timestamp(raw.index.max())
    if end:
        first = pd.Timestamp(end) - pd.Timedelta(days=days)
        last = pd.Timestamp(end)
    else:
        first = station_last - pd.Timedelta(days=days)
        last = station_last
        if not model_full.empty:
            last = max(last, pd.Timestamp(model_full.index.max()))
    # The raw layer keeps both ends: a 5-minute sample stamped at `last` is an
    # instantaneous observation that really happened inside the window.
    #
    # The aggregated layers are right-OPEN, because an hourly mean labelled at T
    # covers [T, T+1h): a value stamped at `last` describes data lying entirely
    # PAST the window the payload declares. The last complete hour inside the
    # window is the one that starts an hour before its end.
    #
    # That bound is measured against the STATION's own last sample, not against
    # `last`: when the model runs ahead `last` is the model's end, and trimming
    # against it publishes the station's trailing PARTIAL hour as a complete
    # hourly mean — an average of however many minutes the logger had written,
    # drawn beside hours built from twelve samples.
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
        decimals = _DECIMALS.get(chart.unit, 2)

        wrf_columns: dict[str, str] = {}
        missing: list[str] = []
        for series in chart.series:
            column = resolve_wrf_column(series, model.columns) if not model.empty else None
            if column:
                wrf_columns[series.id] = column
            # Only when a model was actually loaded. `wrf_pending` means "the
            # extraction does not write this variable yet", and the page states
            # exactly that to the reader, so a run given no model file at all
            # must not populate it: a forgotten `-w` in a cron would otherwise
            # publish a false claim about the pipeline for every variable the
            # extraction does write.
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
            # this instead of showing a legend that is silently one entry short,
            # and it is what makes a future extraction change visible.
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
            # masked, so temperature and humidity have no data in any recent
            # window. Whoever runs the export has to see it.
            typer.echo(f"  [--] {chart.id:18s} SEM DADO na janela — publicado vazio")

    payload = {
        "format": PAYLOAD_FORMAT,
        "version": version,
        "generated_utc": version,
        "commit": run_git(["rev-parse", "--short", "HEAD"], cwd=source_root()),
        "station": {"name": STATION_NAME, "timezone": "America/Bahia"},
        # `end` is the end of what this document CARRIES, which under the
        # accumulating extraction can be later than the newest observation.
        #
        # `station_end` is published beside it because the page derives its
        # visible window as `end - <selected days>`: anchored on `end` alone, a
        # model running three days ahead silently pushes three days of real
        # observations out of the default view. Anchor the START on this field
        # and the MAXIMUM on `end`, and the reader gets the record behind them
        # plus the forecast in front.
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
