"""CLI: publish the rolling-window payload the interactive monitoring page reads.

Writes ``labmim-monitoring-v1`` JSON: one document carrying, for every chart,
the raw station samples, their hourly means and the WRF series where one exists.
The page draws whatever layers the document contains.

This is the **interactive** producer. The static PNGs stay: ``labmim-site-graphs``
(:mod:`micrometeorology.cli.plot_station_graphs`) keeps writing the nine
fixed-name images in the same three-layer style, because those are what goes
into papers.

Sizing, measured on a fully instrumented week: the raw layer is ~105 kB, hourly
~9 kB and WRF ~5 kB, so the whole document is ~117 kB — about a third of the
~380 kB the nine PNGs cost the visitor today, and it arrives as numbers a reader
can hover, toggle and download.

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

from micrometeorology.common.git import run_git
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

# Decimals per unit. Pressure needs one to stay readable at ~1013 hPa;
# precipitation needs three so the 0.254 mm tipping-bucket quantum survives the
# round trip; irradiance is integer because a tenth of a W/m2 is noise.
_DECIMALS = {
    "°C": 2,
    "%": 1,
    "hPa": 2,
    "m/s": 2,
    "°": 1,
    "mm": 3,
    "W/m²": 1,
}

STATION_NAME = "Estação Micrometeorológica LabMiM"


def _round(values: pd.Series, decimals: int) -> list[float | None]:
    """Serialise a series, mapping every non-finite sample to ``null``.

    ``null`` is both correct and cheap here: the writer refuses NaN outright
    (it is not valid JSON), and a gap costs fewer bytes than a real value would.
    """
    return [
        None if not np.isfinite(value) else round(float(value), decimals)
        for value in values.to_numpy(dtype=float)
    ]


def _axis(index: pd.DatetimeIndex) -> dict[str, object]:
    """Publish the time axis as start + step instead of one stamp per sample.

    2,016 ISO-8601 strings cost about 50 kB on their own; a regular grid needs
    only its origin and its cadence. Gaps stay visible because the values carry
    ``null`` at the missing positions.
    """
    step = round((index[1] - index[0]).total_seconds() / 60) if len(index) > 1 else 0
    return {"start": index[0].isoformat(sep=" "), "step_minutes": step, "count": len(index)}


def _layer(frame: pd.DataFrame, columns: dict[str, str], decimals: int) -> dict[str, object] | None:
    """One layer of one chart: its time axis plus a value array per series."""
    present = {sid: column for sid, column in columns.items() if column in frame.columns}
    if frame.empty or not present:
        return None
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"a monitoring layer needs a time index, got {type(index).__name__}")
    return {
        "axis": _axis(index),
        "series": {sid: _round(frame[column], decimals) for sid, column in present.items()},
    }


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

    last = pd.Timestamp(end) if end else pd.Timestamp(raw.index.max())
    first = last - pd.Timedelta(days=days)
    raw = raw.loc[first:last]
    hourly = hourly.loc[first:last]
    typer.echo(f"Janela: {first} .. {last}  ({len(raw):,} amostras brutas, {len(hourly)} horas)")

    model = pd.DataFrame()
    if wrf_path is not None:
        from micrometeorology.cli.export_climatology import read_wrf_series

        model = read_wrf_series(wrf_path).loc[first:last]
        typer.echo(f"WRF: {len(model)} horas na janela, {len(model.columns)} colunas disponiveis")

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
            elif series.wrf:
                missing.append(series.id)

        layers: dict[str, dict[str, object] | None] = {
            "raw": _layer(raw, station_columns, decimals),
            "hourly": _layer(hourly, station_columns, decimals),
            "wrf": _layer(model, wrf_columns, decimals) if wrf_columns else None,
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
        note = f" (WRF ausente para {', '.join(missing)})" if missing else ""
        typer.echo(f"  [ok] {chart.id:18s} camadas: {'+'.join(drawn)}{note}")

    payload = {
        "format": PAYLOAD_FORMAT,
        "version": version,
        "generated_utc": version,
        "commit": run_git(["rev-parse", "--short", "HEAD"]),
        "station": {"name": STATION_NAME, "timezone": "America/Bahia"},
        "window": {"start": str(first), "end": str(last), "days": days},
        "charts": charts,
    }
    path = write_json(Path(output_dir) / PAYLOAD_FILENAME, payload)
    typer.echo(f"\n>> {path} ({path.stat().st_size / 1024:.1f} kB)")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-monitoring``)."""
    app()


if __name__ == "__main__":
    main()
