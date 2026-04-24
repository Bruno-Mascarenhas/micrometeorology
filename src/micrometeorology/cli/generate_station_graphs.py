"""CLI: Generate LabMiM station graphs from raw Campbell Scientific .dat files.

Drop-in replacement for the legacy ``graficos3_UFBA_v1.py`` script.
Reads the same ``.dat`` files produced by the datalogger and outputs
visually identical PNG graphs for the website.

Optionally overlays WRF model output (``series_operacional.dat``) on
applicable graphs as dashed black lines.

Usage::

    python -m micrometeorology.cli.generate_station_graphs \
        --lenta data/LBM_lenta_2022.dat \
        --rain  data/LBM_rain_2022.dat  \
        --output-dir output/figures

    # With optional WRF overlay:
    python -m micrometeorology.cli.generate_station_graphs \
        --lenta data/LBM_lenta_2022.dat \
        --rain  data/LBM_rain_2022.dat  \
        --wrf   data/formatado/series_operacional.dat \
        --output-dir output/figures
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

import click
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")  # headless backend for server

from micrometeorology.common.logging import setup_logging
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.ingestion import read_campbell_dat
from micrometeorology.sensors.plotting import (
    add_labmim_watermark,
    add_timestamp_label,
    add_top_legend,
    create_figure,
    save_figure,
    setup_date_axis,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column names expected in the 2022+ data format
# ---------------------------------------------------------------------------

# Columns to drop from the slow (lenta) file -- raw millivolt / admin columns
LENTA_DROP_COLUMNS = [
    "RECORD",
    "rtime",
    "batt_volt",
    "panel_temp",
    "CM3Up_mv_Avg",
    "CG3Up_mv_Avg",
    "CM3Dn_mv_Avg",
    "CG3Dn_mv_Avg",
    "CNR1TK_Avg",
    "NRLite_Wm2_Avg",
    "NRLite_Wm2Cr_Avg",
    "CMP21_Avg",
    "PAR_Den_Avg",
]

# Columns to drop from the rain file
RAIN_DROP_COLUMNS = [
    "RECORD",
    "rtime(9)",
    "rtime(1)",
    "rtime(4)",
    "rtime(5)",
]

# Precipitation column
RAIN_COLUMN = "PL01_mm_Tot"

# RH_WXT sensor bias offset (legacy: +10.339 in graficos3_v1)
RH_WXT_OFFSET = 10.339

# ---------------------------------------------------------------------------
# WRF series_operacional.dat column mapping
# ---------------------------------------------------------------------------
# Maps graph -> WRF column name.  The WRF file has the first 4 columns as
# year, month, day, hour for datetime construction, then variable columns.
WRF_COLUMNS = {
    "radiacao_difusa": "Sw_dw",
    "temperatura": "T",
    "umidade": "ur",
    "pressao": "pressure",
    "velocidade": "WS",
    "direcao": "WD",
}


# ---------------------------------------------------------------------------
# WRF file reader
# ---------------------------------------------------------------------------


def read_wrf_series(path: str | Path) -> pd.DataFrame:
    """Read a WRF ``series_operacional.dat`` file.

    The file is a CSV where the first 4 columns contain year, month, day,
    hour used to build a DatetimeIndex.  Remaining columns are hourly
    model variables (Sw_dw, T, ur, pressure, WS, WD, etc.).
    """
    p = Path(path)
    logger.info("Reading WRF series: %s", p.name)

    wrf = pd.read_csv(p, sep=",")

    # Build datetime index from the first 4 columns
    dt_cols = wrf.iloc[:, :4]
    dt_cols.columns = ["year", "month", "day", "hour"]
    wrf.index = pd.to_datetime(dt_cols)
    wrf.index.name = None

    logger.info("  -> %d rows, columns: %s", len(wrf), list(wrf.columns[4:]))
    return wrf


def _plot_wrf_overlay(ax, wrf: pd.DataFrame | None, col: str, label: str = "wrf 1h") -> None:
    """Add a dashed black WRF overlay line if data is available."""
    if wrf is None or col not in wrf.columns:
        return
    ax.plot(wrf.index, wrf[col], "--", color="black", label=label)


# ---------------------------------------------------------------------------
# Individual graph generators
# ---------------------------------------------------------------------------


def _plot_radiacao_difusa(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 1 -- Solar Radiation (SW global + diffuse)."""
    fig, ax = create_figure()

    col_sw = "CM3Up_Wm2_Avg"
    col_df = "CMP21_Wm2_Avg"

    if col_sw in raw.columns:
        ax.plot(raw.index, raw[col_sw], "o", color="yellow", markersize=6, label="Media 5 min")
    if col_df in raw.columns:
        ax.plot(raw.index, raw[col_df], "o", color="silver", markersize=6, label="Media 5 min")
    if col_sw in hourly.columns:
        ax.plot(hourly.index, hourly[col_sw], "-vr", label="SW_dw 1h")
    if col_df in hourly.columns:
        ax.plot(hourly.index, hourly[col_df], "-db", label="SW_df 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["radiacao_difusa"], label="SW_dw-wrf 1h")

    ax.set_ylim(0, 1360)
    setup_date_axis(ax)
    plt.ylabel(
        "Radiacao Solar (W/m\u00b2)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=4)
    save_figure(fig, out_dir / "radiacao_difusa.png")


def _plot_balanco(raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime) -> None:
    """Graph 2 -- Radiation Balance (all four components)."""
    fig, ax = create_figure()

    lw_dw = "CG3Up_Wm2Cr_Avg"
    sw_dw = "CM3Up_Wm2_Avg"
    lw_up = "CG3Dn_Wm2Cr_Avg"
    sw_up = "CM3Dn_Wm2_Avg"

    # Raw dots
    if lw_dw in raw.columns:
        ax.plot(raw.index, raw[lw_dw], "o", color="silver", markersize=6)
    if sw_dw in raw.columns:
        ax.plot(raw.index, raw[sw_dw], "o", color="yellow", markersize=6)
    if lw_up in raw.columns:
        ax.plot(raw.index, -raw[lw_up], "o", color="silver", markersize=6)
    if sw_up in raw.columns:
        ax.plot(raw.index, -raw[sw_up], "o", color="silver", markersize=6)

    # Hourly means
    if lw_dw in hourly.columns:
        ax.plot(hourly.index, hourly[lw_dw], "p-", color="black", label="LW_dw")
    if lw_up in hourly.columns:
        ax.plot(hourly.index, -hourly[lw_up], "p-", color="orange", label="LW_up")
    if sw_dw in hourly.columns:
        ax.plot(hourly.index, hourly[sw_dw], "p-", color="red", label="SW_dw")
    if sw_up in hourly.columns:
        ax.plot(hourly.index, -hourly[sw_up], "p-", color="blue", label="SW_up")

    ax.set_ylim(-750, 1200)
    setup_date_axis(ax)
    plt.ylabel(
        "Balanco de Radiacao (W/m\u00b2)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=4)
    save_figure(fig, out_dir / "balanco.png")


def _plot_radiacao_liq(
    raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> None:
    """Graph 3 -- Net Radiation."""
    col = "Net_Wm2_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping radiacao_liq.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "-vr", label="RN 1h")

    ax.set_ylim(-200, 800)
    setup_date_axis(ax)
    plt.ylabel(
        "Radiacao Liquida (W/m\u00b2)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=4)
    save_figure(fig, out_dir / "radiacao_liq.png")


def _plot_radiacao_par(
    raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> None:
    """Graph 4 -- PAR Radiation."""
    col = "PAR_Wm2_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping radiacao_par.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "-*g", label="Media 1 h")

    ax.set_ylim(0, 500)
    setup_date_axis(ax)
    plt.ylabel(
        "Radiacao PAR (W/m\u00b2)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=2)
    save_figure(fig, out_dir / "radiacao_par.png")


def _plot_temperatura(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 5 -- Air Temperature (WXT + CS215)."""
    fig, ax = create_figure()

    col_wxt = "Temp_WXT_Avg"
    col_cs = "Temp1_Avg"

    if col_wxt in raw.columns:
        ax.plot(raw.index, raw[col_wxt], "o", color="silver", markersize=6, label="Media 5 min")
    if col_cs in raw.columns:
        ax.plot(raw.index, raw[col_cs], "o", color="silver", markersize=6)
    if col_wxt in hourly.columns:
        ax.plot(hourly.index, hourly[col_wxt], "^-g", label="WXT 1h")
    if col_cs in hourly.columns:
        ax.plot(hourly.index, hourly[col_cs], "^-r", label="CS215 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["temperatura"])

    ax.set_ylim(18, 32)
    setup_date_axis(ax)
    plt.ylabel(
        "Temperatura do Ar (\u00b0C)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3)
    save_figure(fig, out_dir / "temperatura.png")


def _plot_umidade(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    rh_offset: float,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 6 -- Relative Humidity (WXT + CS215)."""
    fig, ax = create_figure()

    col_wxt = "RH_WXT_Avg"
    col_cs = "RH1_Avg"

    if col_wxt in raw.columns:
        ax.plot(
            raw.index,
            raw[col_wxt] + rh_offset,
            "o",
            color="silver",
            markersize=6,
            label="Media 5 min",
        )
    if col_cs in raw.columns:
        ax.plot(raw.index, raw[col_cs], "o", color="silver", markersize=6)
    if col_wxt in hourly.columns:
        ax.plot(hourly.index, hourly[col_wxt] + rh_offset, "s-b", label="WXT 1h")
    if col_cs in hourly.columns:
        ax.plot(hourly.index, hourly[col_cs], "s-r", label="CS215 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["umidade"])

    ax.set_ylim(50, 100)
    setup_date_axis(ax)
    plt.ylabel(
        "Umidade Relativa do Ar (%)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    # v3 legacy: humidity timestamp at bottom-right (0.88, 0.05)
    ax.text(
        0.88,
        0.05,
        dt.strftime("%Y-%m-%d %H:%M"),
        fontsize=10,
        color="black",
        horizontalalignment="center",
        transform=ax.transAxes,
    )
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3)
    save_figure(fig, out_dir / "umidade.png")


def _plot_pressao(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 7 -- Atmospheric Pressure."""
    col = "Pmb_WXT"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping pressao.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "s-b", label="Media 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["pressao"])

    ax.set_ylim(1000, 1030)
    setup_date_axis(ax)
    plt.ylabel(
        "Pressao Atmosferica (hPa)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3)
    save_figure(fig, out_dir / "pressao.png")


def _plot_velocidade(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 8 -- Wind Speed."""
    col = "WS_WXT_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping velocidade.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "-*k", label="WXT 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["velocidade"])

    ax.set_ylim(0, 10)
    setup_date_axis(ax)
    plt.ylabel(
        "Velocidade do Vento (m/s)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3)
    save_figure(fig, out_dir / "velocidade.png")


def _plot_direcao(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> None:
    """Graph 9 -- Wind Direction."""
    col = "WD_WXT_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping direcao.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "*k", label="WXT 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["direcao"])

    setup_date_axis(ax)
    plt.ylabel(
        "Direcao do Vento (\u00b0)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3)
    save_figure(fig, out_dir / "direcao.png")


def _plot_precipitacao(
    raw_rain: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> None:
    """Graph 10 -- Precipitation."""
    col = RAIN_COLUMN
    if col not in raw_rain.columns:
        logger.warning("Column %s not found -- skipping precipitacao.png", col)
        return

    fig, ax = create_figure()
    ax.plot(raw_rain.index, raw_rain[col], "-", color="grey", lw=2, label="Acumulada 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "o", color="blue", markersize=3, label="Acumulada 1h")

    ax.set_ylim(0, 30)
    setup_date_axis(ax)
    plt.ylabel(
        "Precipitacao (mm)", fontsize=12, horizontalalignment="center", verticalalignment="center"
    )
    add_timestamp_label(ax, dt)
    add_labmim_watermark(ax)
    add_top_legend(ax, ncol=3, loc=1)
    save_figure(fig, out_dir / "precipitacao.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--lenta",
    "-l",
    required=True,
    type=click.Path(exists=True),
    help="Path to LBM_lenta_YYYY.dat (slow sensor data).",
)
@click.option(
    "--rain",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Path to LBM_rain_YYYY.dat (precipitation data).",
)
@click.option("--output-dir", "-o", required=True, help="Output directory for graph PNGs.")
@click.option(
    "--wrf",
    "-w",
    "wrf_path",
    default=None,
    type=click.Path(exists=True),
    help="Optional: path to WRF series_operacional.dat for model overlay.",
)
@click.option(
    "--last-days", default=7, type=int, show_default=True, help="Number of recent days to plot."
)
@click.option(
    "--rh-offset",
    default=RH_WXT_OFFSET,
    type=float,
    show_default=True,
    help="Additive bias correction for RH_WXT sensor.",
)
@click.option("--log-level", default="INFO", show_default=True, help="Logging level.")
def main(
    lenta: str,
    rain: str,
    output_dir: str,
    wrf_path: str | None,
    last_days: int,
    rh_offset: float,
    log_level: str,
) -> None:
    """Generate LabMiM station graphs from raw .dat sensor files.

    Reads Campbell Scientific .dat files directly from the datalogger,
    performs quality control, hourly aggregation, and generates 10 PNG
    graphs for the LabMiM website with the standard watermark and layout.

    Optionally overlays WRF model series (series_operacional.dat) as
    dashed black lines on radiation, temperature, humidity, pressure,
    and wind graphs.
    """
    setup_logging(log_level)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    now = datetime.now()

    # ------------------------------------------------------------------
    # 1. Ingest -- read raw .dat files
    # ------------------------------------------------------------------
    click.echo("Reading sensor data...")

    df_lenta = read_campbell_dat(
        lenta,
        drop_columns=LENTA_DROP_COLUMNS,
    )
    df_rain = read_campbell_dat(
        rain,
        drop_columns=RAIN_DROP_COLUMNS,
    )

    click.echo(f"  lenta: {len(df_lenta)} rows, {len(df_lenta.columns)} columns")
    click.echo(f"  rain:  {len(df_rain)} rows, {len(df_rain.columns)} columns")

    # ------------------------------------------------------------------
    # 1b. Optional WRF series
    # ------------------------------------------------------------------
    wrf: pd.DataFrame | None = None
    if wrf_path:
        wrf = read_wrf_series(wrf_path)
        click.echo(f"  wrf:   {len(wrf)} rows")

    # ------------------------------------------------------------------
    # 2. Merge rain into the main DataFrame for unified aggregation
    # ------------------------------------------------------------------
    if RAIN_COLUMN in df_rain.columns:
        df_lenta[RAIN_COLUMN] = df_rain[RAIN_COLUMN].reindex(df_lenta.index)

    # ------------------------------------------------------------------
    # 3. Filter to last N days
    # ------------------------------------------------------------------
    date_end = pd.Timestamp(now)

    # If current time is beyond the data range, fall back to data end
    data_max = df_lenta.index.max()
    if date_end > data_max:
        logger.info("Current date beyond data range; using data end: %s", data_max)
        date_end = data_max

    date_start = date_end - pd.Timedelta(days=last_days)

    mask_lenta = (df_lenta.index >= date_start) & (df_lenta.index <= date_end)
    mask_rain = (df_rain.index >= date_start) & (df_rain.index <= date_end)

    raw = df_lenta.loc[mask_lenta].copy()
    raw_rain = df_rain.loc[mask_rain].copy()

    click.echo(
        f"  filtered to last {last_days} days: {len(raw)} rows (lenta), {len(raw_rain)} rows (rain)"
    )

    if raw.empty:
        click.echo("[!] No data in the requested date range -- nothing to plot.")
        sys.exit(0)

    # Filter WRF to same date range
    if wrf is not None:
        mask_wrf = (wrf.index >= date_start) & (wrf.index <= date_end)
        wrf = wrf.loc[mask_wrf].copy()
        click.echo(f"  wrf filtered: {len(wrf)} rows")

    # ------------------------------------------------------------------
    # 4. Aggregate to hourly means
    # ------------------------------------------------------------------
    click.echo("Computing hourly aggregates...")

    wind_dir_cols = ["WD_WXT_Avg"]
    wind_speed_map = {"WD_WXT_Avg": "WS_WXT_Avg"}
    sum_cols = [RAIN_COLUMN]

    # Filter to only columns that actually exist
    wind_dir_cols = [c for c in wind_dir_cols if c in raw.columns]
    sum_cols = [c for c in sum_cols if c in raw.columns]
    wind_speed_map = {k: v for k, v in wind_speed_map.items() if k in raw.columns}

    hourly = aggregate_to_hourly(
        raw,
        min_samples=6,
        sum_columns=sum_cols,
        wind_dir_columns=wind_dir_cols,
        wind_speed_column_map=wind_speed_map,
    )
    click.echo(f"  -> {len(hourly)} hourly rows")

    # Use data-end timestamp for graph labels (not wall-clock if historical)
    graph_dt = date_end.to_pydatetime()

    # ------------------------------------------------------------------
    # 5. Generate all graphs
    # ------------------------------------------------------------------
    click.echo("Generating graphs...")

    _plot_radiacao_difusa(raw, hourly, out, graph_dt, wrf=wrf)
    click.echo("  [ok] radiacao_difusa.png")

    _plot_balanco(raw, hourly, out, graph_dt)
    click.echo("  [ok] balanco.png")

    _plot_radiacao_liq(raw, hourly, out, graph_dt)
    click.echo("  [ok] radiacao_liq.png")

    _plot_radiacao_par(raw, hourly, out, graph_dt)
    click.echo("  [ok] radiacao_par.png")

    _plot_temperatura(raw, hourly, out, graph_dt, wrf=wrf)
    click.echo("  [ok] temperatura.png")

    _plot_umidade(raw, hourly, out, graph_dt, rh_offset, wrf=wrf)
    click.echo("  [ok] umidade.png")

    _plot_pressao(raw, hourly, out, graph_dt, wrf=wrf)
    click.echo("  [ok] pressao.png")

    _plot_velocidade(raw, hourly, out, graph_dt, wrf=wrf)
    click.echo("  [ok] velocidade.png")

    _plot_direcao(raw, hourly, out, graph_dt, wrf=wrf)
    click.echo("  [ok] direcao.png")

    _plot_precipitacao(raw_rain, hourly, out, graph_dt)
    click.echo("  [ok] precipitacao.png")

    click.echo(f"\n[ok] All graphs saved to {out}")


if __name__ == "__main__":
    main()
