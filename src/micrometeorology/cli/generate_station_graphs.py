"""CLI: Generate LabMiM station graphs from raw Campbell Scientific .dat files.

Reads the ``.dat`` files the datalogger writes and emits the PNG graphs the
website expects, keeping the layout of the ``graficos3_UFBA_v1.py`` script the
site was built around. WRF model output (``series_operacional.dat``) is
optionally overlaid on the applicable graphs as dashed black lines.

Timestamps are naive station-local throughout, as the datalogger writes them;
the graphs' axes are therefore local wall-clock hours, not UTC ones.

Usage::

    python -m micrometeorology.cli.generate_station_graphs \
        --lenta data/LBM_lenta_2022.dat \
        --rain  data/LBM_rain_2022.dat  \
        --output-dir output/figures

    # With optional WRF overlay:
    python -m micrometeorology.cli.generate_station_graphs \
        --lenta data/LBM_lenta_2025.dat \
        --rain  data/LBM_rain_2025.dat  \
        --wrf   data/series/labmim_series_operacional.dat \
        --output-dir output/figures
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.axes import Axes

matplotlib.use("Agg")

from micrometeorology.cli.plot_station_graphs import GRAPH_SPECS
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir
from micrometeorology.common.timeparse import parse_naive_timestamp
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.ingestion import read_campbell_dat
from micrometeorology.sensors.plotting import (
    BALANCE_COMPONENT_COLORS,
    add_labmim_watermark,
    add_timestamp_label,
    add_top_legend,
    create_figure,
    save_figure,
    setup_date_axis,
)
from micrometeorology.wrf.columns import (
    PSFC_HPA,
    RH_PCT,
    SWDDIF_W_M2,
    SWDOWN_W_M2,
    T2_C,
    WIND_DIR_DEG,
    WIND_SPEED_M_S,
)
from micrometeorology.wrf.operational_record import read_wrf_series

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

# Columns of the 2022+ logger format dropped from the slow (lenta) file: raw
# millivolt channels and administrative fields.
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

RAIN_DROP_COLUMNS = [
    "RECORD",
    "rtime(9)",
    "rtime(1)",
    "rtime(4)",
    "rtime(5)",
]

RAIN_COLUMN = "PL01_mm_Tot"

# Additive bias correction for the RH_WXT sensor, inherited from graficos3_v1.
RH_WXT_OFFSET = 10.339

# Graph name -> column of series_operacional.dat carrying the model series.
# The solar graph draws BOTH shortwave channels, so it carries its own two
# entries: keying a graph named "difusa" to the global flux alone left the only
# model curve on the figure being the global one, which a reader comparing it
# against the measured diffuse below it has no way to tell apart.
WRF_COLUMNS = {
    "radiacao_global": SWDOWN_W_M2,
    "radiacao_difusa": SWDDIF_W_M2,
    "temperatura": T2_C,
    "umidade": RH_PCT,
    "pressao": PSFC_HPA,
    "velocidade": WIND_SPEED_M_S,
    "direcao": WIND_DIR_DEG,
}

#: Fixed y-frames, read from the SIBLING producer of these same nine filenames so
#: the two cannot frame one PNG differently.
SHARED_Y_LIMITS: dict[str, tuple[float, float]] = {
    spec.key: spec.ylim for spec in GRAPH_SPECS if spec.ylim is not None
}


def _plot_wrf_overlay(
    ax,
    wrf: pd.DataFrame | None,
    col: str,
    label: str = "wrf 1h",
    linestyle: str = "--",
    color: str = "black",
    marker: str | None = None,
) -> None:
    """Add a WRF overlay to *ax* if the column is available.

    A figure carrying two model curves must give them different strokes: two
    dashed black lines on one axes are indistinguishable, which is exactly how
    the model's global flux came to be read as its diffuse.

    Parameters
    ----------
    marker:
        Draw markers instead of a joined line. A circular quantity needs this:
        joining a 350 deg hour to the 10 deg hour after it sweeps the axis
        through every bearing between them, none of which the model produced.
    """
    if wrf is None or col not in wrf.columns:
        return
    if marker is not None:
        ax.plot(wrf.index, wrf[col], marker=marker, linestyle="none", color=color, label=label)
        return
    ax.plot(wrf.index, wrf[col], linestyle=linestyle, color=color, label=label)


def _carries_signal(series: pd.Series) -> bool:
    finite = series.dropna().to_numpy(dtype=float)
    return finite.size > 0 and not np.isclose(finite, 0.0).all()


def _draw_radiacao_difusa(
    ax: Axes, raw: pd.DataFrame, hourly: pd.DataFrame, wrf: pd.DataFrame | None
) -> None:
    col_sw = "CM3Up_Wm2_Avg"
    col_diffuse = "CMP21_Wm2_Avg"
    col_diffuse_psp = "PSP_Wm2_Avg"

    # Named per instrument, as every other graph here already does: three clouds
    # sharing one "Media 5 min" entry left the reader unable to tell the global
    # pyranometer from the two diffuse ones on the figure they disagree on.
    # Short spellings because eight entries have to fit the fixed-height bar.
    # A diffuse channel the logger writes as a hard zero (CMP21 in every v22 row,
    # see configs/micromet/calibrations.yaml) is not the diffuse and is not drawn.
    if col_sw in raw.columns:
        ax.plot(raw.index, raw[col_sw], "o", color="yellow", markersize=6, label="5 min (CM3Up)")
    if col_diffuse in raw.columns and _carries_signal(raw[col_diffuse]):
        ax.plot(
            raw.index,
            raw[col_diffuse],
            "o",
            color="silver",
            markersize=6,
            label="5 min (CMP21)",
        )
    if col_diffuse_psp in raw.columns and _carries_signal(raw[col_diffuse_psp]):
        ax.plot(
            raw.index,
            raw[col_diffuse_psp],
            "o",
            color="darkseagreen",
            markersize=6,
            label="5 min (PSP)",
        )
    if col_sw in hourly.columns:
        ax.plot(hourly.index, hourly[col_sw], "-vr", label="SW_dw 1h")
    if col_diffuse in hourly.columns and _carries_signal(hourly[col_diffuse]):
        ax.plot(hourly.index, hourly[col_diffuse], "-db", label="SW_df 1h (CMP21)")
    if col_diffuse_psp in hourly.columns and _carries_signal(hourly[col_diffuse_psp]):
        ax.plot(hourly.index, hourly[col_diffuse_psp], "-dg", label="SW_df 1h (PSP)")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["radiacao_global"], label="SW_dw-wrf 1h")
    _plot_wrf_overlay(
        ax,
        wrf,
        WRF_COLUMNS["radiacao_difusa"],
        label="SW_df-wrf 1h",
        linestyle="-.",
        color="#e07a1f",
    )


def _plot_radiacao_difusa(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
    """Graph 1 -- Solar Radiation (SW global + diffuse)."""
    fig, ax = create_figure()
    _draw_radiacao_difusa(ax, raw, hourly, wrf)

    if not ax.get_lines():
        logger.warning("No solar radiation columns found -- skipping radiacao_difusa.png")
        plt.close(fig)
        return None

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
    return save_figure(fig, out_dir / "radiacao_difusa.png")


def _plot_balanco(
    raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> Path | None:
    """Graph 2 -- Radiation Balance (all four components)."""
    fig, ax = create_figure()

    lw_dw = "CG3Up_Wm2Cr_Avg"
    sw_dw = "CM3Up_Wm2_Avg"
    lw_up = "CG3Dn_Wm2Cr_Avg"
    sw_up = "CM3Dn_Wm2_Avg"

    if lw_dw in raw.columns:
        ax.plot(
            raw.index,
            raw[lw_dw],
            "o",
            color=BALANCE_COMPONENT_COLORS["lw_down"],
            markersize=6,
            alpha=0.35,
        )
    if sw_dw in raw.columns:
        ax.plot(
            raw.index,
            raw[sw_dw],
            "o",
            color=BALANCE_COMPONENT_COLORS["sw_down"],
            markersize=6,
            alpha=0.35,
        )
    if lw_up in raw.columns:
        ax.plot(
            raw.index,
            -raw[lw_up],
            "o",
            color=BALANCE_COMPONENT_COLORS["lw_up"],
            markersize=6,
            alpha=0.35,
        )
    if sw_up in raw.columns:
        ax.plot(
            raw.index,
            -raw[sw_up],
            "o",
            color=BALANCE_COMPONENT_COLORS["sw_up"],
            markersize=6,
            alpha=0.35,
        )

    if lw_dw in hourly.columns:
        ax.plot(
            hourly.index,
            hourly[lw_dw],
            "p-",
            color=BALANCE_COMPONENT_COLORS["lw_down"],
            label="LW_dw",
        )
    if lw_up in hourly.columns:
        ax.plot(
            hourly.index,
            -hourly[lw_up],
            "p-",
            color=BALANCE_COMPONENT_COLORS["lw_up"],
            label="LW_up",
        )
    if sw_dw in hourly.columns:
        ax.plot(
            hourly.index,
            hourly[sw_dw],
            "p-",
            color=BALANCE_COMPONENT_COLORS["sw_down"],
            label="SW_dw",
        )
    if sw_up in hourly.columns:
        ax.plot(
            hourly.index,
            -hourly[sw_up],
            "p-",
            color=BALANCE_COMPONENT_COLORS["sw_up"],
            label="SW_up",
        )

    if not ax.get_lines():
        logger.warning("No radiation balance columns found -- skipping balanco.png")
        plt.close(fig)
        return None

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
    return save_figure(fig, out_dir / "balanco.png")


def _plot_radiacao_liq(
    raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> Path | None:
    """Graph 3 -- Net Radiation."""
    col = "Net_Wm2_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping radiacao_liq.png", col)
        return None

    fig, ax = create_figure()
    ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "-vr", label="RN 1h")

    ax.set_ylim(-200, 900)
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
    return save_figure(fig, out_dir / "radiacao_liq.png")


def _plot_radiacao_par(
    raw: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> Path | None:
    """Graph 4 -- PAR Radiation."""
    col = "PAR_Wm2_Avg"
    if col not in raw.columns:
        logger.warning("Column %s not found -- skipping radiacao_par.png", col)
        return None

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
    return save_figure(fig, out_dir / "radiacao_par.png")


def _plot_temperatura(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
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

    if not ax.get_lines():
        logger.warning("No air temperature columns found -- skipping temperatura.png")
        plt.close(fig)
        return None

    ax.set_ylim(*SHARED_Y_LIMITS["temperatura"])
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
    return save_figure(fig, out_dir / "temperatura.png")


def _plot_umidade(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    rh_offset: float,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
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

    if not ax.get_lines():
        logger.warning("No relative humidity columns found -- skipping umidade.png")
        plt.close(fig)
        return None

    ax.set_ylim(*SHARED_Y_LIMITS["umidade"])
    setup_date_axis(ax)
    plt.ylabel(
        "Umidade Relativa do Ar (%)",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
    )
    # Not add_timestamp_label: graficos3_v1 puts the humidity stamp bottom-right.
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
    return save_figure(fig, out_dir / "umidade.png")


def _plot_pressao(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
    """Graph 7 -- Atmospheric Pressure."""
    col_wxt = "Pmb_WXT"
    col_bp1 = "BP1_mbar_Avg"
    if col_wxt not in raw.columns and col_bp1 not in raw.columns:
        logger.warning("No pressure columns found -- skipping pressao.png")
        return None

    fig, ax = create_figure()
    if col_wxt in raw.columns:
        ax.plot(raw.index, raw[col_wxt], "o", color="silver", markersize=6, label="Media 5 min")
    if col_bp1 in raw.columns:
        ax.plot(
            raw.index, raw[col_bp1], "o", color="silver", markersize=6, label="Media 5 min (BP1)"
        )
    if col_wxt in hourly.columns:
        ax.plot(hourly.index, hourly[col_wxt], "s-b", label="Media 1h")
    if col_bp1 in hourly.columns:
        ax.plot(hourly.index, hourly[col_bp1], "s-c", label="Media 1h (BP1)")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["pressao"])

    # No fixed frame: a 30 hPa window clips a real synoptic passage off the top.
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
    return save_figure(fig, out_dir / "pressao.png")


def _plot_velocidade(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
    """Graph 8 -- Wind Speed."""
    col = "WS_WXT_Avg"
    col_ws1 = "WS1_ms_GMX"
    col_ws2 = "WS_ms"
    if col not in raw.columns and col_ws1 not in raw.columns and col_ws2 not in raw.columns:
        logger.warning("No wind speed columns found -- skipping velocidade.png")
        return None

    fig, ax = create_figure()
    if col in raw.columns:
        ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col_ws1 in raw.columns:
        ax.plot(
            raw.index, raw[col_ws1], "o", color="silver", markersize=6, label="Media 5 min (WS1)"
        )
    if col_ws2 in raw.columns:
        ax.plot(
            raw.index, raw[col_ws2], "o", color="silver", markersize=6, label="Media 5 min (WS2)"
        )

    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "-*k", label="WXT 1h")
    if col_ws1 in hourly.columns:
        ax.plot(hourly.index, hourly[col_ws1], "-*b", label="GMX 1h")
    if col_ws2 in hourly.columns:
        ax.plot(hourly.index, hourly[col_ws2], "-*g", label="WS 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["velocidade"])

    ax.set_ylim(*SHARED_Y_LIMITS["velocidade"])
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
    return save_figure(fig, out_dir / "velocidade.png")


def _plot_direcao(
    raw: pd.DataFrame,
    hourly: pd.DataFrame,
    out_dir: Path,
    dt: datetime,
    wrf: pd.DataFrame | None = None,
) -> Path | None:
    """Graph 9 -- Wind Direction."""
    col = "WD_WXT_Avg"
    col_wd1 = "WindDir1_GMX"
    col_wd2 = "WindDir"
    if col not in raw.columns and col_wd1 not in raw.columns and col_wd2 not in raw.columns:
        logger.warning("No wind direction columns found -- skipping direcao.png")
        return None

    fig, ax = create_figure()
    if col in raw.columns:
        ax.plot(raw.index, raw[col], "o", color="silver", markersize=6, label="Media 5 min")
    if col_wd1 in raw.columns:
        ax.plot(
            raw.index, raw[col_wd1], "o", color="silver", markersize=6, label="Media 5 min (WD1)"
        )
    if col_wd2 in raw.columns:
        ax.plot(
            raw.index, raw[col_wd2], "o", color="silver", markersize=6, label="Media 5 min (WD2)"
        )

    if col in hourly.columns:
        ax.plot(hourly.index, hourly[col], "*k", label="WXT 1h")
    if col_wd1 in hourly.columns:
        ax.plot(hourly.index, hourly[col_wd1], "*b", label="GMX 1h")
    if col_wd2 in hourly.columns:
        ax.plot(hourly.index, hourly[col_wd2], "*g", label="WindDir 1h")

    _plot_wrf_overlay(ax, wrf, WRF_COLUMNS["direcao"], marker="x")

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
    return save_figure(fig, out_dir / "direcao.png")


def _plot_precipitacao(
    raw_rain: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path, dt: datetime
) -> Path | None:
    """Graph 10 -- Precipitation."""
    col = RAIN_COLUMN
    if col not in raw_rain.columns:
        logger.warning("Column %s not found -- skipping precipitacao.png", col)
        return None

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
    return save_figure(fig, out_dir / "precipitacao.png")


def newest_plotted_stamp(raw: pd.DataFrame, raw_rain: pd.DataFrame) -> datetime:
    """The newest sample actually drawn, across both logger tables.

    Stamped onto every figure instead of the wall clock: without ``--start-date``
    the window ends at today, so a record that stopped months ago would publish
    under today's date and read as current.

    Raises
    ------
    ValueError
        When neither frame carries a row, which the caller has already refused.
    """
    plotted = [frame.index.max() for frame in (raw, raw_rain) if not frame.empty]
    if not plotted:
        raise ValueError("neither the lenta nor the rain frame carries a sample to stamp")
    newest: datetime = max(plotted).to_pydatetime()
    return newest


@app.command()
def run(
    lenta: Annotated[
        Path,
        typer.Option(
            "-l", "--lenta", help="Path to LBM_lenta_YYYY.dat (slow sensor data).", exists=True
        ),
    ],
    rain: Annotated[
        Path,
        typer.Option(
            "-r", "--rain", help="Path to LBM_rain_YYYY.dat (precipitation data).", exists=True
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("-o", "--output-dir", help="Output directory for graph PNGs.")
    ],
    wrf_path: Annotated[
        Path | None,
        typer.Option(
            "-w", "--wrf", help="Path to WRF series_operacional.dat for model overlay.", exists=True
        ),
    ] = None,
    start_date: Annotated[
        str | None, typer.Option(help="Initial date YYYY-MM-DD. Defaults to today minus 'days'.")
    ] = None,
    days: Annotated[int, typer.Option(help="Number of days to plot.")] = 7,
    rh_offset: Annotated[
        float, typer.Option(help="Additive bias correction for RH_WXT sensor.")
    ] = RH_WXT_OFFSET,
    strict: Annotated[
        bool, typer.Option("--strict", help="Exit non-zero if any graph was skipped.")
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate LabMiM station graphs from raw .dat sensor files.

    Reads Campbell Scientific .dat files directly from the datalogger, aggregates
    them to hourly means and writes 10 PNG graphs for the LabMiM website with the
    standard watermark and layout.

    **No quality control runs here.** The sentinel table, the physical range
    gates and every radiation mask belong to ``labmim-archive``, which reads the
    same files through the explicit manifest and publishes ``station_5min_qc``;
    this command plots what the logger wrote. A rail such as the BP1 fill value
    therefore reaches the figure, which is the difference between these PNGs and
    the ones ``labmim-site-graphs`` draws from the QC'd database.

    A graph whose source columns are all absent from the .dat file is
    skipped rather than written blank, and reported as `[skip]`; `--strict`
    turns any such skip into a non-zero exit.

    Optionally overlays WRF model series (series_operacional.dat) as
    dashed black lines on radiation, temperature, humidity, pressure,
    and wind graphs.
    """
    setup_logging(log_level)
    out = ensure_dir(output_dir)
    # Campbell .dat timestamps carry no UTC offset, so `read_campbell_dat`
    # builds a naive station-local index. Both date bounds below must stay
    # naive to match it -- an aware Timestamp raises TypeError on comparison.
    now = pd.Timestamp.now()

    typer.echo("Reading sensor data...")

    df_lenta = read_campbell_dat(
        lenta,
        drop_columns=LENTA_DROP_COLUMNS,
    )
    df_rain = read_campbell_dat(
        rain,
        drop_columns=RAIN_DROP_COLUMNS,
    )

    typer.echo(f"  lenta: {len(df_lenta)} rows, {len(df_lenta.columns)} columns")
    typer.echo(f"  rain:  {len(df_rain)} rows, {len(df_rain.columns)} columns")

    wrf: pd.DataFrame | None = None
    if wrf_path:
        wrf = read_wrf_series(wrf_path, consumes=WRF_COLUMNS.values())
        typer.echo(f"  wrf:   {len(wrf)} rows")

    if start_date is not None:
        # Naive for the same reason as `now` above: this bound is compared
        # against the datalogger's naive station-local index. Parsed strictly —
        # an unset `--start-date "$VAR"` in a cron wrapper must fail loudly
        # rather than filter every row away and exit 0.
        try:
            date_start = parse_naive_timestamp(start_date, "%Y-%m-%d")
        except ValueError as exc:
            raise typer.BadParameter(
                f"--start-date must be a YYYY-MM-DD date (got {start_date!r})"
            ) from exc
        date_end = date_start + pd.Timedelta(days=days)
    else:
        date_end = now
        date_start = date_end - pd.Timedelta(days=days)

    mask_lenta = (df_lenta.index >= date_start) & (df_lenta.index <= date_end)
    mask_rain = (df_rain.index >= date_start) & (df_rain.index <= date_end)

    raw = df_lenta.loc[mask_lenta].copy()
    raw_rain = df_rain.loc[mask_rain].copy()

    typer.echo(
        f"  filtered for {days} days ({date_start.date()} to {date_end.date()}): {len(raw)} rows (lenta), {len(raw_rain)} rows (rain)"
    )

    if raw.empty:
        # `--strict` is honoured here too, matching plot_station_graphs, the
        # sibling producer of the same site images: an empty range skips EVERY
        # graph and leaves the previous PNGs in place, so a cron chain must be
        # able to see it instead of reading an exit 0 as a fresh publication.
        typer.echo("[!] No data in the requested date range -- nothing to plot.")
        raise typer.Exit(code=1 if strict else 0)

    if wrf is not None:
        mask_wrf = (wrf.index >= date_start) & (wrf.index <= date_end)
        wrf = wrf.loc[mask_wrf].copy()
        typer.echo(f"  wrf filtered: {len(wrf)} rows")

    typer.echo("Computing hourly aggregates...")

    wind_dir_cols = ["WD_WXT_Avg", "WindDir1_GMX", "WindDir"]
    wind_speed_map = {"WD_WXT_Avg": "WS_WXT_Avg", "WindDir1_GMX": "WS1_ms_GMX", "WindDir": "WS_ms"}

    wind_dir_cols = [c for c in wind_dir_cols if c in raw.columns]
    wind_speed_map = {k: v for k, v in wind_speed_map.items() if k in raw.columns}

    hourly = aggregate_to_hourly(
        raw,
        min_samples=6,
        wind_dir_columns=wind_dir_cols,
        wind_speed_column_map=wind_speed_map,
    )
    # Rain is a SEPARATE logger table on its own stamps, so it is aggregated from
    # its own frame and joined on the hourly grid: reindexing it onto lenta's
    # stamps drops every rain sample at a stamp lenta lacks — measured over
    # 2026-05-27 + 7 d, 126.24 mm against 67.31 mm on one axes.
    # build_archive joins the two tables `how="outer"` for the same reason.
    if RAIN_COLUMN in raw_rain.columns:
        hourly_rain = aggregate_to_hourly(raw_rain, min_samples=6, sum_columns=[RAIN_COLUMN])
        # One of the archive's lenta tables carries the gauge column too, where
        # the generic pass would have averaged it; the rain table's own SUM is
        # the one the bars mean, so it replaces rather than collides.
        hourly = hourly.drop(columns=[RAIN_COLUMN], errors="ignore").join(
            hourly_rain[[RAIN_COLUMN]], how="outer"
        )
        dropped = len(raw_rain.index.difference(raw.index))
        if dropped:
            typer.echo(f"  rain: {dropped} carimbo(s) fora do índice da lenta, preservados")
    typer.echo(f"  -> {len(hourly)} hourly rows")

    # Stamp the newest sample actually drawn, not the wall clock: without
    # `--start-date`, `date_end` is today, so a record that stopped months ago
    # would be published under today's date. plot_station_graphs, the sibling
    # producer of the same images, stamps the data end for the same reason.
    graph_dt = newest_plotted_stamp(raw, raw_rain)

    typer.echo("Generating graphs...")

    written: list[str] = []
    skipped: list[str] = []

    def report(name: str, saved: Path | None) -> None:
        """Echo one graph's outcome, counting only files actually on disk."""
        if saved is None:
            skipped.append(name)
            typer.echo(f"  [skip] {name} (missing source column)")
        else:
            written.append(name)
            typer.echo(f"  [ok] {name}")

    report("radiacao_difusa.png", _plot_radiacao_difusa(raw, hourly, out, graph_dt, wrf=wrf))
    report("balanco.png", _plot_balanco(raw, hourly, out, graph_dt))
    report("radiacao_liq.png", _plot_radiacao_liq(raw, hourly, out, graph_dt))
    report("radiacao_par.png", _plot_radiacao_par(raw, hourly, out, graph_dt))
    report("temperatura.png", _plot_temperatura(raw, hourly, out, graph_dt, wrf=wrf))
    report("umidade.png", _plot_umidade(raw, hourly, out, graph_dt, rh_offset, wrf=wrf))
    report("pressao.png", _plot_pressao(raw, hourly, out, graph_dt, wrf=wrf))
    report("velocidade.png", _plot_velocidade(raw, hourly, out, graph_dt, wrf=wrf))
    report("direcao.png", _plot_direcao(raw, hourly, out, graph_dt, wrf=wrf))
    report("precipitacao.png", _plot_precipitacao(raw_rain, hourly, out, graph_dt))

    total = len(written) + len(skipped)
    if skipped:
        typer.echo(
            f"\n[!] {len(written)}/{total} graphs written to {out}, "
            f"{len(skipped)} skipped: {', '.join(skipped)}"
        )
    else:
        typer.echo(f"\n[ok] All graphs saved to {out}")

    if strict and skipped:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-station-graphs``)."""
    app()


if __name__ == "__main__":
    main()
