"""Plotting utilities for LabMiM station meteorological graphs.

Helpers shared by both station-graph producers: they preserve the layout of the
``graficos1_UFBA_v5.py`` / ``graficos3_UFBA_v1.py`` scripts the site was built
around, behind one set of color, watermark, date-axis and legend conventions.
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

WATERMARK_LINE1 = "LabMiM & LaPO (IF)"
WATERMARK_LINE2 = "UFBA"
WATERMARK_COLOR = "#CFCFCF"
WATERMARK_FONTSIZE = 16

# The figure size graficos3 published; the site's image slots assume it.
DEFAULT_FIGSIZE: tuple[float, float] = (8, 4)

# Okabe-Ito: the color-vision-deficiency-friendly sequence matplotlib 3.11
# exposes through the public registry. Fixing the radiation channels to it keeps
# the two station-graph producers in agreement without touching the global
# scientific colormaps.
_OKABE_ITO = matplotlib.color_sequences["okabe_ito"]
BALANCE_COMPONENT_COLORS = {
    "sw_down": _OKABE_ITO[1],  # orange
    "sw_up": _OKABE_ITO[2],  # sky blue
    "lw_down": _OKABE_ITO[3],  # bluish green
    "lw_up": _OKABE_ITO[7],  # reddish purple
}


def add_labmim_watermark(
    ax,
    line1: str = WATERMARK_LINE1,
    line2: str = WATERMARK_LINE2,
) -> None:
    """Add the LabMiM/UFBA watermark text centred on the axes."""
    ax.text(
        0.5,
        0.55,
        line1,
        fontsize=WATERMARK_FONTSIZE,
        color=WATERMARK_COLOR,
        horizontalalignment="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.5,
        0.45,
        line2,
        fontsize=WATERMARK_FONTSIZE,
        color=WATERMARK_COLOR,
        horizontalalignment="center",
        transform=ax.transAxes,
    )


# Longest span that still gets one tick per day. The site graphs plot a week, so
# the legacy look survives wherever it was ever seen; beyond this a fixed
# DayLocator gives not a denser axis but a BLANK one -- matplotlib refuses above
# 1000 ticks (a decade asks for 3,844) and drops the labels entirely. Reachable
# from `--last-days 0` and from the archive's hourly frame, which spans
# 2016-2026.
DAILY_TICK_MAX_DAYS = 21


def setup_date_axis(ax) -> None:
    """Configure the date axis: day-major / 6 h-minor, thinning on long spans."""
    lo, hi = ax.get_xlim()
    if hi - lo <= DAILY_TICK_MAX_DAYS:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 6)))
    else:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=8, maxticks=40))
    ax.xaxis.grid(True, linestyle="-", which="major", color="grey", alpha=0.5)


def add_timestamp_label(ax, dt) -> None:
    """Add a date/time label in the top-right corner of the axes."""
    ax.text(
        1.0,
        0.95,
        dt.strftime("%Y-%m-%d %H:%M"),
        fontsize=10,
        color="black",
        horizontalalignment="right",
        transform=ax.transAxes,
    )


def add_top_legend(ax, *, ncol: int = 3, loc: int = 3) -> None:
    """Add a legend bar along the top edge of the axes."""
    ax.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),
        loc=loc,
        ncol=ncol,
        mode="expand",
        borderaxespad=0.0,
    )


def create_figure(figsize: tuple[float, float] = DEFAULT_FIGSIZE):
    """Create a new figure + axes pair for a station graph."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def save_figure(fig, path: str | Path, *, dpi: int = 100) -> Path:
    """Save a figure and close it.  Creates parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out.name)
    return out
