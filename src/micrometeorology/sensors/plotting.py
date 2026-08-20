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
_ORANGE = _OKABE_ITO[1]
_SKY_BLUE = _OKABE_ITO[2]
_BLUISH_GREEN = _OKABE_ITO[3]
_REDDISH_PURPLE = _OKABE_ITO[7]
BALANCE_COMPONENT_COLORS = {
    "sw_down": _ORANGE,
    "sw_up": _SKY_BLUE,
    "lw_down": _BLUISH_GREEN,
    "lw_up": _REDDISH_PURPLE,
}


def add_labmim_watermark(
    ax,
    line1: str = WATERMARK_LINE1,
    line2: str = WATERMARK_LINE2,
) -> None:
    """Add the LabMiM/UFBA watermark text centred on the axes.

    Parameters
    ----------
    ax:
        Axes to draw on. The text is placed in axes coordinates, so it stays
        centred whatever the data limits are.
    line1, line2:
        The two stacked lines of the mark.
    """
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
    """Configure the date axis: day-major / 6 h-minor, thinning on long spans.

    Call this **after** the data is plotted: the span is read from the current
    x limits, so on an empty axes it sees matplotlib's placeholder range rather
    than the period being drawn.

    Parameters
    ----------
    ax:
        Axes whose x limits are already those of the plotted period, in
        matplotlib date numbers (days).
    """
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
    """Add a date/time label in the top-right corner of the axes.

    Parameters
    ----------
    ax:
        Axes to draw on.
    dt:
        Anything with ``strftime``. It is printed to the minute and carries the
        station's own naive local clock, matching the x axis rather than UTC.
    """
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
    """Add a legend bar along the top edge of the axes.

    Parameters
    ----------
    ax:
        Axes whose labelled artists go into the legend.
    ncol:
        Columns to spread the entries over. The bar expands to the full axes
        width, so this is what decides whether the labels stay readable.
    loc:
        Anchor corner inside the reserved strip, in matplotlib's numeric
        location code.
    """
    ax.legend(
        bbox_to_anchor=(0.0, 1.0, 1.0, 0.1),
        loc=loc,
        ncol=ncol,
        mode="expand",
        borderaxespad=0.0,
    )


def create_figure(figsize: tuple[float, float] = DEFAULT_FIGSIZE):
    """Create a new figure + axes pair for a station graph.

    Parameters
    ----------
    figsize:
        Width and height in inches. The default is the size the site's image
        slots were built around (:data:`DEFAULT_FIGSIZE`).

    Returns
    -------
    (fig, ax):
        The figure and its single axes. The caller owns the figure until
        :func:`save_figure` closes it; leaving it open leaks a matplotlib
        figure per graph, which matters in the batch that renders the whole
        archive.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    return fig, ax


def save_figure(fig, path: str | Path, *, dpi: int = 100) -> Path:
    """Save a figure and close it.  Creates parent directories as needed.

    Parameters
    ----------
    fig:
        Figure to write. It is closed afterwards and must not be reused.
    path:
        Destination. The extension selects the format matplotlib writes.
    dpi:
        Dots per inch; with :data:`DEFAULT_FIGSIZE` the default gives the
        800x400 px the site's slots expect.

    Returns
    -------
    Path
        The path written, as a ``Path`` whatever was passed in.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", out.name)
    return out
