"""Plotting utilities for LabMiM station meteorological graphs.

Provides helper functions that reproduce the exact visual style of the
legacy ``graficos1_UFBA_v5.py`` / ``graficos3_UFBA_v1.py`` scripts:
watermark placement, date-axis formatting, legend style, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Watermark defaults
# ---------------------------------------------------------------------------

WATERMARK_LINE1 = "LabMiM & LaPO (IF)"
WATERMARK_LINE2 = "UFBA"
WATERMARK_COLOR = "#CFCFCF"
WATERMARK_FONTSIZE = 16

# Default figure size matching legacy graficos3 (8x4)
DEFAULT_FIGSIZE: tuple[float, float] = (8, 4)


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------


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


def setup_date_axis(ax) -> None:
    """Configure day-major / 6 h-minor date axis matching legacy graphs."""
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d - %b"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(np.arange(0, 25, 6)))
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
