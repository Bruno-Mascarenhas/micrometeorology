"""Figures for a paired model/observation frame.

Split out of :mod:`micrometeorology.stats.comparison` so that reading and
scoring a dataset does not import matplotlib: ``labmim-metrics`` needs the
pandas half only, and matplotlib costs it ~0.5 s of startup per run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_comparison(
    paired_df: pd.DataFrame,
    variable: str,
    output_path: str | Path | None = None,
) -> Figure:
    """Create a comparison plot (time series + scatter) for a variable.

    The figure is built off the pyplot state machine, so it is never registered
    with the global figure manager and the caller owns its lifetime.
    """
    obs_col = f"{variable}_obs"
    model_col = f"{variable}_model"

    fig = Figure(figsize=(14, 5))
    axes = fig.subplots(1, 2)

    # Time series
    ax1 = axes[0]
    ax1.plot(paired_df.index, paired_df[obs_col], "b-", label="Observed", alpha=0.7)
    ax1.plot(paired_df.index, paired_df[model_col], "r--", label="Model", alpha=0.7)
    ax1.set_ylabel(variable)
    ax1.legend()
    ax1.set_title(f"{variable} — Time Series")
    ax1.tick_params(axis="x", rotation=45)

    # Scatter
    ax2 = axes[1]
    obs_vals = paired_df[obs_col].dropna()
    mod_vals = paired_df[model_col].reindex(obs_vals.index).dropna()
    common = obs_vals.index.intersection(mod_vals.index)
    ax2.scatter(obs_vals[common], mod_vals[common], alpha=0.5, s=10)
    lims = [
        min(obs_vals[common].min(), mod_vals[common].min()),
        max(obs_vals[common].max(), mod_vals[common].max()),
    ]
    ax2.plot(lims, lims, "k--", alpha=0.5)
    ax2.set_xlabel("Observed")
    ax2.set_ylabel("Model")
    ax2.set_title(f"{variable} — Scatter")
    ax2.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
        logger.info("Saved comparison plot: %s", output_path)

    return fig
