"""Figures for a paired model/observation frame.

Split out of :mod:`micrometeorology.stats.comparison` so that reading and
scoring a dataset does not import matplotlib: ``labmim-metrics`` needs the
pandas half only, and matplotlib costs it ~0.5 s of startup per run.
"""

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

    The figure is built without the pyplot state machine, so it is never
    registered with the global figure manager and the caller owns its lifetime.

    Parameters
    ----------
    paired_df:
        Output of :func:`micrometeorology.stats.comparison.pair_dataframes`,
        carrying ``<variable>_obs`` and ``<variable>_model`` columns on a time
        index.
    variable:
        Base name to draw; it labels the y-axis, so it should read in the
        variable's own unit.
    output_path:
        Written as PNG when given; ``None`` returns the figure undrawn to disk.

    Returns
    -------
    matplotlib.figure.Figure
        The two-panel figure. The caller is responsible for closing it.
    """
    obs_col = f"{variable}_obs"
    model_col = f"{variable}_model"

    fig = Figure(figsize=(14, 5))
    axes = fig.subplots(1, 2)

    series_ax = axes[0]
    series_ax.plot(paired_df.index, paired_df[obs_col], "b-", label="Observed", alpha=0.7)
    series_ax.plot(paired_df.index, paired_df[model_col], "r--", label="Model", alpha=0.7)
    series_ax.set_ylabel(variable)
    series_ax.legend()
    series_ax.set_title(f"{variable} — Time Series")
    series_ax.tick_params(axis="x", rotation=45)

    scatter_ax = axes[1]
    obs_vals = paired_df[obs_col].dropna()
    mod_vals = paired_df[model_col].reindex(obs_vals.index).dropna()
    common = obs_vals.index.intersection(mod_vals.index)
    scatter_ax.scatter(obs_vals[common], mod_vals[common], alpha=0.5, s=10)
    lims = [
        min(obs_vals[common].min(), mod_vals[common].min()),
        max(obs_vals[common].max(), mod_vals[common].max()),
    ]
    scatter_ax.plot(lims, lims, "k--", alpha=0.5)
    scatter_ax.set_xlabel("Observed")
    scatter_ax.set_ylabel("Model")
    scatter_ax.set_title(f"{variable} — Scatter")
    scatter_ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()

    if output_path:
        fig.savefig(str(output_path), bbox_inches="tight")
        logger.info("Saved comparison plot: %s", output_path)

    return fig
