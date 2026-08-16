"""Publication figure of the Kt-Kd plane with the three published diffuse-fraction models.

The paper companion of the ``labmim-ktkd-v1`` artifact the sky page reads: both
come from :mod:`micrometeorology.stats.ktkd`, so a figure in a paper and the
curve on the site cannot disagree about the same record.

Marques Filho et al. (2016) draws as a line because it is a function of Kt alone.
Lemos et al. (2017) and the BRL of Ridley, Boland & Lauret (2010) read four more
predictors, so at one Kt they predict a spread: they are drawn as a median with a
p10-p90 envelope, never as a line.

Usage
-----
::

    uv run python scripts/plot_ktkd.py \\
        -i output/archive/station_hourly.parquet -o output/figures/
"""

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from micrometeorology.common.git import run_git, source_root
from micrometeorology.common.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.stats import ktkd as ktkd_stats
from micrometeorology.stats.sky_condition import sky_condition_summary

logger = logging.getLogger(__name__)

KT_EDGES = np.round(np.arange(0.0, 1.02, 0.02), 2)
KD_EDGES = np.round(np.arange(0.0, 1.02, 0.02), 2)

#: Okabe-Ito, so the model lines stay distinguishable in greyscale and to a
#: colour-blind reader, matching the palette the station figures already use.
MODEL_STYLE = {
    "marques_filho_2016": ("#D55E00", "Marques Filho et al. (2016)"),
    "lemos_2017": ("#0072B2", "Lemos et al. (2017)"),
    "ridley_brl_2010": ("#009E73", "BRL — Ridley et al. (2010)"),
}


def build_figure(hourly: pd.DataFrame) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw the Kt-Kd density with the three models, and return it with its scores.

    Parameters
    ----------
    hourly:
        The hourly database, indexed by naive station-local stamps.

    Returns
    -------
    tuple
        The figure and a dict of ``{model_id: {rmse, mbe, mae, n}}``.
    """
    prepared = ktkd_stats.prepare_ktkd(
        hourly, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    kt, kd = prepared["kt"], prepared["kd"]
    if kt.empty:
        raise ValueError("no Kt-Kd pair survived the gates; nothing to plot")

    predictors = (prepared["ast"], prepared["elevation"], prepared["daily_kt"], prepared["psi"])
    predictions = {
        "marques_filho_2016": ktkd_stats.marques_filho_2016(kt.to_numpy()),
        "lemos_2017": ktkd_stats.lemos_2017(kt.to_numpy(), *predictors),
        "ridley_brl_2010": ktkd_stats.ridley_brl_2010(kt.to_numpy(), *predictors),
    }

    figure, axes = plt.subplots(figsize=(7.2, 5.2))
    *_counts_and_edges, mesh = axes.hist2d(
        kt.to_numpy(), kd.to_numpy(), bins=[KT_EDGES, KD_EDGES], cmap="Greys", norm=LogNorm()
    )
    bar = figure.colorbar(mesh, ax=axes, pad=0.02)
    bar.set_label("horas por célula")

    grid = (KT_EDGES[:-1] + KT_EDGES[1:]) / 2.0
    colour, label = MODEL_STYLE["marques_filho_2016"]
    axes.plot(grid, ktkd_stats.marques_filho_2016(grid), color=colour, lw=2.0, label=label)

    for model_id in ("lemos_2017", "ridley_brl_2010"):
        colour, label = MODEL_STYLE[model_id]
        band = ktkd_stats.model_band(
            kt.to_numpy(), predictions[model_id], KT_EDGES, min_samples_per_bin=30
        )
        centres = np.asarray(band["kt"], dtype=float)
        median = np.array([np.nan if v is None else v for v in band["median"]])
        low = np.array([np.nan if v is None else v for v in band["p10"]])
        high = np.array([np.nan if v is None else v for v in band["p90"]])
        axes.fill_between(centres, low, high, color=colour, alpha=0.20, linewidth=0)
        axes.plot(centres, median, color=colour, lw=2.0, label=label)

    summary = sky_condition_summary(kt.to_numpy())
    for bound in summary["kt_upper_bounds"]:
        axes.axvline(bound, color="#666666", ls=":", lw=1.0)
    for condition in summary["conditions"]:
        lower = condition["kt_range"][0] or 0.0
        upper = condition["kt_range"][1] or 1.0
        axes.annotate(
            condition["id"].upper(),
            xy=((lower + upper) / 2.0, 0.965),
            ha="center",
            fontsize=9,
            color="#444444",
        )

    axes.set_xlabel("Índice de claridade $K_t$")
    axes.set_ylabel("Fração difusa $K_d = H_d/H$")
    axes.set_xlim(0.0, 1.0)
    axes.set_ylim(0.0, 1.0)
    axes.legend(loc="lower left", fontsize=9, framealpha=0.9)
    axes.set_title(f"n = {len(kt):,} médias horárias")
    figure.tight_layout()

    scores = {
        model_id: ktkd_stats.regression_scores(kd.to_numpy(), predicted)
        for model_id, predicted in predictions.items()
    }
    return figure, scores


def render(input_path: Path, output_dir: Path) -> Path:
    """Write the figure for *input_path* into *output_dir* and return its path."""
    hourly = pd.read_parquet(input_path)
    figure, scores = build_figure(hourly)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "ktkd.png"
    commit = run_git(["rev-parse", "--short", "HEAD"], cwd=source_root()) or "unknown"
    figure.savefig(
        destination,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": f"labmim micrometeorology @ {commit}"},
    )
    plt.close(figure)
    for model_id, score in scores.items():
        logger.info(
            "%-20s rmse=%.4f mbe=%+.4f mae=%.4f n=%d",
            model_id,
            score["rmse"],
            score["mbe"],
            score["mae"],
            score["n"],
        )
    logger.info("wrote %s", destination)
    return destination


def main() -> None:
    """Parse the arguments and render the figure."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, required=True, help="Hourly parquet database.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Directory for the PNG.")
    arguments = parser.parse_args()
    render(arguments.input, arguments.output)


if __name__ == "__main__":
    main()
