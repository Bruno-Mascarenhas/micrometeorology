"""Figure of the cumulative clearness index F(Kt) with the Escobedo sky conditions.

The publication companion of the ``clearness_index_cumulative`` artifact the
climatology exporter writes: same computation, same frozen edges, same published
bounds, so a figure in a paper and the curve on the site cannot disagree.

Every number comes from :mod:`micrometeorology.stats.sky_condition` and the
exporter's own spec — this file only arranges axes and labels.

Usage
-----
::

    uv run python scripts/plot_clearness_cumulative.py \\
        -i output/archive/station_hourly.parquet -o output/figures/
"""

import logging
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from micrometeorology.common.git import run_git, source_root
from micrometeorology.common.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.stats import distributions as dist
from micrometeorology.stats import ktkd as ktkd_stats
from micrometeorology.stats.sky_condition import (
    KT_CUMULATIVE_EDGES,
    cumulative_fractions,
    sky_condition_summary,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)

#: Okabe-Ito, the colour-vision-safe palette the station graphs already use, so
#: the four conditions read the same here as on every other published figure.
CONDITION_COLORS = ("#0072B2", "#56B4E9", "#E69F00", "#D55E00")


def build_figure(hourly: pd.DataFrame) -> tuple[plt.Figure, dict[str, Any]]:
    """Draw F(Kt) with the sky-condition bands, and return it with its summary.

    Parameters
    ----------
    hourly:
        The hourly database, indexed by naive station-local stamps, as
        ``labmim-archive`` writes it.

    Returns
    -------
    tuple
        The figure and the sky-condition summary that annotates it.
    """
    values = ktkd_stats.prepare_clearness(
        hourly, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    ).to_numpy()
    if not values.size:
        raise ValueError("no clearness index survived the daylight gate; nothing to plot")

    binned = dist.histogram(values, KT_CUMULATIVE_EDGES)
    total = binned.n + binned.below + binned.above
    cumulative = cumulative_fractions(binned.counts, total=total)
    summary = sky_condition_summary(values)

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    upper_edges = np.asarray(KT_CUMULATIVE_EDGES[1:], dtype=float)
    axes.step(upper_edges, cumulative, where="post", color="#000000", linewidth=1.6)

    lower = 0.0
    for condition, color in zip(summary["conditions"], CONDITION_COLORS, strict=True):
        upper = condition["kt_range"][1] or float(KT_CUMULATIVE_EDGES[-1])
        axes.axvspan(lower, upper, color=color, alpha=0.14)
        share = condition["fraction"]
        axes.annotate(
            f"{condition['id'].upper()}\n{share:.1%}"
            if share is not None
            else condition["id"].upper(),
            xy=((lower + upper) / 2.0, 0.06),
            ha="center",
            va="bottom",
            fontsize=9,
        )
        lower = upper

    axes.set_xlabel("Índice de claridade $K_t$")
    axes.set_ylabel("Frequência acumulada $F(K_t)$")
    axes.set_xlim(float(KT_CUMULATIVE_EDGES[0]), float(KT_CUMULATIVE_EDGES[-1]))
    axes.set_ylim(0.0, 1.0)
    axes.grid(alpha=0.3)
    axes.set_title(f"Condições de céu (Escobedo et al., 2009) — n = {summary['n']:,} horas")
    figure.tight_layout()
    return figure, summary


def _commit() -> str:
    """Commit the figure was produced at, stamped into its metadata."""
    return run_git(["rev-parse", "--short", "HEAD"], cwd=source_root()) or "unknown"


def render(input_path: Path, output_dir: Path) -> Path:
    """Write the figure for *input_path* into *output_dir* and return its path."""
    hourly = pd.read_parquet(input_path)
    figure, summary = build_figure(hourly)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "clearness_cumulative.png"
    figure.savefig(
        destination,
        dpi=300,
        bbox_inches="tight",
        metadata={"Software": f"labmim micrometeorology @ {_commit()}"},
    )
    plt.close(figure)
    logger.info(
        "wrote %s (n=%d, %s)",
        destination,
        summary["n"],
        ", ".join(
            f"{c['id'].upper()}={c['fraction']:.1%}"
            for c in summary["conditions"]
            if c["fraction"] is not None
        ),
    )
    return destination


@app.command()
def run(
    input_path: Annotated[Path, typer.Option("-i", "--input", help="Hourly parquet database.")],
    output_dir: Annotated[Path, typer.Option("-o", "--output", help="Directory for the PNG.")],
) -> None:
    """Render the figure for the hourly database into the output directory."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    render(input_path, output_dir)


def main() -> None:
    """Entry point for ``python scripts/plot_clearness_cumulative.py``."""
    app()


if __name__ == "__main__":
    main()
