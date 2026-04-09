"""CLI: Run model vs. observation comparison.

Usage::

    python scripts/run_comparison.py --obs data/obs/salvador.csv \\
        --model data/model/wrf_series.csv --output output/comparison/
"""

from __future__ import annotations

import click
import matplotlib

matplotlib.use("Agg")

from labmim_micrometeorology.common.logging import setup_logging
from labmim_micrometeorology.common.paths import ensure_dir
from labmim_micrometeorology.stats.comparison import (
    compare_all_variables,
    pair_dataframes,
    plot_comparison,
    read_dataset,
)


@click.command()
@click.option("--obs", required=True, type=click.Path(exists=True), help="Observation data file.")
@click.option("--model", required=True, type=click.Path(exists=True), help="Model data file.")
@click.option("--output", "-o", required=True, help="Output directory.")
@click.option("--separator", default=",", help="Column separator for input files.")
@click.option("--tolerance", default="30min", help="Max time offset for pairing (e.g. 30min, 1h).")
@click.option("--plots/--no-plots", default=True, help="Generate comparison plots.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    obs: str,
    model: str,
    output: str,
    separator: str,
    tolerance: str,
    plots: bool,
    log_level: str,
) -> None:
    """Compare model predictions with observational data."""
    setup_logging(log_level)
    out_dir = ensure_dir(output)

    click.echo(f"Observations: {obs}")
    click.echo(f"Model:        {model}")

    df_obs = read_dataset(obs, separator=separator)
    df_model = read_dataset(model, separator=separator)

    paired = pair_dataframes(df_obs, df_model, tolerance=tolerance)
    if paired.empty:
        click.echo("⚠ No overlapping data found")
        return

    click.echo(f"Paired {len(paired)} time steps")

    # Compute metrics
    metrics_df = compare_all_variables(paired)
    metrics_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    click.echo(f"\n{metrics_df.to_string()}")
    click.echo(f"\n✓ Metrics saved to {metrics_path}")

    # Plots
    if plots:
        variables = sorted({
            c.replace("_obs", "")
            for c in paired.columns
            if c.endswith("_obs") and c.replace("_obs", "_model") in paired.columns
        })
        for var in variables:
            plot_comparison(paired, var, output_path=out_dir / f"comparison_{var}.png")
        click.echo(f"✓ Plots saved to {out_dir}")


if __name__ == "__main__":
    main()
