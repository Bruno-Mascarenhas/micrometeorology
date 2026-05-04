"""CLI: Run model vs. observation comparison.

Examples
--------
Compare model output against observations:
    labmim-comparison --obs data/obs/salvador.csv --model data/model/wrf_series.csv -o output/comparison/
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import Annotated

import matplotlib
import typer

matplotlib.use("Agg")

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import ensure_dir
from micrometeorology.stats.comparison import (
    compare_all_variables,
    pair_dataframes,
    plot_comparison,
    read_dataset,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    obs: Annotated[Path, typer.Option(help="Observation data file.", exists=True)],
    model: Annotated[Path, typer.Option(help="Model data file.", exists=True)],
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory.")],
    separator: Annotated[str, typer.Option(help="Column separator for input files.")] = ",",
    tolerance: Annotated[
        str, typer.Option(help="Max time offset for pairing (e.g. 30min, 1h).")
    ] = "30min",
    plots: Annotated[
        bool, typer.Option("--plots/--no-plots", help="Generate comparison plots.")
    ] = True,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Compare model predictions with observational data."""
    setup_logging(log_level)
    out_dir = ensure_dir(output)

    typer.echo(f"Observations: {obs}")
    typer.echo(f"Model:        {model}")

    df_obs = read_dataset(str(obs), separator=separator)
    df_model = read_dataset(str(model), separator=separator)

    paired = pair_dataframes(df_obs, df_model, tolerance=tolerance)
    if paired.empty:
        typer.echo("Warning: No overlapping data found")
        return

    typer.echo(f"Paired {len(paired)} time steps")

    metrics_df = compare_all_variables(paired)
    metrics_path = out_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path)
    typer.echo(f"\n{metrics_df.to_string()}")
    typer.echo(f"\n>> Metrics saved to {metrics_path}")

    if plots:
        variables = sorted(
            {
                c.replace("_obs", "")
                for c in paired.columns
                if c.endswith("_obs") and c.replace("_obs", "_model") in paired.columns
            }
        )
        for var in variables:
            plot_comparison(paired, var, output_path=out_dir / f"comparison_{var}.png")
        typer.echo(f">> Plots saved to {out_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
