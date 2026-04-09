"""CLI: Compute statistical metrics from two datasets.

This is a general-purpose tool for computing evaluation statistics between
any two CSV/DAT files that share common column names.

Usage::

    python scripts/run_metrics.py \\
        --dataset-a data/salvador-12.95-38.51.dat \\
        --dataset-b data/rio-de-janeiro-22.91-43.17.dat \\
        --columns T2 PSFC Q2 \\
        --output output/metrics_result.csv

    # Or compare all common columns:
    python scripts/run_metrics.py \\
        --dataset-a observations.csv \\
        --dataset-b predictions.csv \\
        --output metrics.csv
"""

from __future__ import annotations

import sys

import click
import pandas as pd

from labmim_micrometeorology.common.logging import setup_logging
from labmim_micrometeorology.stats.comparison import read_dataset
from labmim_micrometeorology.stats.metrics import compute_all


@click.command()
@click.option("--dataset-a", "-a", required=True, type=click.Path(exists=True), help="First dataset (treated as 'observed').")
@click.option("--dataset-b", "-b", required=True, type=click.Path(exists=True), help="Second dataset (treated as 'predicted').")
@click.option("--columns", "-c", multiple=True, default=None, help="Columns to evaluate. If omitted, all common columns are used.")
@click.option("--output", "-o", default=None, help="Output CSV for metrics table. If omitted, prints to stdout.")
@click.option("--separator", "-s", default=",", help="Column separator for input files.")
@click.option("--join", type=click.Choice(["index", "nearest"]), default="index", help="How to align rows: by exact index match or nearest timestamp.")
@click.option("--tolerance", default="30min", help="Max offset for 'nearest' join.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    dataset_a: str,
    dataset_b: str,
    columns: tuple[str, ...],
    output: str | None,
    separator: str,
    join: str,
    tolerance: str,
    log_level: str,
) -> None:
    """Compute statistical metrics between two datasets.

    Reads two CSV/DAT files, finds common columns, and computes RMSE, MAE,
    MBE, R², correlation, d-index, IOA, and NRMSE for each column.
    """
    setup_logging(log_level)

    click.echo(f"Dataset A: {dataset_a}")
    click.echo(f"Dataset B: {dataset_b}")

    df_a = read_dataset(dataset_a, separator=separator)
    df_b = read_dataset(dataset_b, separator=separator)

    # Determine columns to compare
    if columns:
        cols = [c for c in columns if c in df_a.columns and c in df_b.columns]
        missing = [c for c in columns if c not in cols]
        if missing:
            click.echo(f"⚠ Columns not found in both datasets: {missing}")
    else:
        cols = sorted(set(df_a.columns) & set(df_b.columns))

    if not cols:
        click.echo("✗ No common columns found between the two datasets")
        sys.exit(1)

    click.echo(f"Comparing {len(cols)} columns: {cols}")

    # Align datasets
    if join == "nearest" and hasattr(df_a.index, "tz"):
        aligned = pd.merge_asof(
            df_a[cols].sort_index(),
            df_b[cols].sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(tolerance),
            suffixes=("_a", "_b"),
            direction="nearest",
        )
    elif join == "nearest":
        aligned = pd.merge_asof(
            df_a[cols].reset_index().sort_values(df_a.index.name or "index"),
            df_b[cols].reset_index().sort_values(df_b.index.name or "index"),
            on=df_a.index.name or "index",
            tolerance=pd.Timedelta(tolerance),
            suffixes=("_a", "_b"),
            direction="nearest",
        )
    else:
        # Exact index join
        aligned = df_a[cols].join(df_b[cols], lsuffix="_a", rsuffix="_b", how="inner")

    if aligned.empty:
        click.echo("✗ No overlapping data after alignment")
        sys.exit(1)

    click.echo(f"Aligned {len(aligned)} rows")

    # Compute metrics per column
    results: dict[str, dict[str, float]] = {}
    for col in cols:
        a_col = f"{col}_a"
        b_col = f"{col}_b"
        if a_col in aligned.columns and b_col in aligned.columns:
            metrics = compute_all(aligned[a_col].values, aligned[b_col].values)
            results[col] = metrics

    metrics_df = pd.DataFrame(results)

    # Output
    click.echo(f"\n{'═' * 60}")
    click.echo(metrics_df.to_string(float_format="%.4f"))
    click.echo(f"{'═' * 60}")

    if output:
        metrics_df.to_csv(output)
        click.echo(f"\n✓ Saved to {output}")


if __name__ == "__main__":
    main()
