"""CLI: Compute statistical metrics from two datasets.

Examples
--------
Compare two datasets on specific columns:
    labmim-metrics -a data/salvador.dat -b data/rio.dat -c T2,PSFC,Q2 -o output/metrics.csv

Compare all common columns:
    labmim-metrics -a observations.csv -b predictions.csv -o metrics.csv
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.stats.comparison import read_dataset
from micrometeorology.stats.metrics import compute_all


class JoinMethod(StrEnum):
    """How the two datasets are aligned before metrics are computed.

    ``index`` inner-joins on matching index labels; ``nearest`` uses
    ``merge_asof`` to pair each row with the closest index value in the other
    dataset within the configured tolerance.
    """

    by_index = "index"
    nearest = "nearest"


app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


def _parse_csv(value: str | None) -> list[str]:
    """Parse comma-separated strings."""
    if not value:
        return []
    return [x.strip() for x in value.split(",")]


@app.command()
def run(
    dataset_a: Annotated[
        Path,
        typer.Option(
            "-a", "--dataset-a", help="First dataset (treated as 'observed').", exists=True
        ),
    ],
    dataset_b: Annotated[
        Path,
        typer.Option(
            "-b", "--dataset-b", help="Second dataset (treated as 'predicted').", exists=True
        ),
    ],
    columns: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--columns",
            help="Columns to evaluate, comma-separated. If omitted, all common columns.",
        ),
    ] = None,
    output: Annotated[
        Path | None, typer.Option("-o", "--output", help="Output CSV for metrics table.")
    ] = None,
    separator: Annotated[str, typer.Option("-s", help="Column separator for input files.")] = ",",
    join: Annotated[JoinMethod, typer.Option(help="How to align rows.")] = JoinMethod.by_index,
    tolerance: Annotated[str, typer.Option(help="Max offset for 'nearest' join.")] = "30min",
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Compute statistical metrics between two datasets.

    Reads two CSV/DAT files, finds common columns, and computes RMSE, MAE,
    MBE, R2, correlation, d-index, IOA, and NRMSE for each column.
    """
    setup_logging(log_level)

    typer.echo(f"Dataset A: {dataset_a}")
    typer.echo(f"Dataset B: {dataset_b}")

    df_a = read_dataset(str(dataset_a), separator=separator)
    df_b = read_dataset(str(dataset_b), separator=separator)

    col_list = _parse_csv(columns)
    if col_list:
        cols = [c for c in col_list if c in df_a.columns and c in df_b.columns]
        missing = [c for c in col_list if c not in cols]
        if missing:
            typer.echo(f"Warning: Columns not found in both datasets: {missing}")
    else:
        cols = sorted(set(df_a.columns) & set(df_b.columns))

    if not cols:
        typer.echo("Error: No common columns found between the two datasets")
        sys.exit(1)

    typer.echo(f"Comparing {len(cols)} columns: {cols}")

    # Align datasets
    if join == JoinMethod.nearest and hasattr(df_a.index, "tz"):
        aligned = pd.merge_asof(
            df_a[cols].sort_index(),
            df_b[cols].sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(tolerance),
            suffixes=("_a", "_b"),
            direction="nearest",
        )
    elif join == JoinMethod.nearest:
        aligned = pd.merge_asof(
            df_a[cols].reset_index().sort_values(df_a.index.name or "index"),  # type: ignore
            df_b[cols].reset_index().sort_values(df_b.index.name or "index"),  # type: ignore
            on=df_a.index.name or "index",
            tolerance=pd.Timedelta(tolerance),
            suffixes=("_a", "_b"),
            direction="nearest",
        )
    else:
        aligned = df_a[cols].join(df_b[cols], lsuffix="_a", rsuffix="_b", how="inner")

    if aligned.empty:
        typer.echo("Error: No overlapping data after alignment")
        sys.exit(1)

    typer.echo(f"Aligned {len(aligned)} rows")

    results: dict[str, dict[str, float]] = {}
    for col in cols:
        a_col, b_col = f"{col}_a", f"{col}_b"
        if a_col in aligned.columns and b_col in aligned.columns:
            results[col] = compute_all(aligned[a_col].values, aligned[b_col].values)  # type: ignore

    metrics_df = pd.DataFrame(results)

    typer.echo(f"\n{'=' * 60}")
    typer.echo(metrics_df.to_string(float_format="%.4f"))
    typer.echo(f"{'=' * 60}")

    if output:
        metrics_df.to_csv(output)
        typer.echo(f"\n>> Saved to {output}")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-metrics``)."""
    app()


if __name__ == "__main__":
    main()
