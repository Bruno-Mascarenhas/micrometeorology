"""Synthetic benchmark for preprocessing fit/transform.

Examples
--------
Run with default settings:
    python benchmarks/solrad_correction/preprocessing.py

Run with a larger synthetic dataset:
    python benchmarks/solrad_correction/preprocessing.py --rows 100000 --features 48
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from solrad_correction.data.preprocessing import PreprocessingPipeline  # noqa: E402

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    rows: Annotated[int, typer.Option(help="Number of synthetic rows.")] = 20_000,
    features: Annotated[int, typer.Option(help="Number of feature columns.")] = 24,
    nan_rate: Annotated[float, typer.Option(help="Fraction of NaN values.")] = 0.02,
) -> None:
    """Benchmark preprocessing fit/transform performance."""
    df = _make_frame(rows, features, nan_rate)
    midpoint = max(1, int(len(df) * 0.7))
    train, test = df.iloc[:midpoint], df.iloc[midpoint:]
    pipeline = PreprocessingPipeline(
        scaler_type="standard",
        impute_strategy="mean",
        feature_columns=[f"f{i}" for i in range(features)],
        target_column="target",
    )

    started = time.perf_counter()
    train_out = pipeline.fit_transform(train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test_out = pipeline.transform(test)
    transform_seconds = time.perf_counter() - started

    typer.echo(
        {
            "benchmark": "preprocessing",
            "train_shape": train_out.shape,
            "test_shape": test_out.shape,
            "fit_seconds": round(fit_seconds, 6),
            "transform_seconds": round(transform_seconds, 6),
            "dropped_columns": len(pipeline.dropped_columns),
        }
    )


def _make_frame(rows: int, features: int, nan_rate: float) -> pd.DataFrame:
    rng = np.random.default_rng(43)
    values = rng.normal(size=(rows, features)).astype("float32")
    values[rng.random(size=values.shape) < nan_rate] = np.nan
    df = pd.DataFrame(values, columns=[f"f{i}" for i in range(features)])
    df["target"] = rng.normal(size=rows).astype("float32")
    return df


def main() -> None:
    app()


if __name__ == "__main__":
    main()
