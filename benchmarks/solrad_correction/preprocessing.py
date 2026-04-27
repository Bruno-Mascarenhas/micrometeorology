"""Synthetic benchmark for preprocessing fit/transform."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from solrad_correction.data.preprocessing import PreprocessingPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--nan-rate", type=float, default=0.02)
    args = parser.parse_args()

    df = _make_frame(args.rows, args.features, args.nan_rate)
    midpoint = max(1, int(len(df) * 0.7))
    train = df.iloc[:midpoint]
    test = df.iloc[midpoint:]
    pipeline = PreprocessingPipeline(
        scaler_type="standard",
        impute_strategy="mean",
        feature_columns=[f"f{i}" for i in range(args.features)],
        target_column="target",
    )

    started = time.perf_counter()
    train_out = pipeline.fit_transform(train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    test_out = pipeline.transform(test)
    transform_seconds = time.perf_counter() - started

    print(
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
    mask = rng.random(size=values.shape) < nan_rate
    values[mask] = np.nan
    df = pd.DataFrame(values, columns=[f"f{i}" for i in range(features)])
    df["target"] = rng.normal(size=rows).astype("float32")
    return df


if __name__ == "__main__":
    main()
