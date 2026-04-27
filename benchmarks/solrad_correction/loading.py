"""Synthetic benchmark for CSV/Parquet loading paths."""

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

from solrad_correction.data.loaders import load_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--limit-rows", type=int, default=None)
    args = parser.parse_args()

    scratch = ROOT / "scratch" / "benchmarks" / "loading"
    scratch.mkdir(parents=True, exist_ok=True)
    frame = _make_frame(args.rows, args.features)
    path = scratch / f"synthetic.{args.format}"
    if args.format == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    columns = [f"f{i}" for i in range(min(args.features, 8))]
    started = time.perf_counter()
    loaded = load_table(
        path,
        columns=[*columns, "target"],
        datetime_column="timestamp",
        limit_rows=args.limit_rows,
    )
    elapsed = time.perf_counter() - started
    print(
        {
            "benchmark": "loading",
            "format": args.format,
            "rows": len(loaded),
            "cols": len(loaded.columns),
            "seconds": round(elapsed, 6),
        }
    )


def _make_frame(rows: int, features: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.normal(size=rows).astype("float32") for i in range(features)}
    data["target"] = rng.normal(size=rows).astype("float32")
    data["timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="1h")
    return pd.DataFrame(data)


if __name__ == "__main__":
    main()
