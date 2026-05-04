"""Synthetic benchmark for CSV/Parquet loading paths.

Examples
--------
Run with default settings (Parquet):
    python benchmarks/solrad_correction/loading.py

Run with CSV format and row limit:
    python benchmarks/solrad_correction/loading.py --format csv --limit-rows 5000
"""

from __future__ import annotations

import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import pandas as pd
import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from solrad_correction.data.loaders import load_table  # noqa: E402

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


class FileFormat(StrEnum):
    csv = "csv"
    parquet = "parquet"


@app.command()
def run(
    rows: Annotated[int, typer.Option(help="Number of synthetic rows.")] = 10_000,
    features: Annotated[int, typer.Option(help="Number of feature columns.")] = 16,
    fmt: Annotated[FileFormat, typer.Option("--format", help="File format.")] = FileFormat.parquet,
    limit_rows: Annotated[int | None, typer.Option(help="Limit rows loaded.")] = None,
) -> None:
    """Benchmark CSV/Parquet loading throughput."""
    scratch = ROOT / "scratch" / "benchmarks" / "loading"
    scratch.mkdir(parents=True, exist_ok=True)
    frame = _make_frame(rows, features)
    path = scratch / f"synthetic.{fmt.value}"
    if fmt == FileFormat.csv:
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)

    columns = [f"f{i}" for i in range(min(features, 8))]
    started = time.perf_counter()
    loaded = load_table(
        path, columns=[*columns, "target"], datetime_column="timestamp", limit_rows=limit_rows
    )
    elapsed = time.perf_counter() - started
    typer.echo(
        {
            "benchmark": "loading",
            "format": fmt.value,
            "rows": len(loaded),
            "cols": len(loaded.columns),
            "seconds": round(elapsed, 6),
        }
    )


def _make_frame(rows: int, features: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data: dict[str, Any] = {
        f"f{i}": rng.normal(size=rows).astype("float32") for i in range(features)
    }
    data["target"] = rng.normal(size=rows).astype("float32")
    data["timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="1h")
    return pd.DataFrame(data)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
