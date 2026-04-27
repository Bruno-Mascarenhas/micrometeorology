"""Synthetic data-loading tests for CSV/Parquet paths."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solrad_correction.data.loaders import load_sensor_hourly, load_table


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1h"),
            "f1": np.arange(6, dtype=np.float64),
            "f2": np.arange(10, 16, dtype=np.float64),
            "target": np.arange(20, 26, dtype=np.float64),
        }
    )


def test_csv_loading_projection_limit_dates_and_dtype() -> None:
    scratch = Path("scratch") / "test_data_loading_csv"
    path = scratch / "hourly.csv"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        _synthetic_frame().to_csv(path, index=False)

        df = load_sensor_hourly(
            path,
            columns=["f1", "target"],
            datetime_column="timestamp",
            dtype_map={"f1": "float32"},
            limit_rows=3,
        )

        assert list(df.columns) == ["f1", "target"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 3
        assert str(df["f1"].dtype) == "float32"
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_parquet_loading_projection_limit_and_index() -> None:
    scratch = Path("scratch") / "test_data_loading_parquet"
    path = scratch / "hourly.parquet"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        _synthetic_frame().to_parquet(path, index=False)

        df = load_table(
            path,
            columns=["f2", "target"],
            datetime_column="timestamp",
            limit_rows=4,
        )

        assert list(df.columns) == ["f2", "target"]
        assert isinstance(df.index, pd.DatetimeIndex)
        assert len(df) == 4
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_invalid_auto_format_raises() -> None:
    scratch = Path("scratch") / "test_data_loading_invalid"
    path = scratch / "hourly.unsupported"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        path.write_text("timestamp,f1,target\n2024-01-01,1,2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Could not detect"):
            load_table(path)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
