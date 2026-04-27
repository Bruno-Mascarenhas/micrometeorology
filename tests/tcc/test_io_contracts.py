"""I/O, configuration, and serialization contracts."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from typing import Literal

import numpy as np
import pandas as pd
import pytest

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    SplitConfig,
)
from solrad_correction.data.loaders import load_sensor_hourly, load_sensor_raw, load_table


def _synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1h"),
            "f1": np.arange(6, dtype=np.float64),
            "f2": np.arange(10, 16, dtype=np.float64),
            "target": np.arange(20, 26, dtype=np.float64),
        }
    )


def test_default_config_validates_and_supported_models_are_scoped() -> None:
    ExperimentConfig().validate()

    for model_type in ["hgb", "gru", "bogus"]:
        with pytest.raises(ValueError, match=r"model\.model_type"):
            ExperimentConfig(model=ModelConfig(model_type=model_type)).validate()


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            ExperimentConfig(
                model=ModelConfig(model_type="transformer", tf_d_model=10, tf_nhead=3)
            ),
            "divisible",
        ),
        (
            ExperimentConfig(split=SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)),
            "split ratios",
        ),
        (ExperimentConfig(data=DataConfig(source_format="xlsx")), r"data\.source_format"),
        (
            ExperimentConfig(runtime=RuntimeConfig(num_workers=0, prefetch_factor=2)),
            "prefetch_factor",
        ),
        (ExperimentConfig(runtime=RuntimeConfig(limit_rows=0)), "runtime.limit_rows"),
        (ExperimentConfig(runtime=RuntimeConfig(device="quantum")), "runtime.device"),
        (ExperimentConfig(runtime=RuntimeConfig(checkpoint_every=0)), "runtime.checkpoint_every"),
        (ExperimentConfig(data=DataConfig(sensor_min_samples=0)), "data.sensor_min_samples"),
    ],
)
def test_invalid_config_cases_fail_clearly(cfg: ExperimentConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cfg.validate()


@pytest.mark.parametrize("source_format", ["csv", "parquet"])
def test_table_loading_projection_limit_index_and_dtype(
    source_format: Literal["csv", "parquet"],
) -> None:
    scratch = Path("scratch") / f"test_table_loading_{source_format}"
    path = scratch / f"hourly.{source_format}"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        if source_format == "csv":
            _synthetic_frame().to_csv(path, index=False)
        else:
            _synthetic_frame().to_parquet(path, index=False)

        df = load_sensor_hourly(
            path,
            source_format=source_format,
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


def test_raw_sensor_loading_uses_micrometeorology_ingestion_and_resampling() -> None:
    scratch = Path("scratch") / "test_raw_sensor_loading_contract"
    dat_path = scratch / "sensor.dat"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        dat_path.write_text(
            "TOA5\n"
            "TIMESTAMP,f1,target\n"
            "TS,unit,unit\n"
            "meta,meta,meta\n"
            "2024-01-01 00:00:00,1,10\n"
            "2024-01-01 00:30:00,3,14\n"
            "2024-01-01 01:00:00,5,18\n",
            encoding="utf-8",
        )

        df = load_sensor_raw(
            scratch,
            pattern="*.dat",
            resample_freq="1h",
            min_samples=1,
        )

        assert list(df.columns) == ["f1", "target"]
        assert len(df) == 2
        assert df["f1"].iloc[0] == 2.0
        assert df["target"].iloc[0] == 12.0
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_invalid_auto_format_and_csv_cache_contracts() -> None:
    scratch = Path("scratch") / "test_table_cache_contract"
    csv_path = scratch / "hourly.csv"
    bad_path = scratch / "hourly.unsupported"
    cache_dir = scratch / "cache"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        _synthetic_frame().to_csv(csv_path, index=False)
        bad_path.write_text("timestamp,f1,target\n2024-01-01,1,2\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Could not detect"):
            load_table(bad_path)

        df1 = load_table(csv_path, datetime_column="timestamp", cache_dir=str(cache_dir))
        df2 = load_table(csv_path, datetime_column="timestamp", cache_dir=str(cache_dir))

        assert (cache_dir / "hourly.parquet").exists()
        pd.testing.assert_frame_equal(df1, df2)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_csv_cache_refreshes_when_source_is_newer_and_skips_limited_reads() -> None:
    scratch = Path("scratch") / "test_table_cache_refresh_contract"
    csv_path = scratch / "hourly.csv"
    cache_dir = scratch / "cache"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        _synthetic_frame().to_csv(csv_path, index=False)
        load_table(csv_path, datetime_column="timestamp", cache_dir=str(cache_dir))

        time.sleep(0.1)
        csv_path.write_text(
            "timestamp,f1,f2,target\n2024-01-01 00:00:00,99,10,20\n2024-01-01 01:00:00,88,11,21\n",
            encoding="utf-8",
        )
        refreshed = load_table(csv_path, datetime_column="timestamp", cache_dir=str(cache_dir))

        limited_cache = scratch / "limited_cache"
        load_table(
            csv_path,
            datetime_column="timestamp",
            cache_dir=str(limited_cache),
            limit_rows=1,
        )

        assert refreshed["f1"].iloc[0] == 99.0
        assert not (limited_cache / "hourly.parquet").exists()
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
