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
from solrad_correction.utils.io import save_predictions


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


def test_parquet_head_read_preserves_stored_datetime_index_under_projection() -> None:
    """Regression for finding 2: limit_rows + column projection must keep the index."""
    scratch = Path("scratch") / "test_parquet_head_index_contract"
    path = scratch / "hourly.parquet"
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "SWDOWN": np.arange(6, dtype=np.float32),
                "T2": np.arange(10, 16, dtype=np.float32),
                "SW_dif": np.arange(20, 26, dtype=np.float32),
            },
            index=index,
        ).to_parquet(path)

        df = load_sensor_hourly(
            path,
            source_format="parquet",
            columns=["SWDOWN", "SW_dif"],
            limit_rows=3,
        )

        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.equals(index[:3])
        assert list(df.columns) == ["SWDOWN", "SW_dif"]
        np.testing.assert_allclose(df["SWDOWN"], [0.0, 1.0, 2.0])
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_parquet_without_recoverable_datetime_index_raises_instead_of_epoch_fallback() -> None:
    """Regression for finding 2: numeric columns must never become 1970-epoch indexes."""
    scratch = Path("scratch") / "test_parquet_no_index_contract"
    path = scratch / "hourly.parquet"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "SWDOWN": np.arange(6, dtype=np.float32),
                "SW_dif": np.arange(20, 26, dtype=np.float32),
            }
        ).to_parquet(path, index=False)

        for limit_rows in [None, 3]:
            with pytest.raises(ValueError, match="datetime index"):
                load_sensor_hourly(path, source_format="parquet", limit_rows=limit_rows)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_save_predictions_writes_timestamps_and_rejects_misaligned_index() -> None:
    """Regression for finding 15: never silently drop the prediction index."""
    scratch = Path("scratch") / "test_save_predictions_contract"
    path = scratch / "predictions.csv"
    index = pd.date_range("2024-01-01", periods=4, freq="1h")
    y = np.arange(4, dtype=np.float64)
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        save_predictions(y, y, path, index=index)
        written = pd.read_csv(path, index_col=0, parse_dates=True)

        assert isinstance(written.index, pd.DatetimeIndex)
        assert written.index.name == "timestamp"
        assert written.index.equals(index)
        with pytest.raises(ValueError, match="does not match"):
            save_predictions(y, y, path, index=index[:3])
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
