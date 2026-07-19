"""Tests for allsky.dataset — pairing index, day split, torch dataset.

Sensor-side inputs are built as plain DataFrames with the columns that
``allsky.sensors.derive_targets`` contracts to produce (timestamp index,
feature columns, ``kt``, ``diffuse``, ``cloud_class``, ``target_source``) —
no import of ``allsky.sensors`` here.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.config import AllSkyConfig, ModelConfig, SensorConfig
from allsky.dataset import (
    AllSkyDataset,
    FeatureStats,
    build_index,
    infer_feature_columns,
)
from allsky.training import split_days

FEATURE_COLUMNS = ["ghi", "par"]


def make_sensor_cfg(**overrides: Any) -> SensorConfig:
    defaults: dict[str, Any] = {
        "paths": [],
        "ghi_column": "ghi",
        "feature_columns": list(FEATURE_COLUMNS),
        "tolerance_minutes": 5.0,
    }
    defaults.update(overrides)
    return SensorConfig(**defaults)


def make_sensor_df(times: list[str]) -> pd.DataFrame:
    """Sensor frame shaped like the derive_targets output contract."""
    index = pd.DatetimeIndex(pd.to_datetime(times), name="TIMESTAMP")
    n = len(index)
    return pd.DataFrame(
        {
            "ghi": np.linspace(100.0, 800.0, n),
            "par": np.linspace(50.0, 400.0, n),
            "kt": np.linspace(0.2, 0.8, n),
            "diffuse": np.linspace(30.0, 200.0, n),
            "cloud_class": np.arange(n) % 3,
            "target_source": "erbs_pseudo",
        },
        index=index,
    )


def make_manifest(times: list[str], video: str = "allsky-20260101.mp4") -> pd.DataFrame:
    timestamps = pd.to_datetime(times)
    return pd.DataFrame(
        {
            "frame_path": [f"frames/frame-{i}.jpg" for i in range(len(timestamps))],
            "timestamp": timestamps,
            "video": video,
            "index": range(len(timestamps)),
        }
    )


class TestBuildIndex:
    def test_matches_within_tolerance(self):
        # Frame at 10:02 is 3 min from the 10:05 sensor record -> matched.
        manifest = make_manifest(["2026-01-01 10:02"])
        sensor_df = make_sensor_df(["2026-01-01 10:05", "2026-01-01 12:00"])

        index_df = build_index(manifest, sensor_df, make_sensor_cfg())

        assert len(index_df) == 1
        assert index_df["sensor_timestamp"].iloc[0] == pd.Timestamp("2026-01-01 10:05")
        assert index_df["ghi"].iloc[0] == sensor_df["ghi"].iloc[0]
        assert index_df["target_source"].iloc[0] == "erbs_pseudo"

    def test_drops_frames_outside_tolerance(self):
        # Frame at 11:40 is 20 min from the nearest sensor record -> dropped.
        manifest = make_manifest(["2026-01-01 10:02", "2026-01-01 11:40"])
        sensor_df = make_sensor_df(["2026-01-01 10:05", "2026-01-01 12:00"])

        index_df = build_index(manifest, sensor_df, make_sensor_cfg())

        assert index_df["timestamp"].tolist() == [pd.Timestamp("2026-01-01 10:02")]

    def test_drops_night_frames_without_sensor_rows(self):
        # derive_targets drops low-sun sensor rows, so night frames find no
        # match within tolerance and are removed here.
        manifest = make_manifest(["2026-01-01 05:00", "2026-01-01 10:02"])
        sensor_df = make_sensor_df(["2026-01-01 10:05"])

        index_df = build_index(manifest, sensor_df, make_sensor_cfg())

        assert len(index_df) == 1
        assert index_df["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 10:02")

    def test_drops_rows_with_missing_targets(self):
        manifest = make_manifest(["2026-01-01 10:02", "2026-01-01 10:32"])
        sensor_df = make_sensor_df(["2026-01-01 10:05", "2026-01-01 10:35"])
        sensor_df.loc[sensor_df.index[1], "diffuse"] = np.nan

        index_df = build_index(manifest, sensor_df, make_sensor_cfg())

        assert index_df["sensor_timestamp"].tolist() == [pd.Timestamp("2026-01-01 10:05")]

    def test_cloud_class_is_int64(self):
        manifest = make_manifest(["2026-01-01 10:02"])
        index_df = build_index(manifest, make_sensor_df(["2026-01-01 10:05"]), make_sensor_cfg())
        assert index_df["cloud_class"].dtype == np.int64

    def test_missing_feature_column_raises(self):
        manifest = make_manifest(["2026-01-01 10:02"])
        sensor_df = make_sensor_df(["2026-01-01 10:05"]).drop(columns=["par"])
        with pytest.raises(ValueError, match="missing feature columns"):
            build_index(manifest, sensor_df, make_sensor_cfg())

    def test_missing_target_column_raises(self):
        manifest = make_manifest(["2026-01-01 10:02"])
        sensor_df = make_sensor_df(["2026-01-01 10:05"]).drop(columns=["cloud_class"])
        with pytest.raises(ValueError, match="derive_targets"):
            build_index(manifest, sensor_df, make_sensor_cfg())

    def test_accepts_root_config(self):
        manifest = make_manifest(["2026-01-01 10:02"])
        sensor_df = make_sensor_df(["2026-01-01 10:05"])
        cfg = AllSkyConfig(sensor=make_sensor_cfg())

        from_root = build_index(manifest, sensor_df, cfg)
        from_section = build_index(manifest, sensor_df, cfg.sensor)

        pd.testing.assert_frame_equal(from_root, from_section)

    def test_persists_parquet(self, tmp_path: Path):
        manifest = make_manifest(["2026-01-01 10:02"])
        out_path = tmp_path / "index" / "pairing.parquet"

        index_df = build_index(
            manifest, make_sensor_df(["2026-01-01 10:05"]), make_sensor_cfg(), out_path=out_path
        )

        assert out_path.exists()
        pd.testing.assert_frame_equal(pd.read_parquet(out_path), index_df)


def _multi_day_index(n_days: int, rows_per_day: int = 4) -> pd.DataFrame:
    timestamps = [
        pd.Timestamp("2026-01-01 10:00") + pd.Timedelta(days=d, minutes=5 * r)
        for d in range(n_days)
        for r in range(rows_per_day)
    ]
    manifest = make_manifest([str(t) for t in timestamps])
    manifest["ghi"] = np.linspace(100.0, 800.0, len(manifest))
    return manifest


class TestSplitByDay:
    def test_no_shared_days_between_splits(self):
        index_df = _multi_day_index(n_days=6)

        train_df, val_df = split_days(index_df, val_fraction=0.34, seed=0)

        train_days = set(train_df["timestamp"].dt.normalize())
        val_days = set(val_df["timestamp"].dt.normalize())
        assert train_days.isdisjoint(val_days)
        assert len(val_days) == 2
        assert len(train_df) + len(val_df) == len(index_df)
        assert not val_df.empty

    def test_at_least_one_val_day_with_small_fraction(self):
        train_df, val_df = split_days(_multi_day_index(n_days=3), val_fraction=0.01)
        assert len(set(val_df["timestamp"].dt.normalize())) == 1
        assert not train_df.empty

    def test_single_day_raises_so_train_falls_back_explicitly(self):
        # train() catches this and reuses the day with a loud warning.
        index_df = _multi_day_index(n_days=1)
        with pytest.raises(ValueError, match="2 distinct days"):
            split_days(index_df, val_fraction=0.2)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError, match="val_fraction"):
            split_days(_multi_day_index(n_days=2), val_fraction=1.0)


class TestTrainValFractionPropagates:
    def test_bad_val_fraction_raises_not_single_day_fallback(self, tmp_path: Path):
        # Finding F9: train() narrowed its bare `except ValueError` so that only
        # the "fewer than 2 distinct days" case falls back to train==val. A bad
        # val_fraction (here 1.5) must now propagate rather than be silently
        # swallowed into a single-day run. device='cpu' keeps this torch-free:
        # split_days rejects the fraction before any loader/model is built.
        from allsky.config import AllSkyConfig
        from allsky.training import train

        index_path = tmp_path / "index.parquet"
        _multi_day_index(n_days=3).to_parquet(index_path)
        cfg = AllSkyConfig()
        cfg.train.device = "cpu"
        cfg.train.out_dir = str(tmp_path / "run")
        with pytest.raises(ValueError, match="val_fraction"):
            train(cfg, index_path=index_path, val_fraction=1.5)


def test_infer_feature_columns_excludes_reserved():
    index_df = make_manifest(["2026-01-01 10:02"])
    index_df["ghi"] = 100.0
    index_df["par"] = 50.0
    index_df["kt"] = 0.5
    index_df["diffuse"] = 80.0
    index_df["cloud_class"] = 1
    index_df["target_source"] = "erbs_pseudo"
    index_df["sensor_timestamp"] = index_df["timestamp"]

    assert infer_feature_columns(index_df) == ["ghi", "par"]


def test_dataset_module_imports_without_torch():
    """Contract: importing allsky.dataset / allsky.video must not pull torch."""
    code = (
        "import sys\n"
        "import allsky.dataset\n"
        "import allsky.video\n"
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_feature_stats_roundtrip():
    frame = pd.DataFrame({"ghi": [100.0, 300.0], "par": [50.0, 50.0]})
    stats = FeatureStats.from_frame(frame, ["ghi", "par"])

    assert stats.std[1] == 1.0  # constant feature clamped, no div-by-zero
    restored = FeatureStats.from_dict(stats.to_dict())
    assert restored.columns == stats.columns
    np.testing.assert_allclose(restored.mean, stats.mean)
    np.testing.assert_allclose(restored.std, stats.std)


# --------------------------------------------------------------------------
# Torch-dependent tests
# --------------------------------------------------------------------------


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


def make_paired_index(tmp_path: Path, n: int = 4, image_size: int = 48) -> pd.DataFrame:
    """Pairing index with real JPEGs on disk, as produced by build_index."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    timestamps = pd.date_range("2026-01-01 10:00", periods=n, freq="5min")
    rows = []
    for i, ts in enumerate(timestamps):
        frame_path = tmp_path / f"frame-{i}.jpg"
        image = rng.integers(0, 256, size=(image_size, image_size, 3)).astype(np.uint8)
        iio.imwrite(frame_path, image, quality=92)
        rows.append(
            {
                "frame_path": str(frame_path),
                "timestamp": ts,
                "video": "allsky-20260101.mp4",
                "index": i,
                "sensor_timestamp": ts,
                "ghi": 100.0 * (i + 1),
                "par": 10.0 * (i + 1),
                "kt": 0.2 + 0.1 * i,
                "diffuse": 50.0 + 10.0 * i,
                "cloud_class": i % 3,
                "target_source": "erbs_pseudo",
            }
        )
    return pd.DataFrame(rows)


class TestAllSkyDataset:
    def test_item_shapes_and_dtypes(self, torch: Any, tmp_path: Path):
        index_df = make_paired_index(tmp_path)
        dataset = AllSkyDataset(index_df, ModelConfig(image_size=32), train=True)

        assert len(dataset) == 4
        assert dataset.feature_columns == ["ghi", "par"]
        item = dataset[1]
        assert set(item) == {"image", "features", "cloud_class", "diffuse"}
        assert item["image"].shape == (3, 32, 32)
        assert item["image"].dtype == torch.float32
        assert 0.0 <= float(item["image"].min()) <= float(item["image"].max()) <= 1.0
        assert item["features"].shape == (2,)
        assert item["features"].dtype == torch.float32
        assert item["cloud_class"].dtype == torch.long
        assert int(item["cloud_class"]) == 1
        assert item["diffuse"].dtype == torch.float32
        assert float(item["diffuse"]) == pytest.approx(60.0)

    def test_train_features_are_standardized(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        index_df = make_paired_index(tmp_path)
        dataset = AllSkyDataset(index_df, ModelConfig(image_size=32), train=True)

        features = np.stack([dataset[i]["features"].numpy() for i in range(len(dataset))])
        np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-5)
        np.testing.assert_allclose(features.std(axis=0), 1.0, atol=1e-4)

    def test_val_requires_train_stats(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        index_df = make_paired_index(tmp_path)
        with pytest.raises(ValueError, match="leak"):
            AllSkyDataset(index_df, ModelConfig(image_size=32), train=False)

    def test_val_uses_train_stats(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        train_df = make_paired_index(tmp_path / "train", n=4)
        val_df = make_paired_index(tmp_path / "val", n=2)
        cfg = ModelConfig(image_size=32)

        train_ds = AllSkyDataset(train_df, cfg, train=True)
        val_ds = AllSkyDataset(val_df, cfg, train=False, stats=train_ds.stats)

        raw = val_df.loc[:, ["ghi", "par"]].to_numpy(dtype=np.float32)
        expected = (raw - train_ds.stats.mean) / train_ds.stats.std
        actual = np.stack([val_ds[i]["features"].numpy() for i in range(len(val_ds))])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    def test_stats_column_mismatch_raises(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        index_df = make_paired_index(tmp_path)
        wrong = FeatureStats.from_frame(index_df, ["kt"])
        with pytest.raises(ValueError, match="do not match"):
            AllSkyDataset(index_df, ModelConfig(image_size=32), stats=wrong)

    def test_dataloader_collates_batches(self, torch: Any, tmp_path: Path):
        from torch.utils.data import DataLoader

        index_df = make_paired_index(tmp_path)
        dataset = AllSkyDataset(index_df, ModelConfig(image_size=32), train=True)
        # AllSkyDataset deliberately duck-types the map-style Dataset protocol
        # (no torch import at module scope), so the nominal type differs.
        loader: DataLoader[dict[str, Any]] = DataLoader(
            dataset,  # type: ignore[arg-type]
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(loader))

        assert batch["image"].shape == (2, 3, 32, 32)
        assert batch["features"].shape == (2, 2)
        assert batch["cloud_class"].shape == (2,)
        assert batch["cloud_class"].dtype == torch.long
        assert batch["diffuse"].shape == (2,)
