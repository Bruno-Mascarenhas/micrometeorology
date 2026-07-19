"""Tests for allsky.data.datasets: batch contract, train-only stats, torch-free import."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky import solar
from allsky.config import SiteConfig
from allsky.data.datasets import MultimodalEmbeddingDataset, MultimodalImageDataset
from allsky.data.manifest import build_manifest
from allsky.features import resolve_feature_set

type TorchDataset = Any  # runtime type: torch.utils.data.Dataset[dict[str, Any]]

_MET = {
    "AirT1_C_Avg": (20.0, 30.0),
    "DP1_C_Avg": (10.0, 20.0),
    "RH1": (50.0, 90.0),
    "BP1_mbar_Avg": (1005.0, 1015.0),
    "WS_ms": (0.0, 8.0),
    "WindDir": (0.0, 360.0),
}


def _sensor(site: SiteConfig) -> pd.DataFrame:
    index = pd.date_range("2025-03-21 06:00", "2025-03-21 18:00", freq="5min")
    rng = np.random.default_rng(0)
    e0h = solar.extraterrestrial_ghi(index, site)
    data = {k: rng.uniform(lo, hi, len(index)) for k, (lo, hi) in _MET.items()}
    data["CM3Up_Wm2_Avg"] = 0.7 * e0h
    data["PSP_Wm2_Avg"] = 0.2 * e0h
    return pd.DataFrame(data, index=index)


def _build(tmp_path: Path, n: int = 6, image_size: int = 16):
    site = SiteConfig()
    times = pd.date_range("2025-03-21 09:00", periods=n, freq="30min")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for i, ts in enumerate(times):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        iio.imwrite(
            path, rng.integers(0, 256, (image_size, image_size, 3)).astype(np.uint8), quality=90
        )
        rows.append({"frame_path": str(path), "timestamp": ts, "video": "v.mp4", "index": i})
    frames = pd.DataFrame(rows)
    manifest, _ = build_manifest(frames, _sensor(site), site=site, data_root=tmp_path)
    return manifest, tmp_path


def _build_minutely(tmp_path: Path, periods: int = 11):
    """Manifest with 1-min-cadence noon frames (for temporal-window tests)."""
    site = SiteConfig()
    times = pd.date_range("2025-03-21 12:00", periods=periods, freq="1min")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
    rows = []
    for i, ts in enumerate(times):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        iio.imwrite(path, rng.integers(0, 256, (8, 8, 3)).astype(np.uint8), quality=90)
        rows.append({"frame_path": str(path), "timestamp": ts, "video": "v.mp4", "index": i})
    manifest, _ = build_manifest(pd.DataFrame(rows), _sensor(site), site=site, data_root=tmp_path)
    return manifest


class FakeEmbeddingReader:
    """Deterministic hash-based embedding reader (no torch, no I/O)."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim

    def __call__(self, sample_id: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
        return rng.standard_normal(self.dim).astype(np.float32)


def test_datasets_module_imports_without_torch():
    """Contract: importing allsky.data.datasets must not pull torch."""
    code = (
        "import sys\n"
        "import allsky.data.datasets\n"
        "import allsky.data\n"
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def torch() -> Any:
    return pytest.importorskip("torch")


class TestImageDatasetContract:
    def test_item_keys_shapes_dtypes(self, torch: Any, tmp_path: Path):
        manifest, root = _build(tmp_path)
        features = resolve_feature_set("safe")
        dataset = MultimodalImageDataset(
            manifest, features, data_root=root, image_size=16, train=True
        )

        assert len(dataset) == len(manifest)
        item = dataset[0]
        assert set(item) == {"features", "image", "dhi", "kindex", "sky_class", "cloud_fraction"}
        assert item["features"].shape == (len(features),)
        assert item["features"].dtype == torch.float32
        assert item["image"].shape == (3, 16, 16)
        assert 0.0 <= float(item["image"].min()) <= float(item["image"].max()) <= 1.0
        assert item["dhi"].dtype == torch.float32
        assert item["kindex"].dtype == torch.float32
        assert item["sky_class"].dtype == torch.long
        assert item["cloud_fraction"].dtype == torch.float32

    def test_targets_are_raw_physical(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        item = dataset[2]
        # Raw target equals the manifest value (no normalization applied to targets).
        assert float(item["dhi"]) == pytest.approx(float(manifest["target_dhi"].iloc[2]), rel=1e-5)

    def test_cloud_fraction_missing_is_nan(self, torch: Any, tmp_path: Path):
        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        assert bool(torch.isnan(dataset[0]["cloud_fraction"]))

    def test_train_features_standardized(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        feats = np.stack([dataset[i]["features"].numpy() for i in range(len(dataset))])
        raw_std = (
            manifest.loc[:, resolve_feature_set("safe")].to_numpy(dtype=np.float64).std(axis=0)
        )
        varying = raw_std > 1e-3
        np.testing.assert_allclose(feats.mean(axis=0)[varying], 0.0, atol=1e-4)
        np.testing.assert_allclose(feats.std(axis=0)[varying], 1.0, atol=1e-4)

    def test_val_requires_train_stats(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, root = _build(tmp_path)
        with pytest.raises(ValueError, match="leak"):
            MultimodalImageDataset(
                manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=False
            )

    def test_val_uses_train_stats(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, root = _build(tmp_path)
        features = resolve_feature_set("safe")
        train = MultimodalImageDataset(
            manifest, features, data_root=root, image_size=16, train=True
        )
        val = MultimodalImageDataset(
            manifest, features, data_root=root, image_size=16, train=False, stats=train.stats
        )
        np.testing.assert_allclose(val[0]["features"].numpy(), train[0]["features"].numpy())

    def test_dataloader_collates(self, torch: Any, tmp_path: Path):
        from torch.utils.data import DataLoader

        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        loader: DataLoader[dict[str, Any]] = DataLoader(
            cast("TorchDataset", dataset), batch_size=3, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["image"].shape == (3, 3, 16, 16)
        assert batch["features"].shape == (3, len(resolve_feature_set("safe")))
        assert batch["sky_class"].shape == (3,)
        assert batch["sky_class"].dtype == torch.long


class TestEmbeddingDatasetContract:
    def test_item_has_embedding_not_image(self, torch: Any, tmp_path: Path):
        manifest, _ = _build(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        item = dataset[0]
        assert set(item) == {
            "features",
            "embedding",
            "dhi",
            "kindex",
            "sky_class",
            "cloud_fraction",
        }
        assert item["embedding"].shape == (8,)
        assert item["embedding"].dtype == torch.float32
        assert dataset.embedding_dim == 8

    def test_embedding_dim_discovered_without_declared_attr(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, _ = _build(tmp_path)

        def reader(sample_id: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
            return rng.standard_normal(5).astype(np.float32)

        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        assert dataset.embedding_dim == 5  # inferred from the first read

    def test_deterministic_reader_is_repeatable(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, _ = _build(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        np.testing.assert_array_equal(
            dataset[1]["embedding"].numpy(), dataset[1]["embedding"].numpy()
        )

    def test_wrong_dim_raises(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest, _ = _build(tmp_path)
        dims = iter([8, 8, 3])  # third read has the wrong length

        def reader(sample_id: str) -> np.ndarray:  # noqa: ARG001
            return np.zeros(next(dims), dtype=np.float32)

        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        dataset[0]
        dataset[1]
        with pytest.raises(ValueError, match="does not match"):
            dataset[2]


class TestEmbeddingWindowModes:
    def test_center_frame_is_own_embedding(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest = _build_minutely(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )  # window defaults to center_frame
        item = dataset[3]
        assert set(item) == {
            "features",
            "embedding",
            "dhi",
            "kindex",
            "sky_class",
            "cloud_fraction",
        }
        np.testing.assert_array_equal(
            item["embedding"].numpy(), reader(str(manifest["sample_id"].iloc[3]))
        )

    def test_mean_embedding_equals_manual_window_mean(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest = _build_minutely(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest,
            resolve_feature_set("safe"),
            embedding_reader=reader,
            train=True,
            window="mean_embedding",
            window_minutes=5.0,
        )
        # 12:05 window [12:02:30, 12:07:30] -> 12:03..12:07 (1-min cadence).
        idx = int(manifest.index[manifest["sample_id"] == "allsky-20250321-1205"][0])
        members = [f"allsky-20250321-120{m}" for m in range(3, 8)]
        expected = np.mean([reader(m) for m in members], axis=0).astype(np.float32)
        item = dataset[idx]
        assert "embedding_seq" not in item
        np.testing.assert_allclose(item["embedding"].numpy(), expected, rtol=1e-6)

    def test_attention_pooling_emits_padded_seq_and_mask(self, torch: Any, tmp_path: Path):
        manifest = _build_minutely(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest,
            resolve_feature_set("safe"),
            embedding_reader=reader,
            train=True,
            window="attention_pooling",
            window_minutes=5.0,
        )
        assert dataset.seq_len == 6  # ceil(5) + 1
        idx = int(manifest.index[manifest["sample_id"] == "allsky-20250321-1205"][0])
        item = dataset[idx]
        assert "embedding" not in item
        assert item["embedding_seq"].shape == (6, 8)
        assert item["frame_mask"].shape == (6,)
        assert item["frame_mask"].dtype == torch.bool
        assert int(item["frame_mask"].sum()) == 5  # 5 real frames, 1 padded slot
        assert not bool(item["frame_mask"][5])
        # Sequence is time-ordered: first slot is the earliest window frame.
        np.testing.assert_array_equal(
            item["embedding_seq"][0].numpy(), reader("allsky-20250321-1203")
        )
        # The padded trailing slot is zeros.
        np.testing.assert_array_equal(
            item["embedding_seq"][5].numpy(), np.zeros(8, dtype=np.float32)
        )

    def test_own_frame_always_in_window(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest = _build_minutely(tmp_path)
        reader = FakeEmbeddingReader(dim=8)
        dataset = MultimodalEmbeddingDataset(
            manifest,
            resolve_feature_set("safe"),
            embedding_reader=reader,
            train=True,
            window="mean_embedding",
            window_minutes=1.0,  # tiny window -> only the own frame qualifies
        )
        idx = 5
        expected = reader(str(manifest["sample_id"].iloc[idx]))
        np.testing.assert_allclose(dataset[idx]["embedding"].numpy(), expected, rtol=1e-6)

    def test_invalid_window_raises(self, torch: Any, tmp_path: Path):  # noqa: ARG002
        manifest = _build_minutely(tmp_path)
        with pytest.raises(ValueError, match="window"):
            MultimodalEmbeddingDataset(
                manifest,
                resolve_feature_set("safe"),
                embedding_reader=FakeEmbeddingReader(),
                train=True,
                window="bogus",  # type: ignore[arg-type]
            )
