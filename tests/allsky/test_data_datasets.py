"""Tests for allsky.data.datasets: batch contract, train-only stats, torch-free import."""

import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.data.datasets import MultimodalEmbeddingDataset, MultimodalImageDataset
from allsky.data.manifest import build_manifest
from allsky.features import resolve_feature_set
from allsky.preprocessing import IMAGENET_MEAN, IMAGENET_STD, imagenet_standardize
from labmim_core import solar
from labmim_core.site import SiteConfig

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


def _reference_windows(manifest: pd.DataFrame, window_minutes: float) -> list[list[int]]:
    """Per-row window members straight from the docstring definition (O(n^2))."""
    times = pd.DatetimeIndex(manifest["timestamp_utc"]).tz_convert("UTC").tz_localize(None)
    times_ns = times.as_unit("ns").to_numpy().astype("int64")
    days = manifest["day_id"].astype(str).to_numpy()
    half_ns = round(window_minutes / 2.0 * 60_000_000_000)
    windows = []
    for row in range(len(manifest)):
        members = [
            other
            for other in range(len(manifest))
            if days[other] == days[row] and abs(times_ns[other] - times_ns[row]) <= half_ns
        ]
        # time order, ties broken by original position (a stable sort)
        windows.append(sorted(members, key=lambda p: (times_ns[p], p)))
    return windows


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
        # Standardized by the DINOv2 channel stats, so the range is the [0, 1]
        # frame mapped through (x - mean) / std, not [0, 1].
        bounds = np.array(
            [
                (lim - m) / sd
                for lim in (0.0, 1.0)
                for m, sd in zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)
            ],
            dtype=np.float32,
        )
        assert float(item["image"].min()) == pytest.approx(float(bounds.min()), abs=1e-5) or (
            float(bounds.min()) <= float(item["image"].min())
        )
        assert float(item["image"].max()) <= float(bounds.max()) + 1e-5
        assert item["dhi"].dtype == torch.float32
        assert item["kindex"].dtype == torch.float32
        assert item["sky_class"].dtype == torch.long
        assert item["cloud_fraction"].dtype == torch.float32

    @pytest.mark.usefixtures("torch")
    def test_train_features_standardized(self, tmp_path: Path):
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

    @pytest.mark.usefixtures("torch")
    def test_val_requires_train_stats(self, tmp_path: Path):
        manifest, root = _build(tmp_path)
        with pytest.raises(ValueError, match="leak"):
            MultimodalImageDataset(
                manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=False
            )

    @pytest.mark.usefixtures("torch")
    def test_val_uses_train_stats(self, tmp_path: Path):
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


class TestTargetItemTensors:
    """The shared target tensors are views into cached whole-column tensors."""

    def test_values_dtypes_and_shapes_match_the_manifest(self, torch: Any, tmp_path: Path):
        manifest, root = _build(tmp_path)
        features = resolve_feature_set("safe")
        dataset = MultimodalImageDataset(
            manifest, features, data_root=root, image_size=16, train=True
        )
        for idx in range(len(dataset)):
            item = dataset[idx]
            assert item["features"].dtype == torch.float32
            assert item["features"].shape == (len(features),)
            np.testing.assert_array_equal(
                item["features"].numpy(), dataset.stats.transform(manifest)[idx].astype(np.float32)
            )
            assert item["dhi"].shape == ()
            assert float(item["dhi"]) == pytest.approx(
                float(manifest["target_dhi"].iloc[idx]), rel=1e-6
            )
            assert int(item["sky_class"]) == int(manifest["sky_class"].iloc[idx])
            assert item["sky_class"].dtype == torch.long
            assert bool(torch.isnan(item["cloud_fraction"]))

    def test_items_share_storage_but_batches_do_not(self, torch: Any, tmp_path: Path):
        from torch.utils.data import default_collate

        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        batch = default_collate([dataset[0], dataset[1]])
        original = float(dataset[0]["dhi"])
        dataset[0]["dhi"].add_(1.0)  # in-place on a view into the dataset's column
        assert float(dataset[0]["dhi"]) == pytest.approx(original + 1.0)
        # default_collate stacks (and copies), so the batch kept the original value.
        assert float(batch["dhi"][0]) == pytest.approx(original)
        assert batch["dhi"].dtype == torch.float32

    @pytest.mark.usefixtures("torch")
    def test_column_tensors_are_writable(self, tmp_path: Path):
        """A read-only pandas view would make torch.from_numpy warn about UB."""
        manifest, root = _build(tmp_path)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        for array in (
            dataset._features,
            dataset._dhi,
            dataset._kindex,
            dataset._cloud_fraction,
            dataset._sky_class,
        ):
            assert array.flags.writeable


class TestImageDecoding:
    """The PIL decode path must be pixel-identical to the imageio+fromarray one."""

    @staticmethod
    def _imageio_recipe(path: Path, size: int) -> np.ndarray:
        from PIL import Image

        image = iio.imread(path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        if image.shape[0] != size or image.shape[1] != size:
            image = np.asarray(
                Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
            )
        scaled = image.astype(np.float32) / 255.0
        chw = np.ascontiguousarray(scaled.transpose(2, 0, 1))
        return np.ascontiguousarray(imagenet_standardize(chw))

    @pytest.mark.usefixtures("torch")
    def test_matches_imageio_recipe_on_rgb_jpeg(self, tmp_path: Path):
        manifest, root = _build(tmp_path, n=3, image_size=64)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=32, train=True
        )
        for path in manifest["image_path"]:
            loaded = dataset._load_image(root / str(path))
            expected = self._imageio_recipe(root / str(path), 32)
            assert loaded.dtype == expected.dtype
            np.testing.assert_array_equal(loaded, expected)

    @pytest.mark.usefixtures("torch")
    def test_image_mode_feeds_the_backbone_what_the_embedding_path_feeds_it(self, tmp_path: Path):
        """Both routes into DINOv2 must standardize identically.

        The offline embedding path standardizes by the ImageNet channel stats
        while image-mode training fed a raw [0, 1] frame, so a finetune ran
        about 1.3 sigma off the distribution the backbone was pretrained on and
        its features were not comparable with the frozen ones it was measured
        against.
        """
        import imageio.v3 as iio_local

        from allsky.embeddings.backbone import DinoV2Backbone

        manifest, root = _build(tmp_path, n=1, image_size=32)
        path = root / str(manifest["image_path"].iloc[0])
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=32, train=True
        )

        # transform() only reads self.image_size, so the hub weights stay unloaded.
        stub = DinoV2Backbone.__new__(DinoV2Backbone)
        stub.image_size = 32

        from_dataset = dataset._load_image(root / str(manifest["image_path"].iloc[0]))
        from_backbone = stub.transform([iio_local.imread(path)])[0].numpy().astype(np.float32)

        np.testing.assert_allclose(from_dataset, from_backbone, rtol=0, atol=1e-6)

    @pytest.mark.usefixtures("torch")
    def test_grayscale_is_channel_replicated(self, tmp_path: Path):
        manifest, root = _build(tmp_path, n=1, image_size=16)
        gray = root / "frames" / "gray.jpg"
        rng = np.random.default_rng(4)
        iio.imwrite(gray, rng.integers(0, 256, (16, 16), dtype=np.uint8), quality=90)
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        loaded = dataset._load_image(gray)
        assert loaded.shape == (3, 16, 16)
        np.testing.assert_array_equal(loaded, self._imageio_recipe(gray, 16))

    @pytest.mark.usefixtures("torch")
    def test_rgba_source_is_reduced_to_three_channels(self, tmp_path: Path):
        """The old recipe served a 4-channel array the 3-channel encoder cannot take."""
        manifest, root = _build(tmp_path, n=1, image_size=16)
        rgba = root / "frames" / "rgba.png"
        rng = np.random.default_rng(5)
        iio.imwrite(rgba, rng.integers(0, 256, (16, 16, 4), dtype=np.uint8))
        dataset = MultimodalImageDataset(
            manifest, resolve_feature_set("safe"), data_root=root, image_size=16, train=True
        )
        assert dataset._load_image(rgba).shape == (3, 16, 16)


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

    @pytest.mark.usefixtures("torch")
    def test_embedding_dim_discovered_without_declared_attr(self, tmp_path: Path):
        manifest, _ = _build(tmp_path)

        def reader(sample_id: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(sample_id)) % (2**32))
            return rng.standard_normal(5).astype(np.float32)

        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        assert dataset.embedding_dim == 5  # inferred from the first read

    @pytest.mark.usefixtures("torch")
    def test_wrong_dim_raises(self, tmp_path: Path):
        manifest, _ = _build(tmp_path)
        dims = iter([8, 8, 3])  # third read has the wrong length

        def reader(sample_id: str) -> np.ndarray:
            # The EmbeddingReader protocol names this argument, so it stays put;
            # this fake is driven by the length sequence, not by the id.
            del sample_id
            return np.zeros(next(dims), dtype=np.float32)

        dataset = MultimodalEmbeddingDataset(
            manifest, resolve_feature_set("safe"), embedding_reader=reader, train=True
        )
        dataset[0]
        dataset[1]
        with pytest.raises(ValueError, match="does not match"):
            dataset[2]


class TestEmbeddingWindowModes:
    @pytest.mark.usefixtures("torch")
    def test_center_frame_is_own_embedding(self, tmp_path: Path):
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

    @pytest.mark.usefixtures("torch")
    def test_mean_embedding_equals_manual_window_mean(self, tmp_path: Path):
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

    @pytest.mark.usefixtures("torch")
    def test_own_frame_always_in_window(self, tmp_path: Path):
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

    @pytest.mark.usefixtures("torch")
    def test_resolved_windows_match_the_reference_definition(self):
        """Vectorized grouping must equal the per-row definition, adversarially."""
        rng = np.random.default_rng(9)
        for trial in range(25):
            n_days = int(rng.integers(1, 4))
            days: list[np.ndarray] = []
            for day in range(n_days):
                base = pd.Timestamp("2025-03-21 12:00") + pd.Timedelta(days=day)
                # ragged day sizes, duplicate timestamps and gaps
                offsets = rng.integers(0, 14, int(rng.integers(1, 12)))
                days.append((base + pd.to_timedelta(sorted(offsets), unit="m")).to_numpy())
            times = pd.DatetimeIndex(np.concatenate(days))
            order = rng.permutation(len(times))  # shuffled row order
            times = times[order]
            manifest = pd.DataFrame(
                {
                    "sample_id": [f"s{i}" for i in range(len(times))],
                    "timestamp_utc": times.tz_localize("UTC"),
                    "day_id": [f"{t:%Y-%m-%d}" for t in times],
                    "f": rng.normal(size=len(times)),
                    "target_dhi": rng.normal(size=len(times)),
                    "target_kindex": rng.normal(size=len(times)),
                    "cloud_fraction": np.full(len(times), np.nan),
                    "sky_class": rng.integers(-1, 3, len(times)).astype(np.int64),
                }
            )
            window_minutes = float([0.5, 1.0, 7.0, 10.0, 13.0][trial % 5])
            dataset = MultimodalEmbeddingDataset(
                manifest,
                ["f"],
                embedding_reader=FakeEmbeddingReader(dim=2),
                window="mean_embedding",
                window_minutes=window_minutes,
            )
            assert dataset._windows == _reference_windows(manifest, window_minutes)

    @pytest.mark.usefixtures("torch")
    def test_invalid_window_raises(self, tmp_path: Path):
        manifest = _build_minutely(tmp_path)
        with pytest.raises(ValueError, match="window"):
            MultimodalEmbeddingDataset(
                manifest,
                resolve_feature_set("safe"),
                embedding_reader=FakeEmbeddingReader(),
                train=True,
                window="bogus",  # type: ignore[arg-type]
            )
