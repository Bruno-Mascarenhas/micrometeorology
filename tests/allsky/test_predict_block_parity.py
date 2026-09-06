"""``predict_block`` against the dataset itself as the oracle.

The recording tests in ``test_snapshot_predict.py`` pin the batch's shape and
which helper was called with which time; here the tensor
:class:`allsky.data.datasets.MultimodalImageDataset` serves for a block under
``sensor_block`` + ``one_sample_per_block`` is compared plane by plane with
what ``predict_block`` sends the model for the same frames. The synthetic
manifest is rebuilt at a one-minute cadence, so a five-minute block holds five
frames and ``max_frames`` actually engages inside the dataset.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from allsky.config import SITE_UTC_OFFSET_HOURS, ExperimentConfig
from allsky.data.contracts import resolve
from allsky.data.datasets import MultimodalImageDataset
from allsky.data.manifest import build_manifest, write_manifest_parquet
from allsky.data.splits import create_day_splits, save_split_artifact
from allsky.features.policy import SAFE_FEATURES
from allsky.geometry import resolve_geometry_channels
from allsky.preprocessing import PreprocessingPipeline
from allsky.snapshot import _clearsky_dhi_reference, _image_input, block_end_of, predict_block
from allsky.training.checkpointing import load_checkpoint, normalizers_from_checkpoint
from labmim_core.site import SiteConfig
from tests.allsky import _synthetic as synthetic
from tests.allsky._block_probe import stub_image_backbone

BLOCK_MINUTES = 5.0
MAX_FRAMES = 3
IMAGE_PX = 8
NIGHT_FLOOR_DEG = 5.0


def _minute_cadence_dataset(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    site = SiteConfig()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2025-03-20", periods=3, freq="D")
    rows = [
        {
            "frame_path": f"frames/allsky-{ts:%Y%m%d-%H%M%S}.jpg",
            "timestamp": ts,
            "video": "v.mp4",
            "index": index,
        }
        for index, ts in enumerate(
            ts
            for day in days
            for ts in pd.date_range(day + pd.Timedelta(hours=9), periods=60, freq="1min")
        )
    ]
    manifest, meta = build_manifest(
        pd.DataFrame(rows), synthetic._sensor(site, days[0], days[-1]), site=site, data_root=root
    )
    write_manifest_parquet(manifest, meta, root / "manifest.parquet")
    split = create_day_splits(
        manifest["day_id"].tolist(), val_fraction=0.34, test_fraction=0.0, seed=0
    )
    save_split_artifact(split, root / "splits.json")
    synthetic.write_frame_images(root, manifest, image_px=IMAGE_PX)
    return root, manifest


def _probe_config(root: Path, **model: Any) -> dict[str, Any]:
    return {
        "experiment": True,
        "seed": 0,
        "output_dir": str(root / "out"),
        "data": {
            "manifest": "manifest.parquet",
            "data_root": str(root),
            "split_artifact": "splits.json",
            "input_mode": "image",
            "alignment": {
                "strategy": "sensor_block",
                "window_minutes": BLOCK_MINUTES,
                "max_frames": MAX_FRAMES,
                "one_sample_per_block": True,
            },
        },
        "features": {"set": "safe"},
        "targets": {
            "dhi": {"enabled": True, "loss": "huber", "parameterization": "clearsky_index"},
            "sky": {"enabled": True},
        },
        "model": {"name": "image_only", "image_size": IMAGE_PX, **model},
        "train": {
            "epochs": 1,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
            "early_stopping": {"monitor": "val_loss", "patience": 100},
        },
    }


@pytest.fixture(scope="module")
def probe(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pd.DataFrame, Path]:
    from allsky.training.engine import run_experiment

    tmp_path = tmp_path_factory.mktemp("parity")
    root, manifest = _minute_cadence_dataset(tmp_path)
    cfg = ExperimentConfig.model_validate(_probe_config(root))
    run_dir = tmp_path / "run"
    run_experiment(
        cfg, data_root=root, output_dir=run_dir, image_backbone_builder=stub_image_backbone
    )
    return root, manifest, run_dir / "best.ckpt"


def _local(utc: Any) -> pd.Timestamp:
    return pd.Timestamp(utc).tz_convert("UTC").tz_localize(None) + pd.Timedelta(
        hours=SITE_UTC_OFFSET_HOURS
    )


def _block_frames_of(
    manifest: pd.DataFrame, root: Path, served: pd.Series
) -> tuple[list[tuple[Path, pd.Timestamp]], pd.Timestamp]:
    representative = _local(served["timestamp_utc"])
    end = block_end_of(representative, BLOCK_MINUTES)
    frames = [
        (resolve(str(row["image_path"]), root), _local(row["timestamp_utc"]))
        for _, row in manifest.iterrows()
        if row["day_id"] == served["day_id"]
        and block_end_of(_local(row["timestamp_utc"]), BLOCK_MINUTES) == end
    ]
    return frames, end


def _served_dataset(
    root: Path, manifest: pd.DataFrame, checkpoint: dict[str, Any]
) -> MultimodalImageDataset:
    cfg = ExperimentConfig.model_validate(checkpoint["config"])
    feature_normalizer, _ = normalizers_from_checkpoint(checkpoint)
    return MultimodalImageDataset(
        manifest,
        list(checkpoint["feature_columns"]),
        data_root=root,
        image_size=IMAGE_PX,
        train=False,
        stats=feature_normalizer,
        preprocess=PreprocessingPipeline.from_config(cfg),
        dhi_parameterization="clearsky_index",
        window="sensor_block",
        window_minutes=BLOCK_MINUTES,
        window_max_frames=MAX_FRAMES,
        one_sample_per_block=True,
    )


def _recorded_block_batch(
    frames: list[tuple[Path, pd.Timestamp]],
    checkpoint_path: Path,
    end: pd.Timestamp,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from allsky.modeling import registry

    seen: dict[str, Any] = {}

    def recording_model(batch: dict[str, Any]) -> dict[str, Any]:
        seen.update(batch)
        return {"dhi": torch.zeros(1)}

    monkeypatch.setattr(registry, "restore_model", lambda *_a, **_k: recording_model)
    result = predict_block(
        frames,
        checkpoint_path,
        min_solar_elevation_deg=NIGHT_FLOOR_DEG,
        block_end=end,
        trust_checkpoint=True,
        image_backbone_builder=stub_image_backbone,
    )
    return seen, result


def test_a_block_of_five_frames_is_capped_to_three_by_the_dataset_itself(
    probe: tuple[Path, pd.DataFrame, Path],
) -> None:
    root, manifest, checkpoint_path = probe
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)

    dataset = _served_dataset(root, manifest, checkpoint)
    frames, _end = _block_frames_of(manifest, root, dataset.served_manifest.iloc[3])

    assert len(frames) == 5
    assert dataset[3]["frame_mask"].tolist() == [True] * MAX_FRAMES


def test_predict_block_sends_the_image_sequence_the_dataset_serves(
    probe: tuple[Path, pd.DataFrame, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    root, manifest, checkpoint_path = probe
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    dataset = _served_dataset(root, manifest, checkpoint)
    item = dataset[3]
    frames, end = _block_frames_of(manifest, root, dataset.served_manifest.iloc[3])

    seen, _result = _recorded_block_batch(frames, checkpoint_path, end, monkeypatch)

    assert seen["frame_mask"][0].tolist() == item["frame_mask"].tolist()
    assert torch.equal(seen["image_seq"][0], item["image_seq"])


def test_predict_block_takes_the_representative_the_dataset_serves(
    probe: tuple[Path, pd.DataFrame, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, checkpoint_path = probe
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    dataset = _served_dataset(root, manifest, checkpoint)
    served = dataset.served_manifest.iloc[3]
    frames, end = _block_frames_of(manifest, root, served)

    _seen, result = _recorded_block_batch(frames, checkpoint_path, end, monkeypatch)

    assert result["block"]["representative"] == _local(served["timestamp_utc"]).isoformat()


def test_predict_block_scales_dhi_by_the_served_rows_clear_sky_reference(
    probe: tuple[Path, pd.DataFrame, Path],
) -> None:
    root, manifest, checkpoint_path = probe
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    dataset = _served_dataset(root, manifest, checkpoint)
    served = dataset.served_manifest.iloc[3]

    reference = _clearsky_dhi_reference(_local(served["timestamp_utc"]), SiteConfig())

    assert reference == pytest.approx(float(dataset[3]["dhi_scale"]), rel=1e-5)


def test_predict_block_standardizes_the_geometry_features_as_the_dataset_does(
    probe: tuple[Path, pd.DataFrame, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest, checkpoint_path = probe
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    dataset = _served_dataset(root, manifest, checkpoint)
    item = dataset[3]
    frames, end = _block_frames_of(manifest, root, dataset.served_manifest.iloc[3])
    geometry_slots = [
        slot
        for slot, name in enumerate(checkpoint["feature_columns"])
        if SAFE_FEATURES[name] is None
    ]

    seen, _result = _recorded_block_batch(frames, checkpoint_path, end, monkeypatch)

    assert geometry_slots
    np.testing.assert_allclose(
        seen["features"][0, geometry_slots].numpy(),
        item["features"][geometry_slots].numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def test_the_geometry_planes_of_every_co_frame_match_the_datasets(
    probe: tuple[Path, pd.DataFrame, Path],
) -> None:
    root, manifest, _checkpoint_path = probe
    cfg = ExperimentConfig.model_validate(_probe_config(root, geometry_channels=True))
    columns = [name for name, source in SAFE_FEATURES.items() if source is None]
    dataset = MultimodalImageDataset(
        manifest,
        columns,
        data_root=root,
        image_size=IMAGE_PX,
        train=True,
        geometry_channels=resolve_geometry_channels(True),
        window="sensor_block",
        window_minutes=BLOCK_MINUTES,
        window_max_frames=MAX_FRAMES,
        one_sample_per_block=True,
    )
    item = dataset[3]
    served = dataset.served_manifest.iloc[3]
    frames, _end = _block_frames_of(manifest, root, served)
    representative = _local(served["timestamp_utc"])
    kept = [frames[i] for i in np.linspace(0, len(frames) - 1, MAX_FRAMES).round().astype(int)]

    planes = [
        _image_input(path, IMAGE_PX, cfg, timestamp=representative, site=SiteConfig())
        for path, _ in kept
    ]

    for slot, plane in enumerate(planes):
        np.testing.assert_allclose(plane, item["image_seq"][slot].numpy(), rtol=1e-5, atol=1e-5)
