"""A one-epoch image-mode ``sensor_block`` probe, the block checkpoint the serving tests score with."""

from pathlib import Path
from typing import Any

from allsky.config import ExperimentConfig
from tests.allsky import _synthetic as synthetic


def stub_image_backbone() -> Any:
    from tests.allsky.test_engine_findings import TinyConvBackbone

    return TinyConvBackbone(dim=12)


def train_block_probe(tmp_path: Path) -> Path:
    """Train one epoch of an image-mode ``sensor_block`` probe on the synthetic frames."""
    from allsky.training.engine import run_experiment

    root, _manifest, _ = synthetic.make_dataset(
        tmp_path, n_days=3, per_day=6, write_images=True, image_px=8
    )
    cfg = ExperimentConfig.model_validate(
        {
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
                    "window_minutes": 5,
                    "max_frames": 5,
                    "one_sample_per_block": True,
                },
            },
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
            "model": {"name": "image_only", "image_size": 8},
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": {"monitor": "val_loss", "patience": 100},
            },
        }
    )
    run_dir = tmp_path / "block-run"
    run_experiment(
        cfg, data_root=root, output_dir=run_dir, image_backbone_builder=stub_image_backbone
    )
    return run_dir / "best.ckpt"
