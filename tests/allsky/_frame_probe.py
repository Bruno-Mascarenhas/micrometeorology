"""One-epoch image-mode ``center_frame`` probes: the served member and the two controls the publisher tests score with."""

from pathlib import Path
from typing import Any

from allsky.config import ExperimentConfig
from tests.allsky import _synthetic as synthetic
from tests.allsky._block_probe import stub_image_backbone

IMAGE_PX = 8


def _config(root: Path, *, name: str, model: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": True,
            "name": name,
            "seed": 0,
            "output_dir": str(root / "out" / name),
            "data": {
                "manifest": "manifest.parquet",
                "data_root": str(root),
                "split_artifact": "splits.json",
                "input_mode": "image",
                "alignment": {"strategy": "center_frame"},
            },
            "features": {"set": "safe"},
            "targets": {
                "dhi": {"enabled": True, "loss": "huber"},
                "kindex": {"enabled": True, "kind": "kstar", "loss": "huber"},
                "sky": {"enabled": True},
            },
            "model": model,
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": {"monitor": "val_loss", "patience": 100},
            },
        }
    )


def train_frame_probes(tmp_path: Path) -> dict[str, Path]:
    """Train the image-only member, the scalars-only control and the climatology control; return their ``best.ckpt`` paths and the dataset root."""
    from allsky.training.engine import run_experiment

    root, _manifest, _ = synthetic.make_dataset(
        tmp_path, n_days=3, per_day=6, write_images=True, image_px=IMAGE_PX
    )
    checkpoints: dict[str, Path] = {"dataset": root}
    for name, model in (
        ("probe_s0", {"name": "image_only", "image_size": IMAGE_PX}),
        ("sensor_only_s0", {"name": "sensor_only"}),
        ("climatology_s0", {"name": "climatology"}),
    ):
        run_dir = tmp_path / f"{name}-run"
        run_experiment(
            _config(root, name=name, model=model),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=stub_image_backbone,
        )
        checkpoints[name] = run_dir / "best.ckpt"
    return checkpoints
