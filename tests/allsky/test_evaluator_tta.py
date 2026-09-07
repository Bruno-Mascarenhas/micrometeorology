"""Torch-gated tests for test-time rotation in allsky.evaluation.evaluator.

A backbone that pools every plane to its mean is exactly invariant under the
quarter turns four rotations produce, which are pixel permutations: its
averaged outputs must equal its plain ones. The refusal without solar-geometry
planes and the provenance entry are pinned beside it.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from allsky.config import ExperimentConfig
from allsky.evaluation.evaluator import (
    _rotation_averaged_outputs,
    _rotation_fill,
    evaluate_checkpoint,
)
from allsky.evaluation.reports import write_evaluation_report
from allsky.training.engine import run_experiment
from labmim_core.sky import SKY_CLASS_COUNT
from tests.allsky import _synthetic as synthetic


class _MeanBackbone(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.model: Any = nn.Module()
        self.model.patch_embed = nn.Module()
        self.model.patch_embed.proj = nn.Conv2d(3, 3, kernel_size=1)
        self.proj = nn.Linear(3, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(self.model.patch_embed.proj(image).mean(dim=(2, 3)))
        return out


class _OrientedBackbone(nn.Module):
    """Reads every pixel in place, so a turned frame is a different input."""

    def __init__(self, dim: int = 8, side: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.model: Any = nn.Module()
        self.model.patch_embed = nn.Module()
        self.model.patch_embed.proj = nn.Conv2d(3, 3, kernel_size=1)
        self.proj = nn.Linear(3 * side * side, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(self.model.patch_embed.proj(image).flatten(1))
        return out


def _train(
    tmp_path: Path, *, geometry: bool, backbone: type[nn.Module] = _MeanBackbone
) -> tuple[Path, Path]:
    root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)
    model: dict[str, Any] = {"name": "image_only", "image_size": 8, "backbone_frozen": False}
    if geometry:
        model["geometry_channels"] = ["cos_sun_angle"]
    cfg = ExperimentConfig.model_validate(
        {
            "experiment": True,
            "seed": 0,
            "output_dir": str(root / "out"),
            "data": {"data_root": str(root), "input_mode": "image"},
            "features": {"set": "safe"},
            "targets": {
                "dhi": {"enabled": True, "loss": "mae"},
                "kindex": {"enabled": True, "kind": "kstar", "loss": "mae"},
                "sky": {"enabled": True},
            },
            "model": model,
            "train": {
                "epochs": 1,
                "batch_size": 8,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": {"monitor": "val/kindex_mae", "patience": 100},
            },
        }
    )
    run_dir = tmp_path / "run"
    run_experiment(cfg, data_root=root, output_dir=run_dir, image_backbone_builder=backbone)
    return root, run_dir / "best.ckpt"


def test_four_rotations_of_an_oriented_model_change_its_predictions(tmp_path: Path):
    """The invariant model above cannot tell a rotation that never happened from
    one that did; a model reading pixels in place can."""
    root, ckpt = _train(tmp_path, geometry=True, backbone=_OrientedBackbone)

    plain = evaluate_checkpoint(
        ckpt, split="val", data_root=root, image_backbone_builder=_OrientedBackbone
    )
    turned = evaluate_checkpoint(
        ckpt,
        split="val",
        data_root=root,
        image_backbone_builder=_OrientedBackbone,
        tta_rotations=4,
    )

    assert not np.allclose(turned.predictions["pred_kindex"], plain.predictions["pred_kindex"])


def test_four_rotations_of_an_invariant_model_reproduce_its_plain_predictions(tmp_path: Path):
    root, ckpt = _train(tmp_path, geometry=True)

    plain = evaluate_checkpoint(
        ckpt, split="val", data_root=root, image_backbone_builder=lambda: _MeanBackbone()
    )
    turned = evaluate_checkpoint(
        ckpt,
        split="val",
        data_root=root,
        image_backbone_builder=lambda: _MeanBackbone(),
        tta_rotations=4,
    )

    for column in ("pred_dhi", "pred_kindex", "prob_sky_clear"):
        np.testing.assert_allclose(
            turned.predictions[column], plain.predictions[column], rtol=1e-4, atol=1e-5
        )
    assert turned.predictions["pred_sky"].tolist() == plain.predictions["pred_sky"].tolist()
    assert turned.predictions["pred_sky_kt"].tolist() == plain.predictions["pred_sky_kt"].tolist()


def test_the_rotation_count_is_recorded_in_the_report_meta(tmp_path: Path):
    root, ckpt = _train(tmp_path, geometry=True)

    result = evaluate_checkpoint(
        ckpt,
        split="val",
        data_root=root,
        image_backbone_builder=lambda: _MeanBackbone(),
        tta_rotations=3,
    )
    write_evaluation_report(result, tmp_path / "report", predictions=False)

    assert result.meta["tta_rotations"] == 3
    written = json.loads((tmp_path / "report" / "eval_metrics.json").read_text(encoding="utf-8"))
    assert written["meta"]["tta_rotations"] == 3


def test_without_rotations_the_meta_records_zero(tmp_path: Path):
    root, ckpt = _train(tmp_path, geometry=True)

    result = evaluate_checkpoint(
        ckpt, split="val", data_root=root, image_backbone_builder=lambda: _MeanBackbone()
    )

    assert result.meta["tta_rotations"] == 0


def test_rotations_are_refused_on_a_checkpoint_without_solar_geometry_planes(tmp_path: Path):
    root, ckpt = _train(tmp_path, geometry=False)

    with pytest.raises(ValueError, match="geometry_channels"):
        evaluate_checkpoint(
            ckpt,
            split="val",
            data_root=root,
            image_backbone_builder=lambda: _MeanBackbone(),
            tta_rotations=4,
        )


class _CornerReader:
    """Emits the top-left pixel of the red plane as k* and as the first sky logit,
    so each quarter turn about the grid centre hands it a different corner."""

    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        pixel = batch["image"][:, 0, 0, 0]
        zeros = torch.zeros_like(pixel)
        return {
            "kindex": pixel,
            "sky_logits": torch.stack([pixel] + [zeros] * (SKY_CLASS_COUNT - 1), dim=-1),
        }


def _corner_values(frames: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [frames[:, 0, 0, 0], frames[:, 0, 0, -1], frames[:, 0, -1, -1], frames[:, 0, -1, 0]],
        dim=-1,
    )


def _four_quarter_turns(frames: torch.Tensor) -> dict[str, Any]:
    raw = {"image": frames}
    return _rotation_averaged_outputs(
        _CornerReader(), dict(raw), raw, rotations=4, fill=_rotation_fill(0), device="cpu"
    )


def test_a_regression_head_is_averaged_over_the_rotations():
    frames = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 6.0

    out = _four_quarter_turns(frames)

    torch.testing.assert_close(out["kindex"], _corner_values(frames).mean(dim=-1))


def test_the_sky_probabilities_are_the_mean_of_the_per_rotation_probabilities():
    frames = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(0)) * 6.0
    corners = _corner_values(frames)
    per_turn_logits = torch.stack(
        [corners] + [torch.zeros_like(corners)] * (SKY_CLASS_COUNT - 1), dim=-1
    )

    out = _four_quarter_turns(frames)

    torch.testing.assert_close(
        torch.softmax(out["sky_logits"], dim=-1), torch.softmax(per_turn_logits, dim=-1).mean(dim=1)
    )
