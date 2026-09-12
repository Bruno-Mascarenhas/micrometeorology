"""Torch-gated tests for the layer-wise learning-rate decay of the backbone.

A three-block ViT-shaped stub — patch embedding before the blocks, a final norm
after them — pins the tiering: block ``i`` of ``n`` at ``decay ** (n - i)``,
the embedding tier at ``decay ** (n + 1)``, the tail at the base rate.  The engine
tests go through :func:`_build_optimizer` and one image-mode epoch so the
groups the optimizer actually holds, and the labels logged for them, are the
ones under test.  ``None`` must reproduce today's groups exactly.
"""

import logging
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from allsky.config import ExperimentConfig
from allsky.modeling.backbone_families import BackboneCapabilityError
from allsky.modeling.registry import build_model
from allsky.modeling.visual_encoder import ImageEncoder
from allsky.training.checkpointing import load_checkpoint
from allsky.training.engine import _build_optimizer, run_experiment
from allsky.training.errors import TrainingError
from tests.allsky import _synthetic as synthetic

N_BLOCKS = 3
BACKBONE_LR = 1e-3
HEAD_LR = 2e-4
DECAY = 0.5
IMAGE_PX = 8


class TinyViTBackbone(nn.Module):
    """ViT-shaped stub: ``patch_embed.proj`` before ``blocks``, ``norm`` after."""

    def __init__(self, dim: int = 8, n_blocks: int = N_BLOCKS) -> None:
        super().__init__()
        self.dim = dim
        self.patch_embed: Any = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=IMAGE_PX, stride=IMAGE_PX)
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(dim)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: Any) -> Any:
        hidden = self.pool(self.patch_embed.proj(frame)).flatten(1)
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)


class _NoBlocks(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(3, dim, kernel_size=IMAGE_PX, stride=IMAGE_PX)

    def forward(self, frame: Any) -> Any:
        return self.conv(frame).flatten(1)


def _named(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {group["name"]: group for group in groups if "name" in group}


def _ids(params: list[Any]) -> set[int]:
    return {id(p) for p in params}


class TestEncoderTiers:
    def test_three_blocks_at_half_decay_scale_an_eighth_a_quarter_and_a_half(self):
        encoder = ImageEncoder(TinyViTBackbone())

        groups = _named(encoder.param_groups(BACKBONE_LR, layer_decay=DECAY))

        assert [groups[f"backbone_block_{i}"]["lr"] for i in range(N_BLOCKS)] == pytest.approx(
            [BACKBONE_LR * 0.125, BACKBONE_LR * 0.25, BACKBONE_LR * 0.5]
        )

    def test_the_patch_embedding_sits_one_tier_below_the_first_block(self):
        backbone = TinyViTBackbone()
        encoder = ImageEncoder(backbone)

        embed = _named(encoder.param_groups(BACKBONE_LR, layer_decay=DECAY))["backbone_embed"]

        assert embed["lr"] == pytest.approx(BACKBONE_LR * DECAY ** (N_BLOCKS + 1))
        assert _ids(embed["params"]) == _ids(list(backbone.patch_embed.parameters()))

    def test_the_final_norm_keeps_the_base_rate(self):
        backbone = TinyViTBackbone()
        encoder = ImageEncoder(backbone)

        tail = _named(encoder.param_groups(BACKBONE_LR, layer_decay=DECAY))["backbone_tail"]

        assert tail["lr"] == pytest.approx(BACKBONE_LR)
        assert _ids(tail["params"]) == _ids(list(backbone.norm.parameters()))

    def test_each_block_group_holds_exactly_that_block(self):
        backbone = TinyViTBackbone()
        encoder = ImageEncoder(backbone)

        groups = _named(encoder.param_groups(BACKBONE_LR, layer_decay=DECAY))

        for index, block in enumerate(backbone.blocks):
            assert _ids(groups[f"backbone_block_{index}"]["params"]) == _ids(
                list(block.parameters())
            )

    def test_frozen_tiers_yield_no_group(self):
        encoder = ImageEncoder(TinyViTBackbone(), unfreeze_last_n=2)

        groups = _named(encoder.param_groups(BACKBONE_LR, layer_decay=DECAY))

        assert set(groups) == {"backbone_block_1", "backbone_block_2"}

    def test_a_decay_of_one_leaves_every_tier_at_the_base_rate(self):
        encoder = ImageEncoder(TinyViTBackbone())

        groups = _named(encoder.param_groups(BACKBONE_LR, layer_decay=1.0))

        assert all(group["lr"] == pytest.approx(BACKBONE_LR) for group in groups.values())

    def test_no_decay_keeps_one_backbone_group_with_every_backbone_parameter(self):
        backbone = TinyViTBackbone()
        encoder = ImageEncoder(backbone)

        groups = encoder.param_groups(BACKBONE_LR)

        backbone_groups = [group for group in groups if "lr" in group]
        assert len(backbone_groups) == 1
        assert "name" not in backbone_groups[0]
        assert _ids(backbone_groups[0]["params"]) == _ids(list(backbone.parameters()))

    def test_a_backbone_without_blocks_is_refused_rather_than_decayed_inertly(self):
        encoder = ImageEncoder(_NoBlocks())

        with pytest.raises(BackboneCapabilityError, match="layer_decay"):
            encoder.param_groups(BACKBONE_LR, layer_decay=DECAY)


def _image_cfg(root: Path, *, layer_decay: float | None, epochs: int = 1) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": True,
            "seed": 0,
            "output_dir": str(root / "out"),
            "data": {
                "manifest": "manifest.parquet",
                "data_root": str(root),
                "split_artifact": "splits.json",
                "input_mode": "image",
            },
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
            "model": {"name": "image_only", "image_size": IMAGE_PX, "backbone_frozen": False},
            "train": {
                "epochs": epochs,
                "batch_size": 8,
                "num_workers": 0,
                "device": "cpu",
                "lr": HEAD_LR,
                "backbone_lr": BACKBONE_LR,
                "layer_decay": layer_decay,
                "early_stopping": {"monitor": "val_loss", "patience": 100},
            },
        }
    )


def _optimizer_groups(
    root: Path, layer_decay: float | None
) -> tuple[list[dict[str, Any]], list[str]]:
    torch.manual_seed(0)
    cfg = _image_cfg(root, layer_decay=layer_decay)
    model = build_model(cfg, 1, image_backbone=TinyViTBackbone())
    optimizer, labels = _build_optimizer(model, cfg)
    return optimizer.param_groups, labels


class TestEngineGroups:
    def test_no_decay_builds_the_same_groups_as_before(self, tmp_path: Path):
        groups, labels = _optimizer_groups(tmp_path, None)

        assert labels == ["lr_backbone", "lr"]
        assert [group["lr"] for group in groups] == pytest.approx([BACKBONE_LR, HEAD_LR])

    def test_half_decay_builds_one_group_per_tier_shallowest_first(self, tmp_path: Path):
        groups, labels = _optimizer_groups(tmp_path, DECAY)

        assert labels == [
            "lr_backbone_embed",
            "lr_backbone_block_0",
            "lr_backbone_block_1",
            "lr_backbone_block_2",
            "lr_backbone_tail",
            "lr",
        ]
        assert [group["lr"] for group in groups] == pytest.approx(
            [
                BACKBONE_LR * 0.0625,
                BACKBONE_LR * 0.125,
                BACKBONE_LR * 0.25,
                BACKBONE_LR * 0.5,
                BACKBONE_LR,
                HEAD_LR,
            ]
        )

    def test_the_decayed_groups_cover_exactly_the_parameters_of_the_single_group(
        self, tmp_path: Path
    ):
        flat, _ = _optimizer_groups(tmp_path, None)
        tiered, _ = _optimizer_groups(tmp_path, DECAY)

        flat_backbone = [p.shape for p in flat[0]["params"]]
        tiered_backbone = [p.shape for group in tiered[:-1] for p in group["params"]]
        assert sorted(map(str, tiered_backbone)) == sorted(map(str, flat_backbone))

    def test_the_rate_table_is_logged_at_start(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.INFO, logger="allsky.training.engine"):
            _optimizer_groups(tmp_path, DECAY)

        assert "layer decay 0.5" in caplog.text
        assert "lr_backbone_block_0" in caplog.text

    def test_an_image_epoch_trains_under_layer_decay(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=IMAGE_PX
        )
        run_dir = tmp_path / "run"

        summary = run_experiment(
            _image_cfg(root, layer_decay=DECAY),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=TinyViTBackbone,
        )

        assert summary["epochs_ran"] == 1
        assert (run_dir / "last.ckpt").exists()

    def test_the_named_groups_survive_a_resume(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=IMAGE_PX
        )
        run_dir = tmp_path / "run"
        run_experiment(
            _image_cfg(root, layer_decay=DECAY),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=TinyViTBackbone,
        )

        summary = run_experiment(
            _image_cfg(root, layer_decay=DECAY, epochs=2),
            data_root=root,
            output_dir=run_dir,
            resume="auto",
            image_backbone_builder=TinyViTBackbone,
        )

        assert summary["epochs_ran"] == 1
        stored_groups = load_checkpoint(run_dir / "last.ckpt")["optimizer_state"]["param_groups"]
        assert [group["name"] for group in stored_groups[:-1]] == [
            "backbone_embed",
            "backbone_block_0",
            "backbone_block_1",
            "backbone_block_2",
            "backbone_tail",
        ]

    def test_a_resume_that_turns_layer_decay_on_is_refused_naming_the_knob(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=IMAGE_PX
        )
        run_dir = tmp_path / "run"
        run_experiment(
            _image_cfg(root, layer_decay=None),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=TinyViTBackbone,
        )

        with pytest.raises(TrainingError, match=r"train\.layer_decay"):
            run_experiment(
                _image_cfg(root, layer_decay=DECAY, epochs=2),
                data_root=root,
                output_dir=run_dir,
                resume="auto",
                image_backbone_builder=TinyViTBackbone,
            )
