"""Torch-gated regression tests for the allsky-training science findings.

Each test fails against the pre-fix behaviour and pins the corrected semantics:

#. :class:`~allsky.modeling.visual_encoder.ImageEncoder` honours
   ``unfreeze_last_n`` regardless of ``backbone_frozen``, so the shipped V6
   config finetunes the last 2 ViT blocks instead of the whole DINOv2; the
   combinations that cannot be honoured now say so in the log.
#. The epoch loss metrics weight every loss component by its own valid-row count,
   so the reported ``loss`` / ``loss_<head>`` no longer depend on
   ``train.batch_size`` when targets are missing.
#. :class:`~allsky.modeling.baselines.ImageOnlyModel` exposes ``param_groups``,
   so ``train.backbone_lr`` reaches the image backbone instead of being silently
   ignored, and a ``backbone_lr`` that reaches nothing is logged.

All offline and CPU-only: tiny stub backbones, synthetic tensors, no downloads.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from allsky.config import ExperimentConfig, TargetsConfig, load_experiment_config  # noqa: E402
from allsky.data.loading import load_manifest  # noqa: E402
from allsky.data.manifest import write_manifest_parquet  # noqa: E402
from allsky.features import resolve_feature_set  # noqa: E402
from allsky.modeling.registry import build_model  # noqa: E402
from allsky.modeling.visual_encoder import ImageEncoder, coerce_image_backbone  # noqa: E402
from allsky.training.engine import (  # noqa: E402
    _build_optimizer,
    _loss_component_weights,
    _MetricAccumulator,
    run_experiment,
)
from allsky.training.losses import MultitaskLoss  # noqa: E402
from tests.allsky.test_engine import _cfg, _make_dataset, _reader  # noqa: E402
from tests.allsky.test_modeling import TinyConvBackbone, TinyViTBackbone  # noqa: E402

_V6_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "allsky"
    / "experiments"
    / "v6_film_finetune.yaml"
)
_N_FEATURES = len(resolve_feature_set("safe"))
_ENCODER_LOGGER = "allsky.modeling.visual_encoder"


def _trainable_block_indices(backbone: TinyViTBackbone) -> list[int]:
    """Indices of the backbone blocks with at least one gradient-carrying parameter."""
    return [
        index
        for index, block in enumerate(backbone.blocks)
        if any(param.requires_grad for param in block.parameters())
    ]


def _last_blocks(backbone: TinyViTBackbone, n: int) -> list[Any]:
    """Every parameter of the backbone's last *n* blocks."""
    return [param for block in list(backbone.blocks)[-n:] for param in block.parameters()]


def _param_ids(params: Any) -> set[int]:
    return {id(param) for param in params}


# --------------------------------------------------------------------------- #
# Finding 1 — unfreeze_last_n is honoured independently of `frozen`.
# --------------------------------------------------------------------------- #


def test_unfreeze_last_n_applies_without_backbone_frozen():
    # Pre-fix: the unfreeze branch was nested inside `if frozen:`, so this
    # combination (exactly what v6_film_finetune.yaml asks for) left the WHOLE
    # backbone trainable.
    backbone = TinyViTBackbone(dim=16, n_blocks=12)
    encoder = ImageEncoder(backbone, frozen=False, unfreeze_last_n=2)

    assert _trainable_block_indices(backbone) == [10, 11]
    assert all(not param.requires_grad for param in backbone.patch.parameters())
    groups = encoder.param_groups(backbone_lr=1e-5)
    backbone_group = next(group for group in groups if group.get("lr") == 1e-5)
    assert _param_ids(backbone_group["params"]) == _param_ids(_last_blocks(backbone, 2))


def test_frozen_backbone_still_unfreezes_last_blocks():
    # The pre-existing (and only tested) combination keeps working unchanged.
    backbone = TinyViTBackbone(dim=16, n_blocks=12)
    ImageEncoder(backbone, frozen=True, unfreeze_last_n=2)
    assert _trainable_block_indices(backbone) == [10, 11]


def test_v6_config_finetunes_only_the_last_two_vit_blocks():
    # Pins the shipped config's intent, not just the encoder API: V6 is published
    # as "FiLM + last-2-block finetune" and must build exactly that.
    cfg = load_experiment_config(_V6_CONFIG)
    backbone = TinyViTBackbone(dim=16, n_blocks=12)
    model = build_model(cfg, _N_FEATURES, image_backbone=backbone)

    assert _trainable_block_indices(backbone) == [10, 11]
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    assert trainable < sum(p.numel() for p in backbone.parameters())

    optimizer, labels = _build_optimizer(model, cfg)
    # Group ORDER is a resume contract: torch matches param groups positionally.
    assert labels == ["lr_backbone", "lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.train.backbone_lr)
    assert _param_ids(optimizer.param_groups[0]["params"]) == _param_ids(_last_blocks(backbone, 2))
    assert optimizer.param_groups[1]["lr"] == pytest.approx(cfg.train.lr)


def test_unfreeze_last_n_on_a_block_less_backbone_warns_and_keeps_it_trainable(caplog):
    # A backbone with no `blocks` cannot be partially finetuned; that stays a
    # no-op (documented behaviour) but is no longer silent.
    backbone = TinyConvBackbone(dim=12)
    with caplog.at_level(logging.WARNING, logger=_ENCODER_LOGGER):
        ImageEncoder(backbone, frozen=False, unfreeze_last_n=2)

    assert all(param.requires_grad for param in backbone.parameters())
    assert any("no 'blocks' sequence" in record.getMessage() for record in caplog.records)


class _ChunkedViT(nn.Module):
    """Depth-12 transformer whose ``blocks`` are 4 chunks of 3 (DINOv2 block_chunks)."""

    def __init__(self) -> None:
        super().__init__()
        self.n_blocks = 12
        self.blocks = nn.ModuleList(
            nn.ModuleList(nn.Linear(8, 8) for _ in range(3)) for _ in range(4)
        )


class _ChunkedHubBackbone:
    """VisualBackbone-shaped extraction wrapper over a chunked hub module."""

    dim = 8
    pooling = "cls"

    def load_torch_module(self) -> nn.Module:
        return _ChunkedViT()


def test_chunked_backbone_blocks_are_reported(caplog):
    # DINOv2 built with block_chunks > 0 exposes `blocks` as chunks, so
    # unfreeze_last_n=2 would mean 2 chunks (6 blocks), not 2 blocks.
    with caplog.at_level(logging.WARNING, logger=_ENCODER_LOGGER):
        coerce_image_backbone(_ChunkedHubBackbone())

    assert any("chunk" in record.getMessage() for record in caplog.records)


# --------------------------------------------------------------------------- #
# Finding 2 — epoch loss metrics are masked-count weighted, not batch weighted.
# --------------------------------------------------------------------------- #

_IDENTITY_STATS = (0.0, 1.0, 0.0, 1.0)


def _dhi_targets(weight: float = 1.0) -> TargetsConfig:
    return TargetsConfig.model_validate(
        {
            "dhi": {"enabled": True, "loss": "mse", "weight": weight},
            "kindex": {"enabled": False},
            "sky": {"enabled": False},
            "cloud_fraction": {"enabled": False},
        }
    )


def _epoch_metrics(
    targets: TargetsConfig,
    batch: dict[str, Tensor],
    outputs: dict[str, Tensor],
    *,
    batch_size: int,
) -> dict[str, float]:
    """Run one epoch of ``_MetricAccumulator`` over *batch* sliced into chunks."""
    loss_fn = MultitaskLoss(targets, {})
    accumulator = _MetricAccumulator(_IDENTITY_STATS, _loss_component_weights(targets))
    rows = int(batch["features"].shape[0])
    for start in range(0, rows, batch_size):
        window = slice(start, start + batch_size)
        chunk = {key: value[window] for key, value in batch.items()}
        chunk_outputs = {key: value[window] for key, value in outputs.items()}
        accumulator.update(chunk_outputs, chunk, loss_fn(chunk_outputs, chunk))
    return accumulator.result()


def _half_missing_batch() -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """8 rows: 4 with ``dhi=2.0``, 4 with a sensor gap; predictions are all zeros."""
    dhi = torch.tensor([2.0, 2.0, 2.0, 2.0, float("nan"), float("nan"), float("nan"), float("nan")])
    batch = {
        "features": torch.zeros(8, 3),
        "dhi": dhi,
        "kindex": torch.full((8,), float("nan")),
        "sky_class": torch.full((8,), -1, dtype=torch.long),
        "cloud_fraction": torch.full((8,), float("nan")),
    }
    return batch, {"dhi": torch.zeros(8)}


@pytest.mark.parametrize("batch_size", [8, 4, 2, 1])
def test_epoch_loss_is_batch_size_invariant_with_missing_targets(batch_size: int):
    # Pre-fix each component was weighted by the full batch row count while the
    # component itself averages only the non-missing rows, so the reported loss
    # was the masked MSE times the valid fraction — 4.0 at batch_size 8 but 2.0
    # at 4, 2 and 1.
    batch, outputs = _half_missing_batch()
    metrics = _epoch_metrics(_dhi_targets(), batch, outputs, batch_size=batch_size)

    assert metrics["loss_dhi"] == pytest.approx(4.0)  # masked MSE of 0 vs 2.0
    assert metrics["loss"] == pytest.approx(4.0)


def test_epoch_loss_is_the_weighted_sum_of_the_per_head_epoch_means():
    targets = TargetsConfig.model_validate(
        {
            "dhi": {"enabled": True, "loss": "mse", "weight": 2.0},
            "kindex": {"enabled": False},
            "sky": {"enabled": True, "weight": 0.5},
            "cloud_fraction": {"enabled": False},
        }
    )
    batch, outputs = _half_missing_batch()
    # Two labelled sky rows, equal logits over 3 classes -> cross entropy log(3).
    batch["sky_class"] = torch.tensor([0, -1, -1, -1, -1, -1, -1, 1])
    outputs["sky_logits"] = torch.zeros(8, 3)

    metrics = _epoch_metrics(targets, batch, outputs, batch_size=3)

    assert metrics["loss_dhi"] == pytest.approx(4.0)
    assert metrics["loss_sky"] == pytest.approx(float(np.log(3.0)), rel=1e-6)
    assert metrics["loss"] == pytest.approx(2.0 * 4.0 + 0.5 * float(np.log(3.0)), rel=1e-6)


def test_head_without_a_single_label_reports_zero_and_warns(caplog):
    batch, outputs = _half_missing_batch()
    batch["dhi"] = torch.full((8,), float("nan"))

    with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
        metrics = _epoch_metrics(_dhi_targets(), batch, outputs, batch_size=4)

    # The key must stay present: metrics.csv has a column for it and
    # `val/loss_dhi` is a legal early-stopping monitor.
    assert metrics["loss_dhi"] == 0.0
    assert metrics["loss"] == 0.0
    assert any("no valid target row" in record.getMessage() for record in caplog.records)


def test_reported_val_loss_no_longer_depends_on_batch_size(tmp_path: Path):
    # End-to-end through run_experiment: the same climatology model (constant
    # predictions, no optimizer step) on the same val split must report one
    # val_loss, whatever `--batch-size` the cron passes.
    root, manifest, _ = _make_dataset(tmp_path, n_days=3, per_day=20)
    stored, meta = load_manifest(root / "manifest.parquet")
    values = stored["target_dhi"].to_numpy(dtype=float).copy()
    for start in range(0, len(values), 14):  # clustered sensor gaps
        values[start : start + 7] = np.nan
    stored["target_dhi"] = values
    write_manifest_parquet(stored, meta, root / "manifest.parquet")
    assert np.isfinite(values).any()
    assert not np.isfinite(values).all()

    reported = {}
    for batch_size in (64, 8, 3, 1):
        cfg = _cfg(
            root,
            model="climatology",
            epochs=1,
            batch_size=batch_size,
            targets={"dhi": {"enabled": True, "loss": "mse"}},
        )
        summary = run_experiment(
            cfg,
            data_root=root,
            output_dir=tmp_path / f"run_{batch_size}",
            embedding_reader=_reader(manifest),
        )
        reported[batch_size] = summary["final_val_metrics"]["loss"]

    # batch_size=64 puts the whole val split in one batch: the true masked MSE.
    for batch_size, value in reported.items():
        assert value == pytest.approx(reported[64], rel=1e-6), batch_size


# --------------------------------------------------------------------------- #
# Finding 3 — image_only honours train.backbone_lr.
# --------------------------------------------------------------------------- #


def _image_only_cfg(*, backbone_frozen: bool, unfreeze_last_n: int) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "mse"}},
            "data": {"input_mode": "image"},
            "model": {
                "name": "image_only",
                "backbone_frozen": backbone_frozen,
                "unfreeze_last_n": unfreeze_last_n,
            },
            "train": {"lr": 3e-4, "backbone_lr": 1e-5, "device": "cpu"},
        }
    )


def test_image_only_puts_the_unfrozen_blocks_on_the_backbone_lr():
    # Pre-fix ImageOnlyModel had no param_groups, so _build_optimizer fell back to
    # one group and the unfrozen ViT blocks trained at train.lr (30x too large).
    cfg = _image_only_cfg(backbone_frozen=True, unfreeze_last_n=2)
    backbone = TinyViTBackbone(dim=16, n_blocks=12)
    model = build_model(cfg, _N_FEATURES, image_backbone=backbone)

    optimizer, labels = _build_optimizer(model, cfg)

    assert labels == ["lr_backbone", "lr"]
    assert _param_ids(optimizer.param_groups[0]["params"]) == _param_ids(_last_blocks(backbone, 2))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(3e-4)


def test_fully_frozen_backbone_keeps_a_single_group_and_warns(caplog):
    # A wholly frozen backbone has no trainable parameter to put on backbone_lr;
    # the run must say so instead of silently training it at train.lr.
    cfg = _image_only_cfg(backbone_frozen=True, unfreeze_last_n=0)
    model = build_model(cfg, _N_FEATURES, image_backbone=TinyConvBackbone(dim=12))

    with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
        _optimizer, labels = _build_optimizer(model, cfg)

    assert labels == ["lr"]
    assert any("backbone_lr" in record.getMessage() for record in caplog.records)


def test_image_only_forward_still_runs():
    cfg = _image_only_cfg(backbone_frozen=True, unfreeze_last_n=2)
    model = build_model(cfg, _N_FEATURES, image_backbone=TinyViTBackbone(dim=16, n_blocks=12))
    model.eval()
    with torch.no_grad():
        outputs = model(
            {"features": torch.zeros(2, _N_FEATURES), "image": torch.rand(2, 3, 32, 32)}
        )
    assert outputs["dhi"].shape == (2,)
    assert bool(torch.isfinite(outputs["dhi"]).all())
