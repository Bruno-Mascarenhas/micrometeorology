"""Tests for initialising a run from another run's weights.

Every test here is about the failure mode, not the happy path: a transfer that
moves less than the operator believes looks exactly like one that worked, and
stays looking that way until the metrics come back. So what is asserted is that
a partial transfer is refused, that the report names what it left behind, and
that the one legitimate rename — the convolution the geometry adapter wraps —
still finds its source.
"""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from allsky.modeling.geometry_adapter import GeometryPatchProjection
from allsky.modeling.transfer import TransferMismatchError, load_transferable_weights

PATCH = 8
FRAME = 32


class _HubBackboneStub(nn.Module):
    def __init__(self, width: int = 6) -> None:
        super().__init__()
        self.model: Any = nn.Module()
        self.model.patch_embed = nn.Module()
        self.model.patch_embed.proj = nn.Conv2d(3, width, kernel_size=PATCH, stride=PATCH)
        self.model.blocks = nn.ModuleList(nn.Linear(width, width) for _ in range(2))


class _Model(nn.Module):
    """A visual encoder plus a task head, named as the real models name them."""

    def __init__(self, *, width: int = 6, head_out: int = 1, extra_channels: int = 0) -> None:
        super().__init__()
        self.visual_encoder: Any = nn.Module()
        self.visual_encoder.backbone = _HubBackboneStub(width)
        if extra_channels:
            proj = self.visual_encoder.backbone.model.patch_embed.proj
            adapter = GeometryPatchProjection(proj, extra_channels)
            self.visual_encoder.backbone.model.patch_embed.proj = adapter
            self.visual_encoder.extra_channel_projection = adapter
        self.heads = nn.Linear(width, head_out)


def _write_checkpoint(path: Path, model: nn.Module) -> Path:
    torch.save({"model_state": model.state_dict()}, path)
    return path


class TestTransfer:
    def test_a_matching_source_fills_every_tensor(self, tmp_path: Path):
        source = _Model()
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", source)
        target = _Model()

        report = load_transferable_weights(target, checkpoint, trust_pickle=True)

        assert not report.missing
        assert not report.reshaped
        assert torch.equal(
            target.visual_encoder.backbone.model.patch_embed.proj.weight,
            source.visual_encoder.backbone.model.patch_embed.proj.weight,
        )

    def test_a_different_head_is_skipped_and_named(self, tmp_path: Path):
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", _Model(head_out=4))

        report = load_transferable_weights(_Model(head_out=1), checkpoint, trust_pickle=True)

        assert set(report.reshaped) == {"heads.weight", "heads.bias"}
        assert any(name.startswith("visual_encoder.backbone.") for name in report.loaded)

    def test_a_tensor_the_target_has_no_home_for_is_named_not_swallowed(self, tmp_path: Path):
        """``report.unexpected`` is what tells the operator the source carried
        weights this model cannot hold — the symptom of a checkpoint from another
        architecture that still cleared every guard. Nothing asserted it, so the
        field could have been empty for every source.
        """
        source = _Model()
        state = source.state_dict()
        state["aux_head.weight"] = torch.zeros(3, 4)
        path = tmp_path / "source.ckpt"
        torch.save({"model_state": state}, path)

        report = load_transferable_weights(_Model(), path, trust_pickle=True)

        assert report.unexpected == ("aux_head.weight",)
        assert "heads.weight" in report.loaded

    def test_a_backbone_of_another_width_is_refused(self, tmp_path: Path):
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", _Model(width=6))

        with pytest.raises(TransferMismatchError, match="describes a different backbone"):
            load_transferable_weights(_Model(width=10), checkpoint, trust_pickle=True)

    def test_a_source_sharing_only_the_head_is_refused(self, tmp_path: Path):
        other = _Model()
        other.visual_encoder.backbone = nn.Sequential(nn.Conv2d(3, 6, 3), nn.ReLU())
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", other)

        with pytest.raises(TransferMismatchError, match="transfers whole or not at all"):
            load_transferable_weights(_Model(), checkpoint, trust_pickle=True)

    def test_the_wrapped_patch_convolution_still_finds_its_source(self, tmp_path: Path):
        source = _Model()
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", source)
        target = _Model(extra_channels=2)

        report = load_transferable_weights(target, checkpoint, trust_pickle=True)

        adapter = target.visual_encoder.extra_channel_projection
        assert torch.equal(
            adapter.pretrained.weight,
            source.visual_encoder.backbone.model.patch_embed.proj.weight,
        )
        assert float(adapter.extra_proj.weight.abs().sum()) == 0.0
        assert report.moved_anything

    def test_a_checkpoint_without_model_state_is_refused(self, tmp_path: Path):
        path = tmp_path / "not-ours.ckpt"
        torch.save({"something_else": 1}, path)

        with pytest.raises(TransferMismatchError, match="carries no 'model_state'"):
            load_transferable_weights(_Model(), path, trust_pickle=True)

    def test_the_report_reads_as_a_sentence(self, tmp_path: Path):
        checkpoint = _write_checkpoint(tmp_path / "source.ckpt", _Model())

        report = load_transferable_weights(_Model(), checkpoint, trust_pickle=True)

        assert report.describe().startswith("transferred ")
