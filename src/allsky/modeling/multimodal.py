"""Assembled multimodal network: visual + sensor encoders, fusion, trunk, heads.

:class:`MultimodalNet` wires the pieces from :mod:`allsky.modeling` together
according to an experiment config: the visual source (precomputed embedding or
image backbone) chosen by ``input_mode``, the sensor MLP, the named fusion
block, the shared trunk, and the enabled heads.  Cross-attention fusion is
handed the raw standardized feature vector (sliced into per-group tokens); the
other fusions are not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

from torch import Tensor, nn

from allsky.config import TargetsConfig
from allsky.features.policy import active_feature_groups
from allsky.modeling.contracts import ModelOutputs
from allsky.modeling.fusion import build_fusion
from allsky.modeling.heads import Heads, Trunk
from allsky.modeling.sensor_encoder import SensorEncoder
from allsky.modeling.visual_encoder import build_visual_encoder

__all__ = ["MultimodalNet"]


class MultimodalNet(nn.Module):
    """Full multimodal model: ``visual + sensor -> fusion -> trunk -> heads``.

    Parameters
    ----------
    feature_columns:
        Ordered engineered feature names; their count is the sensor encoder's
        input width and they drive cross-attention's per-group tokens.
    targets:
        Which heads to build (:class:`allsky.config.TargetsConfig`).
    fusion_name:
        ``concat`` | ``film`` | ``cross_attention``.
    input_mode:
        ``embedding`` (uses *embedding_dim*) or ``image`` (uses
        *image_backbone*).
    feature_set:
        Feature-set name used to resolve cross-attention groups; the groups are
        intersected with *feature_columns*, so a superset name is harmless.
    embedding_dim, image_backbone:
        Visual-source inputs for the two modes.
    sensor_hidden:
        Sensor-encoder block widths (last is the sensor embedding width).
    visual_out_dim:
        Optional visual-projection width (``None`` = passthrough).
    trunk_hidden, trunk_layers, dropout:
        Trunk shape and dropout, shared by the sensor/visual/fusion dropouts.
    num_heads, token_dim:
        Cross-attention configuration (ignored by the other fusions).
    temporal_pooling:
        How a windowed ``embedding_seq`` is pooled in embedding mode
        (``"mean"`` — default, mask-aware mean — or learned ``"attention"``).
        Chosen from ``cfg.data.alignment.strategy`` by the engine at build time;
        inert in image mode and when the dataset emits a plain ``embedding``.
    backbone_frozen, unfreeze_last_n, backbone_lr:
        Image-backbone fine-tuning controls (``backbone_lr`` only affects
        :meth:`param_groups`).
    """

    def __init__(
        self,
        *,
        feature_columns: Sequence[str],
        targets: TargetsConfig,
        fusion_name: str = "concat",
        input_mode: Literal["image", "embedding"] = "embedding",
        feature_set: str = "safe",
        embedding_dim: int | None = None,
        image_backbone: nn.Module | None = None,
        sensor_hidden: Sequence[int] = (64, 128),
        visual_out_dim: int | None = None,
        trunk_hidden: int = 256,
        trunk_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        token_dim: int | None = None,
        temporal_pooling: Literal["mean", "attention"] = "mean",
        backbone_frozen: bool = False,
        unfreeze_last_n: int = 0,
        backbone_lr: float | None = None,
    ) -> None:
        super().__init__()
        self.feature_columns = list(feature_columns)
        self.input_mode = input_mode
        self.backbone_lr = backbone_lr

        self.sensor_encoder = SensorEncoder(
            len(self.feature_columns), sensor_hidden, dropout=dropout
        )
        self.visual_encoder = build_visual_encoder(
            input_mode,
            embedding_dim=embedding_dim,
            image_backbone=image_backbone,
            out_dim=visual_out_dim,
            frozen=backbone_frozen,
            unfreeze_last_n=unfreeze_last_n,
            dropout=dropout,
            temporal_pooling=temporal_pooling,
        )
        visual_dim = cast("int", self.visual_encoder.out_dim)
        sensor_dim = int(self.sensor_encoder.out_dim)

        groups = active_feature_groups(feature_set) if fusion_name == "cross_attention" else None
        self.fusion = build_fusion(
            fusion_name,
            visual_dim,
            sensor_dim,
            feature_columns=self.feature_columns if fusion_name == "cross_attention" else None,
            groups=groups,
            num_heads=num_heads,
            token_dim=token_dim,
        )
        self.trunk = Trunk(
            cast("int", self.fusion.out_dim), trunk_hidden, trunk_layers, dropout=dropout
        )
        self.heads = Heads(int(self.trunk.out_dim), targets)

    def forward(self, batch: dict[str, Tensor]) -> ModelOutputs:
        """Encode both modalities, fuse, and return the enabled head outputs."""
        sensor = self.sensor_encoder(batch["features"])
        visual = self.visual_encoder(batch)
        if getattr(self.fusion, "needs_features", False):
            fused = self.fusion(
                visual,
                sensor,
                features=batch["features"],
                key_padding_mask=batch.get("group_mask"),
            )
        else:
            fused = self.fusion(visual, sensor)
        outputs: ModelOutputs = self.heads(self.trunk(fused))
        return outputs

    def param_groups(self, backbone_lr: float | None = None) -> list[dict[str, Any]]:
        """Optimizer parameter groups; the image backbone gets its own LR.

        When the visual encoder wraps an image backbone and a ``backbone_lr``
        is available (argument or the constructor value), the backbone
        parameters form one group at that LR and every other trainable
        parameter forms a second group.  Otherwise a single group of all
        trainable parameters is returned.
        """
        lr = backbone_lr if backbone_lr is not None else self.backbone_lr
        get_backbone_groups = getattr(self.visual_encoder, "param_groups", None)
        if lr is None or get_backbone_groups is None:
            return [{"params": [p for p in self.parameters() if p.requires_grad]}]
        backbone_params = {
            id(p): p for group in get_backbone_groups(lr) if "lr" in group for p in group["params"]
        }
        other = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_params]
        groups: list[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": list(backbone_params.values()), "lr": lr})
        if other:
            groups.append({"params": other})
        return groups
