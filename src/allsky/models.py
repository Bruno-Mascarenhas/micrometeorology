"""Multi-task fusion network: sky image + sensor features -> cloud class + diffuse.

``SkyFusionNet`` fuses an all-sky camera frame with a vector of radiation-sensor
features and predicts, jointly:

1. Cloud condition class (clear / partial / overcast) — weak labels derived
   from clearness-index bins.
2. Diffuse horizontal irradiance in W/m2 — until a shaded pyranometer exists
   the training target is an Erbs-decomposition PSEUDO-target derived from GHI
   (see :mod:`allsky.sensors`); treat regression metrics accordingly.

Batch contract (as produced by ``allsky.dataset.AllSkyDataset``):

- ``image``: float32 ``(B, 3, H, W)`` in ``[0, 1]``.
- ``features``: float32 ``(B, F)`` standardized sensor features.
- ``cloud_class``: int64 ``(B,)`` in ``[0, n_classes)``.
- ``diffuse``: float32 ``(B,)`` diffuse irradiance in W/m2 (raw scale).
"""

from __future__ import annotations

import torch
from torch import nn

from allsky.config import ModelConfig

#: Scale (W/m2) applied to diffuse targets/predictions inside the regression
#: loss. SmoothL1's quadratic-to-linear transition (beta=1) then corresponds to
#: a 100 W/m2 error, and the loss magnitude stays comparable to cross-entropy.
DIFFUSE_SCALE_WM2 = 100.0


def _small_backbone() -> tuple[nn.Module, int]:
    """Built-in image encoder: 4x (Conv3x3 stride 2 - BN - ReLU) then GAP."""
    layers: list[nn.Module] = []
    in_channels = 3
    for out_channels in (32, 64, 128, 256):
        layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    return nn.Sequential(*layers), in_channels


def _resnet18_backbone() -> tuple[nn.Module, int]:
    """torchvision resnet18 (random init) with the classifier head removed."""
    try:
        from torchvision.models import resnet18
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "backbone='resnet18' requires torchvision; install it or use backbone='small'"
        ) from exc
    net = resnet18(weights=None)
    net.fc = nn.Identity()
    return net, 512


class SkyFusionNet(nn.Module):
    """Two-branch fusion model with a classification and a regression head.

    Architecture
    ------------
    - Image branch: small conv net (``backbone="small"``) or torchvision
      resnet18 with random weights (``backbone="resnet18"``), projected to
      ``embed_dim``.
    - Sensor branch: MLP ``F -> 64 -> embed_dim``.
    - Fusion: concat -> MLP (``hidden_dim``) -> ``cls_head`` (``n_classes``
      logits) and ``reg_head`` (1 output through ReLU — irradiance is
      non-negative by construction).
    """

    def __init__(self, model_cfg: ModelConfig, n_features: int) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        if model_cfg.backbone == "small":
            backbone, feat_dim = _small_backbone()
        elif model_cfg.backbone == "resnet18":
            backbone, feat_dim = _resnet18_backbone()
        else:
            raise ValueError(
                f"unknown backbone {model_cfg.backbone!r}; expected 'small' or 'resnet18'"
            )
        self.image_branch = backbone
        self.image_proj = nn.Sequential(
            nn.Linear(feat_dim, model_cfg.embed_dim),
            nn.ReLU(inplace=True),
        )
        self.sensor_branch = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, model_cfg.embed_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * model_cfg.embed_dim, model_cfg.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(model_cfg.hidden_dim, model_cfg.n_classes)
        # Final ReLU clamps predictions to >= 0: irradiance cannot be negative.
        self.reg_head = nn.Sequential(nn.Linear(model_cfg.hidden_dim, 1), nn.ReLU())

    def forward(self, image: torch.Tensor, features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return ``logits`` ``(B, n_classes)`` and ``diffuse`` ``(B,)`` in W/m2."""
        image_embed = self.image_proj(self.image_branch(image))
        sensor_embed = self.sensor_branch(features)
        fused = self.fusion(torch.cat([image_embed, sensor_embed], dim=1))
        return {
            "logits": self.cls_head(fused),
            "diffuse": self.reg_head(fused).squeeze(-1),
        }


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_cls: float = 1.0,
    w_reg: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Weighted multi-task loss: cross-entropy + SmoothL1 on scaled diffuse.

    Formula
    -------
    ``loss = w_cls * CE(logits, cloud_class)
    + w_reg * SmoothL1(diffuse_pred / 100, diffuse_true / 100)``

    The division by :data:`DIFFUSE_SCALE_WM2` (100 W/m2) scale-normalizes the
    regression term so it is commensurate with the cross-entropy term and the
    SmoothL1 quadratic region covers errors up to ~100 W/m2. Predictions and
    metrics stay in raw W/m2 — only the loss is scaled.

    Returns a dict with ``loss`` (total), ``loss_cls`` and ``loss_reg``
    (unweighted components).
    """
    cls_loss = nn.functional.cross_entropy(outputs["logits"], batch["cloud_class"])
    diffuse_pred = outputs["diffuse"]
    diffuse_true = batch["diffuse"].to(diffuse_pred.dtype)
    reg_loss = nn.functional.smooth_l1_loss(
        diffuse_pred / DIFFUSE_SCALE_WM2, diffuse_true / DIFFUSE_SCALE_WM2
    )
    return {
        "loss": w_cls * cls_loss + w_reg * reg_loss,
        "loss_cls": cls_loss,
        "loss_reg": reg_loss,
    }
