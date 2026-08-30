"""Feeding extra channels to a pretrained backbone without disturbing its RGB weights.

Widening the pretrained convolution would put the new weights inside the
backbone, where the freeze sweep owns them — ``patch_embed`` is not part of
``blocks``, so ``unfreeze_last_n`` never reaches it, and channels initialised at
zero and frozen at zero are structurally inert.
:class:`GeometryPatchProjection` instead leaves the pretrained convolution
untouched and sums a **separate** zero-initialised one reading the extra
channels: at initialisation the sum equals the pretrained projection exactly,
and the new convolution is a normal trainable module no freeze sweep owns.
"""

import logging
from typing import Any, cast

from torch import Tensor, nn

from allsky.modeling.backbone_families import BackboneCapabilityError

__all__ = [
    "GeometryPatchProjection",
    "PatchProjectionNotFoundError",
    "attach_extra_input_channels",
]

logger = logging.getLogger(__name__)


def _pair(value: Any) -> tuple[int, int]:
    """A Conv2d geometry parameter as the ``(h, w)`` pair torch's stubs expect."""
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


class PatchProjectionNotFoundError(TypeError):
    """Raised when a backbone exposes no first convolution to wrap."""


class GeometryPatchProjection(nn.Module):
    """Patch projection over ``rgb_channels + extra_channels`` input planes.

    Drop-in replacement for a ViT's ``patch_embed.proj``: it takes
    ``(B, C, H, W)`` and returns the same ``(B, embed_dim, H/patch, W/patch)``
    the wrapped convolution would, so the surrounding ``PatchEmbed.forward``
    (which only asserts that the frame divides into whole patches) is unchanged.

    Parameters
    ----------
    pretrained:
        The backbone's own patch-projection convolution. It is kept, not copied,
        so its pretrained weights and whatever ``requires_grad`` the caller set
        on them survive.
    extra_channels:
        Number of channels appended after the pretrained ones.

    Raises
    ------
    ValueError
        If *extra_channels* is not positive (a zero-channel adapter would be a
        silent no-op wearing the name of a wired one), or if the wrapped
        convolution is grouped.
    """

    def __init__(self, pretrained: nn.Conv2d, extra_channels: int) -> None:
        super().__init__()
        if extra_channels <= 0:
            raise ValueError(f"extra_channels must be positive, got {extra_channels}")
        self.pretrained = pretrained
        self.pretrained_channels = int(pretrained.in_channels)
        if pretrained.groups != 1:
            raise ValueError(
                f"the wrapped convolution is grouped (groups={pretrained.groups}); a grouped "
                "stem ties each output channel to a subset of the input ones, so what the "
                "extra planes should feed is a modelling decision, not a default"
            )
        # Every geometric parameter is copied, not just kernel and stride: a
        # ResNet stem is 7x7 stride 2 with padding 3, and an extra branch built
        # without that padding produces a 109x109 map against the pretrained
        # 112x112 — measured, and it fails at the sum rather than silently.
        self.extra_proj = nn.Conv2d(
            extra_channels,
            int(pretrained.out_channels),
            kernel_size=_pair(pretrained.kernel_size),
            stride=_pair(pretrained.stride),
            padding=pretrained.padding
            if isinstance(pretrained.padding, str)
            else _pair(pretrained.padding),
            dilation=_pair(pretrained.dilation),
            padding_mode=pretrained.padding_mode,
            bias=False,
        )
        nn.init.zeros_(self.extra_proj.weight)

    @property
    def in_channels(self) -> int:
        """Total input planes this projection consumes."""
        return self.pretrained_channels + int(self.extra_proj.in_channels)

    def forward(self, image: Tensor) -> Tensor:
        """Project ``(B, C, H, W)`` to ``(B, embed_dim, H/patch, W/patch)``.

        Parameters
        ----------
        image:
            ``(B, in_channels, H, W)`` float32; the first
            ``pretrained_channels`` planes are the standardised RGB frame and
            the rest are the extra maps.

        Returns
        -------
        Tensor
            ``(B, embed_dim, H/patch, W/patch)`` float32 patch embeddings,
            dimensionless.

        Raises
        ------
        ValueError
            If *image* does not carry exactly :attr:`in_channels` planes.
        """
        if image.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels "
                f"({self.pretrained_channels} pretrained + {self.extra_proj.in_channels} extra), "
                f"got {image.shape[1]}"
            )
        split = self.pretrained_channels
        projected: Tensor = self.pretrained(image[:, :split])
        from_extra: Tensor = self.extra_proj(image[:, split:])
        return projected + from_extra


def attach_extra_input_channels(
    backbone: nn.Module, extra_channels: int
) -> GeometryPatchProjection:
    """Wrap *backbone*'s patch projection so it accepts extra input channels.

    Parameters
    ----------
    backbone:
        A module owning (directly or through one wrapper attribute) a
        ``patch_embed`` whose ``proj`` is a :class:`torch.nn.Conv2d`.
    extra_channels:
        Number of channels appended after the backbone's own.

    Returns
    -------
    GeometryPatchProjection
        The installed projection, so the caller can put its trainable parameters
        in the right optimizer group and re-enable them after a freeze sweep.

    Raises
    ------
    PatchProjectionNotFoundError
        If the backbone's family cannot say where the frame enters. Failing here
        is the point: silently skipping the wrap would leave the extra channels
        unconsumed and the experiment would measure nothing.
    """
    owner, attribute = _locate_first_convolution(backbone)
    adapter = GeometryPatchProjection(getattr(owner, attribute), extra_channels)
    setattr(owner, attribute, adapter)
    logger.info(
        "wrapped %s.%s: %d pretrained + %d extra input channels, the extra branch zero-initialised",
        type(owner).__name__,
        attribute,
        adapter.pretrained_channels,
        extra_channels,
    )
    return adapter


def _locate_first_convolution(backbone: nn.Module) -> tuple[nn.Module, str]:
    """``(owner, attribute)`` of the convolution the frame enters.

    A production backbone carries a family that answers this
    (:mod:`allsky.modeling.backbone_families`).  Test stubs do not, so a direct
    ``patch_embed.proj`` is still recognised — that path predates the families
    and the stubs are written against it.

    Raises
    ------
    PatchProjectionNotFoundError
        If neither route finds one. When a family was consulted and refused, its
        own message is chained: "efficientnet_v2_s stem attribute '0' is not a
        convolution" says what to fix, and replacing it with a generic sentence
        would discard the diagnosis.
    """
    family = getattr(backbone, "family", None)
    model = getattr(backbone, "model", None)
    if family is not None:
        try:
            return cast("tuple[nn.Module, str]", family.first_convolution(model))
        except BackboneCapabilityError as exc:
            raise PatchProjectionNotFoundError(
                f"backbone {type(backbone).__name__} exposes no first convolution to wrap: {exc}"
            ) from exc
    for candidate in (backbone, model):
        patch_embed = getattr(candidate, "patch_embed", None)
        if patch_embed is not None and isinstance(getattr(patch_embed, "proj", None), nn.Conv2d):
            return patch_embed, "proj"
    raise PatchProjectionNotFoundError(
        f"backbone {type(backbone).__name__} exposes no first convolution to wrap; "
        "extra input channels cannot reach a backbone that does not say where the frame enters"
    )
