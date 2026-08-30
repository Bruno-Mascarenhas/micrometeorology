"""Feeding extra channels to a pretrained ViT without disturbing its RGB weights.

A DINOv2 backbone tokenises with a single ``Conv2d(3, embed_dim, patch, patch)``
in ``patch_embed.proj``. Extra input channels have to pass through it, and the
two obvious routes both fail:

- widening that convolution and zero-initialising the new slices puts the new
  weights inside the backbone, where :class:`allsky.modeling.visual_encoder.ImageEncoder`
  freezes them — ``patch_embed`` is not part of ``blocks``, so ``unfreeze_last_n``
  never reaches it. Weights initialised at zero and frozen at zero make the extra
  channels **structurally inert**: the arm returns the control's result and the
  experiment reads as a null finding about the channels rather than about the
  wiring;
- initialising them randomly instead perturbs the pretrained tokenisation from
  step 0, so the fine-tune starts from a backbone that is no longer the one the
  pretraining produced.

:class:`GeometryPatchProjection` avoids both: the pretrained convolution keeps
its own weights and its own ``requires_grad``, a **separate** zero-initialised
convolution reads the extra channels, and their outputs are summed. At
initialisation the sum equals the pretrained projection exactly, and the new
convolution is a normal trainable module that no backbone freeze sweep owns.
"""

import logging
from typing import cast

from torch import Tensor, nn

from allsky.modeling.backbone_families import BackboneCapabilityError

__all__ = [
    "GeometryPatchProjection",
    "PatchProjectionNotFoundError",
    "attach_extra_input_channels",
]

logger = logging.getLogger(__name__)


class PatchProjectionNotFoundError(TypeError):
    """Raised when a backbone exposes no ``patch_embed.proj`` convolution to wrap."""


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
        If *extra_channels* is not positive: a zero-channel adapter would be a
        silent no-op wearing the name of a wired one.
    """

    def __init__(self, pretrained: nn.Conv2d, extra_channels: int) -> None:
        super().__init__()
        if extra_channels <= 0:
            raise ValueError(f"extra_channels must be positive, got {extra_channels}")
        self.pretrained = pretrained
        self.pretrained_channels = int(pretrained.in_channels)
        kernel = (int(pretrained.kernel_size[0]), int(pretrained.kernel_size[1]))
        stride = (int(pretrained.stride[0]), int(pretrained.stride[1]))
        self.extra_proj = nn.Conv2d(
            extra_channels,
            int(pretrained.out_channels),
            kernel_size=kernel,
            stride=stride,
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
    located = _locate_first_convolution(backbone)
    if located is None:
        raise PatchProjectionNotFoundError(
            f"backbone {type(backbone).__name__} exposes no first convolution to wrap; "
            "extra input channels cannot reach a backbone whose family does not say where "
            "the frame enters"
        )
    owner, attribute = located
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


def _locate_first_convolution(backbone: nn.Module) -> tuple[nn.Module, str] | None:
    """``(owner, attribute)`` of the convolution the frame enters, or ``None``.

    A production backbone carries a family that answers this
    (:mod:`allsky.modeling.backbone_families`).  Test stubs do not, so a direct
    ``patch_embed.proj`` is still recognised — that path predates the families
    and the stubs are written against it.
    """
    family = getattr(backbone, "family", None)
    model = getattr(backbone, "model", None)
    if family is not None and model is not None:
        try:
            return cast("tuple[nn.Module, str]", family.first_convolution(model))
        except BackboneCapabilityError:
            return None
    for candidate in (backbone, model):
        patch_embed = getattr(candidate, "patch_embed", None)
        if patch_embed is not None and isinstance(getattr(patch_embed, "proj", None), nn.Conv2d):
            return patch_embed, "proj"
    return None
