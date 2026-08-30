"""What fine-tuning needs from a backbone, per architecture family.

Three things are asked of an image backbone here, and each is answered
differently by a vision transformer and by a convolutional network:

1. **pooling** — how a ``(B, C, H, W)`` frame becomes a ``(B, dim)`` embedding.
   A DINOv2 ViT hands back a CLS token and patch tokens from
   ``forward_features``; a ResNet has neither and pools its last feature map.
2. **stages** — what ``unfreeze_last_n`` counts. Transformer blocks and residual
   stages are not the same unit, and a run that unfroze "the last 2" of one
   thinking it got the other would report a fine-tuning depth it never used.
3. **the first convolution** — where extra input channels attach
   (:mod:`allsky.modeling.geometry_adapter`). ``patch_embed.proj`` on a ViT,
   ``conv1`` on a ResNet, the stem of ``features`` on an EfficientNet.

A family that cannot answer one of them raises
:class:`BackboneCapabilityError` rather than guessing. That is the whole point
of the type: a config asking for geometry channels on a backbone that would
silently drop them, or for ``unfreeze_last_n`` on one with no stages to count,
is the inert-input defect this project has already paid for once — the model
trains, the number comes back, and it answers a question nobody asked.
"""

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from torch import Tensor, nn

__all__ = [
    "BackboneCapabilityError",
    "BackboneFamily",
    "ConvNetFamily",
    "VitTokenFamily",
    "family_for",
]

#: Token poolings a vision transformer family accepts.
VIT_POOLINGS = ("cls", "mean", "cls+mean")


class BackboneCapabilityError(TypeError):
    """A backbone family cannot provide something the experiment asked for."""


@runtime_checkable
class BackboneFamily(Protocol):
    """How one architecture family answers the three fine-tuning questions.

    Attributes
    ----------
    name:
        Family identity, for log and error messages.
    stage_unit:
        What one stage IS in this family (``"transformer block"``,
        ``"residual block"``), so a log line about ``unfreeze_last_n`` names the
        unit it counted.
    """

    name: str
    stage_unit: str

    def pooled(self, model: Any, image: Tensor) -> Tensor:
        """Embed ``(B, C, H, W)`` frames as ``(B, dim)``, with gradients."""
        ...

    def stages(self, model: Any) -> Sequence[nn.Module]:
        """The ordered stages ``unfreeze_last_n`` counts, first to last."""
        ...

    def first_convolution(self, model: Any) -> tuple[nn.Module, str]:
        """``(owner, attribute)`` of the convolution the frame enters through."""
        ...


class VitTokenFamily:
    """Vision transformers exposing ``forward_features`` and token outputs.

    Covers the DINOv2 and DINOv3 hub models: both return a dict with
    ``x_norm_clstoken`` and ``x_norm_patchtokens``, tokenise with a
    ``patch_embed.proj`` convolution, and hold their depth in ``blocks``.

    Parameters
    ----------
    pooling:
        ``"cls"``, ``"mean"`` (patch-token mean) or ``"cls+mean"``
        (concatenation, twice the token width).
    name:
        Family identity for messages.

    Raises
    ------
    BackboneCapabilityError
        If *pooling* is not one this family produces.
    """

    stage_unit = "transformer block"

    def __init__(self, pooling: str = "cls", *, name: str = "vit") -> None:
        if pooling not in VIT_POOLINGS:
            raise BackboneCapabilityError(
                f"{name} does not produce pooling {pooling!r}; expected one of {list(VIT_POOLINGS)}"
            )
        self.pooling = pooling
        self.name = name

    def pooled(self, model: Any, image: Tensor) -> Tensor:
        """Run ``forward_features`` and pool its tokens as *pooling* asks."""
        out = model.forward_features(image)
        cls: Tensor = out["x_norm_clstoken"]
        if self.pooling == "cls":
            return cls
        patch_mean: Tensor = out["x_norm_patchtokens"].mean(dim=1)
        if self.pooling == "mean":
            return patch_mean
        import torch

        return torch.cat([cls, patch_mean], dim=-1)

    def stages(self, model: Any) -> Sequence[nn.Module]:
        """The transformer ``blocks``.

        Raises
        ------
        BackboneCapabilityError
            If the model exposes no ``blocks`` sequence.
        """
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise BackboneCapabilityError(
                f"{self.name} exposes no 'blocks' sequence, so there is nothing for "
                "unfreeze_last_n to count"
            )
        return list(blocks)

    def first_convolution(self, model: Any) -> tuple[nn.Module, str]:
        """The patch-embedding projection.

        Raises
        ------
        BackboneCapabilityError
            If ``patch_embed.proj`` is not a convolution.
        """
        patch_embed = getattr(model, "patch_embed", None)
        if not isinstance(patch_embed, nn.Module) or not isinstance(
            getattr(patch_embed, "proj", None), nn.Conv2d
        ):
            raise BackboneCapabilityError(
                f"{self.name} has no patch_embed.proj convolution for extra input channels"
            )
        return patch_embed, "proj"


class ConvNetFamily:
    """Convolutional backbones pooled by global average over the last feature map.

    Covers the torchvision ResNet and EfficientNet families. They produce no
    tokens, so the only pooling they offer is the global average their own
    classifier heads use — asking for ``"cls"`` here is asking for something the
    architecture does not have.

    Parameters
    ----------
    stage_attributes:
        Ordered attribute names holding the stages.  An attribute that is a
        :class:`torch.nn.Sequential` is FLATTENED into its children, so
        ``unfreeze_last_n`` counts the same unit a transformer's ``blocks``
        would: a ResNet50's four ``layerN`` become its 16 bottleneck blocks, and
        an EfficientNetV2-S's ``features`` becomes its 8. Counting the four
        ``layerN`` instead would make ``unfreeze_last_n: 1`` unfreeze a quarter
        of the network.
    first_convolution_path:
        Attribute path to the stem convolution's OWNER, plus the attribute name
        on it.
    stage_unit:
        What one entry of :meth:`stages` is, for the unfreezing log line.
    pooling:
        Must be ``"mean"``.
    name:
        Family identity for messages.

    Raises
    ------
    BackboneCapabilityError
        If *pooling* is anything but ``"mean"``.
    """

    def __init__(
        self,
        *,
        stage_attributes: Sequence[str],
        first_convolution_path: tuple[Sequence[str | int], str],
        stage_unit: str = "residual block",
        pooling: str = "mean",
        name: str = "convnet",
    ) -> None:
        if pooling != "mean":
            raise BackboneCapabilityError(
                f"{name} produces no tokens, so pooling {pooling!r} does not exist for it; "
                "it offers only 'mean', the global average over its last feature map"
            )
        self.pooling = pooling
        self.name = name
        self.stage_unit = stage_unit
        self.stage_attributes = tuple(stage_attributes)
        self.first_convolution_path = first_convolution_path

    def pooled(self, model: Any, image: Tensor) -> Tensor:
        """Embed by the model's own forward with its classifier removed.

        The classifier is replaced by :class:`torch.nn.Identity` when the
        backbone is built (see :mod:`allsky.embeddings.backbone`), so the
        model's own forward already ends at the pooled feature vector — running
        it whole keeps every family-specific detail (EfficientNet's dropout, a
        ResNet's flatten) exactly where its authors put it.
        """
        embedded: Tensor = model(image)
        return embedded

    def stages(self, model: Any) -> Sequence[nn.Module]:
        """The stages named by *stage_attributes*, in order.

        Raises
        ------
        BackboneCapabilityError
            If the model lacks one of them.
        """
        found: list[nn.Module] = []
        for attribute in self.stage_attributes:
            stage = getattr(model, attribute, None)
            if stage is None:
                raise BackboneCapabilityError(
                    f"{self.name} was described with a stage {attribute!r} the model does not have"
                )
            found.extend(stage if isinstance(stage, nn.Sequential) else [stage])
        return found

    def first_convolution(self, model: Any) -> tuple[nn.Module, str]:
        """The stem convolution, resolved through *first_convolution_path*.

        Raises
        ------
        BackboneCapabilityError
            If the path does not land on a convolution.
        """
        path, attribute = self.first_convolution_path
        owner: Any = model
        for step in path:
            owner = owner[step] if isinstance(step, int) else getattr(owner, step, None)
            if owner is None:
                raise BackboneCapabilityError(
                    f"{self.name} has no {'.'.join(str(p) for p in path)} to reach its "
                    "stem convolution"
                )
        if not isinstance(getattr(owner, attribute, None), nn.Conv2d):
            raise BackboneCapabilityError(
                f"{self.name} stem attribute {attribute!r} is not a convolution"
            )
        return owner, attribute


def family_for(name: str, pooling: str) -> BackboneFamily:
    """The family that describes backbone *name*.

    Parameters
    ----------
    name:
        Backbone identity as :func:`allsky.embeddings.backbone.build_backbone`
        knows it.
    pooling:
        Token pooling the experiment asked for.

    Returns
    -------
    BackboneFamily
        The matching family, already validated against *pooling*.

    Raises
    ------
    BackboneCapabilityError
        If no family describes *name*, or the family rejects *pooling*.
    """
    if name.startswith(("dinov2_", "dinov3_")):
        return VitTokenFamily(pooling, name=name)
    if name.startswith("resnet"):
        return ConvNetFamily(
            stage_attributes=("layer1", "layer2", "layer3", "layer4"),
            first_convolution_path=((), "conv1"),
            pooling=pooling,
            name=name,
        )
    if name.startswith("efficientnet"):
        return ConvNetFamily(
            stage_attributes=("features",),
            first_convolution_path=(("features", 0), "0"),
            stage_unit="feature block",
            pooling=pooling,
            name=name,
        )
    raise BackboneCapabilityError(
        f"no backbone family describes {name!r}; extending this function is what adding "
        "an architecture means, and doing it wrong silently is what it must not do"
    )
