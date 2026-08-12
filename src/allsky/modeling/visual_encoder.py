"""Visual-feature sources: precomputed embeddings or an image backbone.

Two abstractions provide the visual branch, discovered dimensions only (no
magic constants):

- :class:`PrecomputedEmbedding` — passthrough (optionally projected) over a
  precomputed visual embedding read from ``batch["embedding"]``, or a pooled
  ``batch["embedding_seq"]`` (masked by ``batch["frame_mask"]`` when a windowed
  alignment strategy produced a per-sample frame window).  Temporal pooling is
  either a mask-aware **mean** (default) or a small **learned attention** pooler
  (a single learnable query over one :class:`torch.nn.MultiheadAttention`),
  selected by ``temporal_pooling``.
- :class:`ImageEncoder` — wraps any ``nn.Module`` exposing an integer ``.dim``
  attribute (the DINOv2 wrapper from
  :func:`allsky.embeddings.backbone.build_backbone`, or a small conv net in
  tests).  Supports a frozen backbone, partial fine-tuning of the last *n*
  ViT blocks when the backbone exposes a ``blocks`` sequence (independently of
  ``frozen``), and a :meth:`ImageEncoder.param_groups` helper for a separate
  backbone learning rate.

:func:`split_backbone_param_groups` builds the two optimizer groups (image
backbone at its own learning rate, then everything else) that any model owning a
visual encoder hands to the engine.

:func:`build_visual_encoder` picks the source from ``input_mode``.
"""

import logging
from typing import Any, Literal, cast

import torch
from torch import Tensor, nn

logger = logging.getLogger(__name__)

#: Temporal pooling modes for a windowed ``embedding_seq``.
TemporalPooling = Literal["mean", "attention"]

__all__ = [
    "ImageEncoder",
    "PrecomputedEmbedding",
    "build_visual_encoder",
    "coerce_image_backbone",
    "split_backbone_param_groups",
]


def _projection(in_dim: int, out_dim: int | None, dropout: float) -> tuple[nn.Module, int]:
    """Return ``(module, resolved_out_dim)``; identity when no projection is needed."""
    if out_dim is None or out_dim == in_dim:
        return nn.Identity(), in_dim
    return (
        nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ),
        out_dim,
    )


class PrecomputedEmbedding(nn.Module):
    """Passthrough/projection over a precomputed visual embedding.

    Reads ``batch["embedding"]`` ``(B, D)`` directly, or pools
    ``batch["embedding_seq"]`` ``(B, T, D)`` over the window dimension.  Two
    temporal poolers are available (``temporal_pooling``):

    - ``"mean"`` (default) — mask-aware mean over the valid frames
      (``batch["frame_mask"]`` ``(B, T)`` bool, True = valid; plain mean when
      absent).  No learned parameters.
    - ``"attention"`` — a single learnable query attends over the window via one
      :class:`torch.nn.MultiheadAttention` with ``key_padding_mask=~frame_mask``.
      A row whose mask is entirely False (no valid frame) cannot be attended, so
      its output falls back to zeros and a warning is logged.

    Parameters
    ----------
    in_dim:
        Embedding dimension ``D`` (discovered from the reader upstream).
    out_dim:
        Projection width; ``None`` or ``== in_dim`` leaves an identity
        passthrough.
    dropout:
        Dropout inside the projection block (unused when identity) and inside the
        attention pooler.
    temporal_pooling:
        ``"mean"`` (default) or ``"attention"`` — how ``embedding_seq`` is pooled.
    num_heads:
        Attention heads for the learned pooler (must divide *in_dim*); ignored by
        the mean pooler.  Defaults to 1 (always valid for any ``in_dim``).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        *,
        dropout: float = 0.0,
        temporal_pooling: TemporalPooling = "mean",
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if temporal_pooling not in ("mean", "attention"):
            raise ValueError(
                f"temporal_pooling must be 'mean' or 'attention', got {temporal_pooling!r}"
            )
        self.in_dim = in_dim
        self.temporal_pooling = temporal_pooling
        self.projection, self._out_dim = _projection(in_dim, out_dim, dropout)
        if temporal_pooling == "attention":
            if in_dim % num_heads != 0:
                raise ValueError(
                    f"num_heads {num_heads} must divide in_dim {in_dim} for attention pooling"
                )
            self.query = nn.Parameter(torch.zeros(1, 1, in_dim))
            nn.init.normal_(self.query, std=0.02)
            self.temporal_attn = nn.MultiheadAttention(
                in_dim, num_heads, dropout=dropout, batch_first=True
            )

    @property
    def out_dim(self) -> int:
        """Width of the visual embedding this source produces."""
        return self._out_dim

    @staticmethod
    def _pool(sequence: Tensor, mask: Tensor | None) -> Tensor:
        """Mean-pool a ``(B, T, D)`` window over ``T`` (masked when *mask* given)."""
        if mask is None:
            pooled: Tensor = sequence.mean(dim=1)
            return pooled
        weights = mask.unsqueeze(-1).to(sequence.dtype)
        summed = (sequence * weights).sum(dim=1)
        count = weights.sum(dim=1).clamp_min(1.0)
        masked: Tensor = summed / count
        return masked

    def _attention_pool(self, sequence: Tensor, mask: Tensor | None) -> Tensor:
        """Learned single-query attention pool of a ``(B, T, D)`` window over ``T``.

        ``key_padding_mask`` is ``~mask`` (True positions are ignored).  A row with
        no valid frame would make attention degenerate (softmax over an all-masked
        row -> NaN); such rows are un-masked at position 0 for a finite forward and
        then overwritten with zeros (with a warning), so the graph stays finite.
        """
        batch_size = sequence.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask: Tensor | None = None
        all_pad: Tensor | None = None
        if mask is not None:
            key_padding_mask = ~mask
            all_pad = key_padding_mask.all(dim=1)
            if bool(all_pad.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_pad, 0] = False
        attended, _ = self.temporal_attn(
            query, sequence, sequence, key_padding_mask=key_padding_mask, need_weights=False
        )
        pooled: Tensor = attended.squeeze(1)
        if all_pad is not None and bool(all_pad.any()):
            logger.warning(
                "attention temporal pooling: %d row(s) had an all-False frame_mask "
                "(no valid window frame); their pooled embedding falls back to zeros",
                int(all_pad.sum()),
            )
            pooled = pooled.clone()
            pooled[all_pad] = 0.0
        return pooled

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Read (and pool, when windowed) the batch's precomputed visual embedding.

        Parameters
        ----------
        batch:
            Batch dict carrying either ``embedding`` ``(B, D)`` float32 or
            ``embedding_seq`` ``(B, T, D)`` float32 plus an optional
            ``frame_mask`` ``(B, T)`` bool (True = valid frame).  Embeddings are
            dimensionless backbone features.

        Returns
        -------
        Tensor
            ``(B, out_dim)`` float32 visual embedding (dimensionless).

        Raises
        ------
        KeyError
            If *batch* carries neither ``embedding`` nor ``embedding_seq``.
        """
        if "embedding" in batch:
            embedding = batch["embedding"]
        elif "embedding_seq" in batch:
            if self.temporal_pooling == "attention":
                embedding = self._attention_pool(batch["embedding_seq"], batch.get("frame_mask"))
            else:
                embedding = self._pool(batch["embedding_seq"], batch.get("frame_mask"))
        else:
            raise KeyError(
                "batch has neither 'embedding' nor 'embedding_seq' for the "
                "precomputed-embedding visual source"
            )
        out: Tensor = self.projection(embedding)
        return out


class ImageEncoder(nn.Module):
    """Wrap an image backbone (``nn.Module`` with an integer ``.dim``).

    Parameters
    ----------
    backbone:
        Any module mapping ``(B, 3, H, W) -> (B, backbone.dim)`` and exposing
        an integer ``dim`` attribute (e.g. the DINOv2 wrapper, or a conv stub).
    frozen:
        Freeze every backbone parameter (``requires_grad=False``).
    unfreeze_last_n:
        When ``> 0`` and the backbone exposes a ``blocks`` sequence, exactly its
        last *n* blocks keep gradients: the backbone is frozen first and those
        blocks are then re-enabled (typical ViT fine-tuning).  That happens
        **independently of** *frozen* — asking for the last *n* blocks is asking
        for partial fine-tuning, so ``frozen=False`` no longer means "train every
        block".  Ignored, with a warning, for a backbone without a ``blocks``
        attribute: there is nothing to unfreeze, so its parameters keep whatever
        *frozen* set.
    out_dim:
        Optional projection width; ``None`` or ``== backbone.dim`` leaves an
        identity passthrough.
    dropout:
        Dropout inside the projection block (unused when identity).
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        frozen: bool = False,
        unfreeze_last_n: int = 0,
        out_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = getattr(backbone, "dim", None)
        if dim is None:
            raise AttributeError("image backbone must expose an integer 'dim' attribute")
        self.backbone = backbone
        self.backbone_dim = int(dim)
        partial_finetune = unfreeze_last_n > 0 and getattr(backbone, "blocks", None) is not None
        if frozen or partial_finetune:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        if unfreeze_last_n > 0:
            self._unfreeze_last_blocks(unfreeze_last_n)
        self.projection, self._out_dim = _projection(self.backbone_dim, out_dim, dropout)

    def _unfreeze_last_blocks(self, n: int) -> None:
        """Re-enable gradients on the last *n* transformer blocks, if any."""
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            logger.warning(
                "unfreeze_last_n=%d is inert: image backbone %s exposes no 'blocks' sequence, "
                "so there is nothing to unfreeze and its parameters keep the requires_grad "
                "'backbone_frozen' set",
                n,
                type(self.backbone).__name__,
            )
            return
        unfrozen = list(blocks)[-n:]
        for block in unfrozen:
            for param in block.parameters():
                param.requires_grad_(True)
        logger.info(
            "image backbone %s: unfroze the last %d of %d block(s) — %d of %d backbone "
            "parameters are trainable",
            type(self.backbone).__name__,
            len(unfrozen),
            len(list(blocks)),
            sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            sum(p.numel() for p in self.backbone.parameters()),
        )

    @property
    def out_dim(self) -> int:
        """Width of the visual embedding this encoder produces."""
        return self._out_dim

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the image backbone over the batch's frames.

        Parameters
        ----------
        batch:
            Batch dict whose ``image`` entry is ``(B, 3, H, W)`` float32,
            channels-first and normalized (dimensionless).

        Returns
        -------
        Tensor
            ``(B, out_dim)`` float32 visual embedding (dimensionless).
        """
        features = self.backbone(batch["image"])
        out: Tensor = self.projection(features)
        return out

    def param_groups(self, backbone_lr: float) -> list[dict[str, Any]]:
        """Optimizer parameter groups putting the backbone on its own learning rate.

        Parameters
        ----------
        backbone_lr:
            Learning rate for the backbone group.

        Returns
        -------
        list[dict[str, Any]]
            Up to two groups of **trainable** parameters: the backbone
            parameters at ``lr=backbone_lr``, then everything else (the
            projection) with no per-group override.  Frozen parameters are
            omitted, so a wholly frozen backbone yields no backbone group.
        """
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        other_params = [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("backbone.")
        ]
        groups: list[dict[str, Any]] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        if other_params:
            groups.append({"params": other_params})
        return groups


def split_backbone_param_groups(
    model: nn.Module, visual_encoder: nn.Module, backbone_lr: float | None
) -> list[dict[str, Any]]:
    """Optimizer groups for *model* with its image backbone on its own rate.

    The group ORDER is part of the contract: ``torch`` matches optimizer groups
    positionally in ``load_state_dict``, so ``--resume`` on an existing checkpoint
    requires the backbone group to stay first.

    Parameters
    ----------
    model:
        The model whose trainable parameters are split.
    visual_encoder:
        *model*'s visual branch; only :class:`ImageEncoder` exposes the
        ``param_groups`` method that identifies the backbone parameters.
    backbone_lr:
        Learning rate for the backbone group; ``None`` disables the split.

    Returns
    -------
    list[dict[str, Any]]
        ``[backbone_at_backbone_lr, everything_else]`` when *visual_encoder*
        exposes ``param_groups`` and *backbone_lr* is set, else a single group
        holding every trainable parameter of *model*.  A wholly frozen backbone
        contributes no parameters, so it collapses to that single group too.
    """
    get_backbone_groups = getattr(visual_encoder, "param_groups", None)
    if backbone_lr is None or get_backbone_groups is None:
        return [{"params": [p for p in model.parameters() if p.requires_grad]}]
    backbone_params = {
        id(p): p
        for group in get_backbone_groups(backbone_lr)
        if "lr" in group
        for p in group["params"]
    }
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_params]
    groups: list[dict[str, Any]] = []
    if backbone_params:
        groups.append({"params": list(backbone_params.values()), "lr": backbone_lr})
    if other:
        groups.append({"params": other})
    return groups


class _HubVisualBackbone(nn.Module):
    """Trainable ``nn.Module`` adapter over a :class:`VisualBackbone`.

    :func:`allsky.embeddings.backbone.build_backbone` returns an *extraction*
    wrapper (``DinoV2Backbone``) whose :meth:`encode` runs under
    ``inference_mode`` on CPU and detaches — unusable as a trainable image
    backbone.  This adapter loads the wrapper's underlying hub module (via
    ``load_torch_module``) and runs ``forward_features`` + token pooling **with
    gradients**, exposing the ``(B, 3, H, W) -> (B, dim)`` contract
    :class:`ImageEncoder` expects (plus ``.dim`` and a ``blocks`` view so the
    last-*n* ViT blocks can be unfrozen).

    Only the DINOv2 production path reaches this class; every test injects an
    ``nn.Module`` stub, which :func:`coerce_image_backbone` passes through
    untouched.
    """

    def __init__(self, backbone: Any, *, pooling: str = "cls") -> None:
        super().__init__()
        self.dim = int(backbone.dim)
        self.pooling = str(getattr(backbone, "pooling", pooling))
        loader = getattr(backbone, "load_torch_module", None)
        if not callable(loader):
            raise TypeError(
                f"image backbone {type(backbone).__name__} is neither an nn.Module nor a "
                "VisualBackbone exposing load_torch_module(); cannot use it for image-mode "
                "training"
            )
        # Typed Any (not nn.Module): nn.Module.__getattr__ returns Tensor | Module,
        # which makes forward_features(...) un-analyzable. Runtime registration as a
        # submodule still happens because the assigned value *is* an nn.Module.
        self.model: Any = loader()
        self._warn_if_blocks_are_chunked()

    def _warn_if_blocks_are_chunked(self) -> None:
        """Warn when ``blocks`` holds chunks instead of one entry per transformer block.

        DINOv2 built with ``block_chunks > 0`` exposes ``blocks`` as a ``ModuleList``
        of ``BlockChunk`` wrappers, so ``unfreeze_last_n=2`` would unfreeze the last
        two *chunks* (six blocks for a depth-12/4-chunk build), not two blocks.  The
        hub module advertises the real depth as ``n_blocks``.
        """
        blocks = getattr(self.model, "blocks", None)
        depth = getattr(self.model, "n_blocks", None)
        if blocks is None or depth is None or len(blocks) == int(depth):
            return
        logger.warning(
            "image backbone %s exposes %d 'blocks' entries for a depth-%d transformer: "
            "'blocks' is chunked, so unfreeze_last_n counts chunks of ~%.1f blocks each, "
            "not single blocks",
            type(self.model).__name__,
            len(blocks),
            int(depth),
            int(depth) / max(len(blocks), 1),
        )

    @property
    def blocks(self) -> Any:
        """The backbone's transformer ``blocks`` sequence (for unfreezing), if any."""
        return getattr(self.model, "blocks", None)

    def forward(self, image: Tensor) -> Tensor:
        """Encode ``(B, 3, H, W)`` frames to a ``(B, dim)`` embedding with gradients."""
        out = self.model.forward_features(image)
        cls = out["x_norm_clstoken"]
        if self.pooling == "cls":
            pooled = cls
        elif self.pooling == "mean":
            pooled = out["x_norm_patchtokens"].mean(dim=1)
        else:
            pooled = torch.cat([cls, out["x_norm_patchtokens"].mean(dim=1)], dim=-1)
        return cast("Tensor", pooled)


def coerce_image_backbone(backbone: Any, *, pooling: str = "cls") -> nn.Module:
    """Return an ``nn.Module`` image backbone from *backbone*.

    Parameters
    ----------
    backbone:
        An ``nn.Module`` (the test stubs, or any already-module backbone), which
        is returned unchanged; or a :class:`VisualBackbone` extraction wrapper
        (e.g. ``DinoV2Backbone`` from
        :func:`allsky.embeddings.backbone.build_backbone`), which is wrapped in
        :class:`_HubVisualBackbone` so it becomes a trainable, gradient-carrying
        image encoder.
    pooling:
        Token pooling (``cls`` | ``mean`` | ``cls+mean``) used only to seed the
        wrapper when the source does not carry its own.

    Returns
    -------
    nn.Module
        A module mapping ``(B, 3, H, W)`` float32 frames to ``(B, dim)`` float32
        embeddings and exposing an integer ``dim``.
    """
    if isinstance(backbone, nn.Module):
        return backbone
    return _HubVisualBackbone(backbone, pooling=pooling)


def build_visual_encoder(
    input_mode: Literal["image", "embedding"],
    *,
    embedding_dim: int | None = None,
    image_backbone: nn.Module | None = None,
    out_dim: int | None = None,
    frozen: bool = False,
    unfreeze_last_n: int = 0,
    dropout: float = 0.0,
    temporal_pooling: TemporalPooling = "mean",
) -> nn.Module:
    """Build the visual source for *input_mode* (``embedding`` or ``image``).

    Parameters
    ----------
    input_mode:
        ``"embedding"`` builds a :class:`PrecomputedEmbedding`; ``"image"``
        builds an :class:`ImageEncoder`.
    embedding_dim:
        Embedding width ``D``, required in embedding mode.
    image_backbone:
        Backbone module, required in image mode.
    out_dim:
        Projection width; ``None`` or equal to the source width leaves an
        identity passthrough.
    frozen, unfreeze_last_n:
        Image-mode fine-tuning controls (see :class:`ImageEncoder`); inert in
        embedding mode.
    dropout:
        Dropout inside the projection block and the attention pooler.
    temporal_pooling:
        How a windowed ``embedding_seq`` is pooled in embedding mode (``"mean"``
        or learned ``"attention"``); inert in image mode.

    Returns
    -------
    nn.Module
        The visual source, exposing an integer ``out_dim`` and a
        ``forward(batch) -> (B, out_dim)``.

    Raises
    ------
    ValueError
        If the required source is missing (``embedding_dim`` for embedding
        mode, ``image_backbone`` for image mode) or *input_mode* is unknown.
    """
    if input_mode == "embedding":
        if embedding_dim is None:
            raise ValueError("input_mode='embedding' requires embedding_dim")
        return PrecomputedEmbedding(
            embedding_dim, out_dim, dropout=dropout, temporal_pooling=temporal_pooling
        )
    if input_mode == "image":
        if image_backbone is None:
            raise ValueError("input_mode='image' requires image_backbone")
        return ImageEncoder(
            image_backbone,
            frozen=frozen,
            unfreeze_last_n=unfreeze_last_n,
            out_dim=out_dim,
            dropout=dropout,
        )
    raise ValueError(f"unknown input_mode {input_mode!r}; expected 'image' or 'embedding'")
