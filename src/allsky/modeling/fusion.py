"""Fusion blocks combining the visual and sensor embeddings.

Three strategies, all exposing an :attr:`out_dim` and a uniform ``forward``:

- :class:`ConcatFusion` — plain concatenation.
- :class:`FiLMFusion` — feature-wise linear modulation of the visual embedding
  conditioned on the sensor embedding, then concatenated with the sensor
  embedding (a residual sensor path).  The modulation generator is zero-init
  (weight *and* bias) so at initialization ``gamma == beta == 0`` and the block
  reduces **exactly** to ``concat(visual, sensor)``.
- :class:`CrossAttentionFusion` — one :class:`torch.nn.MultiheadAttention`
  layer with the visual embedding as the query and one token per sensor
  feature-group (built by small per-group linears) as keys/values; the
  attended visual token is concatenated with the sensor embedding (residual
  sensor path).  Absent groups are masked via ``key_padding_mask``.

The ``needs_features`` class attribute tells the assembling model whether the
fusion must be handed the raw standardized feature vector (only cross-attention
builds per-group tokens from it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from allsky.modeling.contracts import group_slices

__all__ = [
    "ConcatFusion",
    "CrossAttentionFusion",
    "FiLMFusion",
    "build_fusion",
]


class ConcatFusion(nn.Module):
    """Concatenate the visual and sensor embeddings along the feature axis."""

    needs_features = False

    def __init__(self, visual_dim: int, sensor_dim: int) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.sensor_dim = sensor_dim
        self._out_dim = visual_dim + sensor_dim

    @property
    def out_dim(self) -> int:
        """Width of the fused vector (``visual_dim + sensor_dim``)."""
        return self._out_dim

    def forward(self, visual: Tensor, sensor: Tensor) -> Tensor:
        """Return ``concat(visual, sensor)`` ``(B, visual_dim + sensor_dim)``."""
        fused: Tensor = torch.cat([visual, sensor], dim=-1)
        return fused


class FiLMFusion(nn.Module):
    """FiLM-modulate the visual embedding on the sensor embedding, then concat.

    Formula
    -------
    ``gamma, beta = film_gen(sensor)`` (split in half);
    ``conditioned = (1 + gamma) * visual + beta``;
    ``fused = concat(conditioned, sensor)``.

    The final ``film_gen`` layer is zero-initialized (weight and bias), so at
    initialization ``gamma = beta = 0`` and ``conditioned = visual`` — the
    block starts as an exact identity, i.e. equal to
    :class:`ConcatFusion`'s output.
    """

    needs_features = False

    def __init__(self, visual_dim: int, sensor_dim: int) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.sensor_dim = sensor_dim
        self.film_gen = nn.Linear(sensor_dim, 2 * visual_dim)
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)
        self._out_dim = visual_dim + sensor_dim

    @property
    def out_dim(self) -> int:
        """Width of the fused vector (``visual_dim + sensor_dim``)."""
        return self._out_dim

    def forward(self, visual: Tensor, sensor: Tensor) -> Tensor:
        """Return ``concat((1 + gamma) * visual + beta, sensor)``."""
        gamma, beta = self.film_gen(sensor).chunk(2, dim=-1)
        conditioned = (1.0 + gamma) * visual + beta
        fused: Tensor = torch.cat([conditioned, sensor], dim=-1)
        return fused


class CrossAttentionFusion(nn.Module):
    """One-layer cross-attention: visual query attends to sensor group tokens.

    Each active feature group (see
    :func:`allsky.features.policy.active_feature_groups`) becomes one token via
    a small ``Linear(group_size -> token_dim)``; the visual embedding is
    projected to a single query token.  A single
    :class:`torch.nn.MultiheadAttention` (batch-first) attends the query over
    the group tokens; the attended visual token is concatenated with the raw
    sensor embedding (a residual sensor path).

    Parameters
    ----------
    visual_dim, sensor_dim:
        Widths of the visual and sensor embeddings.
    feature_columns:
        Ordered engineered feature names (the standardized vector's columns).
    groups:
        ``group -> member feature names`` (e.g. ``active_feature_groups``); only
        non-empty groups become tokens.
    num_heads:
        Attention heads; ``token_dim`` must be divisible by it.
    token_dim:
        Attention embedding width; defaults to *visual_dim*.

    Notes
    -----
    ``forward`` accepts an optional ``key_padding_mask`` ``(B, n_groups)``
    (True = ignore that group's token) so absent groups contribute nothing —
    changing a masked group's feature values leaves the output unchanged.
    """

    needs_features = True

    def __init__(
        self,
        visual_dim: int,
        sensor_dim: int,
        *,
        feature_columns: Sequence[str],
        groups: Mapping[str, Sequence[str]],
        num_heads: int = 4,
        token_dim: int | None = None,
    ) -> None:
        super().__init__()
        resolved_token_dim = token_dim if token_dim is not None else visual_dim
        if resolved_token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({resolved_token_dim}) must be divisible by num_heads ({num_heads})"
            )
        slices = group_slices(feature_columns, groups)
        if not slices:
            raise ValueError("no non-empty feature groups for cross-attention tokens")
        self.token_dim = resolved_token_dim
        self.sensor_dim = sensor_dim
        self.group_names: list[str] = list(slices)
        self._group_indices: list[list[int]] = [slices[name] for name in self.group_names]
        self.group_proj = nn.ModuleList(
            nn.Linear(len(indices), resolved_token_dim) for indices in self._group_indices
        )
        self.visual_proj = nn.Linear(visual_dim, resolved_token_dim)
        self.attn = nn.MultiheadAttention(resolved_token_dim, num_heads, batch_first=True)
        self._out_dim = resolved_token_dim + sensor_dim

    @property
    def out_dim(self) -> int:
        """Width of the fused vector (``token_dim + sensor_dim``)."""
        return self._out_dim

    def forward(
        self,
        visual: Tensor,
        sensor: Tensor,
        *,
        features: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend the visual query over sensor group tokens; concat with *sensor*.

        Parameters
        ----------
        visual:
            ``(B, visual_dim)`` visual embedding (the query).
        sensor:
            ``(B, sensor_dim)`` sensor embedding (residual path).
        features:
            ``(B, F)`` standardized feature vector, sliced into group tokens.
        key_padding_mask:
            Optional ``(B, n_groups)`` bool mask; True entries are ignored.
        """
        tokens = [
            proj(features[:, indices])
            for proj, indices in zip(self.group_proj, self._group_indices, strict=True)
        ]
        keys = torch.stack(tokens, dim=1)  # (B, n_groups, token_dim)
        query = self.visual_proj(visual).unsqueeze(1)  # (B, 1, token_dim)
        attended, _ = self.attn(
            query, keys, keys, key_padding_mask=key_padding_mask, need_weights=False
        )
        fused: Tensor = torch.cat([attended.squeeze(1), sensor], dim=-1)
        return fused


def build_fusion(
    name: str,
    visual_dim: int,
    sensor_dim: int,
    *,
    feature_columns: Sequence[str] | None = None,
    groups: Mapping[str, Sequence[str]] | None = None,
    num_heads: int = 4,
    token_dim: int | None = None,
) -> nn.Module:
    """Construct a fusion block by *name* (``concat``/``film``/``cross_attention``).

    Raises
    ------
    ValueError
        For an unknown *name*, or when cross-attention is requested without
        *feature_columns* and *groups*.
    """
    if name == "concat":
        return ConcatFusion(visual_dim, sensor_dim)
    if name == "film":
        return FiLMFusion(visual_dim, sensor_dim)
    if name == "cross_attention":
        if feature_columns is None or groups is None:
            raise ValueError("cross_attention fusion requires feature_columns and groups")
        return CrossAttentionFusion(
            visual_dim,
            sensor_dim,
            feature_columns=feature_columns,
            groups=groups,
            num_heads=num_heads,
            token_dim=token_dim,
        )
    raise ValueError(
        f"unknown fusion {name!r}; expected one of 'concat', 'film', 'cross_attention'"
    )
