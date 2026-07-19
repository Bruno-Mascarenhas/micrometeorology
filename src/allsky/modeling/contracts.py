"""Model I/O contracts for the multimodal zoo.

Every model in :mod:`allsky.modeling` speaks one contract: it takes the
new-stack batch dict (see :mod:`allsky.data.datasets`) and returns a
:class:`ModelOutputs` mapping.

Regression outputs (``dhi``, ``kindex``, ``cloud_fraction``) live in
**normalized target space** — the model predicts the standardized quantity and
the training/evaluation engine denormalizes with the train-split
:class:`allsky.features.normalization.TargetNormalizer` before computing losses
in a comparable scale and metrics in physical units.  ``sky_logits`` are raw
class logits and ``dhi_log_var`` is a predicted log-variance (heteroscedastic
head, already clamped by the head).

This module is deliberately **torch-free at runtime**: the ``Tensor``
annotations are typing-only (evaluated lazily under ``from __future__ import
annotations``), so importing it never pulls torch.  :func:`group_slices` is pure
Python and drives the cross-attention sensor tokens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

__all__ = [
    "ModelOutputs",
    "MultimodalModel",
    "group_slices",
]


class ModelOutputs(TypedDict, total=False):
    """Union of every head's output; only enabled heads populate their key.

    Keys
    ----
    dhi:
        Diffuse-horizontal-irradiance prediction in **normalized** space.
    dhi_log_var:
        Predicted log-variance for the heteroscedastic DHI head (clamped).
    kindex:
        Clearness/clear-sky index prediction in **normalized** space.
    sky_logits:
        Raw ``(B, 3)`` class logits (clear / partially_cloudy / overcast).
    cloud_fraction:
        Cloud-fraction prediction in ``[0, 1]`` (sigmoid output; not
        normalized — it is already a bounded fraction).
    """

    dhi: Tensor
    dhi_log_var: Tensor
    kindex: Tensor
    sky_logits: Tensor
    cloud_fraction: Tensor


@runtime_checkable
class MultimodalModel(Protocol):
    """Structural type every zoo model satisfies.

    A model reads whatever it needs from *batch* (``features``, ``image`` or
    ``embedding``/``embedding_seq``) and returns the subset of
    :class:`ModelOutputs` its heads produce.  ``nn.Module`` subclasses satisfy
    this through their ``forward`` (and ``__call__``).
    """

    def forward(self, batch: dict[str, Tensor]) -> ModelOutputs:
        """Map a batch dict to the enabled model outputs."""
        ...


def group_slices(
    feature_columns: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> dict[str, list[int]]:
    """Map each feature group to the column indices it occupies in *feature_columns*.

    Used to slice the standardized feature vector into per-group sensor tokens
    for :class:`allsky.modeling.fusion.CrossAttentionFusion`.  A group is
    included only when at least one of its members is present in
    *feature_columns*; members absent from *feature_columns* are skipped (so a
    ``safe`` feature vector paired with the full :data:`FEATURE_GROUPS` simply
    drops the empty ``radiometry_aux`` group).

    Parameters
    ----------
    feature_columns:
        Ordered engineered feature names (the standardized vector's column
        order), e.g. from
        :func:`allsky.features.policy.resolve_feature_set`.
    groups:
        ``group name -> member feature names`` mapping, e.g. from
        :func:`allsky.features.policy.active_feature_groups`.

    Returns
    -------
    dict[str, list[int]]
        ``group name -> sorted column indices`` for every non-empty group,
        preserving the iteration order of *groups*.

    Notes
    -----
    For the ``safe`` set paired with ``active_feature_groups("safe")`` the
    union of the returned index lists covers every column exactly once.
    """
    index_of = {name: i for i, name in enumerate(feature_columns)}
    slices: dict[str, list[int]] = {}
    for group, members in groups.items():
        indices = sorted(index_of[m] for m in members if m in index_of)
        if indices:
            slices[group] = indices
    return slices
