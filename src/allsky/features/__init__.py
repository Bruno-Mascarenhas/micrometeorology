"""Feature policy, engineering and normalization for the multimodal stack.

Public surface:

- :mod:`~allsky.features.policy` — the anti-leakage feature contract
  (:data:`~allsky.features.policy.SAFE_FEATURES`, ``resolve_feature_set``,
  ``validate_features``, :class:`~allsky.features.policy.ForbiddenFeatureError`).
- :mod:`~allsky.features.engineering` — :func:`~allsky.features.engineering.build_feature_frame`
  (solar geometry + cyclic encodings) in canonical policy order.
- :mod:`~allsky.features.normalization` — train-only
  :class:`~allsky.features.normalization.FeatureNormalizer` and
  :class:`~allsky.features.normalization.TargetNormalizer`.

Pure numpy/pandas; importing this package never pulls torch.
"""

from __future__ import annotations

from allsky.features.engineering import build_feature_frame
from allsky.features.normalization import (
    FeatureNormalizer,
    TargetNormalizer,
    fit_target_normalizers,
)
from allsky.features.policy import (
    EXTENDED_FEATURES,
    FEATURE_GROUPS,
    FORBIDDEN_FEATURES,
    SAFE_FEATURES,
    FeatureSet,
    ForbiddenFeatureError,
    active_feature_groups,
    resolve_feature_set,
    source_column,
    validate_features,
)

__all__ = [
    "EXTENDED_FEATURES",
    "FEATURE_GROUPS",
    "FORBIDDEN_FEATURES",
    "SAFE_FEATURES",
    "FeatureNormalizer",
    "FeatureSet",
    "ForbiddenFeatureError",
    "TargetNormalizer",
    "active_feature_groups",
    "build_feature_frame",
    "fit_target_normalizers",
    "resolve_feature_set",
    "source_column",
    "validate_features",
]
