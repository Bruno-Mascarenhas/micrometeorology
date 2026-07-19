"""Train-only standardization for features and regression targets.

Both normalizers are fit on the **training split only** and reused verbatim for
validation/test (and stored in the checkpoint), so no distributional
information leaks across splits.

- :class:`FeatureNormalizer` — per-column mean/std over an engineered feature
  frame, with near-constant columns clamped to unit std.
- :class:`TargetNormalizer` — scalar mean/std for one regression target; hold a
  ``{target_name: TargetNormalizer}`` mapping for a multi-head model.

Both are frozen dataclasses with :meth:`to_dict`/:meth:`from_dict` for
JSON round-tripping into run manifests and checkpoints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["FeatureNormalizer", "TargetNormalizer", "fit_target_normalizers"]

#: Guard against divide-by-zero when standardizing a (near-)constant channel.
_MIN_STD = 1e-6


@dataclass(frozen=True)
class FeatureNormalizer:
    """Per-column standardization statistics for the engineered feature frame.

    Fit on the training split via :meth:`fit`; apply to any split with
    :meth:`transform`.  ``std`` entries below :data:`_MIN_STD` are clamped to
    1.0 so constant columns pass through as zeros instead of exploding.
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> FeatureNormalizer:
        """Fit mean/std over *columns* (all columns when None).

        Must be called on the **training split only**.  Input values are
        expected finite; NaNs would propagate into the statistics.
        """
        cols = list(columns) if columns is not None else list(frame.columns)
        values = frame.loc[:, cols].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < _MIN_STD, 1.0, std).astype(np.float32)
        return cls(columns=tuple(cols), mean=mean.astype(np.float32), std=std)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Standardize *frame* to ``(n_rows, n_columns)`` float32 in fit order."""
        values = frame.loc[:, list(self.columns)].to_numpy(dtype=np.float32)
        standardized: np.ndarray = (values - self.mean) / self.std
        return standardized

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for manifest/checkpoint metadata."""
        return {
            "columns": list(self.columns),
            "mean": self.mean.astype(np.float64).tolist(),
            "std": self.std.astype(np.float64).tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureNormalizer:
        """Inverse of :meth:`to_dict`."""
        return cls(
            columns=tuple(payload["columns"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


@dataclass(frozen=True)
class TargetNormalizer:
    """Scalar standardization for one regression target (train-split mean/std).

    The model learns/predicts in normalized space; the engine denormalizes
    before metrics/reporting so evaluation is always in physical units.
    """

    mean: float
    std: float

    @classmethod
    def fit(cls, values: Sequence[float] | np.ndarray | pd.Series) -> TargetNormalizer:
        """Fit over finite values of *values* (training split only)."""
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        mean = float(finite.mean()) if finite.size else 0.0
        std = float(finite.std()) if finite.size else 1.0
        if std < _MIN_STD:
            std = 1.0
        return cls(mean=mean, std=std)

    def normalize(self, values: Sequence[float] | np.ndarray | float) -> np.ndarray:
        """Map physical units to normalized space."""
        normalized: np.ndarray = (np.asarray(values, dtype=np.float64) - self.mean) / self.std
        return normalized

    def denormalize(self, values: Sequence[float] | np.ndarray | float) -> np.ndarray:
        """Map normalized space back to physical units."""
        physical: np.ndarray = np.asarray(values, dtype=np.float64) * self.std + self.mean
        return physical

    def to_dict(self) -> dict[str, float]:
        """JSON-serializable form."""
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TargetNormalizer:
        """Inverse of :meth:`to_dict`."""
        return cls(mean=float(payload["mean"]), std=float(payload["std"]))


def fit_target_normalizers(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> dict[str, TargetNormalizer]:
    """Fit one :class:`TargetNormalizer` per target column (training split only)."""
    return {col: TargetNormalizer.fit(frame[col]) for col in columns}
