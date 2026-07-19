"""Regression and classification metrics for the multimodal evaluator.

Both helpers are **NaN- and empty-safe** and torch-free (numpy + scikit-learn +
the shared :mod:`solrad_correction` regression metrics), so importing this module
never pulls a heavy framework.

- :func:`regression_metrics` reuses
  :func:`solrad_correction.evaluation.metrics.compute_regression_metrics`
  (RMSE / MAE / MBE / R² / r / d / MAPE, each NaN-safe) and adds a ``bias`` alias
  of the mean bias error plus ``nmae`` / ``nrmse`` normalized by ``mean(obs)``.
- :func:`classification_metrics` reports accuracy, balanced accuracy, macro-F1
  and a fixed ``n_classes-by-n_classes`` confusion matrix, ignoring rows whose
  true label is the missing sentinel (``< 0``) or out of range.

Keys are lowercase ASCII (``r2`` rather than ``R²``) so the JSON / CSV reports
stay portable.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from solrad_correction.evaluation.metrics import compute_regression_metrics

__all__ = [
    "CLASSIFICATION_METRIC_KEYS",
    "REGRESSION_METRIC_KEYS",
    "classification_metrics",
    "regression_metrics",
]

#: Remap the shared regression-metric keys to lowercase ASCII report keys.
_REGRESSION_KEY_REMAP: dict[str, str] = {
    "RMSE": "rmse",
    "MAE": "mae",
    "MBE": "mbe",
    "R²": "r2",
    "r": "r",
    "d": "d",
    "MAPE": "mape",
}

#: Ordered scalar keys :func:`regression_metrics` always returns (``n`` is the
#: count of finite ``(obs, pred)`` pairs the metrics were computed over).
REGRESSION_METRIC_KEYS: tuple[str, ...] = (
    "rmse",
    "mae",
    "mbe",
    "bias",
    "r2",
    "r",
    "d",
    "mape",
    "nmae",
    "nrmse",
    "n",
)

#: Ordered scalar keys :func:`classification_metrics` always returns (besides the
#: nested ``confusion`` matrix).
CLASSIFICATION_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "n",
)


def regression_metrics(obs: ArrayLike, pred: ArrayLike) -> dict[str, float]:
    """Regression metrics for observed/predicted arrays (physical units).

    Pairs where either value is non-finite are dropped first.  The base metrics
    come from
    :func:`solrad_correction.evaluation.metrics.compute_regression_metrics`
    (already NaN-safe: any metric is ``NaN`` when fewer than two valid pairs
    remain).  Two normalized errors are appended:

    - ``nmae = mae / mean(obs)``
    - ``nrmse = rmse / mean(obs)``

    both computed over the same cleaned observations; when ``mean(obs)`` is zero
    or non-finite they are ``NaN`` (documented, never a divide-by-zero).
    ``bias`` is a plain alias of ``mbe`` (positive = model over-predicts).

    Parameters
    ----------
    obs, pred:
        Observed and predicted values (any shape; flattened).

    Returns
    -------
    dict[str, float]
        The keys in :data:`REGRESSION_METRIC_KEYS`; on empty input every metric
        is ``NaN`` and ``n`` is ``0`` (never an empty dict, so downstream tables
        keep a stable schema).
    """
    observed = np.asarray(obs, dtype=np.float64).ravel()
    predicted = np.asarray(pred, dtype=np.float64).ravel()
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[mask], predicted[mask]
    n = int(observed.size)

    if n == 0:
        empty: dict[str, float] = dict.fromkeys(REGRESSION_METRIC_KEYS, float("nan"))
        empty["n"] = 0.0
        return empty

    base = compute_regression_metrics(observed, predicted)
    metrics: dict[str, float] = {
        report_key: float(base[source_key])
        for source_key, report_key in _REGRESSION_KEY_REMAP.items()
    }
    metrics["bias"] = metrics["mbe"]

    mean_obs = float(np.mean(observed))
    normalizable = np.isfinite(mean_obs) and mean_obs != 0.0
    metrics["nmae"] = metrics["mae"] / mean_obs if normalizable else float("nan")
    metrics["nrmse"] = metrics["rmse"] / mean_obs if normalizable else float("nan")
    metrics["n"] = float(n)
    return metrics


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: int = 3
) -> dict[str, Any]:
    """Classification metrics for integer-labelled predictions.

    Rows whose true label is non-finite, negative (the ``-1`` missing sentinel)
    or ``>= n_classes`` are dropped before scoring.  Metrics are computed with a
    fixed ``labels = range(n_classes)`` so the confusion matrix is always
    ``n_classes-by-n_classes`` even when a class is absent from the split — which
    also makes degenerate single-class inputs safe (macro-F1 uses
    ``zero_division=0``; sklearn's ill-defined-metric warnings are suppressed).

    Parameters
    ----------
    y_true, y_pred:
        True and predicted class integers.
    n_classes:
        Number of classes (default 3: clear / partially_cloudy / overcast).

    Returns
    -------
    dict
        ``accuracy`` / ``balanced_accuracy`` / ``macro_f1`` (floats, ``NaN`` on
        empty input), ``n`` (count of scored rows) and ``confusion`` (a nested
        ``n_classes-by-n_classes`` list of ints, all-zero on empty input).
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
    )

    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    valid = _valid_label_mask(true, n_classes) & _valid_label_mask(pred, n_classes)
    true = true[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    n = int(true.size)
    labels = list(range(n_classes))

    if n == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "n": 0,
            "confusion": [[0] * n_classes for _ in range(n_classes)],
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quiet sklearn ill-defined-metric warnings
        accuracy = float(accuracy_score(true, pred))
        balanced = float(balanced_accuracy_score(true, pred))
        macro_f1 = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
    confusion: NDArray = confusion_matrix(true, pred, labels=labels)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "n": n,
        "confusion": confusion.astype(np.int64).tolist(),
    }


def _valid_label_mask(labels: NDArray, n_classes: int) -> NDArray:
    """Boolean mask of finite class labels in ``[0, n_classes)``."""
    as_float = labels.astype(np.float64)
    return np.isfinite(as_float) & (as_float >= 0) & (as_float < n_classes)
