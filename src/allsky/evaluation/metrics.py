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

import itertools
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from labmim_core.sky import SKY_CLASS_COUNT, SKY_CLASS_NAMES
from solrad_correction.evaluation.metrics import compute_regression_metrics

__all__ = [
    "CLASSIFICATION_METRIC_KEYS",
    "REFERENCE_LABELS",
    "REGRESSION_METRIC_KEYS",
    "SKILL_METRIC_KEYS",
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

#: The baselines every regression target is scored against.  The evaluator names
#: its per-sample reference columns ``<label>_<target>`` from this same tuple, so
#: a baseline cannot be published under a label nothing scores.
REFERENCE_LABELS: tuple[str, ...] = ("persistence", "clearsky")

#: Skill entries appended to a regression target's metrics, one triple per
#: reference: the reference's RMSE, the model's RMSE **over the rows paired with
#: that reference**, and the skill score.  The numerator is published because it
#: is not the whole-split ``rmse``: a reference drops the rows it cannot cover
#: (persistence has no predecessor on a day's first frame), so only these two
#: keys reconcile to ``skill_<label>``.
SKILL_METRIC_KEYS: tuple[str, ...] = tuple(
    f"{statistic}_{label}"
    for label in REFERENCE_LABELS
    for statistic in ("rmse", "rmse_model", "skill")
)


def skill_score(model_rmse: float, reference_rmse: float) -> float:
    """Fraction of a reference forecast's error the model removes.

    Parameters
    ----------
    model_rmse, reference_rmse:
        Root-mean-square errors of the model and of the reference, in the same
        physical unit.

    Returns
    -------
    float
        ``1 - rmse_model / rmse_reference``: ``1`` is a perfect model, ``0`` is
        no better than the reference, and a negative value means the reference
        wins.  NaN when the reference is not finite or is exactly zero, which
        is the honest answer rather than an infinite skill.

    Notes
    -----
    An RMSE on its own does not say whether a model is worth running. The skill
    score states it against a named alternative, which is why irradiance work
    reports it against persistence and a clear-sky model rather than alone.
    """
    if not np.isfinite(model_rmse) or not np.isfinite(reference_rmse) or reference_rmse == 0.0:
        return float("nan")
    return float(1.0 - model_rmse / reference_rmse)


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


#: Reliability-diagram bins for the expected calibration error.
ECE_BINS = 10


def classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    n_classes: int = SKY_CLASS_COUNT,
    *,
    probabilities: ArrayLike | None = None,
) -> dict[str, Any]:
    """Classification metrics for integer-labelled, ordered predictions.

    Rows whose true label is non-finite, negative (the ``-1`` missing sentinel)
    or ``>= n_classes`` are dropped before scoring.  Metrics are computed with a
    fixed ``labels = range(n_classes)`` so the confusion matrix is always
    ``n_classes-by-n_classes`` even when a class is absent from the split — which
    also makes degenerate single-class inputs safe (macro-F1 uses
    ``zero_division=0``; sklearn's ill-defined-metric warnings are suppressed).

    The classes are ordered (the four sky conditions are bins of Kt), so beside
    the nominal scores the ordinal ones are reported: the quadratic-weighted
    kappa of Cohen (1968), which charges a two-class miss four times a
    one-class miss; the share of rows within one class of the truth; and the
    mean class-index distance. With *probabilities* the calibration of the head
    is scored too — negative log-likelihood, the multiclass Brier score and the
    expected calibration error over :data:`ECE_BINS` confidence bins.

    Parameters
    ----------
    y_true, y_pred:
        True and predicted class integers.
    n_classes:
        Number of classes, defaulting to :data:`~labmim_core.sky.SKY_CLASS_COUNT`
        so an absent-column split reports the same confusion shape as a scored one.
    probabilities:
        Optional ``(N, n_classes)`` class probabilities aligned with *y_true*;
        rows dropped for an invalid label are dropped here too.

    Returns
    -------
    dict
        ``accuracy`` / ``balanced_accuracy`` / ``macro_f1`` / ``kappa_quadratic``
        / ``within_one_class`` / ``ordinal_mae`` (floats, ``NaN`` on empty
        input), ``n``, ``confusion`` (a nested ``n_classes-by-n_classes`` list of
        ints), ``per_class`` (``precision`` / ``recall`` / ``f1`` / ``support``
        keyed by class name) and, when *probabilities* is given, ``nll``,
        ``brier`` and ``ece``.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    names = (
        list(SKY_CLASS_NAMES)
        if n_classes == SKY_CLASS_COUNT
        else [str(i) for i in range(n_classes)]
    )
    true = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()
    valid = _valid_label_mask(true, n_classes) & _valid_label_mask(pred, n_classes)
    true = true[valid].astype(np.int64)
    pred = pred[valid].astype(np.int64)
    n = int(true.size)
    labels = list(range(n_classes))

    if n == 0:
        nan = float("nan")
        empty: dict[str, Any] = {
            "accuracy": nan,
            "balanced_accuracy": nan,
            "macro_f1": nan,
            "kappa_quadratic": nan,
            "within_one_class": nan,
            "ordinal_mae": nan,
            "n": 0,
            "confusion": [[0] * n_classes for _ in range(n_classes)],
            "per_class": {
                name: {"precision": nan, "recall": nan, "f1": nan, "support": 0} for name in names
            },
        }
        if probabilities is not None:
            empty.update(nll=nan, brier=nan, ece=nan)
        return empty

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # quiet sklearn ill-defined-metric warnings
        accuracy = float(accuracy_score(true, pred))
        balanced = float(balanced_accuracy_score(true, pred))
        macro_f1 = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
        kappa = float(cohen_kappa_score(true, pred, labels=labels, weights="quadratic"))
        precision, recall, f1, support = precision_recall_fscore_support(
            true, pred, labels=labels, zero_division=0
        )
    confusion: NDArray = confusion_matrix(true, pred, labels=labels)
    distance = np.abs(pred - true)
    report: dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "kappa_quadratic": kappa if np.isfinite(kappa) else float("nan"),
        "within_one_class": float(np.mean(distance <= 1)),
        "ordinal_mae": float(np.mean(distance)),
        "n": n,
        "confusion": confusion.astype(np.int64).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
    }
    if probabilities is not None:
        report.update(
            _calibration_metrics(np.asarray(probabilities, dtype=np.float64)[valid], true)
        )
    return report


def _calibration_metrics(probabilities: NDArray, true: NDArray) -> dict[str, float]:
    """NLL, multiclass Brier and expected calibration error of *probabilities*."""
    n = probabilities.shape[0]
    clipped = np.clip(probabilities, 1e-12, 1.0)
    one_hot = np.zeros_like(clipped)
    one_hot[np.arange(n), true] = 1.0
    confidence = clipped.max(axis=1)
    hit = (clipped.argmax(axis=1) == true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, ECE_BINS + 1)
    ece = 0.0
    for low, high in itertools.pairwise(edges):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(hit[in_bin].mean() - confidence[in_bin].mean())
    return {
        "nll": float(-np.log(clipped[np.arange(n), true]).mean()),
        "brier": float(((clipped - one_hot) ** 2).sum(axis=1).mean()),
        "ece": float(ece),
    }


def _valid_label_mask(labels: NDArray, n_classes: int) -> NDArray:
    """Boolean mask of finite class labels in ``[0, n_classes)``."""
    as_float = labels.astype(np.float64)
    return np.isfinite(as_float) & (as_float >= 0) & (as_float < n_classes)
