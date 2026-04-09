"""Model evaluation metrics.

All metrics follow a uniform signature::

    metric(observed: NDArray, predicted: NDArray) -> float

NaN values are stripped before computation.  If fewer than 2 valid pairs
remain, the metric returns ``NaN``.

Ported from ``wrf/metrics.py`` with added type hints and NaN safety.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _clean_pairs(obs: NDArray, pred: NDArray) -> tuple[NDArray, NDArray]:
    """Remove pairs where either value is NaN."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    return obs[mask], pred[mask]


def rmse(observed: NDArray, predicted: NDArray) -> float:
    """Root Mean Square Error."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    return float(np.sqrt(np.mean((o - p) ** 2)))


def mae(observed: NDArray, predicted: NDArray) -> float:
    """Mean Absolute Error."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    return float(np.mean(np.abs(o - p)))


def mbe(observed: NDArray, predicted: NDArray) -> float:
    """Mean Bias Error (positive = model over-predicts)."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    return float(np.mean(p - o))


def r_squared(observed: NDArray, predicted: NDArray) -> float:
    """Coefficient of determination (R²)."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def correlation(observed: NDArray, predicted: NDArray) -> float:
    """Pearson correlation coefficient."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    corr = np.corrcoef(o, p)
    return float(corr[0, 1])


def d_index(observed: NDArray, predicted: NDArray) -> float:
    """Willmott's index of agreement (d).

    Ranges from 0 (no agreement) to 1 (perfect agreement).
    """
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    o_mean = np.mean(o)
    numerator = np.sum((o - p) ** 2)
    denominator = np.sum((np.abs(p - o_mean) + np.abs(o - o_mean)) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def ia(observed: NDArray, predicted: NDArray) -> float:
    """Index of Agreement (alternative formulation)."""
    return d_index(observed, predicted)


def ioa(observed: NDArray, predicted: NDArray) -> float:
    """Refined Index of Agreement (Willmott et al., 2012).

    Ranges from -1 to 1, with 1 indicating perfect agreement.
    """
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    o_mean = np.mean(o)
    numerator = np.sum(np.abs(p - o))
    denominator = 2.0 * np.sum(np.abs(o - o_mean))
    if denominator == 0:
        return float("nan")
    ratio = numerator / denominator
    if ratio <= 1:
        return float(1.0 - ratio)
    else:
        return float(1.0 / ratio - 1.0)


def nrmse(observed: NDArray, predicted: NDArray) -> float:
    """Normalised RMSE (NRMSE), as RMSE / range(observed)."""
    o, p = _clean_pairs(observed, predicted)
    if len(o) < 2:
        return float("nan")
    obs_range = float(np.max(o) - np.min(o))
    if obs_range == 0:
        return float("nan")
    return rmse(o, p) / obs_range


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

ALL_METRICS = {
    "RMSE": rmse,
    "MAE": mae,
    "MBE": mbe,
    "R²": r_squared,
    "r": correlation,
    "d": d_index,
    "IOA": ioa,
    "NRMSE": nrmse,
}


def compute_all(observed: NDArray, predicted: NDArray) -> dict[str, float]:
    """Compute all available metrics and return as a dict."""
    return {name: fn(observed, predicted) for name, fn in ALL_METRICS.items()}
