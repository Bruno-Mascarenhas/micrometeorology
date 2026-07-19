"""Model evaluation metrics.

All metrics follow a uniform signature::

    metric(observed: NDArray, predicted: NDArray) -> float

NaN values are stripped before computation.  If fewer than 2 valid pairs
remain, the metric returns ``NaN``.

Ported from ``wrf/metrics.py`` with added type hints and NaN safety.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _clean_pairs(obs: NDArray, pred: NDArray) -> tuple[NDArray, NDArray]:
    """Remove pairs where either value is NaN."""
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(pred))
    if mask.all():
        return obs, pred
    return obs[mask], pred[mask]


def rmse(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Root Mean Square Error."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(obs - pred))))


def mae(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Mean Absolute Error."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    return float(np.mean(np.abs(obs - pred)))


def mbe(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Mean Bias Error (positive = model over-predicts)."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    return float(np.mean(pred - obs))


def r_squared(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Coefficient of determination (R²)."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    ss_res = np.sum(np.square(obs - pred))
    ss_tot = np.sum(np.square(obs - np.mean(obs)))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def correlation(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Pearson correlation coefficient."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    if np.isclose(np.std(obs), 0.0) or np.isclose(np.std(pred), 0.0):
        return float("nan")
    corr = np.corrcoef(obs, pred)
    return float(corr[0, 1])


def d_index(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Willmott's index of agreement (d).

    Ranges from 0 (no agreement) to 1 (perfect agreement).
    """
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    obs_mean = np.mean(obs)
    numerator = np.sum(np.square(obs - pred))
    denominator = np.sum(np.square(np.abs(pred - obs_mean) + np.abs(obs - obs_mean)))
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


def ia(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Index of Agreement (alternative formulation)."""
    return d_index(observed, predicted, clean=clean)


def ioa(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Refined Index of Agreement (Willmott et al., 2012).

    Ranges from -1 to 1, with 1 indicating perfect agreement.
    """
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    obs_mean = np.mean(obs)
    numerator = np.sum(np.abs(pred - obs))
    denominator = 2.0 * np.sum(np.abs(obs - obs_mean))
    if denominator == 0:
        return float("nan")
    ratio = numerator / denominator
    if ratio <= 1:
        return float(1.0 - ratio)
    else:
        return float(1.0 / ratio - 1.0)


def nrmse(observed: NDArray, predicted: NDArray, clean: bool = True) -> float:
    """Normalised RMSE (NRMSE), as RMSE / range(observed)."""
    obs, pred = _clean_pairs(observed, predicted) if clean else (observed, predicted)
    if len(obs) < 2:
        return float("nan")
    obs_range = float(np.max(obs) - np.min(obs))
    if obs_range == 0:
        return float("nan")
    return rmse(obs, pred, clean=False) / obs_range


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
    obs, pred = _clean_pairs(observed, predicted)
    if len(obs) < 2:
        return {name: float("nan") for name in ALL_METRICS}
    return {name: fn(obs, pred, clean=False) for name, fn in ALL_METRICS.items()}
