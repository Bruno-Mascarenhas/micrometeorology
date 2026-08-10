"""Model evaluation metrics.

All metrics follow a uniform signature::

    metric(observed: NDArray, predicted: NDArray) -> float

Non-finite values (NaN and ±inf) are stripped pairwise before computation, so
one diverged prediction costs its own pair instead of the whole record.  If
fewer than 2 valid pairs remain, the metric returns ``NaN``.
"""

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def _clean_pairs(obs: NDArray, pred: NDArray) -> tuple[NDArray, NDArray]:
    """Remove pairs where either value is non-finite (NaN or ±inf).

    ``±inf`` has to go with the NaNs: it survives an ``isnan`` filter and then
    poisons every reduction downstream (RMSE ``inf``, R² ``-inf``, IOA a
    plausible-looking ``-1.0``), so a single overflowed model output would
    destroy the metrics of an otherwise usable run.
    """
    obs = np.asarray(obs, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.all():
        return obs, pred
    logger.warning("dropped %d non-finite pairs of %d", int((~mask).sum()), mask.size)
    return obs[mask], pred[mask]


def valid_pairs(observed: NDArray, predicted: NDArray) -> int:
    """Number of pairs that survive the non-finite filter.

    This is the denominator every metric is actually computed over, and it
    differs per variable — a frame paired by ``merge_asof`` keeps one row per
    observation whether or not a model row matched, and each column then loses
    its own gaps. Published beside the metrics so the sample size is recoverable
    from the artifact itself.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    return int(np.count_nonzero(np.isfinite(obs) & np.isfinite(pred)))


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

# Metrics that stay meaningful once an angular residual is wrapped: each is a
# reduction of ``pred - obs`` alone. The rest — R², r, d, IOA, NRMSE — normalise
# by the observed mean or variance, and the arithmetic mean of a bearing has no
# meaning (the mean of 350° and 10° is 180°, the exact opposite heading), so they
# are reported as NaN rather than as a plausible-looking wrong number.
CIRCULAR_METRICS = frozenset({"RMSE", "MAE", "MBE"})

# Bearings whose names the comparison CLIs may meet. Dispersion columns
# (``WindDir_SD``, ``Wd_StdDev``, ``WindDir_SD1_WVT``) are excluded first: a
# standard deviation of direction is an ordinary scalar spread, so suppressing
# its R² and r would throw away valid numbers.
_CIRCULAR_EXACT = frozenset({"wd", "winddir", "wind_dir", "wind_direction", "direction"})
# Every prefix keeps its separator ("dir_", not "dir") so that a direct-beam
# radiation column such as ``Direct_Wm2`` is not mistaken for a bearing.
_CIRCULAR_PREFIXES = ("wd_", "winddir", "wind_dir", "wind_direction", "direction_", "dir_")
_LINEAR_DISPERSION_MARKERS = ("sd", "std")


def is_circular_column(name: str) -> bool:
    """Whether *name* denotes a wind direction, i.e. a bearing in degrees.

    Direction is circular: 0° and 360° are the same heading. Scored on the line,
    a model at 10° against an observation at 350° reads as a 340° error instead
    of 20°, which inflates RMSE and can flip the sign of the bias. Detection is
    by name because that is all a generic ``obs``/``model`` CSV carries.
    """
    label = name.strip().casefold()
    if any(marker in label for marker in _LINEAR_DISPERSION_MARKERS):
        return False
    return label in _CIRCULAR_EXACT or label.startswith(_CIRCULAR_PREFIXES)


def _wrap_to_observation(obs: NDArray, pred: NDArray) -> NDArray:
    """Move each prediction to the revolution nearest its observation.

    The residual ``pred - obs`` is wrapped to ``[-180, 180)`` and added back, so
    every metric downstream sees the shortest angular separation while keeping
    the sign convention of :func:`mbe` (positive = model clockwise of observed).
    """
    wrapped: NDArray = obs + ((pred - obs + 180.0) % 360.0 - 180.0)
    return wrapped


def compute_all(
    observed: NDArray, predicted: NDArray, *, circular: bool = False
) -> dict[str, float]:
    """Compute all available metrics and return as a dict.

    Set *circular* for a bearing in degrees. Residuals are then wrapped to the
    shortest angular separation and only :data:`CIRCULAR_METRICS` are computed;
    the others come back NaN. The key set never changes — it is parsed
    downstream — so a suppressed metric is an empty cell, not a missing row.
    """
    obs, pred = _clean_pairs(observed, predicted)
    if len(obs) < 2:
        return {name: float("nan") for name in ALL_METRICS}
    if circular:
        pred = _wrap_to_observation(obs, pred)
        return {
            name: fn(obs, pred, clean=False) if name in CIRCULAR_METRICS else float("nan")
            for name, fn in ALL_METRICS.items()
        }
    return {name: fn(obs, pred, clean=False) for name, fn in ALL_METRICS.items()}
