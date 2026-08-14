"""Regression metrics - reuses micrometeorology and adds MAPE.

Every metric here has the signature ``(observed, predicted) -> float``, in that
order, over two 1-D arrays of equal length. The order is load-bearing for the
signed metrics: MBE reports predicted minus observed, so swapping the arguments
flips the sign of the reported bias.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from micrometeorology.stats.metrics import (
    correlation,
    d_index,
    mae,
    mbe,
    r_squared,
    rmse,
)

MetricFn = Callable[[NDArray, NDArray], float]


def mape(observed: NDArray, predicted: NDArray) -> float:
    """Mean Absolute Percentage Error.

    Observations with value 0 are excluded to avoid division by zero, and
    non-finite pairs are dropped for the same reason
    :func:`micrometeorology.stats.metrics._clean_pairs` drops them: a single
    ``±inf`` survives an ``isnan`` filter and then poisons the whole mean.

    Parameters
    ----------
    observed, predicted:
        Aligned samples of shape ``(N,)``, in the variable's own physical unit.
        Coerced to ``float64`` here.

    Returns
    -------
    float
        Mean absolute percentage error, in percent; ``NaN`` below two valid
        pairs. For irradiance the exclusion of zeros matters physically: night
        rows are genuine zeros, not missing data, and every one of them would
        otherwise be an infinite relative error, so this metric describes
        daylight only.
    """
    o = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = np.isfinite(o) & np.isfinite(p) & (o != 0)
    o, p = o[mask], p[mask]
    if len(o) < 2:
        return float("nan")
    return float(np.mean(np.abs((o - p) / o)) * 100)


REGRESSION_METRICS: dict[str, MetricFn] = {
    "RMSE": rmse,
    "MAE": mae,
    "MBE": mbe,
    "R²": r_squared,
    "r": correlation,
    "d": d_index,
    "MAPE": mape,
}


def compute_regression_metrics(observed: NDArray, predicted: NDArray) -> dict[str, float]:
    """Compute all regression metrics at once.

    RMSE, MAE and MBE are reported together on purpose: MBE alone hides
    dispersion and RMSE alone hides a systematic bias.

    Parameters
    ----------
    observed, predicted:
        Aligned samples of shape ``(N,)``, in the same unit — the original
        units of the target column, once the pipeline's inverse transform has
        been applied.

    Returns
    -------
    dict of str to float
        One score per entry of :data:`REGRESSION_METRICS`, keyed by its display
        name. The error metrics carry the unit of the inputs; ``R²``, ``r`` and
        ``d`` are dimensionless and ``MAPE`` is a percentage.
    """
    return {name: fn(observed, predicted) for name, fn in REGRESSION_METRICS.items()}
