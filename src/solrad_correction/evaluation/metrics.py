"""Regression metrics — reuses micrometeorology and adds MAPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Re-export from the micrometeorology package
from micrometeorology.stats.metrics import (
    correlation,
    d_index,
    mae,
    mbe,
    r_squared,
    rmse,
)


def mape(observed: NDArray, predicted: NDArray) -> float:
    """Mean Absolute Percentage Error.

    Observations with value 0 are excluded to avoid division by zero.
    """
    o = np.asarray(observed, dtype=float)
    p = np.asarray(predicted, dtype=float)
    mask = ~(np.isnan(o) | np.isnan(p)) & (o != 0)
    o, p = o[mask], p[mask]
    if len(o) < 2:
        return float("nan")
    return float(np.mean(np.abs((o - p) / o)) * 100)


REGRESSION_METRICS = {
    "RMSE": rmse,
    "MAE": mae,
    "MBE": mbe,
    "R²": r_squared,
    "r": correlation,
    "d": d_index,
    "MAPE": mape,
}


def compute_regression_metrics(observed: NDArray, predicted: NDArray) -> dict[str, float]:
    """Compute all regression metrics at once."""
    return {name: fn(observed, predicted) for name, fn in REGRESSION_METRICS.items()}
