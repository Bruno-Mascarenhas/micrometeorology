"""Sequence construction for recurrent / transformer models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for sequential models.

    Parameters
    ----------
    features:
        2-D array of shape ``(n_samples, n_features)``.
    target:
        1-D array of shape ``(n_samples,)``.
    sequence_length:
        Number of time steps per sequence window.

    Returns
    -------
    tuple of (x_sequences, y_targets)
        ``x_sequences``: shape ``(n_sequences, sequence_length, n_features)``
        ``y_targets``: shape ``(n_sequences,)`` — the target at the *end* of each window.
    """
    x = np.asarray(features)
    y = np.asarray(target).flatten()

    if len(x) != len(y):
        raise ValueError(f"features ({len(x)}) and target ({len(y)}) must have same length")
    if sequence_length >= len(x):
        raise ValueError(f"sequence_length ({sequence_length}) >= data length ({len(x)})")

    n = len(x) - sequence_length
    x_out = np.empty((n, sequence_length, x.shape[1]), dtype=np.float32)
    y_out = np.empty(n, dtype=np.float32)

    for i in range(n):
        x_out[i] = x[i : i + sequence_length]
        y_out[i] = y[i + sequence_length]

    return x_out, y_out


def create_sequences_index(
    index: pd.DatetimeIndex,
    sequence_length: int,
) -> pd.DatetimeIndex:
    """Get the DatetimeIndex corresponding to sequence targets.

    Useful for mapping predictions back to timestamps.
    """
    return index[sequence_length:]
