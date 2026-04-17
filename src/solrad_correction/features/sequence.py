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

    # sliding_window_view creates a zero-copy strided view — much faster than a Python loop
    x_windows = np.lib.stride_tricks.sliding_window_view(x, sequence_length, axis=0)
    # x_windows shape: (n, n_features, sequence_length) — need to transpose to (n, seq_len, n_features)
    x_out = np.ascontiguousarray(x_windows[:n].transpose(0, 2, 1), dtype=np.float32)
    y_out = y[sequence_length:].astype(np.float32)

    return x_out, y_out


def create_sequences_index(
    index: pd.DatetimeIndex,
    sequence_length: int,
) -> pd.DatetimeIndex:
    """Get the DatetimeIndex corresponding to sequence targets.

    Useful for mapping predictions back to timestamps.
    """
    return index[sequence_length:]
