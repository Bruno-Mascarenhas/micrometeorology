"""Time-series-aware data splitting — no temporal leakage."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Generator

    import pandas as pd

logger = logging.getLogger(__name__)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame chronologically into train / validation / test.

    Parameters
    ----------
    df:
        DataFrame with sorted DatetimeIndex.
    train_ratio, val_ratio, test_ratio:
        Proportions (must sum to 1.0).
    shuffle:
        If True, shuffles before splitting (NOT recommended for time series).

    Returns
    -------
    tuple of (train_df, val_df, test_df)
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    n = len(df)
    if shuffle:
        logger.warning("⚠ Shuffling time-series data — this may cause data leakage!")
        df = df.sample(frac=1.0)
    else:
        df = df.sort_index()

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Split: train=%d (%s→%s), val=%d, test=%d",
        len(train),
        train.index[0] if len(train) > 0 else "N/A",
        train.index[-1] if len(train) > 0 else "N/A",
        len(val),
        len(test),
    )
    return train, val, test


class ExpandingWindowSplit:
    """Walk-forward expanding window cross-validation.

    At each step the training window grows and validation is the
    next ``val_size`` rows.
    """

    def __init__(
        self,
        initial_train_size: int,
        val_size: int,
        step: int | None = None,
    ) -> None:
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.step = step or val_size

    def split(
        self,
        df: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield ``(train_idx, val_idx)`` tuples."""
        n = len(df)
        start = self.initial_train_size

        while start + self.val_size <= n:
            train_idx = np.arange(0, start)
            val_idx = np.arange(start, min(start + self.val_size, n))
            yield train_idx, val_idx
            start += self.step


class TimeSeriesKFold:
    """K-fold cross-validation respecting temporal order.

    Unlike ``sklearn.model_selection.TimeSeriesSplit``, this provides
    non-overlapping folds where training always precedes validation.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(
        self,
        df: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield ``(train_idx, val_idx)`` tuples."""
        n = len(df)
        fold_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + fold_size, n)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx
