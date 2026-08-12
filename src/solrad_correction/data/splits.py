"""Time-series-aware data splitting with no temporal leakage."""

import logging
from collections.abc import Generator

import numpy as np
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
        Contiguous slices covering the whole frame with no overlap: validation
        begins where training ends and test runs to the last row, in
        chronological order unless ``shuffle`` was set. Row counts come from
        truncated fractions of ``len(df)``, so
        the remainder of the division lands in the test slice. Columns, dtypes
        and units are those of ``df``.

    Raises
    ------
    ValueError
        If the three ratios do not sum to 1.0 within 1e-6.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    n = len(df)
    if shuffle:
        logger.warning("Shuffling time-series data may cause data leakage.")
        df = df.sample(frac=1.0)
    else:
        df = df.sort_index()

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logger.info(
        "Split: train=%d (%s to %s), val=%d, test=%d",
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

    Training always starts at row 0 and ends where validation begins, so no fold
    ever trains on a row that follows its own validation rows.

    Parameters
    ----------
    initial_train_size:
        Rows in the first fold's training window.
    val_size:
        Rows in every validation window.
    step:
        Rows the window advances between folds; defaults to ``val_size``, which
        makes consecutive validation windows adjacent and non-overlapping.
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
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield one ``(train_idx, val_idx)`` pair per expanding-window fold.

        Parameters
        ----------
        df:
            Frame in chronological row order; only its length is read.

        Yields
        ------
        tuple of (train_idx, val_idx)
            Positional row indices into ``df``, ``int64``, both strictly
            increasing: ``train_idx`` has shape ``(start,)`` and grows by
            ``step`` each fold, ``val_idx`` shape ``(val_size,)``. Iteration
            stops once a full validation window no longer fits, so a trailing
            partial window is never yielded.
        """
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

    The frame is cut into ``n_splits + 1`` equal blocks of ``len(df) //
    (n_splits + 1)`` rows; fold ``i`` trains on blocks ``0..i`` and validates on
    block ``i + 1``, so every fold has a validation block and none of them
    overlap.

    Parameters
    ----------
    n_splits:
        Number of folds.
    """

    def __init__(self, n_splits: int = 5) -> None:
        self.n_splits = n_splits

    def split(
        self,
        df: pd.DataFrame,
    ) -> Generator[tuple[np.ndarray, np.ndarray]]:
        """Yield one ``(train_idx, val_idx)`` pair per fold, oldest fold first.

        Parameters
        ----------
        df:
            Frame in chronological row order; only its length is read.

        Yields
        ------
        tuple of (train_idx, val_idx)
            Positional row indices into ``df``, ``int64``. Fold ``i`` yields
            ``train_idx`` of shape ``(fold_size * (i + 1),)`` starting at row 0
            and ``val_idx`` of shape ``(fold_size,)`` immediately after it. Rows
            past the last block boundary are left out of every fold, and the
            arrays are empty when ``df`` is shorter than ``n_splits + 1`` rows.
        """
        n = len(df)
        fold_size = n // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_end = min(train_end + fold_size, n)
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(train_end, val_end)
            yield train_idx, val_idx
