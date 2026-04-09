"""Tests for temporal splits — no leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solrad_correction.data.splits import ExpandingWindowSplit, temporal_train_val_test_split


@pytest.fixture
def time_series_df() -> pd.DataFrame:
    """Create a simple time-series DataFrame."""
    idx = pd.date_range("2020-01-01", periods=100, freq="1h")
    return pd.DataFrame({"value": np.arange(100, dtype=float)}, index=idx)


class TestTemporalSplit:
    def test_sizes(self, time_series_df):
        train, val, test = temporal_train_val_test_split(time_series_df, 0.7, 0.15, 0.15)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap(self, time_series_df):
        train, val, test = temporal_train_val_test_split(time_series_df, 0.7, 0.15, 0.15)
        assert train.index[-1] < val.index[0]
        assert val.index[-1] < test.index[0]

    def test_chronological_order(self, time_series_df):
        train, val, test = temporal_train_val_test_split(time_series_df, 0.7, 0.15, 0.15)
        # Values should be monotonically increasing (no shuffling)
        assert train["value"].iloc[0] < val["value"].iloc[0]
        assert val["value"].iloc[0] < test["value"].iloc[0]

    def test_no_future_leakage(self, time_series_df):
        """Train max timestamp must be < val min timestamp."""
        train, val, test = temporal_train_val_test_split(time_series_df, 0.7, 0.15, 0.15)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_invalid_ratios(self, time_series_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            temporal_train_val_test_split(time_series_df, 0.5, 0.5, 0.5)


class TestExpandingWindow:
    def test_yields_splits(self, time_series_df):
        splitter = ExpandingWindowSplit(initial_train_size=50, val_size=10, step=10)
        splits = list(splitter.split(time_series_df))
        assert len(splits) >= 1
        # First split: train has 50, val has 10
        train_idx, val_idx = splits[0]
        assert len(train_idx) == 50
        assert len(val_idx) == 10
        # No overlap
        assert max(train_idx) < min(val_idx)
