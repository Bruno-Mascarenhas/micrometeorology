"""Tests for sequence dataset construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.features.sequence import create_sequences, create_sequences_index


class TestCreateSequences:
    def test_shapes(self):
        features = np.random.randn(100, 5)
        target = np.random.randn(100)
        x_seq, y_seq = create_sequences(features, target, sequence_length=10)
        assert x_seq.shape == (90, 10, 5)
        assert y_seq.shape == (90,)

    def test_target_alignment(self):
        """The target for each sequence should be the value AFTER the window."""
        features = np.arange(50).reshape(50, 1).astype(float)
        target = np.arange(50).astype(float) * 10
        _x_seq, y_seq = create_sequences(features, target, sequence_length=5)
        # First sequence: features[0:5], target should be target[5]
        assert y_seq[0] == pytest.approx(50.0)
        # Last sequence: features[44:49], target should be target[49]
        assert y_seq[-1] == pytest.approx(490.0)

    def test_sequence_content(self):
        features = np.arange(20).reshape(20, 1).astype(float)
        target = np.ones(20)
        x_seq, _y_seq = create_sequences(features, target, sequence_length=3)
        # First window should be [0, 1, 2]
        np.testing.assert_array_equal(x_seq[0, :, 0], [0, 1, 2])
        # Second window should be [1, 2, 3]
        np.testing.assert_array_equal(x_seq[1, :, 0], [1, 2, 3])

    def test_too_short_raises(self):
        features = np.random.randn(5, 3)
        target = np.random.randn(5)
        with pytest.raises(ValueError, match="sequence_length"):
            create_sequences(features, target, sequence_length=10)

    def test_length_mismatch_raises(self):
        features = np.random.randn(10, 3)
        target = np.random.randn(8)
        with pytest.raises(ValueError, match="same length"):
            create_sequences(features, target, sequence_length=3)

    def test_sequence_target_index_starts_after_window(self):
        index = pd.date_range("2024-01-01", periods=10, freq="1h")

        seq_index = create_sequences_index(index, sequence_length=3)

        assert seq_index.equals(index[3:])

    def test_tabular_dataset_preserves_full_index(self):
        index = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame({"feature": np.arange(10), "target": np.arange(10)}, index=index)

        dataset = TabularDataset.from_dataframe(df, ["feature"], "target")

        assert dataset.index is not None
        assert dataset.index.equals(index)
