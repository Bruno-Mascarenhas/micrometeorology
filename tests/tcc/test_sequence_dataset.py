"""Tests for sequence dataset construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solrad_correction.datasets.sequence import (
    WindowedSequenceDataset,
    WindowedSequenceDatasetMeta,
)
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


class TestWindowedSequenceDataset:
    def test_matches_dense_create_sequences_output(self):
        rng = np.random.default_rng(42)
        features = rng.normal(size=(30, 4)).astype(np.float32)
        target = rng.normal(size=30).astype(np.float32)

        dense_x, dense_y = create_sequences(features, target, sequence_length=6)
        lazy = WindowedSequenceDataset(features, target, sequence_length=6)

        assert len(lazy) == len(dense_x)
        for idx in [0, 3, len(lazy) - 1]:
            x_item, y_item = lazy[idx]
            np.testing.assert_array_equal(x_item.numpy(), dense_x[idx])
            assert y_item.item() == pytest.approx(float(dense_y[idx]))

    def test_window_content_and_target_alignment(self):
        features = np.arange(20, dtype=np.float32).reshape(10, 2)
        target = (np.arange(10, dtype=np.float32) * 10).astype(np.float32)

        dataset = WindowedSequenceDataset(features, target, sequence_length=3)
        x0, y0 = dataset[0]
        x1, y1 = dataset[1]

        np.testing.assert_array_equal(x0.numpy(), features[0:3])
        assert y0.item() == pytest.approx(30.0)
        np.testing.assert_array_equal(x1.numpy(), features[1:4])
        assert y1.item() == pytest.approx(40.0)

    def test_custom_target_offset(self):
        features = np.arange(20, dtype=np.float32).reshape(10, 2)
        target = np.arange(10, dtype=np.float32)

        dataset = WindowedSequenceDataset(features, target, sequence_length=3, target_offset=4)

        _x0, y0 = dataset[0]
        assert len(dataset) == 6
        assert y0.item() == pytest.approx(4.0)

    def test_short_input_raises(self):
        features = np.random.randn(5, 3).astype(np.float32)
        target = np.random.randn(5).astype(np.float32)

        with pytest.raises(ValueError, match="sequence_length"):
            WindowedSequenceDataset(features, target, sequence_length=5)

    def test_dataloader_batch_shape(self):
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader

        features = np.arange(60, dtype=np.float32).reshape(20, 3)
        target = np.arange(20, dtype=np.float32)
        dataset = WindowedSequenceDataset(features, target, sequence_length=4)

        x_batch, y_batch = next(iter(DataLoader(dataset, batch_size=5)))

        assert tuple(x_batch.shape) == (5, 4, 3)
        assert tuple(y_batch.shape) == (5,)
        assert x_batch.dtype == torch.float32
        assert y_batch.dtype == torch.float32

    def test_stores_base_matrix_without_dense_window_materialization(self):
        features = np.arange(200, dtype=np.float32).reshape(100, 2)
        target = np.arange(100, dtype=np.float32)
        dataset = WindowedSequenceDataset(features, target, sequence_length=12)

        assert dataset.X.shape == features.shape
        assert dataset.X.ndim == 2
        assert np.shares_memory(dataset.X, features)

    def test_supports_torch_tensor_inputs(self):
        torch = pytest.importorskip("torch")

        features = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        target = torch.arange(10, dtype=torch.float32)
        dataset = WindowedSequenceDataset(features, target, sequence_length=3)

        x0, y0 = dataset[0]

        assert isinstance(x0, torch.Tensor)
        assert tuple(x0.shape) == (3, 4)
        assert y0.item() == pytest.approx(3.0)

    def test_supports_memmap_inputs_without_window_materialization(self):
        scratch = Path("scratch")
        scratch.mkdir(exist_ok=True)
        feature_path = scratch / "windowed_sequence_features.memmap"
        target_path = scratch / "windowed_sequence_target.memmap"
        features = None
        target = None
        dataset = None
        x0 = None
        try:
            features = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(20, 3))
            target = np.memmap(target_path, dtype=np.float32, mode="w+", shape=(20,))
            features[:] = np.arange(60, dtype=np.float32).reshape(20, 3)
            target[:] = np.arange(20, dtype=np.float32)
            features.flush()
            target.flush()

            dataset = WindowedSequenceDataset(features, target, sequence_length=5)
            x0, y0 = dataset[0]

            assert dataset.X.shape == (20, 3)
            np.testing.assert_array_equal(x0.numpy(), np.asarray(features[0:5]))
            assert y0.item() == pytest.approx(5.0)
        finally:
            del x0
            del dataset
            del features
            del target
            feature_path.unlink(missing_ok=True)
            target_path.unlink(missing_ok=True)

    def test_prediction_index_matches_lazy_targets(self):
        index = pd.date_range("2024-01-01", periods=12, freq="1h")
        features = np.arange(24, dtype=np.float32).reshape(12, 2)
        target = np.arange(12, dtype=np.float32)
        dataset = WindowedSequenceDataset(features, target, sequence_length=4)

        seq_index = create_sequences_index(index, sequence_length=4)

        assert len(dataset) == len(seq_index)
        assert seq_index[0] == index[4]

    def test_saves_lazy_cache_without_dense_sequences_file(self):
        scratch = Path("scratch") / "windowed_sequence_cache_test"
        try:
            features = np.arange(40, dtype=np.float32).reshape(10, 4)
            target = np.arange(10, dtype=np.float32)
            dataset = WindowedSequenceDataset(features, target, sequence_length=3)

            dataset.save(scratch, feature_names=["a", "b", "c", "d"])

            assert (scratch / "windowed_sequences.npz").exists()
            assert not (scratch / "sequences.npz").exists()

            loaded_meta = WindowedSequenceDatasetMeta.load(scratch)
            loaded = loaded_meta.to_torch_dataset()
            x0, y0 = loaded[0]

            np.testing.assert_array_equal(x0.numpy(), features[:3])
            assert y0.item() == pytest.approx(3.0)
            assert loaded_meta.feature_names == ["a", "b", "c", "d"]
        finally:
            if scratch.exists():
                for child in scratch.iterdir():
                    child.unlink()
                scratch.rmdir()

