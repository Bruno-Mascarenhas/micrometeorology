"""Smoke tests for training pipelines — SVM and LSTM."""

from __future__ import annotations

import numpy as np
import pytest

from solrad_correction.datasets.tabular import TabularDataset


@pytest.fixture
def synthetic_tabular():
    """Synthetic tabular data for quick training."""
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (200, 5)).astype(np.float32)
    targets = (features[:, 0] * 2 + features[:, 1] + rng.normal(0, 0.1, 200)).astype(np.float32)
    return TabularDataset(X=features, y=targets, feature_names=[f"f{i}" for i in range(5)])


class TestSVMSmoke:
    def test_full_pipeline(self, synthetic_tabular):
        from solrad_correction.models.svm import SVMRegressor

        model = SVMRegressor(kernel="rbf", C=10.0)
        model.fit(synthetic_tabular)
        preds = model.predict(synthetic_tabular)
        metrics = model.evaluate(synthetic_tabular)

        assert len(preds) == 200
        assert metrics["RMSE"] < 1.0  # Should be a reasonable fit


class TestLSTMSmoke:
    def test_full_pipeline(self):
        pytest.importorskip("torch")
        from solrad_correction.datasets.sequence import SequenceDataset
        from solrad_correction.models.lstm import LSTMRegressor

        # Synthetic sequence data
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (200, 3, 5)).astype(np.float32)
        targets = rng.normal(0, 1, 200).astype(np.float32)

        train_ds = SequenceDataset(features[:150], targets[:150])
        val_ds = SequenceDataset(features[150:], targets[150:])

        model = LSTMRegressor(input_size=5, hidden_size=16, num_layers=1, device="cpu")

        from solrad_correction.config import ModelConfig

        config = ModelConfig(
            model_type="lstm",
            lstm_hidden_size=16,
            lstm_num_layers=1,
            max_epochs=2,
            batch_size=32,
            patience=5,
        )

        model.fit(train_ds, val_ds, config)
        preds = model.predict(val_ds)
        assert preds.shape == (50,)


class TestTransformerSmoke:
    def test_full_pipeline(self):
        pytest.importorskip("torch")
        from solrad_correction.datasets.sequence import SequenceDataset
        from solrad_correction.models.transformer import TransformerRegressor

        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (200, 8, 4)).astype(np.float32)
        targets = rng.normal(0, 1, 200).astype(np.float32)

        train_ds = SequenceDataset(features[:150], targets[:150])
        val_ds = SequenceDataset(features[150:], targets[150:])

        from solrad_correction.config import ModelConfig

        config = ModelConfig(
            model_type="transformer",
            tf_d_model=16,
            tf_nhead=2,
            tf_num_encoder_layers=1,
            tf_dim_feedforward=32,
            max_epochs=2,
            batch_size=32,
            patience=5,
        )

        model = TransformerRegressor(
            input_size=4,
            d_model=16,
            nhead=2,
            num_encoder_layers=1,
            dim_feedforward=32,
            device="cpu",
        )
        model.fit(train_ds, val_ds, config)
        preds = model.predict(val_ds)
        assert preds.shape == (50,)
