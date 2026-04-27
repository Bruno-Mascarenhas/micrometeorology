"""Smoke tests for training pipelines — SVM and LSTM."""

from __future__ import annotations

from pathlib import Path

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
        from solrad_correction.utils.serialization import load_torch_checkpoint

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

        checkpoint_path = Path("scratch") / "lstm_smoke_checkpoint.pt"
        try:
            model.save(checkpoint_path)
            checkpoint = load_torch_checkpoint(checkpoint_path)
            assert checkpoint["epoch"] == 2
            assert "optimizer_state_dict" in checkpoint
            assert "scheduler_state_dict" in checkpoint

            loaded = LSTMRegressor.load(checkpoint_path)
            loaded_preds = loaded.predict(val_ds)
            assert loaded_preds.shape == (50,)

            from solrad_correction.config import RuntimeConfig

            resume_config = ModelConfig(
                model_type="lstm",
                lstm_hidden_size=16,
                lstm_num_layers=1,
                max_epochs=1,
                batch_size=32,
                patience=5,
            )
            resume_runtime = RuntimeConfig(device="cpu", num_workers=0, resume=str(checkpoint_path))
            resumed = LSTMRegressor(input_size=5, hidden_size=16, num_layers=1, device="cpu")
            resumed.fit(train_ds, val_ds, resume_config, runtime=resume_runtime)

            resumed_path = Path("scratch") / "lstm_resumed_checkpoint.pt"
            try:
                resumed.save(resumed_path)
                resumed_checkpoint = load_torch_checkpoint(resumed_path)
                assert resumed_checkpoint["epoch"] == 3
                assert "optimizer_state_dict" in resumed_checkpoint
                assert "scheduler_state_dict" in resumed_checkpoint
            finally:
                resumed_path.unlink(missing_ok=True)
        finally:
            checkpoint_path.unlink(missing_ok=True)

    def test_runtime_checkpoints_and_resume(self):
        pytest.importorskip("torch")
        from solrad_correction.config import ModelConfig, RuntimeConfig
        from solrad_correction.datasets.sequence import SequenceDataset
        from solrad_correction.models.lstm import LSTMRegressor
        from solrad_correction.utils.serialization import load_torch_checkpoint

        rng = np.random.default_rng(123)
        features = rng.normal(0, 1, (80, 3, 4)).astype(np.float32)
        targets = rng.normal(0, 1, 80).astype(np.float32)
        train_ds = SequenceDataset(features[:60], targets[:60])
        val_ds = SequenceDataset(features[60:], targets[60:])

        checkpoint_dir = Path("scratch") / "lstm_runtime_checkpoints"
        try:
            model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
            config = ModelConfig(model_type="lstm", max_epochs=1, batch_size=16, patience=3)
            runtime = RuntimeConfig(
                device="cpu",
                num_workers=0,
                amp=False,
                checkpoint_dir=str(checkpoint_dir),
            )

            model.fit(train_ds, val_ds, config, runtime=runtime)

            best_path = checkpoint_dir / "best.pt"
            last_path = checkpoint_dir / "last.pt"
            assert best_path.exists()
            assert last_path.exists()
            last_checkpoint = load_torch_checkpoint(last_path)
            assert last_checkpoint["epoch"] == 1
            assert last_checkpoint["metadata"]["checkpoint_kind"] == "last"

            resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
            resume_config = ModelConfig(model_type="lstm", max_epochs=1, batch_size=16, patience=3)
            resume_runtime = RuntimeConfig(
                device="cpu",
                num_workers=0,
                amp=False,
                checkpoint_dir=str(checkpoint_dir),
                resume=str(last_path),
            )
            resumed.fit(train_ds, val_ds, resume_config, runtime=resume_runtime)
            resumed_checkpoint = load_torch_checkpoint(last_path)
            assert resumed_checkpoint["epoch"] == 2
        finally:
            if checkpoint_dir.exists():
                for child in checkpoint_dir.iterdir():
                    child.unlink()
                checkpoint_dir.rmdir()


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
