"""Training, registry, and checkpoint contracts."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from solrad_correction.config import ModelConfig, RuntimeConfig
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.models.registry import MODEL_REGISTRY, build_model, get_model_spec
from solrad_correction.models.svm import SVMRegressor
from solrad_correction.training.dataloaders import resolve_dataloader_settings, resolve_device


@pytest.fixture
def synthetic_tabular() -> TabularDataset:
    rng = np.random.default_rng(42)
    features = rng.normal(0, 1, (120, 5)).astype(np.float32)
    targets = (features[:, 0] * 2 + features[:, 1] + rng.normal(0, 0.1, 120)).astype(np.float32)
    return TabularDataset(X=features, y=targets, feature_names=[f"f{i}" for i in range(5)])


def test_registry_contract_for_supported_models_only() -> None:
    assert set(MODEL_REGISTRY) == {"svm", "lstm", "transformer"}
    assert get_model_spec("svm").kind == "tabular"
    assert get_model_spec("lstm").kind == "sequence"

    svm = build_model(ModelConfig(model_type="svm"))
    lstm = build_model(ModelConfig(model_type="lstm"), input_size=3, device="cpu")
    transformer = build_model(
        ModelConfig(model_type="transformer", tf_d_model=8, tf_nhead=2),
        input_size=3,
        device="cpu",
    )

    assert "SVM" in svm.name
    assert lstm.name == "LSTM"
    assert transformer.name == "Transformer"
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model_spec("hgb")


def test_runtime_dataloader_resolution_and_cuda_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = resolve_dataloader_settings(
        RuntimeConfig(device="cpu", num_workers=0, pin_memory=False, amp=False),
    )

    assert settings.device == "cpu"
    assert settings.num_workers == 0
    assert settings.pin_memory is False
    assert settings.prefetch_factor is None
    assert settings.amp is False

    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")


def test_svm_fit_predict_evaluate_and_save_load(synthetic_tabular: TabularDataset) -> None:
    path = Path("scratch") / "svm_contract.joblib"
    try:
        model = SVMRegressor(kernel="rbf", C=10.0)
        model.fit(synthetic_tabular)
        preds_before = model.predict(synthetic_tabular)
        metrics = model.evaluate(synthetic_tabular)
        model.save(path)
        loaded = SVMRegressor.load(path)

        assert preds_before.shape == (120,)
        assert metrics["RMSE"] < 1.0
        np.testing.assert_allclose(loaded.predict(synthetic_tabular), preds_before)
    finally:
        path.unlink(missing_ok=True)


def test_lstm_runtime_checkpoints_and_resume() -> None:
    pytest.importorskip("torch")
    from solrad_correction.datasets.sequence import SequenceDataset
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import load_torch_checkpoint

    checkpoint_dir = Path("scratch") / "lstm_runtime_contract"
    try:
        rng = np.random.default_rng(123)
        features = rng.normal(0, 1, (80, 3, 4)).astype(np.float32)
        targets = rng.normal(0, 1, 80).astype(np.float32)
        train_ds = SequenceDataset(features[:60], targets[:60])
        val_ds = SequenceDataset(features[60:], targets[60:])

        model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
        config = ModelConfig(model_type="lstm", max_epochs=1, batch_size=16, patience=3)
        runtime = RuntimeConfig(
            device="cpu", num_workers=0, amp=False, checkpoint_dir=str(checkpoint_dir)
        )
        model.fit(train_ds, val_ds, config, runtime=runtime)

        last_path = checkpoint_dir / "last.pt"
        assert (checkpoint_dir / "best.pt").exists()
        assert last_path.exists()
        assert load_torch_checkpoint(last_path)["epoch"] == 1

        resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
        resume_runtime = RuntimeConfig(
            device="cpu",
            num_workers=0,
            amp=False,
            checkpoint_dir=str(checkpoint_dir),
            resume=str(last_path),
        )
        resumed.fit(train_ds, val_ds, config, runtime=resume_runtime)

        assert load_torch_checkpoint(last_path)["epoch"] == 2
    finally:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
