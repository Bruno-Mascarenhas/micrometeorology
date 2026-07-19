"""Training, registry, and checkpoint contracts."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from solrad_correction.config import ModelConfig, RuntimeConfig
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.models.registry import MODEL_REGISTRY, build_model, get_model_spec
from solrad_correction.models.svm import SVMRegressor
from solrad_correction.training.dataloaders import resolve_dataloader_settings, resolve_device
from solrad_correction.training.progress import TrainingProgress
from solrad_correction.utils.memory import assert_array_size


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


def test_solrad_memory_guard_fails_before_large_array_materialization() -> None:
    with pytest.raises(MemoryError, match="dense test array"):
        assert_array_size(
            (1024, 1024),
            np.float32,
            context="dense test array",
            max_gb=0.001,
            multiplier=2.0,
        )


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


def _synthetic_sequence_split(seed: int = 123) -> tuple:
    from solrad_correction.datasets.sequence import SequenceDataset

    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, (80, 3, 4)).astype(np.float32)
    targets = rng.normal(0, 1, 80).astype(np.float32)
    return (
        SequenceDataset(features[:60], targets[:60]),
        SequenceDataset(features[60:], targets[60:]),
    )


def test_lstm_runtime_checkpoints_and_resume_trains_only_remaining_epochs() -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import load_torch_checkpoint

    checkpoint_dir = Path("scratch") / "lstm_runtime_contract"
    try:
        train_ds, val_ds = _synthetic_sequence_split()

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

        # max_epochs is the TOTAL epoch budget: resuming from epoch 1 with
        # max_epochs=2 trains exactly one additional epoch, not two more.
        resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
        resume_config = ModelConfig(model_type="lstm", max_epochs=2, batch_size=16, patience=3)
        resume_runtime = RuntimeConfig(
            device="cpu",
            num_workers=0,
            amp=False,
            checkpoint_dir=str(checkpoint_dir),
            resume=str(last_path),
        )
        resumed.fit(train_ds, val_ds, resume_config, runtime=resume_runtime)

        assert load_torch_checkpoint(last_path)["epoch"] == 2
        assert len(resumed.training_history["train_loss"]) == 1
    finally:
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)


def test_resume_at_max_epochs_exits_cleanly_without_training(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import load_torch_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    train_ds, val_ds = _synthetic_sequence_split()
    config = ModelConfig(model_type="lstm", max_epochs=1, batch_size=16, patience=3)

    model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    model.fit(
        train_ds,
        val_ds,
        config,
        runtime=RuntimeConfig(
            device="cpu", num_workers=0, amp=False, checkpoint_dir=str(checkpoint_dir)
        ),
    )
    last_path = checkpoint_dir / "last.pt"

    resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    with caplog.at_level(logging.WARNING, logger="solrad_correction.training.trainer"):
        result = resumed.fit(
            train_ds,
            val_ds,
            config,
            runtime=RuntimeConfig(
                device="cpu",
                num_workers=0,
                amp=False,
                checkpoint_dir=str(checkpoint_dir),
                resume=str(last_path),
            ),
        )

    assert "nothing left to train" in caplog.text
    assert result.history["train_loss"] == []
    assert load_torch_checkpoint(last_path)["epoch"] == 1


def test_compiled_training_persists_plain_state_dict_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkpoints/saves from a torch.compile run must be loadable uncompiled."""
    torch = pytest.importorskip("torch")
    from torch import nn

    from solrad_correction.models.lstm import LSTMRegressor

    class FakeCompiledModule(nn.Module):
        """Mimics torch._dynamo.OptimizedModule key prefixing."""

        def __init__(self, orig: nn.Module) -> None:
            super().__init__()
            self._orig_mod = orig

        def forward(self, *args: object, **kwargs: object) -> object:
            return self._orig_mod(*args, **kwargs)

    monkeypatch.setattr(torch, "compile", lambda module, **_kwargs: FakeCompiledModule(module))

    train_ds, val_ds = _synthetic_sequence_split(seed=7)
    checkpoint_dir = tmp_path / "checkpoints"
    model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    model.fit(
        train_ds,
        val_ds,
        ModelConfig(model_type="lstm", max_epochs=1, batch_size=16, patience=3),
        runtime=RuntimeConfig(
            device="cpu",
            num_workers=0,
            amp=False,
            torch_compile=True,
            checkpoint_dir=str(checkpoint_dir),
        ),
    )

    for name in ("best.pt", "last.pt"):
        raw = torch.load(checkpoint_dir / name, map_location="cpu", weights_only=True)
        assert not any(key.startswith("_orig_mod.") for key in raw["model_state_dict"]), name

    model_path = tmp_path / "model.pt"
    model.save(model_path)
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    assert not any(key.startswith("_orig_mod.") for key in raw["model_state_dict"])
    reloaded = LSTMRegressor.load(model_path)
    assert reloaded.predict(val_ds).shape == (len(val_ds),)


def test_compiled_prefixed_checkpoint_remains_loadable(tmp_path: Path) -> None:
    """Existing checkpoints with `_orig_mod.` keys keep working on load."""
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import save_torch_checkpoint

    source = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    prefixed_state = {f"_orig_mod.{k}": v for k, v in source._module.state_dict().items()}
    path = tmp_path / "last.pt"
    save_torch_checkpoint(
        model_state=prefixed_state, optimizer_state=None, config=None, epoch=3, path=path
    )

    target = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    target._load_resume_checkpoint(str(path))

    assert target._start_epoch == 3
    np.testing.assert_allclose(
        target._module.state_dict()["lstm.weight_ih_l0"].numpy(),
        source._module.state_dict()["lstm.weight_ih_l0"].numpy(),
    )


def test_resume_seeds_best_metric_from_checkpoint_metadata(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import load_torch_checkpoint, save_torch_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    train_ds, val_ds = _synthetic_sequence_split(seed=21)
    model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    model.fit(
        train_ds,
        val_ds,
        ModelConfig(model_type="lstm", max_epochs=2, batch_size=16, patience=5),
        runtime=RuntimeConfig(
            device="cpu", num_workers=0, amp=False, checkpoint_dir=str(checkpoint_dir)
        ),
    )

    best_meta = load_torch_checkpoint(checkpoint_dir / "best.pt")["metadata"]
    last_meta = load_torch_checkpoint(checkpoint_dir / "last.pt")["metadata"]
    assert last_meta["best_metric"] == pytest.approx(best_meta["monitor_metric"])

    resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    resumed._load_resume_checkpoint(str(checkpoint_dir / "last.pt"))
    assert resumed._best_metric == pytest.approx(best_meta["monitor_metric"])

    # Legacy checkpoints without best metadata fall back to sibling best.pt.
    legacy_path = checkpoint_dir / "legacy_last.pt"
    save_torch_checkpoint(
        model_state=model._module.state_dict(),
        optimizer_state=None,
        config=None,
        epoch=2,
        path=legacy_path,
    )
    legacy = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    legacy._load_resume_checkpoint(str(legacy_path))
    assert legacy._best_metric == pytest.approx(best_meta["monitor_metric"])


def test_resumed_best_state_prevents_best_clobber_and_seeds_early_stopping(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMNet
    from solrad_correction.training.trainer import Trainer

    train_ds, val_ds = _synthetic_sequence_split(seed=42)
    checkpoint_dir = tmp_path / "checkpoints"
    trainer = Trainer(
        model=LSTMNet(input_size=4, hidden_size=8, num_layers=1),
        device="cpu",
        config=ModelConfig(model_type="lstm", max_epochs=5, batch_size=16, patience=1),
        runtime=RuntimeConfig(
            device="cpu", num_workers=0, amp=False, checkpoint_dir=str(checkpoint_dir)
        ),
        start_epoch=1,
        best_metric=0.0,  # unbeatable best from the interrupted run
        best_epoch=1,
    )
    _, history = trainer.train(train_ds, val_ds)

    # A worse resumed epoch must never overwrite the previous best.pt ...
    assert not (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "last.pt").exists()
    assert trainer.best_metric == 0.0
    # ... and early stopping measures patience against the resumed best.
    assert len(history["train_loss"]) == 1


def test_early_stopping_counter_restored_on_resume() -> None:
    """Regression: the no-improvement counter must survive a resume.

    With an unbeatable seeded best metric every epoch counts as
    "no improvement", so the number of epochs trained before early stopping is
    exactly ``patience - epochs_no_improve``.
    """
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMNet
    from solrad_correction.training.trainer import Trainer

    def run(epochs_no_improve: int) -> tuple[int, int]:
        train_ds, val_ds = _synthetic_sequence_split(seed=42)
        trainer = Trainer(
            model=LSTMNet(input_size=4, hidden_size=8, num_layers=1),
            device="cpu",
            config=ModelConfig(model_type="lstm", max_epochs=10, batch_size=16, patience=3),
            start_epoch=2,
            best_metric=-1e9,  # unbeatable → every epoch is "no improvement"
            best_epoch=1,
            epochs_no_improve=epochs_no_improve,
        )
        _, history = trainer.train(train_ds, val_ds)
        return len(history["train_loss"]), trainer.epochs_no_improve

    # Restored counter=2 with patience=3 stops after exactly 1 more epoch.
    trained, counter = run(2)
    assert trained == 1
    assert counter == 3
    # A lost counter (reset to 0) would instead need the full patience.
    assert run(0)[0] == 3


def test_resume_persists_and_restores_early_stopping_counter(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.utils.serialization import save_torch_checkpoint

    checkpoint_dir = tmp_path / "checkpoints"
    train_ds, val_ds = _synthetic_sequence_split(seed=21)
    model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    model.fit(
        train_ds,
        val_ds,
        ModelConfig(model_type="lstm", max_epochs=3, batch_size=16, patience=5),
        runtime=RuntimeConfig(
            device="cpu", num_workers=0, amp=False, checkpoint_dir=str(checkpoint_dir)
        ),
    )

    from solrad_correction.utils.serialization import load_torch_checkpoint

    last_meta = load_torch_checkpoint(checkpoint_dir / "last.pt")["metadata"]
    assert "epochs_no_improve" in last_meta

    resumed = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    resumed._load_resume_checkpoint(str(checkpoint_dir / "last.pt"))
    assert resumed._epochs_no_improve == last_meta["epochs_no_improve"]

    # Legacy checkpoint predating the field: derive the counter from the gap
    # between the completed epoch and the best epoch.
    legacy_path = checkpoint_dir / "legacy_last.pt"
    save_torch_checkpoint(
        model_state=model._module.state_dict(),
        optimizer_state=None,
        config=None,
        epoch=5,
        path=legacy_path,
        metadata={"best_metric": 0.5, "best_epoch": 2, "monitor_metric": 0.9},
    )
    legacy = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    legacy._load_resume_checkpoint(str(legacy_path))
    assert legacy._epochs_no_improve == 3  # 5 completed - best at epoch 2


def test_evaluate_windowed_dataset_uses_aligned_targets() -> None:
    pytest.importorskip("torch")
    from solrad_correction.datasets.sequence import WindowedSequenceDataset
    from solrad_correction.models.lstm import LSTMRegressor

    rng = np.random.default_rng(11)
    features = rng.normal(0, 1, (40, 4)).astype(np.float32)
    targets = rng.normal(0, 1, 40).astype(np.float32)
    dataset = WindowedSequenceDataset(features, targets, sequence_length=8)

    model = LSTMRegressor(input_size=4, hidden_size=8, num_layers=1, device="cpu")
    metrics = model.evaluate(dataset)

    assert len(model.predict(dataset)) == len(dataset)
    assert set(metrics) >= {"RMSE", "MAE", "MBE"}
    assert np.isfinite(metrics["RMSE"])


def test_predict_returns_float32_and_ignores_amp() -> None:
    pytest.importorskip("torch")
    from solrad_correction.models.lstm import LSTMRegressor
    from solrad_correction.training.dataloaders import DataLoaderSettings

    model = LSTMRegressor(input_size=3, hidden_size=8, num_layers=1, device="cpu")
    model._dataloader_settings = DataLoaderSettings(
        device="cpu",
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        amp=True,  # autocast must NOT quantize inference outputs
        torch_compile=False,
        gradient_clip=None,
    )

    rng = np.random.default_rng(3)
    preds = model.predict(rng.normal(0, 1, (6, 8, 3)).astype(np.float32))

    assert preds.dtype == np.float32
    assert preds.shape == (6,)


def test_tensorboard_import_is_lazy() -> None:
    """factories must not require tensorboard unless log_dir is configured."""
    pytest.importorskip("torch")
    import builtins
    import importlib

    import solrad_correction.training.factories as factories

    real_import = builtins.__import__

    def blocking_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if "tensorboard" in name:
            raise ModuleNotFoundError("No module named 'tensorboard'")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = blocking_import
    try:
        reloaded = importlib.reload(factories)
        assert reloaded.create_summary_writer(None) is None
        with pytest.raises(ModuleNotFoundError):
            reloaded.create_summary_writer("scratch/tb-lazy")
    finally:
        builtins.__import__ = real_import
        importlib.reload(factories)


def test_training_progress_uses_absolute_epoch_budget_on_resume(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = TrainingProgress(total_epochs=10, start_epoch=8)
    progress.start_epoch(8)
    progress.update_batch(1, 2)
    progress.end_epoch(0.5)
    progress.start_epoch(9)
    progress.update_batch(2, 2)
    progress.end_epoch(0.4)
    progress.finish()

    out = capsys.readouterr().out
    assert "Epoch 9/10" in out
    assert "Epoch 10/10" in out
    assert "Overall: 100.0%" in out
    assert "/18" not in out
