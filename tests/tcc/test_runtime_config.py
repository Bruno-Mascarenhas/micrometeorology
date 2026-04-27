"""Tests for runtime and model registry helpers."""

from __future__ import annotations

import pytest

from solrad_correction.config import ModelConfig, RuntimeConfig
from solrad_correction.models.registry import MODEL_REGISTRY, build_model, get_model_spec
from solrad_correction.training.dataloaders import resolve_dataloader_settings, resolve_device


def test_resolve_cpu_dataloader_settings():
    settings = resolve_dataloader_settings(
        RuntimeConfig(device="cpu", num_workers=0, pin_memory=False, amp=False),
    )

    assert settings.device == "cpu"
    assert settings.num_workers == 0
    assert settings.pin_memory is False
    assert settings.prefetch_factor is None
    assert settings.amp is False
    assert settings.torch_compile is False


def test_runtime_compile_controls_compile_setting():
    settings = resolve_dataloader_settings(
        RuntimeConfig(device="cpu", torch_compile=True),
    )

    assert settings.torch_compile is True


def test_cuda_request_fails_when_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")


def test_registry_contains_existing_models():
    assert set(MODEL_REGISTRY) >= {"svm", "lstm", "transformer"}
    assert get_model_spec("svm").kind == "tabular"
    assert get_model_spec("lstm").kind == "sequence"


def test_registry_builds_existing_models():
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


def test_registry_unknown_model_fails_clearly():
    with pytest.raises(ValueError, match="Unknown model type"):
        get_model_spec("unknown")
