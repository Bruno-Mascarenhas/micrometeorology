"""Unit tests for allsky.models (SkyFusionNet + multitask_loss).

Pure synthetic tensors — no video/sensor/dataset modules involved.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from allsky.config import ModelConfig  # noqa: E402
from allsky.models import SkyFusionNet, multitask_loss  # noqa: E402

N_FEATURES = 6


@pytest.fixture
def model_cfg() -> ModelConfig:
    return ModelConfig(image_size=64, backbone="small", embed_dim=32, hidden_dim=64, n_classes=3)


@pytest.fixture
def batch() -> dict:
    generator = torch.Generator().manual_seed(0)
    return {
        "image": torch.rand((2, 3, 64, 64), generator=generator),
        "features": torch.randn((2, N_FEATURES), generator=generator),
        "cloud_class": torch.tensor([0, 2], dtype=torch.int64),
        "diffuse": torch.tensor([120.0, 30.0]),
    }


def test_forward_shapes(model_cfg, batch):
    torch.manual_seed(0)
    model = SkyFusionNet(model_cfg, n_features=N_FEATURES)
    model.eval()
    outputs = model(batch["image"], batch["features"])
    assert set(outputs) == {"logits", "diffuse"}
    assert outputs["logits"].shape == (2, model_cfg.n_classes)
    assert outputs["diffuse"].shape == (2,)
    assert outputs["logits"].dtype == torch.float32


def test_reg_head_non_negative(model_cfg):
    """Irradiance predictions must be >= 0 even for adversarial inputs."""
    torch.manual_seed(1)
    model = SkyFusionNet(model_cfg, n_features=N_FEATURES)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            image = torch.randn(4, 3, 64, 64)  # includes negative pixel values
            features = torch.randn(4, N_FEATURES) * 10.0
            outputs = model(image, features)
            assert bool((outputs["diffuse"] >= 0).all())


def test_multitask_loss_finite_and_backward(model_cfg, batch):
    torch.manual_seed(0)
    model = SkyFusionNet(model_cfg, n_features=N_FEATURES)
    model.train()
    outputs = model(batch["image"], batch["features"])
    losses = multitask_loss(outputs, batch)
    assert set(losses) == {"loss", "loss_cls", "loss_reg"}
    assert bool(torch.isfinite(losses["loss"]))
    losses["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "backward produced no gradients"
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_multitask_loss_weights(model_cfg, batch):
    torch.manual_seed(0)
    model = SkyFusionNet(model_cfg, n_features=N_FEATURES)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"], batch["features"])
        losses = multitask_loss(outputs, batch, w_cls=1.0, w_reg=0.0)
        assert torch.allclose(losses["loss"], losses["loss_cls"])
        losses = multitask_loss(outputs, batch, w_cls=0.0, w_reg=2.0)
        assert torch.allclose(losses["loss"], 2.0 * losses["loss_reg"])


def test_multitask_loss_zero_for_perfect_regression(batch):
    """Regression term is 0 when predictions equal targets (scale-invariant check)."""
    outputs = {
        "logits": torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]]),
        "diffuse": batch["diffuse"].clone(),
    }
    losses = multitask_loss(outputs, batch)
    assert float(losses["loss_reg"]) == pytest.approx(0.0)
    assert float(losses["loss_cls"]) == pytest.approx(0.0, abs=1e-6)


def test_unknown_backbone_raises(model_cfg):
    cfg = model_cfg.model_copy(update={"backbone": "vgg"})
    with pytest.raises(ValueError, match="unknown backbone"):
        SkyFusionNet(cfg, n_features=N_FEATURES)


def test_invalid_n_features_raises(model_cfg):
    with pytest.raises(ValueError, match="n_features"):
        SkyFusionNet(model_cfg, n_features=0)


def test_resnet18_backbone_forward(model_cfg, batch):
    pytest.importorskip("torchvision")
    cfg = model_cfg.model_copy(update={"backbone": "resnet18"})
    torch.manual_seed(0)
    model = SkyFusionNet(cfg, n_features=N_FEATURES)
    model.eval()
    with torch.no_grad():
        outputs = model(batch["image"], batch["features"])
    assert outputs["logits"].shape == (2, cfg.n_classes)
    assert outputs["diffuse"].shape == (2,)
