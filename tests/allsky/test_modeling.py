"""Torch-gated tests for the allsky.modeling zoo: builds, forwards, invariants.

Every test is offline and CPU-only, uses synthetic tensors and tiny stub
backbones (no DINOv2 download), and runs in well under a couple of seconds.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from allsky.config import ExperimentConfig  # noqa: E402
from allsky.features import active_feature_groups, resolve_feature_set  # noqa: E402
from allsky.features.normalization import TargetNormalizer  # noqa: E402
from allsky.modeling.baselines import ClimatologyModel  # noqa: E402
from allsky.modeling.contracts import MultimodalModel  # noqa: E402
from allsky.modeling.fusion import (  # noqa: E402
    ConcatFusion,
    CrossAttentionFusion,
    FiLMFusion,
)
from allsky.modeling.heads import DHIHeteroscedasticHead, Heads, Trunk  # noqa: E402
from allsky.modeling.multimodal import MultimodalNet  # noqa: E402
from allsky.modeling.registry import MODEL_BUILDERS, build_model  # noqa: E402
from allsky.modeling.sensor_encoder import SensorEncoder  # noqa: E402
from allsky.modeling.visual_encoder import ImageEncoder, PrecomputedEmbedding  # noqa: E402

SAFE_FEATURES = resolve_feature_set("safe")
N_FEATURES = len(SAFE_FEATURES)
EMBED_DIM = 32
BATCH = 4

ALL_HEADS = {
    "dhi": {"enabled": True, "loss": "heteroscedastic"},
    "kindex": {"enabled": True, "kind": "kstar"},
    "sky": {"enabled": True},
    "cloud_fraction": {"enabled": True},
}
ALL_HEAD_KEYS = {"dhi", "dhi_log_var", "kindex", "sky_logits", "cloud_fraction"}


class TinyConvBackbone(nn.Module):
    """Minimal conv encoder exposing ``.dim`` (stands in for DINOv2)."""

    def __init__(self, dim: int = 24) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Any) -> Any:
        return self.pool(self.conv(x)).flatten(1)


class TinyViTBackbone(nn.Module):
    """Stub with a ``blocks`` sequence so unfreeze_last_n has something to grip."""

    def __init__(self, dim: int = 16, n_blocks: int = 3) -> None:
        super().__init__()
        self.dim = dim
        self.patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_blocks))

    def forward(self, x: Any) -> Any:
        h = self.pool(self.patch(x)).flatten(1)
        for block in self.blocks:
            h = block(h)
        return h


def _cfg(model_name: str, *, input_mode: str = "embedding", targets: dict | None = None) -> Any:
    return ExperimentConfig.model_validate(
        {
            "features": {"set": "safe"},
            "targets": targets if targets is not None else ALL_HEADS,
            "model": {"name": model_name},
            "data": {"input_mode": input_mode},
        }
    )


def _batch(*, image: bool = False) -> dict[str, Any]:
    torch.manual_seed(0)
    batch: dict[str, Any] = {"features": torch.randn(BATCH, N_FEATURES)}
    if image:
        batch["image"] = torch.rand(BATCH, 3, 32, 32)
    else:
        batch["embedding"] = torch.randn(BATCH, EMBED_DIM)
    return batch


# --------------------------------------------------------------------------- #
# Every registry model builds and forwards with the right outputs.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_name", list(MODEL_BUILDERS))
def test_registry_model_builds_and_forwards_all_heads(model_name: str):
    input_mode = "image" if model_name == "image_only" else "embedding"
    cfg = _cfg(model_name, input_mode=input_mode)
    backbone = TinyConvBackbone() if input_mode == "image" else None
    model = build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM, image_backbone=backbone)
    model.eval()

    out = model(_batch(image=input_mode == "image"))

    assert set(out) == ALL_HEAD_KEYS
    assert out["dhi"].shape == (BATCH,)
    assert out["dhi_log_var"].shape == (BATCH,)
    assert out["kindex"].shape == (BATCH,)
    assert out["sky_logits"].shape == (BATCH, 3)
    assert out["cloud_fraction"].shape == (BATCH,)
    for key, value in out.items():
        assert value.dtype == torch.float32, key
        assert bool(torch.isfinite(value).all()), key
    assert isinstance(model, MultimodalModel)  # structural contract


@pytest.mark.parametrize("model_name", list(MODEL_BUILDERS))
def test_registry_model_backward_runs(model_name: str):
    """Loss over the outputs backpropagates (climatology's dummy param included)."""
    input_mode = "image" if model_name == "image_only" else "embedding"
    cfg = _cfg(model_name, input_mode=input_mode)
    backbone = TinyConvBackbone() if input_mode == "image" else None
    model = build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM, image_backbone=backbone)

    out = model(_batch(image=input_mode == "image"))
    loss = out["dhi"].mean() + out["sky_logits"].sum() + out["cloud_fraction"].mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(bool(torch.isfinite(g).all()) for g in grads)


def test_cloud_fraction_output_is_bounded():
    cfg = _cfg("concat")
    model = build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM)
    model.eval()
    out = model(_batch())
    cf = out["cloud_fraction"]
    assert bool((cf >= 0).all())
    assert bool((cf <= 1).all())


# --------------------------------------------------------------------------- #
# Heads on/off combinations produce exactly the enabled keys.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("targets", "expected"),
    [
        ({"dhi": {"enabled": True, "loss": "huber"}}, {"dhi"}),
        ({"dhi": {"enabled": True, "loss": "heteroscedastic"}}, {"dhi", "dhi_log_var"}),
        (
            {"dhi": {"enabled": False}, "kindex": {"enabled": True}},
            {"kindex"},
        ),
        (
            {"dhi": {"enabled": False}, "sky": {"enabled": True}},
            {"sky_logits"},
        ),
        (
            {
                "dhi": {"enabled": True, "loss": "huber"},
                "cloud_fraction": {"enabled": True},
            },
            {"dhi", "cloud_fraction"},
        ),
    ],
)
def test_heads_on_off_combinations(targets: dict, expected: set[str]):
    cfg = _cfg("concat", targets=targets)
    model = build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM)
    model.eval()
    assert set(model(_batch())) == expected


# --------------------------------------------------------------------------- #
# FiLM starts as an exact identity == concat(visual, sensor).
# --------------------------------------------------------------------------- #


def test_film_initial_identity_equals_concat():
    visual_dim, sensor_dim = EMBED_DIM, 20
    concat = ConcatFusion(visual_dim, sensor_dim)
    film = FiLMFusion(visual_dim, sensor_dim)
    film.eval()
    torch.manual_seed(1)
    visual = torch.randn(BATCH, visual_dim)
    sensor = torch.randn(BATCH, sensor_dim)

    film_out = film(visual, sensor)
    assert torch.allclose(film_out, concat(visual, sensor))
    assert torch.allclose(film_out, torch.cat([visual, sensor], dim=-1))
    assert film.out_dim == visual_dim + sensor_dim


def test_film_diverges_from_identity_after_a_step():
    """Once film_gen weights move, FiLM is no longer the identity."""
    film = FiLMFusion(EMBED_DIM, 16)
    with torch.no_grad():
        film.film_gen.weight.add_(0.1)
        film.film_gen.bias.add_(0.1)
    visual = torch.randn(BATCH, EMBED_DIM)
    sensor = torch.randn(BATCH, 16)
    assert not torch.allclose(film(visual, sensor), torch.cat([visual, sensor], dim=-1))


# --------------------------------------------------------------------------- #
# Cross-attention: masking a group makes it irrelevant; heads configurable.
# --------------------------------------------------------------------------- #


def _cross_fusion(num_heads: int = 4, token_dim: int = EMBED_DIM, sensor_dim: int = 16):
    return CrossAttentionFusion(
        EMBED_DIM,
        sensor_dim,
        feature_columns=SAFE_FEATURES,
        groups=active_feature_groups("safe"),
        num_heads=num_heads,
        token_dim=token_dim,
    )


def test_cross_attention_masked_group_changes_nothing():
    fusion = _cross_fusion()
    fusion.eval()
    torch.manual_seed(2)
    visual = torch.randn(BATCH, EMBED_DIM)
    sensor = torch.randn(BATCH, 16)
    features = torch.randn(BATCH, N_FEATURES)
    n_groups = len(fusion.group_names)
    masked = 1
    mask = torch.zeros(BATCH, n_groups, dtype=torch.bool)
    mask[:, masked] = True

    out1 = fusion(visual, sensor, features=features, key_padding_mask=mask)
    changed = features.clone()
    idx = fusion._group_indices[masked]
    changed[:, idx] = torch.randn(BATCH, len(idx)) * 10.0
    out2 = fusion(visual, sensor, features=changed, key_padding_mask=mask)

    assert torch.allclose(out1, out2, atol=1e-6)
    assert bool(torch.isfinite(out1).all())
    assert out1.shape == (BATCH, EMBED_DIM + 16)


def test_cross_attention_unmasked_group_matters():
    fusion = _cross_fusion()
    fusion.eval()
    torch.manual_seed(3)
    visual = torch.randn(BATCH, EMBED_DIM)
    sensor = torch.randn(BATCH, 16)
    features = torch.randn(BATCH, N_FEATURES)
    mask = torch.zeros(BATCH, len(fusion.group_names), dtype=torch.bool)
    mask[:, 1] = True  # mask group 1; perturb the unmasked group 0

    out1 = fusion(visual, sensor, features=features, key_padding_mask=mask)
    changed = features.clone()
    changed[:, fusion._group_indices[0]] += 5.0
    out2 = fusion(visual, sensor, features=changed, key_padding_mask=mask)
    assert not torch.allclose(out1, out2)


@pytest.mark.parametrize("num_heads", [1, 2, 4, 8])
def test_cross_attention_head_count_configurable(num_heads: int):
    fusion = _cross_fusion(num_heads=num_heads, token_dim=8)
    fusion.eval()
    out = fusion(
        torch.randn(BATCH, EMBED_DIM),
        torch.randn(BATCH, 16),
        features=torch.randn(BATCH, N_FEATURES),
    )
    assert out.shape == (BATCH, 8 + 16)
    assert bool(torch.isfinite(out).all())


def test_cross_attention_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="divisible"):
        _cross_fusion(num_heads=3, token_dim=32)


def test_cross_attention_works_end_to_end_in_multimodal():
    cfg = _cfg("cross_attention", targets={"dhi": {"enabled": True, "loss": "huber"}})
    model = build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM)
    model.eval()
    out = model(_batch())
    assert set(out) == {"dhi"}
    assert bool(torch.isfinite(out["dhi"]).all())


# --------------------------------------------------------------------------- #
# Climatology baseline: constant outputs and class-frequency logits.
# --------------------------------------------------------------------------- #


def test_climatology_fit_constants_and_frequency_logits():
    cfg = _cfg("climatology")
    model = ClimatologyModel(cfg.targets)
    import numpy as np

    dhi = np.array([100.0, 200.0, 300.0, np.nan])  # mean of finite = 200
    kindex = np.array([0.4, 0.6, 0.8])  # mean 0.6
    cloud = np.array([0.1, 0.3])  # mean 0.2
    sky = np.array([2, 2, 2, 1, 0, -1])  # class 2 most frequent, -1 ignored
    model.fit_from_targets(dhi=dhi, kindex=kindex, cloud_fraction=cloud, sky_class=sky)
    model.eval()

    out = model(_batch())
    assert torch.allclose(out["dhi"], torch.full((BATCH,), 200.0))
    assert torch.allclose(out["kindex"], torch.full((BATCH,), 0.6))
    assert torch.allclose(out["cloud_fraction"], torch.full((BATCH,), 0.2))
    # every row identical (constant model)
    assert torch.allclose(out["dhi"], out["dhi"][0])
    # frequency logits: class 2 dominates, softmax matches empirical frequency
    freq = torch.softmax(out["sky_logits"][0].detach(), dim=-1)
    assert int(out["sky_logits"][0].argmax()) == 2
    assert float(freq[2]) == pytest.approx(3 / 5, abs=1e-5)


def test_climatology_uses_normalized_space_means():
    cfg = _cfg("climatology", targets={"dhi": {"enabled": True, "loss": "huber"}})
    model = ClimatologyModel(cfg.targets)
    import numpy as np

    normalizer = TargetNormalizer(mean=100.0, std=50.0)
    model.fit_from_targets(dhi=np.array([200.0, 200.0]), target_normalizers={"dhi": normalizer})
    model.eval()
    out = model(_batch())
    # (200 - 100) / 50 == 2.0 in normalized space
    assert torch.allclose(out["dhi"], torch.full((BATCH,), 2.0))


def test_climatology_has_a_trainable_parameter():
    cfg = _cfg("climatology")
    model = ClimatologyModel(cfg.targets)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params, "climatology must expose a dummy parameter for the optimizer"


# --------------------------------------------------------------------------- #
# Encoders, trunk, heads, param groups.
# --------------------------------------------------------------------------- #


def test_sensor_encoder_default_and_custom_dims():
    encoder = SensorEncoder(N_FEATURES)
    assert encoder.out_dim == 128
    assert encoder(torch.randn(BATCH, N_FEATURES)).shape == (BATCH, 128)

    custom = SensorEncoder(N_FEATURES, (32, 48, 16))
    assert custom.out_dim == 16
    assert custom(torch.randn(BATCH, N_FEATURES)).shape == (BATCH, 16)


def test_sensor_encoder_rejects_bad_dims():
    with pytest.raises(ValueError, match="in_dim"):
        SensorEncoder(0)


def test_precomputed_embedding_passthrough_and_projection():
    passthrough = PrecomputedEmbedding(EMBED_DIM)
    assert passthrough.out_dim == EMBED_DIM
    emb = torch.randn(BATCH, EMBED_DIM)
    assert torch.allclose(passthrough({"embedding": emb}), emb)  # identity

    projected = PrecomputedEmbedding(EMBED_DIM, 8)
    assert projected.out_dim == 8
    assert projected({"embedding": emb}).shape == (BATCH, 8)


def test_precomputed_embedding_masked_mean_pool():
    encoder = PrecomputedEmbedding(EMBED_DIM)
    seq = torch.randn(BATCH, 3, EMBED_DIM)
    mask = torch.tensor([[True, True, False]] * BATCH)
    pooled = encoder({"embedding_seq": seq, "frame_mask": mask})
    expected = seq[:, :2, :].mean(dim=1)
    assert torch.allclose(pooled, expected, atol=1e-6)

    # unmasked mean pooling falls back to a plain mean
    plain = encoder({"embedding_seq": seq})
    assert torch.allclose(plain, seq.mean(dim=1), atol=1e-6)


def test_precomputed_embedding_missing_key_raises():
    encoder = PrecomputedEmbedding(EMBED_DIM)
    with pytest.raises(KeyError, match="embedding"):
        encoder({"features": torch.randn(BATCH, N_FEATURES)})


def test_precomputed_embedding_learned_attention_pool_forward_backward():
    encoder = PrecomputedEmbedding(EMBED_DIM, temporal_pooling="attention")
    seq = torch.randn(BATCH, 4, EMBED_DIM, requires_grad=True)
    mask = torch.tensor([[True, True, False, False]] * BATCH)
    out = encoder({"embedding_seq": seq, "frame_mask": mask})
    assert out.shape == (BATCH, EMBED_DIM)
    out.sum().backward()  # gradients flow to the learnable query
    assert encoder.query.grad is not None
    assert torch.isfinite(out).all()


def test_attention_pool_ignores_masked_frames():
    torch.manual_seed(0)
    encoder = PrecomputedEmbedding(EMBED_DIM, temporal_pooling="attention").eval()
    seq = torch.randn(BATCH, 4, EMBED_DIM)
    mask = torch.tensor([[True, True, False, False]] * BATCH)
    with torch.no_grad():
        base = encoder({"embedding_seq": seq, "frame_mask": mask})
        perturbed = seq.clone()
        perturbed[:, 2:, :] += 100.0  # change only the masked-out frames
        after = encoder({"embedding_seq": perturbed, "frame_mask": mask})
    assert torch.allclose(base, after, atol=1e-5)


def test_attention_pool_all_masked_row_falls_back_to_zero():
    encoder = PrecomputedEmbedding(EMBED_DIM, temporal_pooling="attention").eval()
    seq = torch.randn(2, 3, EMBED_DIM)
    mask = torch.tensor([[True, False, False], [False, False, False]])  # row 1 all-pad
    with torch.no_grad():
        out = encoder({"embedding_seq": seq, "frame_mask": mask})
    assert torch.isfinite(out).all()
    assert torch.allclose(out[1], torch.zeros(EMBED_DIM), atol=1e-6)


def test_attention_pool_rejects_indivisible_heads():
    with pytest.raises(ValueError, match="num_heads"):
        PrecomputedEmbedding(EMBED_DIM, temporal_pooling="attention", num_heads=5)  # 32 % 5


def test_multimodal_attention_temporal_pooling_forward():
    cfg = _cfg("concat", targets={"dhi": {"enabled": True, "loss": "huber"}})
    model = MultimodalNet(
        feature_columns=SAFE_FEATURES,
        targets=cfg.targets,
        fusion_name="concat",
        input_mode="embedding",
        embedding_dim=EMBED_DIM,
        temporal_pooling="attention",
    )
    batch = {
        "features": torch.randn(BATCH, N_FEATURES),
        "embedding_seq": torch.randn(BATCH, 3, EMBED_DIM),
        "frame_mask": torch.tensor([[True, True, False]] * BATCH),
    }
    out = model(batch)
    assert out["dhi"].shape == (BATCH,)


def test_build_model_warns_on_unknown_hyperparameter(caplog):
    cfg = ExperimentConfig.model_validate(
        {
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "huber"}},
            "model": {"name": "sensor_only", "droput": 0.5},  # typo of 'dropout'
            "data": {"input_mode": "embedding"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="allsky.modeling.registry"):
        build_model(cfg, N_FEATURES)
    assert any("droput" in record.getMessage() for record in caplog.records)


def test_build_model_no_warning_for_known_hyperparameters(caplog):
    cfg = ExperimentConfig.model_validate(
        {
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "huber"}},
            "model": {
                "name": "concat",
                "sensor_hidden": [64, 128],
                "visual_out_dim": None,
                "trunk_hidden": 256,
                "dropout": 0.1,
            },
            "data": {"input_mode": "embedding"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="allsky.modeling.registry"):
        build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM)
    assert not any("unknown hyper-parameter" in record.getMessage() for record in caplog.records)


def test_trunk_shape_and_residual():
    trunk = Trunk(160)
    assert trunk.out_dim == 256
    assert trunk(torch.randn(BATCH, 160)).shape == (BATCH, 256)
    # residual block is active where widths match (256 -> 256 second layer)
    assert bool(trunk.blocks[1].residual)
    assert not bool(trunk.blocks[0].residual)


def test_heads_heteroscedastic_log_var_clamped():
    head = DHIHeteroscedasticHead(4)
    with torch.no_grad():
        head.linear.weight.zero_()
        head.linear.bias[1] = 1000.0  # push log_var far past the ceiling
    out = head(torch.zeros(BATCH, 4))
    assert torch.all(out["dhi_log_var"] <= 10.0)
    assert torch.all(out["dhi_log_var"] >= -10.0)
    assert set(out) == {"dhi", "dhi_log_var"}


def test_heads_empty_when_nothing_enabled():
    cfg = _cfg("concat", targets={"dhi": {"enabled": False}})
    heads = Heads(64, cfg.targets)
    assert len(heads.heads) == 0
    assert heads(torch.randn(BATCH, 64)) == {}


def test_image_encoder_param_groups_separate_backbone_lr():
    backbone = TinyConvBackbone(dim=24)
    encoder = ImageEncoder(backbone, out_dim=16)  # projection has its own params
    groups = encoder.param_groups(backbone_lr=1e-5)

    assert len(groups) == 2
    backbone_group = next(g for g in groups if g.get("lr") == 1e-5)
    other_group = next(g for g in groups if "lr" not in g)
    backbone_ids = {id(p) for p in backbone.parameters()}
    assert all(id(p) in backbone_ids for p in backbone_group["params"])
    assert all(id(p) not in backbone_ids for p in other_group["params"])


def test_image_encoder_frozen_unfreezes_last_blocks():
    backbone = TinyViTBackbone(dim=16, n_blocks=3)
    encoder = ImageEncoder(backbone, frozen=True, unfreeze_last_n=1)
    # only the last block is trainable
    assert all(not p.requires_grad for p in backbone.patch.parameters())
    assert all(not p.requires_grad for p in backbone.blocks[0].parameters())
    assert all(p.requires_grad for p in backbone.blocks[-1].parameters())
    # forward still works and discovers the dim
    assert encoder.out_dim == 16
    assert encoder({"image": torch.rand(BATCH, 3, 32, 32)}).shape == (BATCH, 16)


def test_image_encoder_requires_dim_attribute():
    with pytest.raises(AttributeError, match="dim"):
        ImageEncoder(nn.Linear(3, 4))


def test_build_model_unknown_name_lists_available():
    cfg = _cfg("nonexistent")
    with pytest.raises(ValueError, match="unknown model") as exc:
        build_model(cfg, N_FEATURES, embedding_dim=EMBED_DIM)
    message = str(exc.value)
    assert "cross_attention" in message
    assert "climatology" in message


def test_multimodal_param_groups_split_backbone():
    cfg = _cfg("concat", input_mode="image")
    model = build_model(cfg, N_FEATURES, image_backbone=TinyConvBackbone(dim=24))
    assert isinstance(model, MultimodalNet)
    groups = model.param_groups(backbone_lr=1e-5)
    assert any(g.get("lr") == 1e-5 for g in groups)
    # without a backbone_lr it collapses to a single group
    assert len(model.param_groups()) == 1
