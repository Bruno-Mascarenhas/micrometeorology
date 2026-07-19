"""Contract tests for the shipped ``configs/allsky/`` YAML tree (Wave C5).

Every experiment under ``configs/allsky/experiments/`` must:

- load through :func:`allsky.config.load_experiment_config` with its ``extends:``
  chain resolved and ``extra="forbid"`` satisfied (no stray keys);
- name a model registered in :data:`allsky.modeling.registry.MODEL_BUILDERS`;
- build via :func:`allsky.modeling.registry.build_model` and run a forward pass
  on a dummy batch in its configured input mode, emitting the heads its
  ``targets`` block enables.

The ``models/`` fragments must load and name a real model, and
``data/local_prepare.yaml`` must load through
:func:`allsky.config.load_prepare_config`.

Torch is required to import the registry (the model zoo imports torch at module
scope), so the whole module is skipped when torch is unavailable — offline and
CPU-only otherwise; no dataset, embeddings or network are touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from torch import Tensor, nn  # noqa: E402

from allsky.config import (  # noqa: E402
    load_experiment_config,
    load_prepare_config,
)
from allsky.modeling.registry import MODEL_BUILDERS, build_model  # noqa: E402

_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "allsky"
_EXPERIMENTS = sorted((_CONFIGS / "experiments").glob("v*.yaml"))
_FRAGMENTS = sorted((_CONFIGS / "models").glob("*.yaml"))

#: Feature count of the ``safe`` policy set (all shipped experiments use it).
_N_FEATURES = 13
#: Embedding width used for the embedding-mode forward probes.
_EMBED_DIM = 32
_BATCH = 4


class _StubBackbone(nn.Module):
    """Tiny image backbone (``.dim`` attribute) for image-mode forward probes.

    Pools any ``(B, 3, H, W)`` input to ``(B, 3)`` and projects to ``dim`` — no
    downloads, no ``blocks`` (so ``unfreeze_last_n`` is a harmless no-op).
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(3, dim)

    def forward(self, image: Tensor) -> Tensor:
        out: Tensor = self.proj(image.mean(dim=(2, 3)))
        return out


def test_repo_has_eight_experiments() -> None:
    """V0-V7 are all present (guards against an accidentally dropped config)."""
    names = [path.name for path in _EXPERIMENTS]
    assert len(names) == 8, names
    assert names[0].startswith("v0_")
    assert names[-1].startswith("v7_")


def test_local_prepare_config_loads() -> None:
    """``data/local_prepare.yaml`` is a valid PrepareConfig with the pinned knobs."""
    cfg = load_prepare_config(_CONFIGS / "data" / "local_prepare.yaml")
    assert cfg.output.dataset_dir == "output/allsky-mm/dataset"
    assert cfg.output.dataset_version == "2"
    assert cfg.features.feature_set == "safe"
    assert cfg.sensor.ghi_column == "CM3Up_Wm2_Avg"
    assert cfg.targets.diffuse_column == "PSP_Wm2_Avg"
    assert cfg.targets.kindex_kind == "kstar"
    assert cfg.alignment.strategy == "center_frame"
    assert cfg.embeddings.backbone == "dinov2_vits14"
    assert cfg.embeddings.pooling == "cls"
    assert cfg.embeddings.shard_size == 2048
    assert cfg.embeddings.dtype == "fp16"
    assert cfg.splits.val_fraction == pytest.approx(0.15)
    assert cfg.splits.test_fraction == pytest.approx(0.15)
    assert cfg.splits.seed == 42


@pytest.mark.parametrize("fragment", _FRAGMENTS, ids=lambda p: p.name)
def test_model_fragment_loads_and_names_a_real_model(fragment: Path) -> None:
    """Each ``models/*.yaml`` fragment loads and names a registered model."""
    cfg = load_experiment_config(fragment)
    assert cfg.model.name in MODEL_BUILDERS, cfg.model.name


@pytest.mark.parametrize("experiment", _EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_loads_and_names_a_real_model(experiment: Path) -> None:
    """Each experiment loads (extends resolved, extra=forbid) with the pinned invariants."""
    cfg = load_experiment_config(experiment)
    assert cfg.experiment is True
    assert cfg.seed == 42
    assert cfg.features.feature_set == "safe"
    assert cfg.model.name in MODEL_BUILDERS, cfg.model.name
    assert cfg.output_dir == f"output/allsky-mm/experiments/{experiment.stem}"
    # The default embedding path, plus the one image-mode finetune experiment.
    expected_mode = "image" if experiment.stem == "v6_film_finetune" else "embedding"
    assert cfg.data.input_mode == expected_mode


@pytest.mark.parametrize("experiment", _EXPERIMENTS, ids=lambda p: p.name)
def test_data_paths_resolve_without_doubling_data_root(experiment: Path) -> None:
    """manifest/split/embeddings are BARE names: resolving contains data_root once.

    The engine/evaluator resolve each path as ``data_root / name`` (unless the
    name is absolute).  If a config repeated ``data_root`` in the leaf paths the
    resolved location would embed it twice (``.../dataset/.../dataset/...``) —
    this guards against that regression.
    """
    cfg = load_experiment_config(experiment)
    data_root = cfg.data.data_root
    assert data_root, data_root
    assert not Path(data_root).is_absolute(), data_root

    def _resolve(name: str) -> Path:
        candidate = Path(name)
        return candidate if candidate.is_absolute() else Path(data_root) / candidate

    leaves = [cfg.data.manifest, cfg.data.split_artifact]
    if cfg.data.embeddings_dir is not None:
        leaves.append(cfg.data.embeddings_dir)
    for name in leaves:
        # A bare leaf must not itself carry the data_root prefix.
        assert data_root not in name, f"{name!r} repeats data_root {data_root!r}"
        resolved = _resolve(name).as_posix()
        assert resolved.count(data_root) == 1, resolved


@pytest.mark.parametrize("experiment", _EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_builds_and_forwards(experiment: Path) -> None:
    """Each experiment builds and forwards a dummy batch, emitting its enabled heads."""
    cfg = load_experiment_config(experiment)

    if cfg.data.input_mode == "image":
        model = build_model(cfg, _N_FEATURES, image_backbone=_StubBackbone())
        batch = {
            "features": torch.randn(_BATCH, _N_FEATURES),
            "image": torch.randn(_BATCH, 3, 8, 8),
        }
    else:
        model = build_model(cfg, _N_FEATURES, embedding_dim=_EMBED_DIM)
        batch = {
            "features": torch.randn(_BATCH, _N_FEATURES),
            "embedding": torch.randn(_BATCH, _EMBED_DIM),
        }

    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    assert cfg.targets.dhi.enabled  # every shipped experiment predicts DHI
    assert outputs["dhi"].shape == (_BATCH,)
    if cfg.targets.dhi.loss == "heteroscedastic":
        assert outputs["dhi_log_var"].shape == (_BATCH,)
    if cfg.targets.kindex.enabled:
        assert outputs["kindex"].shape == (_BATCH,)
    if cfg.targets.sky.enabled:
        assert outputs["sky_logits"].shape == (_BATCH, 3)
    # cloud_fraction stays disabled everywhere (no ground truth yet).
    assert "cloud_fraction" not in outputs
