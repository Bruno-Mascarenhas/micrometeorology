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

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from allsky.config import (
    load_experiment_config,
    load_prepare_config,
    model_param,
)
from allsky.features.policy import resolve_feature_set
from allsky.modeling.registry import MODEL_BUILDERS, build_model
from labmim_core.sky import SKY_CLASS_COUNT

_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "allsky"
_EXPERIMENTS = sorted((_CONFIGS / "experiments").glob("v*.yaml"))
#: Every experiment the repo ships, the nineteen subfamily trees included. The
#: v* glob alone left them out of the load, the path and the forward checks —
#: which is where the shipped arms live, not in v0..v7.
_ALL_EXPERIMENTS = sorted(
    path for path in (_CONFIGS / "experiments").rglob("*.yaml") if not path.name.startswith("_")
)
#: The prepare configs the experiments train on, keyed by the dataset they build.
_PREPARE_CONFIGS = sorted((_CONFIGS / "data").glob("local_prepare*.yaml"))
_FRAGMENTS = sorted((_CONFIGS / "models").glob("*.yaml"))

#: Embedding width used for the embedding-mode forward probes.
_EMBED_DIM = 32
_BATCH = 4


class _StubBackbone(nn.Module):
    """Tiny image backbone (``.dim`` attribute) for image-mode forward probes.

    Pools any ``(B, C, H, W)`` input to ``(B, C)`` and projects to ``dim`` — no
    downloads, no ``blocks`` (so ``unfreeze_last_n`` is a harmless no-op). It
    carries a ``patch_embed.proj`` because the geometry-channel arms wrap the
    first convolution to admit their extra channels, and a backbone without one
    is refused rather than silently left unwrapped.
    """

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.dim = dim
        self.model: Any = nn.Module()
        self.model.patch_embed = nn.Module()
        self.model.patch_embed.proj = nn.Conv2d(3, 3, kernel_size=1)
        self.proj = nn.Linear(3, dim)

    def forward(self, image: Tensor) -> Tensor:
        patched = self.model.patch_embed.proj(image)
        out: Tensor = self.proj(patched.mean(dim=(2, 3)))
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
    # Not `safe`: the Gill MetSENS1 thermohygrometer has been railed since
    # 2025-12-19, so the three channels safe adds are NaN across the camera
    # archive and the row-wise finite filter would drop 99.98% of the dataset.
    # Not `minimal` either: the same fault took the barometer on 2026-08-10.
    assert cfg.features.feature_set == "bare"
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


def test_the_experiments_train_on_the_set_the_prepare_config_builds() -> None:
    """The two configs pin the same feature set, or training dies on the manifest.

    The engine resolves its feature columns from the EXPERIMENT config while the
    manifest carries the ones the PREPARE config asked for, and nothing
    reconciles them: dropping only the prepare side to ``bare`` cost a full
    dataset build before ``manifest is missing feature columns`` surfaced.
    """
    prepared = load_prepare_config(_CONFIGS / "data" / "local_prepare.yaml")
    for experiment in _EXPERIMENTS:
        trained = load_experiment_config(experiment)
        assert trained.features.feature_set == prepared.features.feature_set, experiment.name


@pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS, ids=lambda p: p.name)
def test_every_shipped_experiment_loads_and_names_a_real_model(experiment: Path) -> None:
    """What must hold for every arm, v0..v7 and the subfamily trees alike. The
    invariants the v* family carries on top of this (its output_dir naming, its
    embedding input mode) stay on their own test.
    """
    cfg = load_experiment_config(experiment)

    assert cfg.experiment is True
    assert cfg.model.name in MODEL_BUILDERS, cfg.model.name
    assert cfg.features.feature_set == "bare"


@pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS, ids=lambda p: p.name)
def test_every_experiment_trains_on_a_dataset_some_prepare_config_builds(
    experiment: Path,
) -> None:
    """An arm pointed at a data_root no prepare config produces trains on
    whatever happens to be in that directory — the previous experiment's
    dataset, or nothing.
    """
    cfg = load_experiment_config(experiment)
    prepared = {load_prepare_config(path).output.dataset_dir for path in _PREPARE_CONFIGS}

    if experiment.parent.name == "folsom":
        # Known and open: the UCSD-Folsom adapter ships in allsky.data.folsom but
        # no configs/allsky/data/*.yaml builds `dataset-folsom`, so this arm's
        # data_root has no producer in the repo. Asserted the other way round so
        # that writing that config removes the exception rather than hiding it.
        assert cfg.data.data_root not in prepared
        return
    assert cfg.data.data_root in prepared, cfg.data.data_root


#: Arms known to resolve to the same run and left in place deliberately, as
#: ``family/stem`` pairs. control_sNN and loss_mae_sNN differ only in ``name`` and
#: ``output_dir`` after ``extends`` resolves — same seed, same data_root, same
#: model, same loss, same schedule — so running both produces two bit-identical
#: runs and "control against loss_mae" compares a run with itself. Which arm is
#: the control, and where seeds 45-47 belong, is an experiment-design decision.
_KNOWN_IDENTICAL_ARMS = {
    ("control/control_s42", "loss/loss_mae_s42"),
    ("control/control_s43", "loss/loss_mae_s43"),
    ("control/control_s44", "loss/loss_mae_s44"),
}


def test_no_two_experiments_resolve_to_the_same_run() -> None:
    """Two arms that resolve identically are one measurement served twice, and
    comparing them measures nothing. New duplicates fail here; the pair already
    shipped is listed above so removing it removes the exception too.
    """
    resolved: dict[str, list[str]] = {}
    for path in _ALL_EXPERIMENTS:
        dumped = load_experiment_config(path).model_dump(mode="json")
        dumped.pop("name", None)
        dumped.pop("output_dir", None)
        key = json.dumps(dumped, sort_keys=True)
        resolved.setdefault(key, []).append(f"{path.parent.name}/{path.stem}")

    duplicates = {tuple(sorted(arms)) for arms in resolved.values() if len(arms) > 1}
    unexpected = {
        pair
        for group in duplicates
        for pair in (
            (group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))
        )
        if pair not in _KNOWN_IDENTICAL_ARMS
    }
    assert unexpected == set(), sorted(unexpected)


def test_the_isotropic_calibration_matches_the_crop_and_pad_that_produced_it() -> None:
    """``allsky.lens`` derives three constants from this config's crop and pad,
    and the derivation lives only in a comment. Moving ``crop.left``,
    ``crop.width`` or ``pad.top`` here leaves ``isotropic_calibration()``
    returning the old centre, and every geometry channel is then drawn around a
    point the frames no longer have.
    """
    from allsky.lens import _ISOTROPIC_CROP_LEFT, _ISOTROPIC_PAD_TOP, ISOTROPIC_FRAME_PX

    cfg = load_prepare_config(_CONFIGS / "data" / "local_prepare_iso.yaml")
    sensor_width = 1920

    width, height = cfg.crop.width, cfg.crop.height
    assert width is not None
    assert height is not None

    assert width == ISOTROPIC_FRAME_PX
    assert (sensor_width - width) // 2 + cfg.crop.left == _ISOTROPIC_CROP_LEFT
    assert cfg.pad.top == _ISOTROPIC_PAD_TOP
    # The pad squares the frame: the padded height must equal the cropped width.
    assert height + cfg.pad.top + cfg.pad.bottom == ISOTROPIC_FRAME_PX


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
    # Has to be the set data/local_prepare.yaml BUILDS with, or the engine asks
    # the manifest for feature columns it does not carry.
    assert cfg.features.feature_set == "bare"
    assert cfg.model.name in MODEL_BUILDERS, cfg.model.name
    assert cfg.output_dir == f"output/allsky-mm/experiments/{experiment.stem}"
    # The default embedding path, plus the one image-mode finetune experiment.
    expected_mode = "image" if experiment.stem == "v6_film_finetune" else "embedding"
    assert cfg.data.input_mode == expected_mode


def test_no_two_experiments_write_into_one_run_directory() -> None:
    """Two seeds sharing an output_dir overwrite each other's checkpoints and
    metrics, and the second run reads as the first having diverged.
    """
    outputs = [load_experiment_config(path).output_dir for path in _ALL_EXPERIMENTS]

    assert len(set(outputs)) == len(outputs), sorted(
        name for name in outputs if outputs.count(name) > 1
    )


@pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS, ids=lambda p: p.name)
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


@pytest.mark.parametrize("experiment", _ALL_EXPERIMENTS, ids=lambda p: p.name)
def test_experiment_builds_and_forwards(experiment: Path) -> None:
    """Each experiment builds and forwards a dummy batch, emitting its enabled heads."""
    cfg = load_experiment_config(experiment)
    # Read the width off the config's own policy set rather than pinning it: the
    # engine sizes the sensor branch the same way, so a hardcoded 13 turned a
    # switch to `minimal` into a shape error in the test instead of in the code.
    n_features = len(resolve_feature_set(cfg.features.feature_set))

    if cfg.data.input_mode == "image":
        model = build_model(cfg, n_features, image_backbone=_StubBackbone())
        # The geometry arms widen the frame: the model wraps the first
        # convolution to admit the maps it is configured for, so the probe batch
        # has to be as wide as the wrapped convolution now expects.
        channels = getattr(getattr(model, "visual_encoder", None), "extra_channel_projection", None)
        n_channels = 3 if channels is None else channels.in_channels
        batch = {
            "features": torch.randn(_BATCH, n_features),
            "image": torch.randn(_BATCH, n_channels, 8, 8),
        }
    else:
        model = build_model(cfg, n_features, embedding_dim=_EMBED_DIM)
        batch = {
            "features": torch.randn(_BATCH, n_features),
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
        assert outputs["sky_logits"].shape == (_BATCH, SKY_CLASS_COUNT)
    # cloud_fraction stays disabled everywhere (no ground truth yet).
    assert "cloud_fraction" not in outputs


class TestTransferDirection:
    @staticmethod
    def _arms(family: str) -> list[Path]:
        return sorted((_CONFIGS / "experiments" / family).glob("*.yaml"))

    def test_the_source_never_initialises_from_anywhere(self):
        for path in self._arms("folsom"):
            cfg = load_experiment_config(path)

            assert model_param(cfg, "init_from", None) is None, path.name

    def test_the_source_trains_on_folsom_and_nothing_else(self):
        for path in self._arms("folsom"):
            cfg = load_experiment_config(path)

            assert cfg.data.data_root.endswith("dataset-folsom"), path.name

    def test_every_transfer_arm_starts_from_a_folsom_checkpoint(self):
        arms = self._arms("transfer")

        assert arms, "the transfer family has no configs"
        for path in arms:
            cfg = load_experiment_config(path)
            source = str(model_param(cfg, "init_from", "") or "")

            assert "/folsom/" in source, f"{path.name} initialises from {source!r}"
            assert source.endswith(".ckpt"), path.name

    def test_every_transfer_arm_fine_tunes_on_this_station(self):
        for path in self._arms("transfer"):
            cfg = load_experiment_config(path)

            assert cfg.data.data_root.endswith("dataset-iso"), path.name

    def test_no_station_arm_outside_the_transfer_family_initialises_from_anything(self):
        for path in sorted((_CONFIGS / "experiments").glob("*/*.yaml")):
            if path.parent.name in {"transfer", "folsom"}:
                continue
            cfg = load_experiment_config(path)

            assert model_param(cfg, "init_from", None) is None, path.name
