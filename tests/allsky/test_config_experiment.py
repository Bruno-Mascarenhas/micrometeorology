"""Tests for the Wave C1b experiment/prepare configs and ``extends:`` composition.

Torch-free: only pydantic + YAML on disk (``tmp_path``), strictly offline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from allsky.config import (
    ExperimentConfig,
    PrepareConfig,
    _deep_merge,
    is_experiment_config,
    load_experiment_config,
    load_prepare_config,
)

# ---------------------------------------------------------------------------
# _deep_merge unit semantics
# ---------------------------------------------------------------------------


def test_deep_merge_scalar_dict_and_list_semantics():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2, 3]}
    override = {"a": 9, "nested": {"y": 20, "z": 30}, "lst": [9]}
    merged = _deep_merge(base, override)
    # scalar overwrite, dict deep-merge, list wholesale overwrite
    assert merged == {"a": 9, "nested": {"x": 1, "y": 20, "z": 30}, "lst": [9]}


def test_deep_merge_does_not_mutate_inputs():
    base = {"nested": {"x": 1}}
    override = {"nested": {"y": 2}}
    _deep_merge(base, override)
    assert base == {"nested": {"x": 1}}
    assert override == {"nested": {"y": 2}}


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_experiment_defaults_sane():
    cfg = ExperimentConfig()
    assert cfg.experiment is False
    assert cfg.seed == 42
    assert cfg.features.feature_set == "safe"
    assert cfg.data.input_mode == "image"
    assert cfg.data.alignment.strategy == "center_frame"
    assert cfg.targets.dhi.enabled is True
    assert cfg.targets.dhi.loss == "huber"
    assert cfg.targets.kindex.enabled is False
    assert cfg.targets.kindex.kind == "kstar"
    assert cfg.train.epochs == 20
    assert cfg.train.amp.enabled is False
    assert cfg.model.name == "concat"


def test_prepare_defaults_sane():
    cfg = PrepareConfig()
    assert cfg.output.dataset_version == "2"
    assert cfg.splits.val_fraction == pytest.approx(0.2)
    assert cfg.splits.test_fraction == pytest.approx(0.1)
    assert cfg.embeddings.backbone == "dinov2_vits14"
    assert cfg.embeddings.pooling == "cls"
    assert cfg.targets.diffuse_column == "PSP_Wm2_Avg"


def test_features_set_alias_accepts_yaml_key(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("experiment: true\nfeatures:\n  set: extended\n", encoding="utf-8")
    cfg = load_experiment_config(path)
    assert cfg.features.feature_set == "extended"


# ---------------------------------------------------------------------------
# extends composition
# ---------------------------------------------------------------------------


def test_extends_two_level_chain(tmp_path):
    (tmp_path / "grandbase.yaml").write_text(
        "name: base\nseed: 1\ntrain:\n  epochs: 3\n", encoding="utf-8"
    )
    (tmp_path / "base.yaml").write_text("extends: [grandbase.yaml]\nseed: 2\n", encoding="utf-8")
    (tmp_path / "top.yaml").write_text("extends: [base.yaml]\nexperiment: true\n", encoding="utf-8")
    cfg = load_experiment_config(tmp_path / "top.yaml")
    # name propagates from the grandparent, seed overridden by the parent
    assert cfg.name == "base"
    assert cfg.seed == 2
    assert cfg.experiment is True
    assert cfg.train.epochs == 3


def test_extends_deep_merges_nested_dicts(tmp_path):
    (tmp_path / "base.yaml").write_text("train:\n  epochs: 5\n  batch_size: 8\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: [base.yaml]\ntrain:\n  epochs: 99\n", encoding="utf-8"
    )
    cfg = load_experiment_config(tmp_path / "child.yaml")
    assert cfg.train.epochs == 99  # child overrides
    assert cfg.train.batch_size == 8  # base preserved via deep merge
    assert cfg.train.lr == pytest.approx(3e-4)  # untouched default


def test_extends_list_overwrites(tmp_path):
    (tmp_path / "base.yaml").write_text("sensor:\n  paths: [a.dat, b.dat]\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: [base.yaml]\nsensor:\n  paths: [only.dat]\n", encoding="utf-8"
    )
    cfg = load_prepare_config(tmp_path / "child.yaml")
    assert cfg.sensor.paths == ["only.dat"]


def test_extends_string_form(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 7\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\n", encoding="utf-8")
    cfg = load_experiment_config(tmp_path / "child.yaml")
    assert cfg.seed == 7


def test_extends_cycle_raises(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: [b.yaml]\nname: a\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: [a.yaml]\nname: b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Cyclic"):
        load_experiment_config(tmp_path / "a.yaml")


# ---------------------------------------------------------------------------
# extra="forbid" strictness
# ---------------------------------------------------------------------------


def test_extra_forbid_rejects_top_level_typo(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("experiment: true\nnamee: oops\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="namee"):
        load_experiment_config(path)


def test_extra_forbid_rejects_nested_typo(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("features:\n  sett: extended\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="sett"):
        load_experiment_config(path)


def test_prepare_extra_forbid_rejects_typo(tmp_path):
    path = tmp_path / "prep.yaml"
    path.write_text("embeddings:\n  poolng: cls\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="poolng"):
        load_prepare_config(path)


# ---------------------------------------------------------------------------
# is_experiment_config dispatch helper
# ---------------------------------------------------------------------------


def test_is_experiment_config_true_from_dict():
    assert is_experiment_config({"experiment": True, "name": "v4"}) is True


def test_is_experiment_config_false_without_marker():
    assert is_experiment_config({"name": "legacy"}) is False
    assert is_experiment_config({"experiment": False}) is False


def test_is_experiment_config_from_path_resolves_extends(tmp_path):
    (tmp_path / "base.yaml").write_text("experiment: true\n", encoding="utf-8")
    (tmp_path / "top.yaml").write_text("extends: [base.yaml]\nname: v4\n", encoding="utf-8")
    assert is_experiment_config(tmp_path / "top.yaml") is True
    (tmp_path / "legacy.yaml").write_text("video:\n  start_time: '06:00'\n", encoding="utf-8")
    assert is_experiment_config(tmp_path / "legacy.yaml") is False
