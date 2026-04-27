"""Tests for experiment config validation and CLI config-only modes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from solrad_correction.cli import run_experiment_cli
from solrad_correction.config import ExperimentConfig, ModelConfig, SplitConfig


def test_existing_default_config_validates():
    cfg = ExperimentConfig()

    cfg.validate()


@pytest.mark.parametrize(
    ("model_config", "message"),
    [
        (ModelConfig(model_type="bogus"), "model.model_type"),
        (ModelConfig(evaluation_policy="bogus"), "model.evaluation_policy"),
        (ModelConfig(sequence_length=0), "model.sequence_length"),
        (ModelConfig(batch_size=0), "model.batch_size"),
        (ModelConfig(max_epochs=0), "model.max_epochs"),
        (ModelConfig(model_type="transformer", tf_d_model=10, tf_nhead=3), "divisible"),
    ],
)
def test_invalid_model_config_fails_clearly(model_config, message):
    cfg = ExperimentConfig(model=model_config)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_invalid_split_ratio_fails_clearly():
    cfg = ExperimentConfig(split=SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5))

    with pytest.raises(ValueError, match="split ratios"):
        cfg.validate()


def test_to_dict_contains_resolved_defaults():
    cfg = ExperimentConfig(name="resolved_test")

    resolved = cfg.to_dict()

    assert resolved["name"] == "resolved_test"
    assert resolved["model"]["model_type"] == "svm"
    assert resolved["split"]["train_ratio"] == 0.7


def test_cli_validate_config_exits_without_training():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "valid_config_cli.yaml"
    try:
        config_path.write_text(yaml.safe_dump({"name": "cli_valid"}), encoding="utf-8")

        result = CliRunner().invoke(
            run_experiment_cli, ["--config", str(config_path), "--validate-config"]
        )

        assert result.exit_code == 0, result.output
        assert "Config is valid." in result.output
    finally:
        config_path.unlink(missing_ok=True)


def test_cli_print_config_outputs_resolved_json():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "print_config_cli.yaml"
    try:
        config_path.write_text(yaml.safe_dump({"name": "cli_print"}), encoding="utf-8")

        result = CliRunner().invoke(
            run_experiment_cli, ["--config", str(config_path), "--print-config"]
        )

        assert result.exit_code == 0, result.output
        assert '"name": "cli_print"' in result.output
        assert '"model_type": "svm"' in result.output
    finally:
        config_path.unlink(missing_ok=True)


def test_cli_invalid_config_reports_click_error():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "invalid_config_cli.yaml"
    try:
        config_path.write_text(
            yaml.safe_dump({"model": {"model_type": "bad_model"}}),
            encoding="utf-8",
        )

        result = CliRunner().invoke(
            run_experiment_cli, ["--config", str(config_path), "--validate-config"]
        )

        assert result.exit_code != 0
        assert "Invalid experiment config" in result.output
    finally:
        config_path.unlink(missing_ok=True)
