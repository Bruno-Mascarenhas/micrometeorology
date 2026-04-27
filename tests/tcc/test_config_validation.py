"""Tests for experiment config validation and CLI config-only modes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from solrad_correction.cli import run_experiment_cli
from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    SplitConfig,
)


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


def test_invalid_data_source_format_fails_clearly():
    cfg = ExperimentConfig(data=DataConfig(source_format="xlsx"))

    with pytest.raises(ValueError, match=r"data\.source_format"):
        cfg.validate()


@pytest.mark.parametrize(
    ("runtime_config", "message"),
    [
        (RuntimeConfig(device="quantum"), "runtime.device"),
        (RuntimeConfig(num_workers=-1), "runtime.num_workers"),
        (RuntimeConfig(prefetch_factor=0), "runtime.prefetch_factor"),
        (RuntimeConfig(num_workers=0, prefetch_factor=2), "prefetch_factor"),
        (RuntimeConfig(gradient_clip=-1.0), "runtime.gradient_clip"),
        (RuntimeConfig(checkpoint_every=0), "runtime.checkpoint_every"),
        (RuntimeConfig(limit_rows=0), "runtime.limit_rows"),
    ],
)
def test_invalid_runtime_config_fails_clearly(runtime_config, message):
    cfg = ExperimentConfig(runtime=runtime_config)

    with pytest.raises(ValueError, match=message):
        cfg.validate()


def test_to_dict_contains_resolved_defaults():
    cfg = ExperimentConfig(name="resolved_test")

    resolved = cfg.to_dict()

    assert resolved["name"] == "resolved_test"
    assert resolved["model"]["model_type"] == "svm"
    assert resolved["split"]["train_ratio"] == 0.7
    assert resolved["runtime"]["device"] == "auto"


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


def test_cli_dry_run_does_not_require_data_load():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "dry_run_config_cli.yaml"
    try:
        config_path.write_text(
            yaml.safe_dump(
                {
                    "name": "cli_dry",
                    "data": {"hourly_data_path": "does-not-need-to-exist.csv"},
                }
            ),
            encoding="utf-8",
        )

        result = CliRunner().invoke(run_experiment_cli, ["--config", str(config_path), "--dry-run"])

        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output
    finally:
        config_path.unlink(missing_ok=True)


def test_cli_runtime_overrides_show_in_print_config():
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    config_path = scratch / "runtime_override_cli.yaml"
    try:
        config_path.write_text(yaml.safe_dump({"name": "cli_runtime"}), encoding="utf-8")

        result = CliRunner().invoke(
            run_experiment_cli,
            [
                "--config",
                str(config_path),
                "--print-config",
                "--device",
                "cpu",
                "--num-workers",
                "0",
                "--no-pin-memory",
                "--no-amp",
                "--no-compile",
                "--limit-rows",
                "10",
                "--profile",
                "--output-dir",
                "scratch/cli-runtime-output",
            ],
        )

        assert result.exit_code == 0, result.output
        assert '"device": "cpu"' in result.output
        assert '"num_workers": 0' in result.output
        assert '"pin_memory": false' in result.output
        assert '"amp": false' in result.output
        assert '"torch_compile": false' in result.output
        assert '"limit_rows": 10' in result.output
        assert '"profile": true' in result.output
        assert '"output_dir": "scratch/cli-runtime-output"' in result.output
    finally:
        config_path.unlink(missing_ok=True)


def test_cli_smoke_dry_run_does_not_need_config():
    result = CliRunner().invoke(run_experiment_cli, ["--smoke-test", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
