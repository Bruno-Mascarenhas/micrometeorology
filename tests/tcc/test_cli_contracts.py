"""CLI and Colab wrapper contracts."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from solrad_correction.cli import run_experiment_cli
from solrad_correction.cli_colab import load_colab_config, run_colab_cli


@pytest.fixture
def scratch_config() -> Generator[Path, None, None]:
    scratch = Path("scratch")
    scratch.mkdir(exist_ok=True)
    path = scratch / "cli_contract.yaml"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def test_cli_validate_print_and_dry_run_config_modes(scratch_config: Path) -> None:
    scratch_config.write_text(yaml.safe_dump({"name": "cli_contract"}), encoding="utf-8")
    runner = CliRunner()

    validate = runner.invoke(
        run_experiment_cli, ["--config", str(scratch_config), "--validate-config"]
    )
    printed = runner.invoke(run_experiment_cli, ["--config", str(scratch_config), "--print-config"])
    dry_run = runner.invoke(run_experiment_cli, ["--config", str(scratch_config), "--dry-run"])

    assert validate.exit_code == 0, validate.output
    assert "Config is valid." in validate.output
    assert printed.exit_code == 0, printed.output
    assert '"name": "cli_contract"' in printed.output
    assert dry_run.exit_code == 0, dry_run.output
    assert "Dry run" in dry_run.output


def test_cli_invalid_config_reports_click_error(scratch_config: Path) -> None:
    scratch_config.write_text(
        yaml.safe_dump({"model": {"model_type": "bad_model"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        run_experiment_cli, ["--config", str(scratch_config), "--validate-config"]
    )

    assert result.exit_code != 0
    assert "Invalid experiment config" in result.output


def test_cli_runtime_overrides_show_in_print_config(scratch_config: Path) -> None:
    scratch_config.write_text(yaml.safe_dump({"name": "cli_runtime"}), encoding="utf-8")

    result = CliRunner().invoke(
        run_experiment_cli,
        [
            "--config",
            str(scratch_config),
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
    for expected in [
        '"device": "cpu"',
        '"num_workers": 0',
        '"pin_memory": false',
        '"amp": false',
        '"torch_compile": false',
        '"limit_rows": 10',
        '"profile": true',
        '"output_dir": "scratch/cli-runtime-output"',
    ]:
        assert expected in result.output


def test_cli_smoke_dry_run_does_not_need_config() -> None:
    result = CliRunner().invoke(run_experiment_cli, ["--smoke-test", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output


def test_colab_wrapper_loads_config_and_applies_runtime_overrides(scratch_config: Path) -> None:
    scratch_config.write_text(
        yaml.safe_dump({"name": "colab_base", "model": {"model_type": "lstm"}}),
        encoding="utf-8",
    )
    cfg = load_colab_config(
        config=str(scratch_config),
        name="colab_override",
        output_dir="scratch/colab-output",
        device="cpu",
        num_workers=0,
        pin_memory=False,
        amp=False,
        torch_compile=False,
        resume=None,
        limit_rows=20,
        profile=True,
    )

    assert cfg.name == "colab_override"
    assert cfg.output_dir == "scratch/colab-output"
    assert cfg.runtime.device == "cpu"
    assert cfg.runtime.num_workers == 0
    assert cfg.runtime.pin_memory is False
    assert cfg.runtime.amp is False
    assert cfg.runtime.torch_compile is False
    assert cfg.runtime.limit_rows == 20
    assert cfg.runtime.profile is True


def test_colab_click_entrypoint_prints_resolved_config(scratch_config: Path) -> None:
    scratch_config.write_text(
        yaml.safe_dump({"name": "colab_print", "model": {"model_type": "lstm"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        run_colab_cli,
        ["--config", str(scratch_config), "--print-config", "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    assert '"name": "colab_print"' in result.output
    assert '"device": "cpu"' in result.output
