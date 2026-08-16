"""CLI and Colab wrapper contracts."""

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from solrad_correction.cli import app as solrad_app
from solrad_correction.cli_colab import app as colab_app
from solrad_correction.cli_colab import load_colab_config


@pytest.fixture
def scratch_config(tmp_path: Path) -> Path:
    return tmp_path / "cli_contract.yaml"


@pytest.fixture(autouse=True)
def isolated_root_logger() -> Iterator[logging.Logger]:
    """Yield the root logger, restoring it after a command reconfigures it.

    ``solrad-run`` calls ``setup_logging``, which replaces the root handlers
    with ``force=True``; restoring them keeps that out of unrelated tests.
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = root.handlers[:]
    try:
        yield root
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_cli_validate_and_dry_run_config_modes(scratch_config: Path) -> None:
    scratch_config.write_text(yaml.safe_dump({"name": "cli_contract"}), encoding="utf-8")
    runner = CliRunner()

    validate = runner.invoke(solrad_app, ["--config", str(scratch_config), "--validate-config"])
    dry_run = runner.invoke(solrad_app, ["--config", str(scratch_config), "--dry-run"])

    assert validate.exit_code == 0, validate.output
    assert "Config is valid." in validate.output
    assert dry_run.exit_code == 0, dry_run.output
    assert "Dry run" in dry_run.output


def test_cli_configures_logging_so_info_diagnostics_are_emitted(
    scratch_config: Path, isolated_root_logger: logging.Logger
) -> None:
    """solrad-run must configure logging; otherwise every INFO diagnostic is discarded."""
    scratch_config.write_text(yaml.safe_dump({"name": "cli_logging"}), encoding="utf-8")
    runner = CliRunner()
    root = isolated_root_logger

    root.setLevel(logging.WARNING)
    default_level = runner.invoke(
        solrad_app, ["--config", str(scratch_config), "--validate-config"]
    )
    assert default_level.exit_code == 0, default_level.output
    assert root.level == logging.INFO
    assert root.handlers

    root.setLevel(logging.WARNING)
    debug_level = runner.invoke(
        solrad_app,
        ["--config", str(scratch_config), "--validate-config", "--log-level", "DEBUG"],
    )
    assert debug_level.exit_code == 0, debug_level.output
    assert root.level == logging.DEBUG


def test_cli_print_config_stdout_stays_parseable_json(scratch_config: Path) -> None:
    """External cron pipes --print-config into a JSON reader; log records must not pollute it."""
    scratch_config.write_text(yaml.safe_dump({"name": "cli_json"}), encoding="utf-8")

    result = CliRunner().invoke(solrad_app, ["--config", str(scratch_config), "--print-config"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["name"] == "cli_json"


def test_cli_invalid_config_reports_error(scratch_config: Path) -> None:
    scratch_config.write_text(
        yaml.safe_dump({"model": {"model_type": "bad_model"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(solrad_app, ["--config", str(scratch_config), "--validate-config"])

    assert result.exit_code != 0
    assert "Invalid experiment config" in result.output


def test_cli_runtime_overrides_show_in_print_config(scratch_config: Path) -> None:
    scratch_config.write_text(yaml.safe_dump({"name": "cli_runtime"}), encoding="utf-8")

    result = CliRunner().invoke(
        solrad_app,
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
        f'"output_dir": "{str(Path("scratch/cli-runtime-output")).replace(chr(92), chr(92) * 2)}"',
    ]:
        assert expected in result.output


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


def test_colab_entrypoint_prints_resolved_config(scratch_config: Path) -> None:
    scratch_config.write_text(
        yaml.safe_dump({"name": "colab_print", "model": {"model_type": "lstm"}}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        colab_app,
        ["--config", str(scratch_config), "--print-config", "--device", "cpu"],
    )

    assert result.exit_code == 0, result.output
    assert '"name": "colab_print"' in result.output
    assert '"device": "cpu"' in result.output


def test_console_script_main_functions_parse_argv(
    scratch_config: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The console scripts must target main(), which routes argv through typer.

    pyproject [project.scripts] must reference ``solrad_correction.cli:main``
    and ``solrad_correction.cli_colab:main``; wiring the bare command
    functions ignores every flag.
    """
    import sys

    from solrad_correction import cli, cli_colab

    monkeypatch.setattr(sys, "argv", ["solrad-run", "--smoke-test", "--dry-run"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code in (0, None)
    assert "Dry run" in capsys.readouterr().out

    scratch_config.write_text(
        yaml.safe_dump({"name": "colab_main", "model": {"model_type": "lstm"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["solrad-colab", "--config", str(scratch_config), "--print-config", "--device", "cpu"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli_colab.main()
    assert excinfo.value.code in (0, None)
    assert '"name": "colab_main"' in capsys.readouterr().out


def test_colab_fails_fast_when_cuda_unavailable(
    scratch_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """solrad-colab with device=cuda must abort before any data is loaded."""
    torch = pytest.importorskip("torch")
    from solrad_correction.experiments import runner

    scratch_config.write_text(
        yaml.safe_dump({"name": "colab_nogpu", "model": {"model_type": "lstm"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pipeline_calls: list[object] = []
    monkeypatch.setattr(runner, "run_experiment", pipeline_calls.append)

    # Device defaults to cuda on the Colab entry point.
    result = CliRunner().invoke(colab_app, ["--config", str(scratch_config)])

    combined = result.output
    # On older click versions stderr is merged into output and .stderr raises.
    with contextlib.suppress(AttributeError, ValueError):
        combined += result.stderr

    assert result.exit_code != 0
    assert "CUDA is not available" in combined
    assert "--device cpu" in combined
    assert pipeline_calls == []
