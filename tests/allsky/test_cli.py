"""Tests for the allsky CLI surface and the torch-free training helpers.

Deliberately torch-free: the heavy commands are exercised only via ``--help``
(their torch/imageio imports are lazy), and :func:`resolve_device` imports torch
lazily too.
"""

import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from allsky.cli import app
from allsky.training import resolve_device

runner = CliRunner()


RETIRED_COMMANDS = ("info", "build-index")


def test_help_does_not_list_retired_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in RETIRED_COMMANDS:
        assert command not in result.output


def test_extract_frames_help_is_torch_free():
    result = runner.invoke(app, ["extract-frames", "--help"])
    assert result.exit_code == 0
    assert "--out" in result.output


def test_train_rejects_non_experiment_config(tmp_path):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("train:\n  epochs: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "experiment" in result.output


def test_train_without_config_is_rejected():
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "experiment" in result.output


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_core_modules_import_without_torch():
    """Contract: importing the core allsky modules must not pull torch."""
    code = (
        "import sys\n"
        "import allsky.video\n"
        "import allsky.data\n"
        "import allsky.cli\n"
        "assert 'torch' not in sys.modules, 'torch was imported eagerly'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
    )
    assert result.returncode == 0, result.stderr
