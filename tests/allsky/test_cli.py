"""Tests for the allsky CLI surface and the torch-free training helpers.

Deliberately torch-free: the heavy commands are exercised only via ``--help``
(their torch/imageio imports are lazy), and :func:`resolve_device` imports torch
lazily too.
"""

from __future__ import annotations

import subprocess
import sys

from typer.testing import CliRunner

from allsky.cli import app
from allsky.training import resolve_device

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI surface (multimodal v2 command set)
# ---------------------------------------------------------------------------

#: The full command set registered on the ``allsky`` app after the v0 retirement.
EXPECTED_COMMANDS = (
    "extract-frames",
    "validate-dataset",
    "prepare-local",
    "export-colab-bundle",
    "precompute-embeddings",
    "train",
    "evaluate",
)

#: Retired v0 commands that must NOT appear anymore.
RETIRED_COMMANDS = ("info", "build-index")


def test_help_lists_the_surviving_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_help_does_not_list_retired_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in RETIRED_COMMANDS:
        assert command not in result.output


def test_extract_frames_help_is_torch_free():
    # --help resolves without importing imageio/torch (imports are lazy).
    result = runner.invoke(app, ["extract-frames", "--help"])
    assert result.exit_code == 0
    assert "--out" in result.output


def test_train_rejects_non_experiment_config(tmp_path):
    # A config without 'experiment: true' is rejected with a clear pointer to
    # the experiment configs — the legacy SkyFusionNet path is gone.
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("train:\n  epochs: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["train", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "experiment" in result.output


def test_train_without_config_is_rejected():
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "experiment" in result.output


# ---------------------------------------------------------------------------
# training helpers (torch-free)
# ---------------------------------------------------------------------------


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
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
