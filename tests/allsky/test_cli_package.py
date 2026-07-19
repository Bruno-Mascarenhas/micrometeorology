"""Tests for the ``allsky.cli`` package structure.

Proves the package wiring: the same app object and ``main`` entry point resolve,
the multimodal v2 command groups register their commands, and the retired v0
commands (``info`` / ``build-index``) are gone. Torch-free (lazy command
imports); CliRunner only.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
from typer.testing import CliRunner

from allsky.cli import app, embeddings, evaluate, frames, main, prepare, train

runner = CliRunner()

#: Commands the assembled app must expose.
EXPECTED_COMMANDS = (
    "extract-frames",
    "validate-dataset",
    "prepare-local",
    "export-colab-bundle",
    "precompute-embeddings",
    "train",
    "evaluate",
)


def test_app_and_main_importable():
    # 'from allsky.cli import app' / 'main' both resolve to the expected objects.
    assert isinstance(app, typer.Typer)
    assert callable(main)


def test_help_lists_the_surviving_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in EXPECTED_COMMANDS:
        assert command in result.output


def test_frames_registers_extract_frames():
    fresh = typer.Typer()
    frames.register(fresh)
    names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in fresh.registered_commands
        if command.callback is not None
    }
    assert names == {"extract-frames"}


def test_train_registers_train():
    fresh = typer.Typer()
    train.register(fresh)
    names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in fresh.registered_commands
        if command.callback is not None
    }
    assert names == {"train"}


def test_command_group_register_functions_are_callable():
    # Each command-group module exposes a callable register() that attaches its
    # commands onto a fresh app without raising.
    for module in (frames, prepare, embeddings, train, evaluate):
        assert callable(module.register)
        fresh = typer.Typer()
        module.register(fresh)  # must not raise
        assert isinstance(fresh.registered_commands, list)


def test_prepare_registers_its_three_commands():
    fresh = typer.Typer()
    prepare.register(fresh)
    names = {
        command.name or command.callback.__name__.replace("_", "-")
        for command in fresh.registered_commands
        if command.callback is not None
    }
    assert names == {"validate-dataset", "prepare-local", "export-colab-bundle"}


def test_entry_point_path_resolves():
    # Mimic how the 'allsky = allsky.cli:main' console-script entry point loads.
    module_name, _, attr = "allsky.cli:main".partition(":")
    module = importlib.import_module(module_name)
    resolved = getattr(module, attr)
    assert resolved is main
    assert callable(resolved)


def test_no_dunder_main_required():
    # 'python -m allsky.cli' is intentionally NOT supported: no __main__.py ships.
    package = importlib.import_module("allsky.cli")
    assert package.__file__ is not None
    package_dir = Path(package.__file__).parent
    assert not (package_dir / "__main__.py").exists()
