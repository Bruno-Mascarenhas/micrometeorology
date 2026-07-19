"""Tests for the Wave C1b ``allsky.cli`` package split.

Proves the module -> package conversion is behaviour-preserving: the same app
object and ``main`` entry point resolve, the four legacy commands still appear
in ``--help``, and the C2/C4 stub modules expose callable no-op ``register``
functions. Torch-free (lazy command imports); CliRunner only.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import typer
from typer.testing import CliRunner

from allsky.cli import app, embeddings, evaluate, legacy, main, prepare

runner = CliRunner()


def test_app_and_main_importable():
    # 'from allsky.cli import app' / 'main' both resolve to the expected objects.
    assert isinstance(app, typer.Typer)
    assert callable(main)


def test_help_lists_legacy_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("info", "extract-frames", "build-index", "train"):
        assert command in result.output


def test_legacy_register_adds_exactly_four_commands():
    fresh = typer.Typer()
    legacy.register(fresh)
    assert len(fresh.registered_commands) == 4
    # Resolve each command's effective name (explicit name, else the callback's
    # function name with underscores -> hyphens, as Typer derives it).
    names = set()
    for command in fresh.registered_commands:
        assert command.callback is not None
        names.add(command.name or command.callback.__name__.replace("_", "-"))
    assert names == {"info", "extract-frames", "build-index", "train"}
    # Registering onto a fresh app yields identical --help command listing.
    result = runner.invoke(fresh, ["--help"])
    assert result.exit_code == 0
    for name in ("info", "extract-frames", "build-index", "train"):
        assert name in result.output


def test_command_group_register_functions_are_callable():
    # Each command-group module exposes a callable register() that attaches its
    # commands (or is a no-op stub) onto a fresh app without raising. Filled and
    # still-stub modules both satisfy this — the counts are asserted per module
    # by their own wave's tests.
    for module in (prepare, embeddings, evaluate):
        assert callable(module.register)
        fresh = typer.Typer()
        module.register(fresh)  # must not raise
        assert isinstance(fresh.registered_commands, list)


def test_prepare_registers_its_three_commands():
    # Wave C2b fills prepare.register with the three prepare-family commands.
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
