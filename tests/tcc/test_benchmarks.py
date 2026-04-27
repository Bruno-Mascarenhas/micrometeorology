"""Smoke tests for synthetic benchmark entry points."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("module_name", "argv"),
    [
        ("benchmarks.solrad_correction.loading", ["loading.py", "--rows", "64", "--features", "4"]),
        (
            "benchmarks.solrad_correction.preprocessing",
            ["preprocessing.py", "--rows", "64", "--features", "4"],
        ),
        (
            "benchmarks.solrad_correction.sequence_dataloader",
            [
                "sequence_dataloader.py",
                "--rows",
                "64",
                "--features",
                "4",
                "--sequence-length",
                "4",
                "--max-batches",
                "2",
            ],
        ),
        (
            "benchmarks.solrad_correction.artifact_checkpoint",
            ["artifact_checkpoint.py", "--hidden-size", "4", "--layers", "1"],
        ),
    ],
)
def test_benchmark_scripts_smoke(
    module_name: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", argv)

    module.main()

    captured = capsys.readouterr()
    assert "benchmark" in captured.out
