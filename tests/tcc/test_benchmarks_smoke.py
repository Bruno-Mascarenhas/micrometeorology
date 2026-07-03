"""Smoke tests keeping benchmarks/ runnable against the current APIs.

The benchmark harnesses are not exercised anywhere else, so interface drift
would rot them silently; each runs here with minimal sizes (<1s apiece).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "solrad_correction"

CASES = [
    ("loading.py", ["--rows", "200", "--features", "4"]),
    ("preprocessing.py", ["--rows", "300", "--features", "4"]),
    ("sequence_dataloader.py", ["--rows", "400", "--features", "4", "--sequence-length", "8"]),
    ("artifact_checkpoint.py", ["--hidden-size", "8", "--layers", "1"]),
]


@pytest.mark.parametrize(("script", "args"), CASES, ids=[c[0] for c in CASES])
def test_benchmark_runs_and_reports(script: str, args: list[str]) -> None:
    if script in {"sequence_dataloader.py", "artifact_checkpoint.py"}:
        pytest.importorskip("torch")
    proc = subprocess.run(
        [sys.executable, str(BENCH_DIR / script), *args],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "'benchmark':" in proc.stdout, proc.stdout[-500:]
