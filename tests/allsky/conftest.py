"""Collection gate and shared fixtures for the torch-backed allsky tests.

The modules listed below need torch to run: most because ``allsky.modeling`` /
``allsky.training`` / ``allsky.evaluation`` import it at module scope, and two
(``test_cli_train_dispatch``, ``test_e2e_experiment``) because their bodies
drive real CLI training runs even though ``allsky.cli`` imports torch lazily.
Gating at *collection* time lets each module keep its imports at the top of the
file rather than stranding them below a mid-module ``pytest.importorskip`` shim,
and it reproduces the shim's behaviour exactly: on a dev install without the
``allsky`` extra these modules are skipped instead of failing.

Keep the tuple in sync -- a module that needs torch but is missing here fails
collection with an ImportError (or errors at runtime), and one listed by mistake
silently stops running for anyone without the extra. One direction of that is
now enforced: ``test_torch_gate.py`` reads every module in this directory and
requires each one importing torch at module scope to be listed. The other
direction stays a judgement call -- the CLI-driving modules import torch only
through the run they launch, so they are here without saying ``import torch``.

The synthetic ``prepare-local`` inputs below keep their mp4/TOA5 writers inside
the fixture bodies, so a dev install without the ``allsky`` extra still imports
this file.
"""

import importlib.util
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

_TORCH_BACKED = (
    "test_checkpointing.py",
    "test_clearsky_index_target.py",
    "test_cli_evaluate.py",
    "test_cli_train_dispatch.py",
    "test_configs_repo.py",
    "test_e2e_experiment.py",
    "test_engine.py",
    "test_engine_audit.py",
    "test_engine_findings.py",
    "test_evaluator.py",
    "test_evaluator_findings.py",
    "test_geometry_channels.py",
    "test_losses.py",
    "test_modeling.py",
    "test_science_findings.py",
    "test_temporal_window.py",
    "test_transfer.py",
)

collect_ignore = [] if importlib.util.find_spec("torch") is not None else list(_TORCH_BACKED)


_SAFE_COLUMNS = ("AirT1_C_Avg", "DP1_C_Avg", "RH1", "BP1_mbar_Avg", "WS_ms", "WindDir")


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 8-frame 64x64 mp4 named like a real all-sky file (2026-01-01)."""
    videos = tmp_path_factory.mktemp("videos")
    path = videos / "allsky-20260101.mp4"
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(8, 64, 64, 3)).astype(np.uint8)
    iio.imwrite(path, frames, fps=25)
    return path


def _write_toa5(path: Path, columns: list[str], rows: list[tuple[str, dict[str, str]]]) -> Path:
    """Write a Campbell TOA5 file: the four header lines, then *rows*.

    Each row is ``(timestamp, {column: literal})``; a column the row omits is
    written as the bare ``NAN`` token the logger emits for a missing sample.
    """
    header = ["TIMESTAMP", *columns]
    lines = [
        '"TOA5","LBM","CR5000","0","std","prog","sig","table"',
        ",".join(f'"{name}"' for name in header),
        ",".join('"unit"' for _ in header),
        ",".join('"Avg"' for _ in header),
    ]
    for stamp, values in rows:
        lines.append(",".join([f'"{stamp}"', *(values.get(name, "NAN") for name in columns)]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def two_dat_files(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Two .dat files whose windows OVERLAP and whose column sets differ.

    The shared fixtures write one file, one day and constant values, so
    ``sensor.paths`` was never a list of more than one and the per-column merge
    ran in no test that drives the CLI. Here the later file adds a column the
    earlier one never carried and disagrees with it on a shared timestamp: the
    documented resolution keeps the earlier value where both have one and takes
    the later file's column where only it does. The earlier file also carries a
    bare ``NAN`` and both kinds of rail — one below the reader's own floor and
    one finite, which only the archive's sentinel table catches — none of which
    the constant fixture ever produced.
    """
    directory = tmp_path_factory.mktemp("two_sensors")
    early_columns = ["AirT1_C_Avg", "RH1", "CM3Up_Wm2_Avg"]
    late_columns = [*early_columns, "PSP_Wm2_Avg"]

    early = _write_toa5(
        directory / "early.dat",
        early_columns,
        [
            (
                "2026-01-01 06:00:00",
                {"AirT1_C_Avg": "25.0", "RH1": "70.0", "CM3Up_Wm2_Avg": "120.0"},
            ),
            # A missing sample, written as the logger writes it.
            ("2026-01-01 06:01:00", {"AirT1_C_Avg": "25.1", "CM3Up_Wm2_Avg": "121.0"}),
            # The 1000 degC rail: finite, so read_campbell_dat's own -900 floor
            # lets it through and only the archive's sentinel table catches it.
            (
                "2026-01-01 06:02:00",
                {"AirT1_C_Avg": "1000", "RH1": "70.2", "CM3Up_Wm2_Avg": "122.0"},
            ),
            # -7999 sits below the reader's own floor, so it arrives as a gap
            # and the later file's reading fills it.
            (
                "2026-01-01 06:03:00",
                {"AirT1_C_Avg": "-7999", "RH1": "70.9", "CM3Up_Wm2_Avg": "124.0"},
            ),
        ],
    )
    late = _write_toa5(
        directory / "late.dat",
        late_columns,
        [
            (
                "2026-01-01 06:02:00",
                {
                    "AirT1_C_Avg": "99.9",
                    "RH1": "11.1",
                    "CM3Up_Wm2_Avg": "999.0",
                    "PSP_Wm2_Avg": "30.0",
                },
            ),
            (
                "2026-01-01 06:03:00",
                {
                    "AirT1_C_Avg": "25.3",
                    "RH1": "70.3",
                    "CM3Up_Wm2_Avg": "123.0",
                    "PSP_Wm2_Avg": "31.0",
                },
            ),
        ],
    )
    return early, late


@pytest.fixture(scope="module")
def synthetic_dat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal Campbell TOA5 .dat covering 2026-01-01 06:00-06:10."""
    path = tmp_path_factory.mktemp("sensors") / "synthetic.dat"
    columns = ["TIMESTAMP", *_SAFE_COLUMNS, "CM3Up_Wm2_Avg", "PSP_Wm2_Avg"]
    header = ",".join(f'"{c}"' for c in columns)
    units = ",".join('"unit"' for _ in columns)
    process = ",".join('"Avg"' for _ in columns)
    values = {
        "AirT1_C_Avg": 25.0,
        "DP1_C_Avg": 15.0,
        "RH1": 70.0,
        "BP1_mbar_Avg": 1010.0,
        "WS_ms": 3.0,
        "WindDir": 180.0,
        "CM3Up_Wm2_Avg": 120.0,
        "PSP_Wm2_Avg": 30.0,
    }
    lines = [
        '"TOA5","LBM","CR5000","0","std","prog","sig","table"',
        header,
        units,
        process,
    ]
    for ts in pd.date_range("2026-01-01 06:00", "2026-01-01 06:10", freq="1min"):
        row = [f'"{ts:%Y-%m-%d %H:%M:%S}"', *(str(values[c]) for c in columns[1:])]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
