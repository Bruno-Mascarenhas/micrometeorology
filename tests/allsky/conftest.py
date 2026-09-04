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


# ---------------------------------------------------------------------------
# Synthetic inputs shared by the modules that drive ``prepare-local``
#
# A conftest fixture is injected by name, so a module needing these no longer
# imports another test module's fixture and then shadows the import with a
# same-named parameter (an unused-import + redefinition pair). The mp4/TOA5
# writers stay inside the fixture bodies, so a dev install without the allsky
# extra still imports this conftest.
# ---------------------------------------------------------------------------

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
