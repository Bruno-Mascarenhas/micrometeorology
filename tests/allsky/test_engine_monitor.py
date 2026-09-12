"""Torch-gated tests for the direction of the early-stopping monitor.

The engine decides ``min`` against ``max`` from the monitor's name; the
physical-unit errors and the losses must be minimised and the sky accuracy
maximised, and a run monitored on ``val/kindex_mae`` must pick the epoch with
the smallest value in its own ``metrics.csv``.
"""

from pathlib import Path

import pandas as pd
import pytest

from allsky.config import ExperimentConfig
from allsky.training.engine import _monitor_key, _monitor_mode, run_experiment
from tests.allsky import _synthetic as synthetic


@pytest.mark.parametrize(
    ("monitor", "mode"),
    [
        ("val/kindex_mae", "min"),
        ("val/dhi_mae", "min"),
        ("val_loss", "min"),
        ("val/loss_sky", "min"),
        ("val/sky_acc", "max"),
    ],
)
def test_the_monitor_name_decides_its_direction(monitor: str, mode: str):
    assert _monitor_mode(_monitor_key(monitor)) == mode


def test_a_kindex_mae_monitor_picks_the_epoch_with_the_smallest_value(tmp_path: Path):
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    payload = synthetic.make_config(
        root,
        epochs=4,
        targets={
            "dhi": {"enabled": True, "loss": "huber"},
            "kindex": {"enabled": True, "kind": "kstar"},
        },
    ).model_dump()
    payload["train"]["early_stopping"] = {"monitor": "val/kindex_mae", "patience": 100}
    run_dir = tmp_path / "run"

    summary = run_experiment(
        ExperimentConfig.model_validate(payload),
        data_root=root,
        output_dir=run_dir,
        embedding_reader=synthetic.reader_for(manifest),
    )

    rows = pd.read_csv(run_dir / "metrics.csv")
    assert summary["best_metric"]["name"] == "kindex_mae"
    assert summary["best_metric"]["value"] == pytest.approx(rows["val_kindex_mae"].min())
    assert summary["best_metric"]["epoch"] == int(rows["val_kindex_mae"].idxmin()) + 1
