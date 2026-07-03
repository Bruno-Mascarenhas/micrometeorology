"""Tests for the allsky CLI and the pure training helpers.

Deliberately torch-free and independent of allsky.video/sensors/dataset:
only ``allsky info`` runs end-to-end here; the heavy commands are exercised
via ``--help`` (their imports are lazy).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from allsky.cli import app
from allsky.training import resolve_device, split_days

runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("info", "extract-frames", "build-index", "train"):
        assert command in result.output


def test_info_default_config():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "frame 0 at:     06:00" in result.output
    assert "CM3Up_Wm2_Avg" in result.output
    # PSP is the live diffuse sensor (CMP21's W/m2 logger channel is
    # currently zero-filled — see SensorConfig.diffuse_column).
    assert "PSP_Wm2_Avg" in result.output
    assert '"minutes_per_frame": 1.0' in result.output


def test_info_with_config_file(tmp_path):
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "video:\n  start_time: '05:30'\n  minutes_per_frame: 2.0\n"
        "sensor:\n  diffuse_column: 'Diffuse_Wm2_Avg'\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["info", "--config", str(config_path)])
    assert result.exit_code == 0
    assert "frame 0 at:     05:30" in result.output
    assert "2 minute(s) per frame" in result.output
    assert "measured column Diffuse_Wm2_Avg" in result.output


def test_info_missing_config_fails():
    result = runner.invoke(app, ["info", "--config", "does/not/exist.yaml"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# training helpers (pure pandas — no torch)
# ---------------------------------------------------------------------------


def _make_index(days, rows_per_day: int = 5) -> pd.DataFrame:
    timestamps = [
        pd.Timestamp(day) + pd.Timedelta(hours=8, minutes=30 * row)
        for day in days
        for row in range(rows_per_day)
    ]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "frame_path": [f"frame-{i:04d}.jpg" for i in range(len(timestamps))],
        }
    )


def test_split_days_no_shared_days():
    index_df = _make_index(pd.date_range("2026-06-01", periods=10))
    train_df, val_df = split_days(index_df, val_fraction=0.3, seed=0)
    train_days = set(pd.to_datetime(train_df["timestamp"]).dt.normalize())
    val_days = set(pd.to_datetime(val_df["timestamp"]).dt.normalize())
    assert train_days.isdisjoint(val_days), "day leaked across train/val"
    assert len(val_days) == 3
    assert len(train_days) == 7
    assert len(train_df) + len(val_df) == len(index_df)


def test_split_days_keeps_whole_days_together():
    index_df = _make_index(pd.date_range("2026-06-01", periods=4), rows_per_day=7)
    train_df, val_df = split_days(index_df, val_fraction=0.25, seed=123)
    # Every day's rows land entirely on one side.
    for part in (train_df, val_df):
        counts = pd.to_datetime(part["timestamp"]).dt.normalize().value_counts()
        assert (counts == 7).all()


def test_split_days_deterministic():
    index_df = _make_index(pd.date_range("2026-06-01", periods=8))
    first_train, first_val = split_days(index_df, val_fraction=0.25, seed=42)
    second_train, second_val = split_days(index_df, val_fraction=0.25, seed=42)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_val, second_val)


def test_split_days_at_least_one_day_each_side():
    index_df = _make_index(pd.date_range("2026-06-01", periods=2))
    train_df, val_df = split_days(index_df, val_fraction=0.9, seed=0)
    assert not train_df.empty
    assert not val_df.empty


def test_split_days_single_day_raises():
    index_df = _make_index([np.datetime64("2026-06-25")])
    with pytest.raises(ValueError, match="2 distinct days"):
        split_days(index_df, val_fraction=0.2, seed=0)


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.5])
def test_split_days_bad_fraction_raises(val_fraction):
    index_df = _make_index(pd.date_range("2026-06-01", periods=5))
    with pytest.raises(ValueError, match="val_fraction"):
        split_days(index_df, val_fraction=val_fraction, seed=0)


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}
