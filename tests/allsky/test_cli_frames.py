"""Tests for the ``extract-frames`` CLI command.

Offline: the stamped timelapses :mod:`tests.allsky._archive_fake` encodes, plus a
64 px video whose frames are too narrow for the overlay reader.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from allsky.cli import app
from tests.allsky._archive_fake import write_overlay_video

runner = CliRunner()

_STAMPS = ("20260101052923", "20260101053500", "20260101120000")
_CAPTURE_TIMES = ["2026-01-01 05:29:23", "2026-01-01 05:35:00", "2026-01-01 12:00:00"]


def _timestamps(out_dir: Path) -> list[str]:
    manifest = pd.read_parquet(out_dir / "manifest.parquet")
    return sorted(str(value) for value in manifest["timestamp"])


def test_frames_are_stamped_from_the_overlay_when_the_config_says_overlay(tmp_path: Path):
    video = write_overlay_video(tmp_path / "allsky-20260101.mp4", _STAMPS)
    out_dir = tmp_path / "frames"

    result = runner.invoke(app, ["extract-frames", str(video), "--out", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert _timestamps(out_dir) == _CAPTURE_TIMES


def test_frames_follow_the_modelled_cadence_when_the_config_asks_for_it(tmp_path: Path):
    video = write_overlay_video(tmp_path / "allsky-20260101.mp4", _STAMPS)
    config = tmp_path / "modelled.yaml"
    config.write_text("video:\n  timestamps: 'modelled'\n", encoding="utf-8")
    out_dir = tmp_path / "frames"

    result = runner.invoke(
        app, ["extract-frames", str(video), "--out", str(out_dir), "--config", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert _timestamps(out_dir) == [
        "2026-01-01 06:00:00",
        "2026-01-01 06:01:00",
        "2026-01-01 06:02:00",
    ]


def test_a_video_named_by_a_configured_date_format_is_stamped_from_the_overlay(tmp_path: Path):
    video = write_overlay_video(tmp_path / "skycam_20260101.mp4", _STAMPS)
    config = tmp_path / "named.yaml"
    config.write_text("video:\n  filename_date_format: 'skycam_%Y%m%d'\n", encoding="utf-8")
    out_dir = tmp_path / "frames"

    result = runner.invoke(
        app, ["extract-frames", str(video), "--out", str(out_dir), "--config", str(config)]
    )

    assert result.exit_code == 0, result.output
    assert _timestamps(out_dir) == _CAPTURE_TIMES


def test_a_video_whose_overlay_cannot_be_read_exits_one_with_the_reason(tmp_path: Path):
    video = tmp_path / "allsky-20260101.mp4"
    iio.imwrite(video, np.full((4, 64, 64, 3), 3, dtype=np.uint8), fps=25)

    result = runner.invoke(app, ["extract-frames", str(video), "--out", str(tmp_path / "frames")])

    assert result.exit_code == 1
    assert "px wide" in result.output
