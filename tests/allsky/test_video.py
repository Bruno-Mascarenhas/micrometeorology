"""Tests for allsky.video — frame/time mapping, streaming, extraction."""

from __future__ import annotations

import itertools
from datetime import date
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.config import AllSkyConfig, VideoConfig
from allsky.video import (
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    extract_frames,
    frame_timestamps,
    iter_frames,
    video_date,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_VIDEO = REPO_ROOT / "data" / "all-sky" / "allsky-20260625.mp4"

N_SYNTHETIC_FRAMES = 8


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 8-frame 64x64 mp4 named like a real all-sky file."""
    path = tmp_path_factory.mktemp("video") / "allsky-20260101.mp4"
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(N_SYNTHETIC_FRAMES, 64, 64, 3)).astype(np.uint8)
    iio.imwrite(path, frames, fps=25)
    return path


@pytest.fixture
def cfg() -> VideoConfig:
    return VideoConfig()


class TestVideoDate:
    def test_parses_date_from_filename(self, cfg: VideoConfig):
        assert video_date(Path("data/all-sky/allsky-20260625.mp4"), cfg) == date(2026, 6, 25)

    def test_accepts_root_config(self):
        assert video_date("allsky-20260101.mp4", AllSkyConfig()) == date(2026, 1, 1)

    def test_rejects_non_matching_name(self, cfg: VideoConfig):
        with pytest.raises(ValueError, match="does not match"):
            video_date("sky_video.mp4", cfg)


class TestFrameTimestamps:
    def test_frame0_is_start_time(self, cfg: VideoConfig):
        ts = frame_timestamps(4, date(2026, 1, 1), cfg)
        assert ts[0] == pd.Timestamp("2026-01-01 06:00")

    def test_minutes_per_frame_scales_spacing(self):
        vcfg = VideoConfig(start_time="08:30", minutes_per_frame=2.5)
        ts = frame_timestamps(3, date(2026, 1, 1), vcfg)
        assert ts[0] == pd.Timestamp("2026-01-01 08:30")
        assert ts[2] == pd.Timestamp("2026-01-01 08:35")

    def test_timestamps_are_naive(self, cfg: VideoConfig):
        ts = frame_timestamps(2, date(2026, 1, 1), cfg)
        assert ts.tz is None


class TestIterFrames:
    def test_streams_all_frames(self, synthetic_video: Path, cfg: VideoConfig):
        records = list(iter_frames(synthetic_video, cfg))
        assert [r.index for r in records] == list(range(N_SYNTHETIC_FRAMES))
        assert records[0].timestamp == pd.Timestamp("2026-01-01 06:00")
        assert records[3].timestamp == pd.Timestamp("2026-01-01 06:03")
        for record in records:
            assert record.image.shape == (64, 64, 3)
            assert record.image.dtype == np.uint8

    def test_step_skips_frames(self, synthetic_video: Path, cfg: VideoConfig):
        records = list(iter_frames(synthetic_video, cfg, step=3))
        assert [r.index for r in records] == [0, 3, 6]
        assert [r.timestamp for r in records] == [
            pd.Timestamp("2026-01-01 06:00"),
            pd.Timestamp("2026-01-01 06:03"),
            pd.Timestamp("2026-01-01 06:06"),
        ]

    def test_step_must_be_positive(self, synthetic_video: Path, cfg: VideoConfig):
        with pytest.raises(ValueError, match="step"):
            next(iter_frames(synthetic_video, cfg, step=0))

    def test_accepts_root_config(self, synthetic_video: Path):
        record = next(iter_frames(synthetic_video, AllSkyConfig(), step=1))
        assert record.index == 0
        assert record.timestamp == pd.Timestamp("2026-01-01 06:00")


class TestExtractFrames:
    def test_writes_expected_names_and_manifest(
        self, synthetic_video: Path, cfg: VideoConfig, tmp_path: Path
    ):
        out_dir = tmp_path / "frames"
        manifest = extract_frames(synthetic_video, out_dir, cfg)

        expected_names = [f"allsky-20260101-060{i}.jpg" for i in range(N_SYNTHETIC_FRAMES)]
        assert sorted(p.name for p in out_dir.glob("*.jpg")) == expected_names
        assert list(manifest.columns) == list(MANIFEST_COLUMNS)
        assert len(manifest) == N_SYNTHETIC_FRAMES
        assert manifest["index"].tolist() == list(range(N_SYNTHETIC_FRAMES))
        assert (manifest["video"] == synthetic_video.name).all()
        assert manifest["timestamp"].iloc[0] == pd.Timestamp("2026-01-01 06:00")
        for frame_path in manifest["frame_path"]:
            assert Path(frame_path).exists()

        persisted = pd.read_parquet(out_dir / MANIFEST_FILENAME)
        pd.testing.assert_frame_equal(persisted, manifest)

    def test_step_and_resize(self, synthetic_video: Path, cfg: VideoConfig, tmp_path: Path):
        manifest = extract_frames(synthetic_video, tmp_path, cfg, step=3, resize=32)
        assert manifest["index"].tolist() == [0, 3, 6]
        image = iio.imread(manifest["frame_path"].iloc[0])
        assert image.shape == (32, 32, 3)

    def test_creates_out_dir(self, synthetic_video: Path, cfg: VideoConfig, tmp_path: Path):
        out_dir = tmp_path / "a" / "b"
        extract_frames(synthetic_video, out_dir, cfg, step=4)
        assert (out_dir / MANIFEST_FILENAME).exists()


@pytest.mark.skipif(not REAL_VIDEO.exists(), reason="real all-sky video not available")
def test_iter_frames_real_video():
    """Smoke on the real 2026-06-25 timelapse: 3 frames at step=600."""
    cfg = VideoConfig()
    records = list(itertools.islice(iter_frames(REAL_VIDEO, cfg, step=600), 3))

    assert len(records) == 3
    assert [r.index for r in records] == [0, 600, 1200]
    for record in records:
        assert record.image.shape == (1080, 1920, 3)
        assert record.image.dtype == np.uint8

    spacing = pd.Timedelta(minutes=600 * cfg.minutes_per_frame)
    assert records[1].timestamp - records[0].timestamp == spacing
    assert records[2].timestamp - records[1].timestamp == spacing
    assert records[0].timestamp == pd.Timestamp("2026-06-25 06:00")
