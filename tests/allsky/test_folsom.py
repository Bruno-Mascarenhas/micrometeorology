"""Tests for reading UCSD-Folsom into this project's manifest.

Everything asserted here is a defect the dataset's own users measured and
published, encoded so this project cannot walk into it: the frame time is the
file's modification time and not its name (25 W/m2 of RMSE apart), frames whose
two timestamps disagree are dropped rather than mispaired, and the irradiance is
interpolated onto the frame's own instant instead of rounded to the minute.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allsky.data.folsom import (
    FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S,
    FOLSOM_SITE,
    FOLSOM_TIMESTAMP_OFFSET_S,
    folsom_sensor_at,
    read_folsom_frames,
    read_folsom_sensor,
)

UTC_MINUS_8 = pd.Timedelta(hours=-8)


def _write_irradiance(path: Path, periods: int = 240) -> Path:
    times = pd.date_range("2014-06-01 14:00:00", periods=periods, freq="1min")
    ramp = np.linspace(0.0, 900.0, periods)
    pd.DataFrame({"timeStamp": times, "ghi": ramp, "dni": ramp * 0.6, "dhi": ramp * 0.3}).to_csv(
        path, index=False
    )
    return path


def _write_frames(root: Path, named_utc: pd.DatetimeIndex, drift_s: np.ndarray) -> Path:
    """Frames whose NAME says one time and whose mtime says that plus *drift_s*."""
    day = root / "20140601"
    day.mkdir(parents=True, exist_ok=True)
    for stamp, drift in zip(named_utc, drift_s, strict=True):
        path = day / f"{stamp:%Y%m%d_%H%M%S}.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        modified = (stamp + pd.Timedelta(seconds=float(drift))).timestamp()
        os.utime(path, (modified, modified))
    return root


class TestFolsomSensor:
    def test_the_irradiance_lands_on_the_site_clock(self, tmp_path: Path):
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))

        assert pd.DatetimeIndex(sensor.index).tz is None
        assert list(sensor.columns[:3]) == ["ghi", "dni", "dhi"]
        assert sensor.index[0] == pd.Timestamp("2014-06-01 14:00:00") + UTC_MINUS_8

    def test_a_file_missing_the_declared_columns_is_refused(self, tmp_path: Path):
        path = tmp_path / "wrong.csv"
        pd.DataFrame({"timeStamp": ["2014-01-01 00:00:00"], "ghi": [0.0]}).to_csv(path, index=False)

        with pytest.raises(ValueError, match="missing the Folsom columns"):
            read_folsom_sensor(path)


class TestFrameTimestamps:
    def test_the_frame_time_is_the_modification_time_not_the_name(self, tmp_path: Path):
        """File-name alignment costs 62.52 W/m2 of RMSE against 37.21 for
        date-modified on this very dataset. The name is an assigned label."""
        named = pd.date_range("2014-06-01 18:00:00", periods=5, freq="1min", tz="UTC")
        drift = np.full(5, 12.0)
        _write_frames(tmp_path, named, drift)

        frames = read_folsom_frames(tmp_path)

        expected = (named[0] + pd.Timedelta(seconds=12)).tz_convert(None) + UTC_MINUS_8
        assert frames["timestamp"].iloc[0] == expected

    def test_frames_whose_two_clocks_disagree_are_dropped(self, tmp_path: Path):
        """The disagreement grows to about 700 s across this dataset's three
        years, which is a clock never resynchronised, not noise."""
        named = pd.date_range("2014-06-01 18:00:00", periods=6, freq="1min", tz="UTC")
        drift = np.array([1.0, 2.0, 900.0, 3.0, -700.0, 4.0])
        _write_frames(tmp_path, named, drift)

        frames = read_folsom_frames(tmp_path)

        assert len(frames) == 4

    def test_keeping_every_frame_is_possible_but_not_the_default(self, tmp_path: Path):
        named = pd.date_range("2014-06-01 18:00:00", periods=3, freq="1min", tz="UTC")
        _write_frames(tmp_path, named, np.array([1.0, 900.0, 2.0]))

        assert len(read_folsom_frames(tmp_path, max_disagreement_s=None)) == 3
        assert len(read_folsom_frames(tmp_path)) == 2

    def test_an_archive_extracted_without_its_times_fails_loudly(self, tmp_path: Path):
        """``tar -m`` discards modification times, which would silently leave
        every frame stamped with the extraction moment."""
        named = pd.date_range("2014-06-01 18:00:00", periods=4, freq="1min", tz="UTC")
        _write_frames(tmp_path, named, np.full(4, 86_400.0))

        with pytest.raises(ValueError, match="without its modification times"):
            read_folsom_frames(tmp_path)

    def test_a_directory_with_no_frames_is_refused(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no Folsom frames"):
            read_folsom_frames(tmp_path)


class TestIrradianceInterpolation:
    def test_the_irradiance_is_evaluated_at_the_frame_instant(self, tmp_path: Path):
        """A frame taken at :30 paired with the :00 sample is half a minute off,
        and that error is largest exactly when cloud moves fastest."""
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))
        named = pd.date_range("2014-06-01 14:10:00", periods=3, freq="1min", tz="UTC")
        _write_frames(tmp_path / "frames", named, np.full(3, 30.0))
        frames = read_folsom_frames(tmp_path / "frames")

        aligned = folsom_sensor_at(sensor, frames)

        assert len(aligned) == len(frames)
        assert list(aligned.index) == list(frames["timestamp"])
        assert bool(np.isfinite(aligned["ghi"]).all())

    def test_the_measured_clock_offset_is_applied(self, tmp_path: Path):
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))
        named = pd.date_range("2014-06-01 14:10:00", periods=1, freq="1min", tz="UTC")
        _write_frames(tmp_path / "frames", named, np.zeros(1))
        frames = read_folsom_frames(tmp_path / "frames")

        aligned = folsom_sensor_at(sensor, frames)

        # The frame sits on a whole minute; with the irradiance shifted by
        # FOLSOM_TIMESTAMP_OFFSET_S it no longer coincides with a sample, so the
        # value is interpolated and lands between the two neighbours.
        moment = pd.Timestamp(frames["timestamp"].iloc[0])
        neighbours = sensor["ghi"].reindex([moment, moment + pd.Timedelta("1min")], method="ffill")
        low, high = float(neighbours.iloc[0]), float(neighbours.iloc[1])
        assert low < float(aligned["ghi"].iloc[0]) < high
        assert FOLSOM_TIMESTAMP_OFFSET_S < 0


class TestFolsomSite:
    def test_the_site_carries_its_own_clock_not_the_stations(self):
        """The offset travels with the coordinates precisely so this cannot
        compute California geometry on a Salvador clock."""
        assert FOLSOM_SITE.utc_offset_hours == -8.0
        assert FOLSOM_SITE.latitude == pytest.approx(38.642)

    def test_the_disagreement_gate_matches_the_published_one(self):
        assert FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S == 30.0
