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
    FOLSOM_LOST_TIMESTAMPS_S,
    FOLSOM_SITE,
    FOLSOM_TIMESTAMP_OFFSET_S,
    folsom_manifest_kwargs,
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


def _write_frames_with_mtime_drift(
    root: Path, named_utc: pd.DatetimeIndex, drift_s: np.ndarray
) -> Path:
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
        named = pd.date_range("2014-06-01 18:00:00", periods=5, freq="1min", tz="UTC")
        drift = np.full(5, 12.0)
        _write_frames_with_mtime_drift(tmp_path, named, drift)

        frames = read_folsom_frames(tmp_path)

        expected = (named[0] + pd.Timedelta(seconds=12)).tz_convert(None) + UTC_MINUS_8
        assert frames["timestamp"].iloc[0] == expected

    def test_a_drifting_clock_keeps_every_frame_by_default(self, tmp_path: Path):
        """Measured on the extracted 2014 archive: the two clocks already differ
        by a median of 14 s and a p95 of 34 s in the opening days, and the
        published Fig. 7 has that growing to ~700 s by late 2016. A 30 s filter
        would delete most of the archive in silence, so the default keeps
        everything — the modification time IS the capture instant, so there is
        nothing to arbitrate."""
        named = pd.date_range("2014-06-01 18:00:00", periods=6, freq="1min", tz="UTC")
        _write_frames_with_mtime_drift(
            tmp_path, named, np.array([1.0, 2.0, 300.0, 3.0, -700.0, 4.0])
        )

        assert len(read_folsom_frames(tmp_path)) == 6

    def test_a_filename_without_a_stamp_is_refused_because_the_check_compares_against_it(
        self, tmp_path: Path
    ):
        named = pd.date_range("2014-06-01 18:00:00", periods=2, freq="1min", tz="UTC")
        _write_frames_with_mtime_drift(tmp_path, named, np.zeros(2))
        (tmp_path / "20140601" / "semdata.jpg").write_bytes(b"\xff\xd8\xff\xd9")

        with pytest.raises(ValueError, match="no YYYYMMDDHHMMSS stamp"):
            read_folsom_frames(tmp_path)

    def test_an_archive_extracted_with_tar_m_fails_loudly(self, tmp_path: Path):
        named = pd.date_range("2014-06-01 18:00:00", periods=4, freq="1min", tz="UTC")
        _write_frames_with_mtime_drift(tmp_path, named, np.full(4, 86_400.0))

        with pytest.raises(ValueError, match="extracted without its modification times"):
            read_folsom_frames(tmp_path)

    def test_a_directory_with_no_frames_is_refused(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="no Folsom frames"):
            read_folsom_frames(tmp_path)


class TestIrradianceInterpolation:
    def test_the_irradiance_is_evaluated_at_the_frame_instant(self, tmp_path: Path):
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))
        named = pd.date_range("2014-06-01 14:10:00", periods=3, freq="1min", tz="UTC")
        _write_frames_with_mtime_drift(tmp_path / "frames", named, np.full(3, 30.0))
        frames = read_folsom_frames(tmp_path / "frames")

        aligned = folsom_sensor_at(sensor, frames)

        assert len(aligned) == len(frames)
        assert list(aligned.index) == list(frames["timestamp"])
        assert bool(np.isfinite(aligned["ghi"]).all())

    def test_the_measured_clock_offset_is_applied(self, tmp_path: Path):
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))
        named = pd.date_range("2014-06-01 14:10:00", periods=1, freq="1min", tz="UTC")
        _write_frames_with_mtime_drift(tmp_path / "frames", named, np.zeros(1))
        frames = read_folsom_frames(tmp_path / "frames")

        aligned = folsom_sensor_at(sensor, frames)

        moment = pd.Timestamp(frames["timestamp"].iloc[0])
        neighbours = sensor["ghi"].reindex([moment, moment + pd.Timedelta("1min")], method="ffill")
        low, high = float(neighbours.iloc[0]), float(neighbours.iloc[1])
        assert low < float(aligned["ghi"].iloc[0]) < high
        assert FOLSOM_TIMESTAMP_OFFSET_S < 0


class TestFolsomSite:
    def test_the_site_carries_its_own_clock_not_the_stations(self):
        assert FOLSOM_SITE.utc_offset_hours == -8.0
        assert FOLSOM_SITE.latitude == pytest.approx(38.642)

    def test_the_lost_timestamp_threshold_only_asks_whether_the_archive_kept_its_times(self):
        assert FOLSOM_LOST_TIMESTAMPS_S >= 3600.0


class TestTheWeatherJoin:
    """The met half of the Folsom adapter was never exercised: the irradiance
    file alone leaves `bare` short of its anemometer, so the join is what makes
    the transfer dataset buildable at all."""

    @staticmethod
    def _write_weather(path: Path, periods: int = 240) -> Path:
        times = pd.date_range("2014-06-01 14:00:00", periods=periods, freq="1min")
        pd.DataFrame(
            {
                "timeStamp": times,
                "windsp": np.full(periods, 3.5),
                "winddir": np.full(periods, 190.0),
                "air_temp": np.full(periods, 24.0),
                "relhum": np.full(periods, 55.0),
                "press": np.full(periods, 1011.0),
            }
        ).to_csv(path, index=False)
        return path

    def test_the_met_columns_arrive_renamed_on_the_site_clock(self, tmp_path: Path):
        sensor = read_folsom_sensor(
            _write_irradiance(tmp_path / "irr.csv"),
            self._write_weather(tmp_path / "weather.csv"),
        )

        assert {"WS_ms", "WindDir", "AirT1_C_Avg", "RH1", "BP1_mbar_Avg"} <= set(sensor.columns)
        assert sensor["WS_ms"].iloc[0] == pytest.approx(3.5)
        assert sensor.index[0] == pd.Timestamp("2014-06-01 14:00:00") + UTC_MINUS_8

    def test_the_joined_frame_carries_every_source_the_bare_tier_needs(self, tmp_path: Path):
        """Which is the whole point of the join: without it the tier is short."""
        from allsky.features.policy import resolve_feature_set, source_column

        sensor = read_folsom_sensor(
            _write_irradiance(tmp_path / "irr.csv"),
            self._write_weather(tmp_path / "weather.csv"),
        )
        needed = {
            column
            for name in resolve_feature_set("bare")
            if (column := source_column(name)) is not None
        }

        assert needed <= set(sensor.columns)

    def test_a_weather_file_the_run_does_not_have_leaves_the_columns_absent(self, tmp_path: Path):
        sensor = read_folsom_sensor(_write_irradiance(tmp_path / "irr.csv"))

        assert "WS_ms" not in sensor.columns


def test_the_folsom_manifest_arguments_name_its_own_site_and_id_prefix():
    """`folsom_manifest_kwargs` has no caller in the repository, so nothing else
    would notice it drifting from the format the frames are actually named in —
    and a sample_id without the prefix could be mistaken for one of the
    station's in a transfer workflow."""
    kwargs = folsom_manifest_kwargs()

    assert kwargs["site"] is FOLSOM_SITE
    assert kwargs["ghi_column"] == "ghi"
    assert kwargs["diffuse_column"] == "dhi"
    assert kwargs["feature_set"] == "bare"
    # Seconds, because the frames are stamped by modification time and land
    # anywhere inside the minute.
    assert kwargs["sample_id_format"] == "folsom-%Y%m%d-%H%M%S"
    assert "%S" in kwargs["sample_id_format"]
