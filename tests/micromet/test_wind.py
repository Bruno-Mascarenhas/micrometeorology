"""Tests for wind vector averaging."""

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors.wind import (
    wind_components,
    wind_direction_from_components,
)


class TestWindComponents:
    def test_north_wind(self):
        """Wind FROM the north (0°) should give u=0, v<0."""
        u, v = wind_components(1.0, 0.0)
        assert abs(u) < 1e-10
        assert v == pytest.approx(-1.0)

    def test_east_wind(self):
        """Wind FROM the east (90°) should give u<0, v≈0."""
        u, v = wind_components(1.0, 90.0)
        assert u == pytest.approx(-1.0)
        assert abs(v) < 1e-10

    def test_south_wind(self):
        """Wind FROM the south (180°) should give u≈0, v>0."""
        u, v = wind_components(1.0, 180.0)
        assert abs(u) < 1e-10
        assert v == pytest.approx(1.0)


class TestVectorMeanDirection:
    def test_elementwise_direction(self):
        from micrometeorology.sensors.wind import wind_direction_from_components

        speeds = np.array([1.0, 1.0, 1.0])
        dirs = np.array([0.0, 90.0, 180.0])
        u, v = wind_components(speeds, dirs)
        out_dirs = np.asarray(wind_direction_from_components(u, v))
        assert out_dirs.shape == (3,)
        assert out_dirs[0] == pytest.approx(0.0, abs=0.1) or out_dirs[0] == pytest.approx(
            360.0, abs=0.1
        )
        assert out_dirs[1] == pytest.approx(90.0, abs=0.1)
        assert out_dirs[2] == pytest.approx(180.0, abs=0.1)


class TestZeroResultantIsMissing:
    """A zero-length resultant carries no bearing, so it must not report one.

    ``arctan2(0, 0)`` is 0, which used to make the helper answer a confident
    270 degrees (or 90, depending on the sign bits) for a vector that holds no
    directional information at all.
    """

    def test_zero_resultant_is_nan(self):
        assert np.isnan(float(np.asarray(wind_direction_from_components(0.0, 0.0))))

    def test_signed_zero_resultant_is_nan(self):
        """Pins the signed-zero artifact: (-0.0, 0.0) used to answer 90, not 270."""
        for u, v in [(-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0)]:
            assert np.isnan(float(np.asarray(wind_direction_from_components(u, v))))

    def test_calm_hour_aggregates_to_missing_direction(self):
        """A stalled anemometer (12 x 0.0 m/s) used to publish WindDir = 270."""
        from micrometeorology.sensors.aggregation import aggregate_to_hourly

        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        calm = pd.DataFrame({"WindDir": [123.0] * 12, "WS_ms": [0.0] * 12}, index=idx)
        hourly = aggregate_to_hourly(
            calm,
            min_samples=6,
            wind_dir_columns=["WindDir"],
            wind_speed_column_map={"WindDir": "WS_ms"},
        )
        assert np.isnan(hourly["WindDir"].iloc[0])
        assert hourly["WS_ms"].iloc[0] == 0.0

    def test_near_cancelling_unit_weighted_hour_still_reports_a_bearing(self):
        """Pins the deliberate scope limit of the guard.

        6 x 000 deg + 6 x 180 deg leaves a round-off resultant of ~6e-17, which
        is not a bit-exact zero, so the helper still answers 90. Masking that
        case needs the window's mean scalar speed (a calm threshold in
        ``aggregate_to_hourly``), not an epsilon on the resultant, because an
        epsilon cannot tell calm from strongly variable wind that cancelled.
        """
        from micrometeorology.sensors.aggregation import aggregate_to_hourly

        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        anti = pd.DataFrame({"WindDir": [0.0, 180.0] * 6}, index=idx)
        hourly = aggregate_to_hourly(anti, min_samples=6, wind_dir_columns=["WindDir"])
        assert hourly["WindDir"].iloc[0] == pytest.approx(90.0)
