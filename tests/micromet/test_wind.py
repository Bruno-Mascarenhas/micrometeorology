"""Tests for wind vector averaging."""

from __future__ import annotations

import numpy as np
import pytest

from labmim_micrometeorology.sensors.wind import (
    vector_mean_direction,
    wind_components,
    wind_speed_from_components,
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

    def test_array_input(self):
        speeds = np.array([1.0, 2.0, 3.0])
        dirs = np.array([0.0, 90.0, 180.0])
        u, v = wind_components(speeds, dirs)
        assert u.shape == (3,)
        assert v.shape == (3,)


class TestVectorMeanDirection:
    def test_uniform_north(self):
        """Average of all-north winds should be ~0° (or 360°)."""
        u, v = wind_components(np.ones(10), np.zeros(10))
        mean_dir = vector_mean_direction(u, v)
        assert mean_dir == pytest.approx(0.0, abs=2.0) or mean_dir == pytest.approx(360.0, abs=2.0)

    def test_wrap_around(self):
        """Averaging 350° and 10° should give ~0°, not 180°."""
        speeds = np.array([1.0, 1.0])
        dirs = np.array([350.0, 10.0])
        u, v = wind_components(speeds, dirs)
        mean_dir = vector_mean_direction(u, v)
        # Should be near 0 or 360
        assert mean_dir < 10.0 or mean_dir > 350.0

    def test_elementwise_direction(self):
        from labmim_micrometeorology.sensors.wind import wind_direction_from_components

        speeds = np.array([1.0, 1.0, 1.0])
        dirs = np.array([0.0, 90.0, 180.0])
        u, v = wind_components(speeds, dirs)
        out_dirs = wind_direction_from_components(u, v)
        assert out_dirs.shape == (3,)
        assert out_dirs[0] == pytest.approx(0.0, abs=0.1) or out_dirs[0] == pytest.approx(
            360.0, abs=0.1
        )
        assert out_dirs[1] == pytest.approx(90.0, abs=0.1)
        assert out_dirs[2] == pytest.approx(180.0, abs=0.1)


class TestWindSpeed:
    def test_known_value(self):
        u = np.array([3.0, 0.0])
        v = np.array([4.0, 5.0])
        speed = wind_speed_from_components(u, v)
        assert speed[0] == pytest.approx(5.0)
        assert speed[1] == pytest.approx(5.0)
