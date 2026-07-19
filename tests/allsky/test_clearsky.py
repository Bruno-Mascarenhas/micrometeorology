"""Tests for allsky.clearsky (Haurwitz clear-sky GHI and clear-sky index k*)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allsky import clearsky, solar
from allsky.config import SiteConfig


@pytest.fixture
def site() -> SiteConfig:
    """Default site: LabMiM/UFBA, Salvador-BA (lat -13.00, lon -38.51)."""
    return SiteConfig()


class TestHaurwitzGhi:
    def test_zero_at_night(self, site: SiteConfig):
        night = pd.DatetimeIndex(["2025-06-25 00:00:00", "2025-06-25 03:00:00"])
        ghi = clearsky.haurwitz_ghi(night, site)
        assert (ghi == 0.0).all()

    def test_nonnegative_everywhere(self, site: SiteConfig):
        day = pd.date_range("2025-01-01", "2025-12-31 23:55", freq="3h")
        ghi = clearsky.haurwitz_ghi(day, site)
        assert (ghi >= 0.0).all()
        assert np.isfinite(ghi).all()

    def test_plausible_clear_noon_magnitude(self, site: SiteConfig):
        # Near a zenith-crossing day the clear-sky peak sits ~1000-1050 W/m2.
        day = pd.date_range("2025-02-15 05:00", "2025-02-15 19:00", freq="1min")
        ghi = clearsky.haurwitz_ghi(day, site)
        assert 1000.0 < ghi.max() < 1100.0

    def test_never_exceeds_amplitude(self, site: SiteConfig):
        day = pd.date_range("2025-01-01", "2025-12-31 23:55", freq="1h")
        ghi = clearsky.haurwitz_ghi(day, site)
        assert ghi.max() <= clearsky.HAURWITZ_A_WM2

    def test_tz_aware_timestamps_rejected(self, site: SiteConfig):
        times = pd.date_range("2025-06-25", periods=3, freq="1h", tz="UTC")
        with pytest.raises(ValueError, match="naive"):
            clearsky.haurwitz_ghi(times, site)


class TestClearSkyIndex:
    def test_unity_on_synthetic_clear_sky(self, site: SiteConfig):
        # Feeding the Haurwitz reference back in must yield k* == 1 wherever the
        # index is defined (sun high enough).
        day = pd.date_range("2025-06-25 05:00", "2025-06-25 19:00", freq="5min")
        ghi_cs = clearsky.haurwitz_ghi(day, site)
        kstar = clearsky.clear_sky_index(ghi_cs, day, site)
        defined = np.isfinite(kstar)
        assert defined.any()
        np.testing.assert_allclose(kstar[defined], 1.0, rtol=1e-9)

    def test_nan_below_elevation_threshold(self, site: SiteConfig):
        day = pd.date_range("2025-06-25 00:00", "2025-06-25 23:55", freq="5min")
        ghi_cs = clearsky.haurwitz_ghi(day, site)
        elevation = solar.solar_elevation(day, site)
        kstar = clearsky.clear_sky_index(ghi_cs, day, site, min_elevation_deg=10.0)
        assert np.isnan(kstar[elevation < 10.0]).all()
        assert np.isfinite(kstar[elevation >= 10.0]).all()

    def test_overcast_and_enhancement_range(self, site: SiteConfig):
        # k* is unclipped: overcast < 1, cloud enhancement > 1 both survive.
        day = pd.date_range("2025-06-25 12:00", periods=4, freq="1min")
        ghi_cs = clearsky.haurwitz_ghi(day, site)
        kstar = clearsky.clear_sky_index(np.array([0.2, 0.6, 1.0, 1.25]) * ghi_cs, day, site)
        np.testing.assert_allclose(kstar, [0.2, 0.6, 1.0, 1.25], rtol=1e-9)

    def test_nan_ghi_propagates(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 12:00:00"])
        kstar = clearsky.clear_sky_index([np.nan], times, site)
        assert np.isnan(kstar).all()

    def test_length_mismatch_raises(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 12:00:00", "2025-06-25 13:00:00"])
        with pytest.raises(ValueError, match="does not match"):
            clearsky.clear_sky_index([500.0], times, site)
