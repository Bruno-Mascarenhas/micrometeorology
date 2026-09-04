"""Tests for allsky.clearsky (Haurwitz clear-sky GHI and clear-sky index k*)."""

import math

import numpy as np
import pandas as pd
import pytest

from allsky import clearsky
from labmim_core import solar
from labmim_core.site import SiteConfig


@pytest.fixture
def site() -> SiteConfig:
    """Default site: LabMiM/UFBA, Salvador-BA (lat -13.0055, lon -38.5089)."""
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

    def test_the_clear_noon_peak_near_a_zenith_crossing_sits_between_1000_and_1100_w_m2(
        self, site: SiteConfig
    ):
        day = pd.date_range("2025-02-15 05:00", "2025-02-15 19:00", freq="1min")
        ghi = clearsky.haurwitz_ghi(day, site)
        assert 1000.0 < ghi.max() < 1100.0

    @pytest.mark.parametrize("cos_zenith_value", [1.0, 0.5, 0.25])
    def test_the_model_is_the_published_one_term_fit(self, cos_zenith_value: float):
        """``GHI_cs = 1098 * cos(z) * exp(-0.057 / cos(z))`` — Haurwitz (1945).

        Written out here so the two coefficients are pinned against arithmetic:
        every other test in this file feeds the function's own output back in or
        checks a 100 W/m2 window, and both survive a different one-term fit.
        """
        expected = 1098.0 * cos_zenith_value * math.exp(-0.057 / cos_zenith_value)

        modelled = clearsky.haurwitz_ghi_from_cos_zenith(np.array([cos_zenith_value]))

        assert modelled[0] == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("cos_zenith_value", [0.0, -0.3])
    def test_the_sun_below_the_horizon_is_exactly_zero_not_extrapolated(
        self, cos_zenith_value: float
    ):
        """Dividing by a cosine at or below zero is the trap the clamp exists for,
        and zero is a physical irradiance, not a missing one.
        """
        assert clearsky.haurwitz_ghi_from_cos_zenith(np.array([cos_zenith_value]))[0] == 0.0

    def test_tz_aware_timestamps_rejected(self, site: SiteConfig):
        times = pd.date_range("2025-06-25", periods=3, freq="1h", tz="UTC")
        with pytest.raises(ValueError, match="naive"):
            clearsky.haurwitz_ghi(times, site)


class TestClearSkyIndex:
    def test_k_star_is_the_ratio_to_the_reference_where_it_is_defined(self, site: SiteConfig):
        """Feeding the reference back in pins the division and the elevation mask,
        not the model: ``f / f == 1`` holds for any positive ``f`` in Haurwitz's
        place, and the coefficients themselves are pinned by the tests above.
        """
        day = pd.date_range("2025-06-25 05:00", "2025-06-25 19:00", freq="5min")
        ghi_cs = clearsky.haurwitz_ghi(day, site)
        kstar = clearsky.clear_sky_index(ghi_cs, day, site)
        defined = np.isfinite(kstar)
        assert defined.any()
        np.testing.assert_allclose(kstar[defined], 1.0, rtol=1e-9)

    def test_nan_below_elevation_threshold(self, site: SiteConfig):
        day = pd.date_range("2025-06-25 00:00", "2025-06-25 23:55", freq="5min")
        ghi_cs = clearsky.haurwitz_ghi(day, site)
        elevation = solar.solar_elevation_deg(day, site)
        kstar = clearsky.clear_sky_index(ghi_cs, day, site, min_elevation_deg=10.0)
        assert np.isnan(kstar[elevation < 10.0]).all()
        assert np.isfinite(kstar[elevation >= 10.0]).all()

    def test_kstar_is_unclipped_so_overcast_and_cloud_enhancement_both_survive(
        self, site: SiteConfig
    ):
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
