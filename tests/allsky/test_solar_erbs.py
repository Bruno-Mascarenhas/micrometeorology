"""Tests for allsky.solar (NOAA solar position) and allsky.erbs (diffuse fraction)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from allsky import erbs, solar
from allsky.config import SiteConfig


@pytest.fixture
def site() -> SiteConfig:
    """Default site: LabMiM/UFBA, Salvador-BA (lat -13.00, lon -38.51)."""
    return SiteConfig()


# ---------------------------------------------------------------------------
# solar.py
# ---------------------------------------------------------------------------


class TestSolarPosition:
    def test_zenith_smaller_at_noon_than_morning(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 08:00:00", "2025-06-25 12:00:00"])
        cosz = solar.cos_zenith(times, site)
        zenith = np.rad2deg(np.arccos(cosz))
        assert zenith[1] < zenith[0]
        # Both daytime for Salvador.
        elevation = solar.solar_elevation(times, site)
        assert (elevation > 0).all()

    def test_solar_noon_near_1130_local(self, site: SiteConfig):
        # Salvador (lon -38.51) sits east of the UTC-3 meridian (-45 deg):
        # true solar noon falls ~26 min before clock noon.
        day = pd.date_range("2025-06-25 05:00", "2025-06-25 19:00", freq="1min")
        elevation = solar.solar_elevation(day, site)
        peak = day[int(np.argmax(elevation))]
        assert pd.Timestamp("2025-06-25 11:15") <= peak <= pd.Timestamp("2025-06-25 12:00")

    def test_declination_annual_range(self):
        days = pd.date_range("2025-01-01 12:00", "2025-12-31 12:00", freq="1D")
        decl_deg = np.rad2deg(solar.solar_declination(days))
        assert np.abs(decl_deg).max() <= 24.0
        # June solstice: northern declination; December: southern.
        assert decl_deg[days.get_loc(pd.Timestamp("2025-06-21 12:00"))] > 20.0
        assert decl_deg[days.get_loc(pd.Timestamp("2025-12-21 12:00"))] < -20.0

    def test_equation_of_time_range(self):
        days = pd.date_range("2025-01-01 12:00", "2025-12-31 12:00", freq="1D")
        eqtime = solar.equation_of_time(days)
        assert eqtime.min() > -15.0
        assert eqtime.max() < 17.5

    def test_eccentricity_correction_range(self):
        days = pd.date_range("2025-01-01 12:00", "2025-12-31 12:00", freq="1D")
        e0 = solar.eccentricity_correction(days)
        assert e0.min() > 0.96
        assert e0.max() < 1.04

    def test_tz_aware_timestamps_rejected(self, site: SiteConfig):
        times = pd.date_range("2025-06-25", periods=3, freq="1h", tz="UTC")
        with pytest.raises(ValueError, match="naive"):
            solar.cos_zenith(times, site)


class TestExtraterrestrialGhi:
    def test_zero_at_night_positive_at_noon(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 00:00:00", "2025-06-25 12:00:00"])
        e0h = solar.extraterrestrial_ghi(times, site)
        assert e0h[0] == 0.0
        assert e0h[1] > 500.0

    def test_never_exceeds_solar_constant_envelope(self, site: SiteConfig):
        day = pd.date_range("2025-01-01", "2025-12-31 23:55", freq="6h")
        e0h = solar.extraterrestrial_ghi(day, site)
        assert (e0h >= 0.0).all()
        assert e0h.max() <= solar.SOLAR_CONSTANT_WM2 * 1.04


class TestClearnessIndex:
    def test_kt_recovers_fraction_and_nan_at_night(self, site: SiteConfig):
        day = pd.date_range("2025-06-25 00:00", "2025-06-25 23:55", freq="5min")
        e0h = solar.extraterrestrial_ghi(day, site)
        ghi = 0.75 * e0h
        kt = solar.clearness_index(ghi, day, site)

        sun_up = e0h > 0
        assert sun_up.any()
        assert (~sun_up).any()
        np.testing.assert_allclose(kt[sun_up], 0.75, rtol=1e-9)
        assert np.isnan(kt[~sun_up]).all()
        # Physically plausible range for finite values.
        finite = kt[np.isfinite(kt)]
        assert (finite >= 0.0).all()
        assert (finite <= 1.2).all()

    def test_nan_ghi_propagates(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 12:00:00"])
        kt = solar.clearness_index([np.nan], times, site)
        assert np.isnan(kt).all()

    def test_length_mismatch_raises(self, site: SiteConfig):
        times = pd.DatetimeIndex(["2025-06-25 12:00:00", "2025-06-25 13:00:00"])
        with pytest.raises(ValueError, match="length"):
            solar.clearness_index([500.0], times, site)


# ---------------------------------------------------------------------------
# erbs.py
# ---------------------------------------------------------------------------


class TestErbsDiffuseFraction:
    def test_bounds_on_dense_grid(self):
        kt = np.linspace(0.0, 1.5, 151)
        kd = erbs.erbs_diffuse_fraction(kt)
        assert not np.isnan(kd).any()
        assert (kd >= 0.0).all()
        assert (kd <= 1.0).all()

    def test_piecewise_values(self):
        kd = erbs.erbs_diffuse_fraction(np.array([0.0, 0.1, 0.5, 0.9, 1.2]))
        assert kd[0] == pytest.approx(1.0)
        assert kd[1] == pytest.approx(1.0 - 0.09 * 0.1)  # low-kt linear piece
        assert kd[2] == pytest.approx(0.6592, abs=1e-3)  # quartic piece
        assert kd[3] == pytest.approx(0.165)  # high-kt plateau
        assert kd[4] == pytest.approx(0.165)

    def test_monotonic_nonincreasing(self):
        # Overcast (low kt) skies are diffuse-dominated; clear (high kt)
        # skies have a small diffuse fraction.
        kt = np.linspace(0.0, 1.2, 121)
        kd = erbs.erbs_diffuse_fraction(kt)
        # Strictly non-increasing until the quartic's small dip below the
        # 0.165 plateau just before kt=0.8 — a known artifact of the
        # published Erbs polynomial, not an implementation bug.
        assert (np.diff(kd[kt <= 0.78]) <= 1e-12).all()
        assert (np.diff(kd) <= 1e-3).all()
        assert kd[0] > 0.9
        assert kd[-1] == pytest.approx(0.165)

    def test_nan_safe(self):
        kd = erbs.erbs_diffuse_fraction(np.array([np.nan, 0.5, np.nan]))
        assert np.isnan(kd[0])
        assert np.isfinite(kd[1])
        assert np.isnan(kd[2])

    def test_scalar_input(self):
        kd = erbs.erbs_diffuse_fraction(0.9)
        assert kd == pytest.approx(0.165)


class TestPseudoDiffuse:
    def test_bounded_by_ghi(self):
        ghi = np.array([0.0, 200.0, 600.0, 900.0])
        kt = np.array([0.1, 0.3, 0.6, 0.85])
        dhi = erbs.pseudo_diffuse(ghi, kt)
        assert (dhi >= 0.0).all()
        assert (dhi <= ghi).all()

    def test_clear_sky_fraction(self):
        dhi = erbs.pseudo_diffuse(np.array([1000.0]), np.array([0.9]))
        assert dhi[0] == pytest.approx(165.0)

    def test_nan_propagates(self):
        dhi = erbs.pseudo_diffuse(np.array([np.nan, 500.0]), np.array([0.5, np.nan]))
        assert np.isnan(dhi).all()
