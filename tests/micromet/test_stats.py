"""Tests for micrometeorology.stats climatology + radiation helpers.

Offline and fast: synthetic pandas objects only, no I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from micrometeorology.stats.climatology import (
    daily_totals,
    diurnal_cycle,
    monthly_means,
    seasonal_groups,
)
from micrometeorology.stats.radiation import clearness_index, diffuse_fraction


@pytest.fixture
def hourly_year() -> pd.DataFrame:
    """One full year of hourly data with a deterministic diurnal signal."""
    idx = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="1h")
    hour = idx.hour.to_numpy()
    return pd.DataFrame(
        {
            "temp": 20.0 + 5.0 * np.sin(2 * np.pi * hour / 24.0),
            "rain": np.ones(len(idx)),
        },
        index=idx,
    )


class TestDiurnalCycle:
    def test_indexed_by_hour_0_to_23(self, hourly_year):
        result = diurnal_cycle(hourly_year, columns=["temp"])
        assert list(result.index) == list(range(24))
        assert list(result.columns) == ["temp"]

    def test_recovers_the_injected_hourly_signal(self, hourly_year):
        result = diurnal_cycle(hourly_year, columns=["temp"])
        expected = 20.0 + 5.0 * np.sin(2 * np.pi * np.arange(24) / 24.0)
        np.testing.assert_allclose(result["temp"].to_numpy(), expected, atol=1e-9)

    def test_missing_columns_are_ignored(self, hourly_year):
        result = diurnal_cycle(hourly_year, columns=["temp", "does_not_exist"])
        assert list(result.columns) == ["temp"]

    def test_none_selects_every_column(self, hourly_year):
        result = diurnal_cycle(hourly_year)
        assert set(result.columns) == {"temp", "rain"}

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"temp": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            diurnal_cycle(df)


class TestMonthlyMeans:
    def test_indexed_by_month(self, hourly_year):
        result = monthly_means(hourly_year, columns=["temp"])
        assert list(result.index) == list(range(1, 13))

    def test_constant_column_mean_is_constant(self, hourly_year):
        result = monthly_means(hourly_year, columns=["rain"])
        np.testing.assert_allclose(result["rain"].to_numpy(), 1.0)


class TestSeasonalGroups:
    def test_keys_are_the_four_seasons(self, hourly_year):
        groups = seasonal_groups(hourly_year)
        assert set(groups) == {"DJF", "MAM", "JJA", "SON"}

    def test_groups_partition_the_frame(self, hourly_year):
        groups = seasonal_groups(hourly_year)
        assert sum(len(g) for g in groups.values()) == len(hourly_year)

    def test_each_group_holds_only_its_months(self, hourly_year):
        groups = seasonal_groups(hourly_year)
        jja = pd.DatetimeIndex(groups["JJA"].index)
        djf = pd.DatetimeIndex(groups["DJF"].index)
        assert set(jja.month.unique()) == {6, 7, 8}
        assert set(djf.month.unique()) == {12, 1, 2}


class TestDailyTotals:
    def test_sum_counts_all_hours(self, hourly_year):
        result = daily_totals(hourly_year, columns=["rain"], agg="sum")
        # Each full day has 24 hourly ones.
        assert result["rain"].iloc[0] == 24.0

    def test_mean_of_constant_is_the_constant(self, hourly_year):
        result = daily_totals(hourly_year, columns=["rain"], agg="mean")
        assert result["rain"].iloc[0] == 1.0


class TestClearnessIndex:
    def test_basic_ratio(self):
        ghi = pd.Series([0.0, 250.0, 500.0])
        e0 = pd.Series([1000.0, 1000.0, 1000.0])
        kt = clearness_index(ghi, e0)
        np.testing.assert_allclose(kt.to_numpy(), [0.0, 0.25, 0.5])

    def test_nonpositive_extraterrestrial_is_nan(self):
        ghi = pd.Series([100.0, 100.0])
        e0 = pd.Series([0.0, -5.0])
        kt = clearness_index(ghi, e0)
        assert kt.isna().all()

    def test_out_of_range_ratio_is_nan(self):
        # ratio 2.0 exceeds the 1.5 ceiling; negative ratio also rejected.
        ghi = pd.Series([2000.0, -100.0])
        e0 = pd.Series([1000.0, 1000.0])
        kt = clearness_index(ghi, e0)
        assert kt.isna().all()

    def test_index_is_preserved(self):
        idx = pd.date_range("2025-01-01", periods=3, freq="h")
        ghi = pd.Series([100.0, 200.0, 300.0], index=idx)
        e0 = pd.Series([1000.0, 1000.0, 1000.0], index=idx)
        kt = clearness_index(ghi, e0)
        assert kt.index.equals(idx)


class TestDiffuseFraction:
    def test_basic_ratio(self):
        dif = pd.Series([50.0, 300.0])
        ghi = pd.Series([100.0, 600.0])
        kd = diffuse_fraction(dif, ghi)
        np.testing.assert_allclose(kd.to_numpy(), [0.5, 0.5])

    def test_nonpositive_global_is_nan(self):
        dif = pd.Series([10.0, 10.0])
        ghi = pd.Series([0.0, -1.0])
        kd = diffuse_fraction(dif, ghi)
        assert kd.isna().all()

    def test_out_of_range_ratio_is_nan(self):
        dif = pd.Series([1000.0])  # ratio 2.0 > 1.5 ceiling
        ghi = pd.Series([500.0])
        kd = diffuse_fraction(dif, ghi)
        assert kd.isna().all()

    def test_valid_fraction_within_bounds_survives(self):
        dif = pd.Series([100.0, 400.0])
        ghi = pd.Series([500.0, 500.0])
        kd = diffuse_fraction(dif, ghi)
        assert kd.notna().all()
        assert ((kd >= 0) & (kd <= 1.5)).all()
