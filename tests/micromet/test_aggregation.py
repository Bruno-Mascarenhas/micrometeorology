"""Tests for temporal aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors.aggregation import aggregate_to_hourly


@pytest.fixture
def sample_5min_data() -> pd.DataFrame:
    """Create synthetic 5-minute data spanning 2 hours."""
    idx = pd.date_range("2024-01-01 00:00", "2024-01-01 01:55", freq="5min")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Temp": 25.0 + rng.normal(0, 1, len(idx)),
            "precip": rng.exponential(0.1, len(idx)),
            "WD_WXT": rng.uniform(0, 360, len(idx)),
            "WS_WXT": rng.uniform(0, 5, len(idx)),
        },
        index=idx,
    )


class TestAggregateToHourly:
    def test_output_length(self, sample_5min_data):
        result = aggregate_to_hourly(sample_5min_data)
        assert len(result) == 2  # 2 hours

    def test_mean_columns(self, sample_5min_data):
        result = aggregate_to_hourly(sample_5min_data)
        assert "Temp" in result.columns
        # Mean of 12 samples should be close to 25
        assert abs(result["Temp"].iloc[0] - 25.0) < 3.0

    def test_sum_columns(self, sample_5min_data):
        result = aggregate_to_hourly(
            sample_5min_data,
            sum_columns=["precip"],
        )
        assert "precip" in result.columns
        # Sum of the first hour's 12 samples
        raw_sum = sample_5min_data["precip"][:12].sum()
        assert result["precip"].iloc[0] == pytest.approx(raw_sum, abs=0.001)

    def test_min_samples_filter(self):
        """If fewer than min_samples valid values exist, result should be NaN."""
        idx = pd.date_range("2024-01-01 00:00", "2024-01-01 00:55", freq="5min")
        data = pd.DataFrame(
            {
                "Temp": [
                    25.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            },
            index=idx,
        )
        result = aggregate_to_hourly(data, min_samples=6)
        assert np.isnan(result["Temp"].iloc[0])

    def test_wind_direction_vector_mean(self, sample_5min_data):
        result = aggregate_to_hourly(
            sample_5min_data,
            wind_dir_columns=["WD_WXT"],
        )
        assert "WD_WXT" in result.columns
        # Direction should be between 0 and 360
        for val in result["WD_WXT"].dropna():
            assert 0 <= val < 360

        # Ensure hours are distinct (no global mean flattening bug)
        dir1 = result["WD_WXT"].iloc[0]
        dir2 = result["WD_WXT"].iloc[1]
        assert abs(dir1 - dir2) > 1.0, f"Hours 1 and 2 identical: {dir1} == {dir2}"

        # The vector mean must not coincide with the arithmetic mean
        arithmetic = sample_5min_data["WD_WXT"][:12].mean()
        assert dir1 != pytest.approx(arithmetic, abs=1.0)

    def test_wind_direction_wraps_around_north(self):
        """350° and 10° in the same hour average to 0°, not to the arithmetic 180°."""
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [350.0, 10.0] * 6, "WS_WXT": [3.0] * 12}, index=idx)
        result = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "WS_WXT"},
        )
        d = float(result["WD_WXT"].iloc[0])
        assert d < 1.0 or d > 359.0, d

    def test_speed_weighted_mean_differs_from_unit_speed(self):
        """The mapped speed column weights the components; without it speeds are unit."""
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [90.0, 180.0] * 6, "WS_WXT": [10.0, 1.0] * 6}, index=idx)
        weighted = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "WS_WXT"},
        )
        unit = aggregate_to_hourly(df, min_samples=6, wind_dir_columns=["WD_WXT"])
        # The 10 m/s easterly dominates the 1 m/s southerly
        assert weighted["WD_WXT"].iloc[0] == pytest.approx(95.71, abs=0.5)
        # Unit speeds give the plain bisector of 090 and 180
        assert unit["WD_WXT"].iloc[0] == pytest.approx(135.0, abs=0.5)

    def test_unit_speed_fallback_when_mapped_speed_column_is_absent(self):
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [90.0, 180.0] * 6, "WS_WXT": [10.0, 1.0] * 6}, index=idx)
        result = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "MISSING"},
        )
        assert result["WD_WXT"].iloc[0] == pytest.approx(135.0, abs=0.5)
