"""Tests for calibration application."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors.calibration import apply_calibrations, unify_sensor_columns


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create synthetic sensor data spanning 2019."""
    idx = pd.date_range("2018-06-01", "2019-06-01", freq="1h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CM3Up_Wm2_Avg": rng.uniform(100, 500, len(idx)),
            "PSP1_Wm2_Avg": rng.uniform(100, 500, len(idx)),
            "CMP21_Wm2_Avg": rng.uniform(100, 500, len(idx)),
        },
        index=idx,
    )


class TestApplyCalibrations:
    def test_multiplicative_factor(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "test",
            }
        ]
        original = sample_data["CM3Up_Wm2_Avg"].copy()
        apply_calibrations(sample_data, cals)

        # A date-only end_date is inclusive of the WHOLE boundary day: every
        # sub-daily sample of 2018-12-31 is calibrated, and the correction stops
        # cleanly at the day boundary (2019-01-01 00:00 onward is untouched).
        mask_before = sample_data.index <= pd.Timestamp("2018-12-31 23:59:59")
        mask_after = sample_data.index >= pd.Timestamp("2019-01-01")

        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_before, "CM3Up_Wm2_Avg"],
            original[mask_before] * 0.5,
        )
        # Data from the next day on should be unchanged.
        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_after, "CM3Up_Wm2_Avg"],
            original[mask_after],
        )

    def test_null_factor_sets_nan(self, sample_data):
        cals = [
            {
                "column": "CMP21_Wm2_Avg",
                "start_date": None,
                "end_date": "2019-01-01",
                "factor": None,
                "description": "sensor not installed",
            }
        ]
        apply_calibrations(sample_data, cals)
        # The whole boundary day (2019-01-01) is NaN'd, not just its midnight
        # sample; the day after is left intact.
        assert sample_data.loc["2019-01-01", "CMP21_Wm2_Avg"].isna().all()
        assert sample_data.loc["2019-01-02", "CMP21_Wm2_Avg"].notna().all()

    def test_missing_column_skipped(self, sample_data):
        cals = [
            {
                "column": "NONEXISTENT",
                "start_date": None,
                "end_date": None,
                "factor": 2.0,
                "description": "should skip",
            }
        ]
        # Should not raise
        apply_calibrations(sample_data, cals)


@pytest.fixture
def five_min_data() -> pd.DataFrame:
    """5-minute cadence spanning the 2018→2019 boundary day."""
    idx = pd.date_range("2018-12-30 00:00", "2019-01-02 23:55", freq="5min")
    return pd.DataFrame({"CM3Up_Wm2_Avg": np.full(len(idx), 100.0)}, index=idx)


class TestBoundaryDayCalibration:
    """A date-only end_date must cover every sample of the boundary day."""

    def test_factor_applies_to_all_samples_of_end_day(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": "2018-12-30",
                    "end_date": "2018-12-31",
                    "factor": 0.5,
                    "description": "boundary",
                }
            ],
        )
        end_day = five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"]
        assert len(end_day) == 288  # all 5-min samples of the day
        assert (end_day == 50.0).all()
        # Correction stops exactly at the day boundary.
        assert (five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"] == 100.0).all()

    def test_null_factor_nans_whole_end_day(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": None,
                    "end_date": "2018-12-31",
                    "factor": None,
                    "description": "invalid",
                }
            ],
        )
        assert five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"].isna().all()
        assert five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"].notna().all()

    def test_explicit_time_end_date_is_honored_exactly(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": "2018-12-31 00:00",
                    "end_date": "2018-12-31 12:00",
                    "factor": 0.5,
                    "description": "explicit",
                }
            ],
        )
        # Inclusive up to exactly 12:00; the very next sample is untouched.
        assert five_min_data.loc["2018-12-31 12:00", "CM3Up_Wm2_Avg"] == 50.0
        assert five_min_data.loc["2018-12-31 12:05", "CM3Up_Wm2_Avg"] == 100.0
        assert (five_min_data.loc["2018-12-31 00:00":"2018-12-31 12:00"] == 50.0).all().all()

    def test_unify_has_no_boundary_day_hole(self, five_min_data):
        five_min_data["B"] = 20.0
        five_min_data = five_min_data.rename(columns={"CM3Up_Wm2_Avg": "A"})
        unify_sensor_columns(
            five_min_data,
            [
                {
                    "unified_name": "U",
                    "mappings": [
                        {"column": "A", "start_date": "2018-12-30", "end_date": "2018-12-31"},
                        {"column": "B", "start_date": "2019-01-01", "end_date": "2019-01-02"},
                    ],
                }
            ],
        )
        # The abutting mappings leave no unfilled hole on the boundary day.
        assert five_min_data.loc["2018-12-31", "U"].notna().all()
        assert (five_min_data.loc["2018-12-31", "U"] == 100.0).all()
        assert (five_min_data.loc["2019-01-01", "U"] == 20.0).all()


class TestUnifySensorColumns:
    def test_basic_switch(self):
        idx = pd.date_range("2018-01-01", "2019-06-01", freq="1D")
        df = pd.DataFrame(
            {
                "sensor_A": np.ones(len(idx)) * 10,
                "sensor_B": np.ones(len(idx)) * 20,
            },
            index=idx,
        )
        switches = [
            {
                "unified_name": "unified",
                "mappings": [
                    {"column": "sensor_A", "start_date": "2018-01-01", "end_date": "2018-12-31"},
                    {"column": "sensor_B", "start_date": "2019-01-01", "end_date": "2019-06-01"},
                ],
            }
        ]
        unify_sensor_columns(df, switches)
        assert "unified" in df.columns
        assert df.loc["2018-06-01", "unified"] == 10
        assert df.loc["2019-03-01", "unified"] == 20


class TestOverlapGuard:
    """Inclusive end dates make same-day-abutting records a config error."""

    def test_same_day_abutment_raises_clear_error(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "first",
            },
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-12-31",
                "end_date": "2019-06-01",
                "factor": 0.9,
                "description": "second",
            },
        ]
        with pytest.raises(ValueError, match=r"Overlapping calibrations.*2018-12-31"):
            apply_calibrations(sample_data, cals)

    def test_next_day_abutment_is_clean(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "first",
            },
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2019-01-01",
                "end_date": "2019-06-01",
                "factor": 0.9,
                "description": "second",
            },
        ]
        original = sample_data["CM3Up_Wm2_Avg"].copy()
        apply_calibrations(sample_data, cals)
        # Whole boundary day gets the FIRST factor; next day starts the second.
        end_day = sample_data.loc["2018-12-31", "CM3Up_Wm2_Avg"]
        np.testing.assert_allclose(end_day, original.loc["2018-12-31"] * 0.5)
        next_day = sample_data.loc["2019-01-01", "CM3Up_Wm2_Avg"]
        np.testing.assert_allclose(next_day, original.loc["2019-01-01"] * 0.9)

    def test_unify_same_day_abutment_raises(self, sample_data):
        df = sample_data.rename(columns={"CM3Up_Wm2_Avg": "sensor_A", "PSP1_Wm2_Avg": "sensor_B"})
        switches = [
            {
                "unified_name": "unified",
                "mappings": [
                    {"column": "sensor_A", "start_date": "2018-06-01", "end_date": "2018-12-31"},
                    {"column": "sensor_B", "start_date": "2018-12-31", "end_date": "2019-06-01"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="Overlapping sensor-switch mappings"):
            unify_sensor_columns(df, switches)
