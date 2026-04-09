"""Tests for calibration application."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from labmim_micrometeorology.sensors.calibration import apply_calibrations, unify_sensor_columns


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

        # Data before 2019 should be halved
        mask_before = sample_data.index <= pd.Timestamp("2018-12-31")
        mask_after = sample_data.index > pd.Timestamp("2018-12-31")

        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_before, "CM3Up_Wm2_Avg"],
            original[mask_before] * 0.5,
        )
        # Data after should be unchanged
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
        mask = sample_data.index <= pd.Timestamp("2019-01-01")
        assert sample_data.loc[mask, "CMP21_Wm2_Avg"].isna().all()

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
