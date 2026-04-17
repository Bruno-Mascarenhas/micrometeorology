"""Tests for ML feature engineering."""

import numpy as np
import pandas as pd
import pytest

from solrad_correction.features.engineering import (
    add_diff_features,
    add_lag_features,
    add_rolling_features,
)
from solrad_correction.features.temporal import (
    add_all_cyclic_encodings,
    add_cyclic_encoding,
    add_temporal_features,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.DataFrame(
        {"target": np.arange(10, dtype=float), "feature": np.arange(10, 20, dtype=float)}, index=idx
    )


class TestEngineering:
    def test_add_lag_features(self, sample_df):
        df_out = add_lag_features(sample_df, ["target"], [1, 2])
        assert "target_lag_1" in df_out.columns
        assert "target_lag_2" in df_out.columns
        assert np.isnan(df_out["target_lag_1"].iloc[0])
        assert df_out["target_lag_1"].iloc[1] == 0.0
        assert np.isnan(df_out["target_lag_2"].iloc[1])
        assert df_out["target_lag_2"].iloc[2] == 0.0

    def test_add_rolling_features(self, sample_df):
        df_out = add_rolling_features(sample_df, ["target"], [3], ["mean"])
        assert "target_roll_mean_3" in df_out.columns
        # min_periods=1, so first element is just the element
        assert df_out["target_roll_mean_3"].iloc[0] == 0.0
        assert df_out["target_roll_mean_3"].iloc[2] == pytest.approx(1.0)  # mean of 0, 1, 2

    def test_add_diff_features(self, sample_df):
        df_out = add_diff_features(sample_df, ["feature"], 1)
        assert "feature_diff_1" in df_out.columns
        assert np.isnan(df_out["feature_diff_1"].iloc[0])
        assert df_out["feature_diff_1"].iloc[1] == 1.0


class TestTemporal:
    def test_add_temporal_features(self, sample_df):
        df_out = add_temporal_features(sample_df)
        assert all(c in df_out.columns for c in ["hour", "day_of_year", "month", "weekday"])
        assert df_out["hour"].iloc[0] == 0
        assert df_out["hour"].iloc[1] == 1

    def test_add_cyclic_encoding(self, sample_df):
        df_out = add_temporal_features(sample_df)
        df_out = add_cyclic_encoding(df_out, "hour", 24.0)
        assert "hour_sin" in df_out.columns
        assert "hour_cos" in df_out.columns
        # hour 0 -> sin(0) = 0, cos(0) = 1
        assert df_out["hour_sin"].iloc[0] == pytest.approx(0.0)
        assert df_out["hour_cos"].iloc[0] == pytest.approx(1.0)

    def test_add_all_cyclic_encodings(self, sample_df):
        df_out = add_temporal_features(sample_df)
        df_out = add_all_cyclic_encodings(df_out)
        expected_cols = [
            "hour_sin",
            "hour_cos",
            "day_of_year_sin",
            "day_of_year_cos",
            "month_sin",
            "month_cos",
        ]
        assert all(c in df_out.columns for c in expected_cols)
