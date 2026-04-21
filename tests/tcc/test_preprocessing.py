"""Tests for PreprocessingPipeline to ensure no data leakage."""

import numpy as np
import pandas as pd
import pytest

from solrad_correction.data.preprocessing import PreprocessingPipeline


@pytest.fixture
def train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, np.nan, 40.0, 50.0],  # 1 NaN (20%)
            "C": [np.nan, np.nan, np.nan, 4.0, 5.0],  # 3 NaNs (60%)
        }
    )


@pytest.fixture
def test_df() -> pd.DataFrame:
    # Notice different means and NaN counts
    return pd.DataFrame(
        {
            "A": [100.0, 200.0],
            "B": [np.nan, np.nan],
            "C": [1.0, 2.0],
        }
    )


class TestPreprocessingPipeline:
    def test_fit_transform_drops_high_na_columns(self, train_df):
        pipeline = PreprocessingPipeline(drop_na_threshold=0.5)
        out = pipeline.fit_transform(train_df)
        assert "C" not in out.columns  # 60% NaNs > 50% threshold
        assert "A" in out.columns
        assert "B" in out.columns

    def test_leakage_prevention_scaling(self, train_df, test_df):
        pipeline = PreprocessingPipeline(
            scaler_type="standard", impute_strategy="mean", drop_na_threshold=0.5
        )
        pipeline.fit_transform(train_df)

        # Test set transformation MUST use train_df's mean/std
        test_out = pipeline.transform(test_df)

        train_mean_a = train_df["A"].mean()
        train_std_a = train_df["A"].std()

        expected_test_a = (test_df["A"] - train_mean_a) / train_std_a
        np.testing.assert_array_almost_equal(test_out["A"].values, expected_test_a.values)  # type: ignore

    def test_leakage_prevention_imputation(self, train_df, test_df):
        pipeline = PreprocessingPipeline(
            scaler_type="none", impute_strategy="mean", drop_na_threshold=0.5
        )
        pipeline.fit(train_df)

        # Train mean for B is 30.0 (excluding NaN)
        # Test set has 2 NaNs for B. They should be filled with 30.0.
        test_out = pipeline.transform(test_df)
        assert test_out["B"].iloc[0] == 30.0
        assert test_out["B"].iloc[1] == 30.0

    def test_transform_unseen_data_structure(self, train_df):
        pipeline = PreprocessingPipeline(
            scaler_type="none", impute_strategy="drop", drop_na_threshold=0.5
        )
        pipeline.fit(train_df)

        # DataFrame missing column A, but having extra column D
        weird_df = pd.DataFrame({"B": [10.0], "D": [99.0]})
        out = pipeline.transform(weird_df)

        # Should only contain A and B (A will be NaN and dropped if strategy is drop)
        # Wait, if strategy is drop, and A is NaN, the whole row is dropped!
        assert len(out) == 0

        # With mean imputation, A should be filled with train mean (3.0)
        pipeline = PreprocessingPipeline(
            scaler_type="none", impute_strategy="mean", drop_na_threshold=0.5
        )
        pipeline.fit(train_df)
        out = pipeline.transform(weird_df)
        assert "D" not in out.columns
        assert out["A"].iloc[0] == 3.0
        assert out["B"].iloc[0] == 10.0

    def test_inverse_transform(self, train_df):
        pipeline = PreprocessingPipeline(scaler_type="minmax", impute_strategy="mean")
        out = pipeline.fit_transform(train_df)

        # A goes from 1.0 to 5.0
        # A mapped to 0.0 to 1.0
        assert out["A"].min() == 0.0
        assert out["A"].max() == 1.0

        recovered = pipeline.inverse_transform_column(out["A"].values, "A")  # type: ignore
        np.testing.assert_array_almost_equal(recovered, train_df.loc[out.index, "A"].values)
