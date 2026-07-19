"""Scientific semantics that must not drift during solrad refactors."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.data.splits import ExpandingWindowSplit, temporal_train_val_test_split
from solrad_correction.datasets.sequence import WindowedSequenceDataset
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.policy import align_test_frame, prediction_index
from solrad_correction.features.sequence import create_sequences, create_sequences_index


def test_chronological_split_sizes_order_and_no_overlap() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({"value": np.arange(100, dtype=float)}, index=index)

    train, val, test = temporal_train_val_test_split(df, 0.7, 0.15, 0.15)

    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        temporal_train_val_test_split(df, 0.5, 0.5, 0.5)


def test_expanding_window_split_keeps_validation_after_train() -> None:
    df = pd.DataFrame(
        {"value": np.arange(100, dtype=float)},
        index=pd.date_range("2024-01-01", periods=100, freq="1h"),
    )

    train_idx, val_idx = next(
        ExpandingWindowSplit(initial_train_size=50, val_size=10, step=10).split(df)
    )

    assert len(train_idx) == 50
    assert len(val_idx) == 10
    assert max(train_idx) < min(val_idx)


def test_preprocessing_uses_train_only_state_and_strict_schema() -> None:
    scratch = Path("scratch") / "test_preprocessing_contract"
    path = scratch / "preprocessing.joblib"
    train = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [10.0, 20.0, 30.0, 40.0, 50.0],  # target: authoritative, complete
            "C": [np.nan, np.nan, np.nan, 4.0, 5.0],
        }
    )
    # One test row has a feature (A) gap with a valid target; the other has a
    # MISSING target and must be excluded rather than fabricated.
    test = pd.DataFrame({"A": [np.nan, 200.0], "B": [45.0, np.nan], "C": [1.0, 2.0]})
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        pipeline = PreprocessingPipeline(
            scaler_type="standard",
            impute_strategy="mean",
            drop_na_threshold=0.5,
            feature_columns=["A"],
            target_column="B",
        )
        transformed_train = pipeline.fit_transform(train)
        transformed_test = pipeline.transform(test)

        assert "C" not in transformed_train.columns
        # The missing-target row is dropped; only the valid-target row survives.
        assert len(transformed_test) == 1
        # Its feature gap is imputed with the TRAIN mean, then standardized.
        expected_a = (train["A"].mean() - train["A"].mean()) / train["A"].std()
        assert transformed_test["A"].iloc[0] == pytest.approx(expected_a)
        # The observed target is passed through (standardized), never fabricated.
        expected_b = (45.0 - train["B"].mean()) / train["B"].std()
        assert transformed_test["B"].iloc[0] == pytest.approx(expected_b)
        with pytest.raises(ValueError, match="Input schema does not match"):
            pipeline.transform(pd.DataFrame({"A": [1.0], "B": [2.0], "extra": [3.0]}))

        pipeline.save(path)
        loaded = PreprocessingPipeline.load(path)
        pd.testing.assert_frame_equal(transformed_train, loaded.transform(train))
        recovered = loaded.inverse_transform_column(transformed_train["A"].to_numpy(), "A")
        np.testing.assert_allclose(recovered, train.loc[transformed_train.index, "A"])
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


@pytest.mark.parametrize("strategy", ["ffill", "mean", "interpolate"])
def test_target_column_is_authoritative_and_never_imputed(strategy: str) -> None:
    """Regression: the target is ground truth — imputation must not fabricate it.

    Under any non-``drop`` strategy the feature columns are filled but rows with
    a missing observed target are dropped, so metrics/val-loss are never scored
    against invented targets.
    """
    index = pd.date_range("2024-01-01", periods=6, freq="1h")
    train = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "B": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
        index=index,
    )
    pipeline = PreprocessingPipeline(
        scaler_type="none",
        impute_strategy=strategy,
        feature_columns=["A"],
        target_column="B",
    )
    pipeline.fit(train)

    # Row 2 has a missing TARGET; row 4 has an internal FEATURE gap.
    test = pd.DataFrame(
        {"A": [10.0, 20.0, 30.0, 40.0, np.nan, 60.0], "B": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]},
        index=index,
    )
    out = pipeline.transform(test)

    # The missing-target row is EXCLUDED, and no target value is fabricated.
    assert index[2] not in out.index
    assert out["B"].notna().all()
    np.testing.assert_array_equal(out["B"].to_numpy(), [1.0, 2.0, 4.0, 5.0, 6.0])
    # Feature gaps are still imputed: the surviving feature-gap row has a value.
    assert index[4] in out.index
    assert not pd.isna(out.loc[index[4], "A"])


def test_fit_transform_drops_missing_target_rows_like_transform() -> None:
    """Train split obeys the same rule: missing-target rows are dropped, not filled."""
    index = pd.date_range("2024-01-01", periods=4, freq="1h")
    train = pd.DataFrame(
        {"A": [1.0, np.nan, 3.0, 4.0], "B": [10.0, 20.0, np.nan, 40.0]}, index=index
    )
    pipeline = PreprocessingPipeline(
        scaler_type="none",
        impute_strategy="mean",
        feature_columns=["A"],
        target_column="B",
    )
    out = pipeline.fit_transform(train)

    # Row 2 (missing target) dropped; row 1 (missing feature) kept and imputed.
    assert index[2] not in out.index
    assert index[1] in out.index
    assert out["B"].notna().all()
    assert not pd.isna(out.loc[index[1], "A"])
    # fit-time row-count metadata reflects the target-aware drop.
    assert pipeline.state.row_counts["fit_output_rows"] == len(out)


def test_preprocessing_non_strict_schema_projects_to_fitted_columns() -> None:
    train = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
    pipeline = PreprocessingPipeline(
        scaler_type="none", impute_strategy="mean", strict_schema=False
    )

    out = pipeline.fit(train).transform(pd.DataFrame({"A": [np.nan], "B": [8.0], "extra": [9.0]}))

    assert list(out.columns) == ["A", "B"]
    assert out["A"].iloc[0] == 1.5


def test_minmax_inverse_transform_preserves_target_values() -> None:
    train = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
    pipeline = PreprocessingPipeline(scaler_type="minmax", impute_strategy="drop")

    transformed = pipeline.fit_transform(train)
    recovered = pipeline.inverse_transform_column(transformed["target"].to_numpy(), "target")

    np.testing.assert_allclose(recovered, train["target"])


def test_transform_ffill_imputation_is_causal() -> None:
    train = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"A": [10.0, np.nan, 100.0]})
    pipeline = PreprocessingPipeline(scaler_type="none", impute_strategy="ffill")

    out = pipeline.fit(train).transform(test)

    assert out["A"].iloc[1] == 10.0


def test_interpolate_fills_internal_gaps_time_linearly() -> None:
    """Regression for finding 14: 'interpolate' must interpolate, not alias ffill."""
    index = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 04:00"]
    )
    train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=index)
    test = pd.DataFrame({"A": [10.0, np.nan, np.nan, 40.0]}, index=index)
    pipeline = PreprocessingPipeline(scaler_type="none", impute_strategy="interpolate")

    out = pipeline.fit(train).transform(test)

    # Time-based interpolation between 10.0 @ 00:00 and 40.0 @ 04:00.
    assert out["A"].iloc[1] == pytest.approx(17.5)
    assert out["A"].iloc[2] == pytest.approx(25.0)


def test_interpolate_never_extrapolates_trailing_or_leading_nans() -> None:
    """Trailing/leading NaNs stay unfilled (no implied ffill) and are dropped."""
    index = pd.date_range("2024-01-01", periods=5, freq="1h")
    train = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
    test = pd.DataFrame({"A": [np.nan, 10.0, np.nan, 30.0, np.nan]}, index=index)
    pipeline = PreprocessingPipeline(scaler_type="none", impute_strategy="interpolate")

    out = pipeline.fit(train).transform(test)

    assert list(out.index) == list(index[1:4])
    np.testing.assert_allclose(out["A"], [10.0, 20.0, 30.0])


def test_sequence_targets_and_lazy_dataset_match_dense_contract() -> None:
    """Window rows [i, i+L) predict y[i+L-1] — the last row inside the window."""
    index = pd.date_range("2024-01-01", periods=10, freq="1h")
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    target = (np.arange(10, dtype=np.float32) * 10).astype(np.float32)

    dense_x, dense_y = create_sequences(features, target, sequence_length=3)
    lazy = WindowedSequenceDataset(features, target, sequence_length=3)
    seq_index = create_sequences_index(index, sequence_length=3)

    assert seq_index.equals(index[2:])
    assert len(dense_x) == 8  # every window whose target lies inside the data
    assert len(lazy) == len(dense_x)
    assert lazy.target_offset == 2
    x0, y0 = lazy[0]
    np.testing.assert_array_equal(x0.numpy(), dense_x[0])
    assert y0.item() == pytest.approx(float(dense_y[0]))
    # First window covers rows 0..2, so its target is concurrent with row 2.
    assert dense_y[0] == pytest.approx(20.0)
    assert lazy.target_values()[0] == pytest.approx(20.0)
    np.testing.assert_array_equal(lazy.target_values(), dense_y)


def test_sequence_dataset_short_input_and_custom_target_offset_contracts() -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    target = np.arange(10, dtype=np.float32)

    dataset = WindowedSequenceDataset(features, target, sequence_length=3, target_offset=4)

    assert len(dataset) == 6
    assert dataset[0][1].item() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="sequence_length"):
        WindowedSequenceDataset(features[:3], target[:3], sequence_length=3)


def test_windowed_dataset_drops_windows_spanning_temporal_gaps() -> None:
    """Regression for finding 8: windows must not mix discontinuous history."""
    full_index = pd.date_range("2024-01-01", periods=12, freq="1h")
    # Simulate NaN rows removed by impute_strategy=drop: rows 5 and 6 missing.
    keep = np.array([0, 1, 2, 3, 4, 7, 8, 9, 10, 11])
    index = full_index[keep]
    features = np.arange(24, dtype=np.float32).reshape(12, 2)[keep]
    target = (np.arange(12, dtype=np.float32) * 10)[keep]

    gapless = WindowedSequenceDataset(features, target, sequence_length=3)
    gapped = WindowedSequenceDataset(features, target, sequence_length=3, index=index)

    # Without an index, all 8 windows exist; the 2 that straddle the gap
    # (rows 3-4-7 and 4-7-8) must be dropped when the index is provided.
    assert len(gapless) == 8
    assert len(gapped) == 6
    pred_index = gapped.prediction_index()
    assert pred_index is not None
    for x_window, y_value in [gapped[i] for i in range(len(gapped))]:
        rows = (x_window.numpy()[:, 0] / 2).astype(int)  # feature col 0 is 2 * base row
        assert np.all(np.diff(rows) == 1)  # contiguous hourly history, no gap spanned
        assert y_value.item() == pytest.approx(rows[-1] * 10)
    np.testing.assert_array_equal(gapped.target_values(), [20.0, 30.0, 40.0, 90.0, 100.0, 110.0])
    assert pred_index.equals(index[np.array([2, 3, 4, 7, 8, 9])])

    # An explicit max_gap override can re-allow spanning the gap.
    relaxed = WindowedSequenceDataset(
        features, target, sequence_length=3, index=index, max_gap="3h"
    )
    assert len(relaxed) == 8


def test_tabular_dataset_preserves_full_prediction_index() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    dataset = TabularDataset.from_dataframe(df, ["feature"], "target")

    assert dataset.index is not None
    assert dataset.index.equals(index)


def test_model_native_policy_preserves_model_rows_and_returns_aligned_index() -> None:
    """Regression for finding 15: model_native must return timestamps, not None."""
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = align_test_frame(
        test_df,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="model_native",
    )

    assert selected.index.equals(index)
    assert prediction_index(
        index,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="model_native",
    ).equals(index)
    assert prediction_index(
        index,
        model_type="lstm",
        sequence_length=3,
        evaluation_policy="model_native",
    ).equals(index[2:])


def test_common_sequence_horizon_aligns_tabular_rows_to_sequence_targets() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = align_test_frame(
        test_df,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="common_sequence_horizon",
    )
    pred_index = prediction_index(
        index,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="common_sequence_horizon",
    )

    # Sequence window targets start at position sequence_length - 1.
    assert selected.index.equals(index[2:])
    assert pred_index is not None
    assert pred_index.equals(index[2:])
    with pytest.raises(ValueError, match="Unknown evaluation_policy"):
        align_test_frame(test_df, model_type="svm", sequence_length=3, evaluation_policy="bad")


def test_common_sequence_horizon_does_not_trim_sequence_model_test_frame() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = align_test_frame(
        test_df,
        model_type="lstm",
        sequence_length=3,
        evaluation_policy="common_sequence_horizon",
    )

    assert selected.index.equals(index)
    with pytest.raises(ValueError, match="Unknown evaluation_policy"):
        prediction_index(index, model_type="lstm", sequence_length=3, evaluation_policy="bad")
