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
            "B": [10.0, 20.0, np.nan, 40.0, 50.0],
            "C": [np.nan, np.nan, np.nan, 4.0, 5.0],
        }
    )
    test = pd.DataFrame({"A": [100.0, 200.0], "B": [np.nan, np.nan], "C": [1.0, 2.0]})
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
        assert transformed_test["B"].iloc[0] == pytest.approx(0.0)
        expected_a = (test["A"] - train["A"].mean()) / train["A"].std()
        np.testing.assert_allclose(transformed_test["A"], expected_a)
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


@pytest.mark.parametrize("strategy", ["ffill", "interpolate"])
def test_transform_imputation_is_causal(strategy: str) -> None:
    train = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"A": [10.0, np.nan, 100.0]})
    pipeline = PreprocessingPipeline(scaler_type="none", impute_strategy=strategy)

    out = pipeline.fit(train).transform(test)

    assert out["A"].iloc[1] == 10.0


def test_sequence_targets_and_lazy_dataset_match_dense_contract() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="1h")
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    target = (np.arange(10, dtype=np.float32) * 10).astype(np.float32)

    dense_x, dense_y = create_sequences(features, target, sequence_length=3)
    lazy = WindowedSequenceDataset(features, target, sequence_length=3)
    seq_index = create_sequences_index(index, sequence_length=3)

    assert seq_index.equals(index[3:])
    assert len(lazy) == len(dense_x)
    x0, y0 = lazy[0]
    np.testing.assert_array_equal(x0.numpy(), dense_x[0])
    assert y0.item() == pytest.approx(float(dense_y[0]))
    assert lazy.target_values()[0] == pytest.approx(30.0)


def test_sequence_dataset_short_input_and_custom_target_offset_contracts() -> None:
    features = np.arange(20, dtype=np.float32).reshape(10, 2)
    target = np.arange(10, dtype=np.float32)

    dataset = WindowedSequenceDataset(features, target, sequence_length=3, target_offset=4)

    assert len(dataset) == 6
    assert dataset[0][1].item() == pytest.approx(4.0)
    with pytest.raises(ValueError, match="sequence_length"):
        WindowedSequenceDataset(features[:3], target[:3], sequence_length=3)


def test_tabular_dataset_preserves_full_prediction_index() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    dataset = TabularDataset.from_dataframe(df, ["feature"], "target")

    assert dataset.index is not None
    assert dataset.index.equals(index)


def test_model_native_policy_preserves_model_rows() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = align_test_frame(
        test_df,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="model_native",
    )

    assert selected.index.equals(index)
    assert (
        prediction_index(
            index,
            model_type="svm",
            sequence_length=3,
            evaluation_policy="model_native",
        )
        is None
    )


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

    assert selected.index.equals(index[3:])
    assert pred_index is not None
    assert pred_index.equals(index[3:])
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
