"""Tests for explicit experiment evaluation row policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solrad_correction.experiments.runner import (
    _prediction_index_for_policy,
    _test_frame_for_policy,
)


def test_model_native_policy_preserves_existing_svm_test_rows():
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = _test_frame_for_policy(
        test_df,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="model_native",
    )
    prediction_index = _prediction_index_for_policy(
        test_df.index,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="model_native",
    )

    assert selected.index.equals(index)
    assert prediction_index is None


def test_common_sequence_horizon_policy_aligns_svm_to_sequence_targets():
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    selected = _test_frame_for_policy(
        test_df,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="common_sequence_horizon",
    )
    prediction_index = _prediction_index_for_policy(
        test_df.index,
        model_type="svm",
        sequence_length=3,
        evaluation_policy="common_sequence_horizon",
    )

    assert selected.index.equals(index[3:])
    assert prediction_index is not None
    assert prediction_index.equals(index[3:])


def test_unknown_evaluation_policy_raises():
    index = pd.date_range("2024-01-01", periods=8, freq="1h")
    test_df = pd.DataFrame({"feature": np.arange(8), "target": np.arange(8)}, index=index)

    with pytest.raises(ValueError, match="Unknown evaluation_policy"):
        _test_frame_for_policy(
            test_df,
            model_type="svm",
            sequence_length=3,
            evaluation_policy="surprise",
        )
