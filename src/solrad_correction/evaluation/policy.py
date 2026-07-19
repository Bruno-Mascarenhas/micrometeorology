"""Evaluation row-alignment policy for experiment predictions."""

from __future__ import annotations

import pandas as pd

from solrad_correction.models.registry import get_model_spec


def prediction_index(
    index: pd.DatetimeIndex,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DatetimeIndex:
    """Return the prediction timestamps aligned with the model's evaluated rows.

    Sequence windows cover rows ``[i, i + sequence_length)`` and predict the
    target at the window's last row, so window targets start at position
    ``sequence_length - 1``. Under ``model_native`` tabular models keep the
    full processed index; predictions therefore always carry timestamps.
    """
    if evaluation_policy == "model_native":
        if get_model_spec(model_type).kind == "sequence":
            return index[sequence_length - 1 :]
        return index
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    return index[sequence_length - 1 :]


def align_test_frame(
    test_df: pd.DataFrame,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DataFrame:
    """Apply the selected evaluation row policy to the processed test frame.

    Under ``common_sequence_horizon`` tabular rows are trimmed to the sequence
    target rows, which start at position ``sequence_length - 1`` (the last row
    inside the first window).
    """
    if evaluation_policy == "model_native":
        return test_df
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    if get_model_spec(model_type).kind == "tabular":
        return test_df.iloc[sequence_length - 1 :]
    return test_df
