"""Evaluation row-alignment policy for experiment predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def prediction_index(
    index: pd.DatetimeIndex,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DatetimeIndex | None:
    """Return the explicit prediction index for the selected evaluation policy."""
    _ = model_type
    if evaluation_policy == "model_native":
        return None
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    return index[sequence_length:]


def align_test_frame(
    test_df: pd.DataFrame,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DataFrame:
    """Apply the selected evaluation row policy to the processed test frame."""
    if evaluation_policy == "model_native":
        return test_df
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    if model_type == "svm":
        return test_df.iloc[sequence_length:]
    return test_df
