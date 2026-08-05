"""Evaluation row-alignment policy for experiment predictions."""

import numpy as np
import pandas as pd

from solrad_correction.models.registry import get_model_spec


def sequence_target_positions(
    index: pd.Index,
    *,
    sequence_length: int,
    max_gap: pd.Timedelta | str | None = None,
) -> np.ndarray | None:
    """Return the row positions a sequence model over ``index`` actually predicts.

    This mirrors ``WindowedSequenceDataset`` for the target offset the pipeline
    always uses (the last row inside the window): windows cover rows
    ``[i, i + sequence_length)``, predict row ``i + sequence_length - 1``, and
    are dropped when any step inside them exceeds ``max_gap`` — by default the
    inferred base frequency (median timestamp delta of the same frame).

    ``None`` means the index carries no usable time information, which is
    exactly when the pipeline hands the sequence dataset ``index=None`` and
    every window survives; callers must then fall back to the positional
    horizon ``sequence_length - 1``.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    window_target_offset = sequence_length - 1
    timestamps = index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    deltas = np.diff(timestamps)
    limit = float(np.median(deltas)) if max_gap is None else float(pd.Timedelta(max_gap).value)
    gap_after = np.zeros(len(index), dtype=np.int64)
    gap_after[1:] = deltas > limit
    gaps_before = np.cumsum(gap_after)
    starts = np.arange(max(len(index) - sequence_length + 1, 0))
    gapless_starts = starts[(gaps_before[starts + window_target_offset] - gaps_before[starts]) == 0]
    target_positions: np.ndarray = gapless_starts + window_target_offset
    return target_positions


def prediction_index(
    index: pd.DatetimeIndex,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
    max_gap: pd.Timedelta | str | None = None,
) -> pd.DatetimeIndex:
    """Return the prediction timestamps aligned with the model's evaluated rows.

    Sequence windows cover rows ``[i, i + sequence_length)`` and predict the
    target at the window's last row, and windows straddling a temporal gap are
    dropped, so the sequence horizon is the gap-aware set of window target rows
    — not every row from position ``sequence_length - 1`` on. Under
    ``model_native`` tabular models keep the full processed index; predictions
    therefore always carry timestamps.
    """
    if evaluation_policy not in {"model_native", "common_sequence_horizon"}:
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    if evaluation_policy == "model_native" and get_model_spec(model_type).kind == "tabular":
        return index
    target_positions = sequence_target_positions(
        index,
        sequence_length=sequence_length,
        max_gap=max_gap,
    )
    if target_positions is None:
        return index[sequence_length - 1 :]
    return index[target_positions]


def align_test_frame(
    test_df: pd.DataFrame,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
    max_gap: pd.Timedelta | str | None = None,
) -> pd.DataFrame:
    """Apply the selected evaluation row policy to the processed test frame.

    Under ``common_sequence_horizon`` tabular rows are trimmed to the rows a
    sequence model over the *same* frame would predict: the last row of every
    window that spans no temporal gap. Rows are selected positionally, so a
    timestamp repeated by the source table cannot multiply them, and the gap
    limit is inferred from the untrimmed frame both branches see.

    Raises
    ------
    ValueError
        If the policy is unknown, or if every window spans a temporal gap — the
        tabular counterpart of ``WindowedSequenceDataset``'s refusal to build a
        run with no windows, instead of writing metrics over zero rows.
    """
    if evaluation_policy == "model_native":
        return test_df
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    if get_model_spec(model_type).kind != "tabular":
        return test_df
    target_positions = sequence_target_positions(
        test_df.index,
        sequence_length=sequence_length,
        max_gap=max_gap,
    )
    if target_positions is None:
        return test_df.iloc[sequence_length - 1 :]
    if len(target_positions) == 0:
        raise ValueError(
            "No sequence windows can be generated on the test split: every window of "
            f"{sequence_length} rows spans a temporal gap, so evaluation_policy="
            "common_sequence_horizon has no rows in common with a sequence model"
        )
    return test_df.iloc[target_positions]
