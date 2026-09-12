"""The no-change forecast, in the units the project scores it in.

Three readers publish a persistence baseline and each measures it on its own
support: the evaluator on the previous frame of the same acquisition day, the
Colab block scorer on the previous datalogger block of the day it holds, and
the model card on the logger row exactly one interval before the paired one.
The first two are one rule, the shift within a day in a caller-chosen order;
the third is a lookup by row end, which a missing row leaves empty rather than
bridging. Each caller names its unit in what it publishes; the rules live here
so no reader rebuilds one.
"""

import logging

import numpy as np
import pandas as pd

__all__ = ["previous_logger_row", "previous_same_day"]

logger = logging.getLogger(__name__)


def previous_same_day(values: np.ndarray, *, day_id: np.ndarray, order: np.ndarray) -> np.ndarray:
    """The previous value of the same day in *order*; NaN for a day's first.

    Parameters
    ----------
    values:
        ``(N,)`` observations in the caller's row order.
    day_id:
        ``(N,)`` acquisition day of each row; the shift never crosses a day,
        since the night gap between two days is not a persistence horizon.
    order:
        ``(N,)`` sortable key giving the time order within a day (a timestamp,
        a block end, a position).

    Returns
    -------
    numpy.ndarray
        ``(N,)`` float64 in the caller's row order.
    """
    ordered = pd.DataFrame(
        {
            "day_id": np.asarray(day_id),
            "order": np.asarray(order),
            "observed": np.asarray(values, dtype=np.float64),
        }
    ).sort_values(["day_id", "order"])
    shifted = ordered.groupby("day_id", sort=False)["observed"].shift(1)
    return shifted.sort_index().to_numpy(dtype=np.float64)


def previous_logger_row(
    rows: pd.DataFrame, column: str, *, interval_minutes: float, one_value_per_row: bool
) -> np.ndarray:
    """*column* on the logger row one *interval_minutes* before each row's ``row_end``, same day.

    Parameters
    ----------
    rows:
        One row per sample with ``day_id``, ``row_end`` (naive local end of
        the paired logger row) and *column*; several samples may share a row.
    column:
        The observation to look up.
    interval_minutes:
        The logger's averaging interval, the distance to the previous row.
    one_value_per_row:
        Whether *column* is one number per logger row by construction; a row
        holding several values is then logged, and its mean stands for it.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` float64, NaN where the previous row of the day is absent.
    """
    by_row = rows.groupby(["day_id", "row_end"], sort=False)[column].agg(["mean", "nunique"])
    mixed = int((by_row["nunique"] > 1).sum())
    if mixed and one_value_per_row:
        logger.warning(
            "%d paired logger row(s) hold more than one %s value (a frame paired past a missing "
            "row); the row mean stands for the row",
            mixed,
            column,
        )
    previous_ends = rows["row_end"] - pd.Timedelta(minutes=interval_minutes)
    previous = pd.MultiIndex.from_arrays([rows["day_id"], previous_ends])
    return by_row["mean"].reindex(previous).to_numpy(dtype=np.float64)
