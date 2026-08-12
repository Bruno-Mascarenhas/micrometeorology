"""Temporal feature extraction: calendar features and cyclic encoding."""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from the DatetimeIndex.

    Parameters
    ----------
    df:
        Frame whose index carries the observation timestamps.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with integer columns ``hour`` (0-23), ``day_of_year``
        (1-366), ``month`` (1-12) and ``weekday`` (0=Monday). The values are read
        straight off the index, so they are in whatever clock the index carries.

    Raises
    ------
    TypeError
        If the index is not a ``DatetimeIndex``.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")

    out["hour"] = out.index.hour
    out["day_of_year"] = out.index.dayofyear
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday
    return out


def add_all_cyclic_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for standard temporal features.

    Each calendar column is mapped onto the unit circle so that the last value
    of a period is adjacent to the first (hour 23 next to hour 0), which a raw
    integer column cannot express.

    Parameters
    ----------
    df:
        Frame carrying any of ``hour``, ``day_of_year`` and ``month`` — run
        :func:`add_temporal_features` first. Columns that are absent are skipped.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with a ``{col}_sin``/``{col}_cos`` pair per encoded
        column, dimensionless in ``[-1, 1]``. Periods are 24 h, 365.25 days and
        12 months; the quarter-day drift of the fixed 365.25-day year is below
        the resolution of hourly features.
    """
    new_cols = {}

    if "hour" in df.columns:
        val = 2 * np.pi * df["hour"] / 24.0
        new_cols["hour_sin"] = np.sin(val)
        new_cols["hour_cos"] = np.cos(val)
    if "day_of_year" in df.columns:
        val = 2 * np.pi * df["day_of_year"] / 365.25
        new_cols["day_of_year_sin"] = np.sin(val)
        new_cols["day_of_year_cos"] = np.cos(val)
    if "month" in df.columns:
        val = 2 * np.pi * df["month"] / 12.0
        new_cols["month_sin"] = np.sin(val)
        new_cols["month_cos"] = np.cos(val)

    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.copy()
