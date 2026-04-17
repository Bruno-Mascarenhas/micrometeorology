"""Temporal feature extraction: calendar features and cyclic encoding."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from the DatetimeIndex.

    Adds ``hour``, ``day_of_year``, ``month``, ``weekday``.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")

    out["hour"] = out.index.hour
    out["day_of_year"] = out.index.dayofyear
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday
    return out


def add_cyclic_encoding(
    df: pd.DataFrame,
    column: str,
    period: float,
) -> pd.DataFrame:
    """Encode a column as sin/cos pair for cyclical representation.

    Parameters
    ----------
    column:
        Column to encode (e.g. ``"hour"``).
    period:
        The natural period (e.g. 24 for hours, 365 for day_of_year).
    """
    values = 2 * np.pi * df[column] / period
    new_cols = {
        f"{column}_sin": np.sin(values),
        f"{column}_cos": np.cos(values),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def add_all_cyclic_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for standard temporal features.

    Expects columns: ``hour``, ``day_of_year``, ``month``.
    Call ``add_temporal_features()`` first.
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
