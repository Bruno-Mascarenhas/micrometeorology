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
    out = df.copy()
    values = 2 * np.pi * out[column] / period
    out[f"{column}_sin"] = np.sin(values)
    out[f"{column}_cos"] = np.cos(values)
    return out


def add_all_cyclic_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic encodings for standard temporal features.

    Expects columns: ``hour``, ``day_of_year``, ``month``.
    Call ``add_temporal_features()`` first.
    """
    out = df.copy()
    if "hour" in out.columns:
        out = add_cyclic_encoding(out, "hour", 24.0)
    if "day_of_year" in out.columns:
        out = add_cyclic_encoding(out, "day_of_year", 365.25)
    if "month" in out.columns:
        out = add_cyclic_encoding(out, "month", 12.0)
    return out
