"""Feature engineering: lags, rolling statistics, differences."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of specified columns.

    Creates columns named ``{col}_lag_{n}`` for each column and lag value.
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    aggs: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling statistics for specified columns.

    Creates columns named ``{col}_roll_{agg}_{window}`` for each combination.
    """
    if aggs is None:
        aggs = ["mean", "std"]

    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        for window in windows:
            roller = out[col].rolling(window, min_periods=1)
            for agg in aggs:
                out[f"{col}_roll_{agg}_{window}"] = getattr(roller, agg)()
    return out


def add_diff_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: int = 1,
) -> pd.DataFrame:
    """Add first-difference features."""
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            continue
        out[f"{col}_diff_{periods}"] = out[col].diff(periods)
    return out
