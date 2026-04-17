"""Feature engineering: lags, rolling statistics, differences."""

from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of specified columns.

    Creates columns named ``{col}_lag_{n}`` for each column and lag value.
    """
    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            new_cols[f"{col}_lag_{lag}"] = df[col].shift(lag)

    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.copy()


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

    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            roller = df[col].rolling(window, min_periods=1)
            for agg in aggs:
                new_cols[f"{col}_roll_{agg}_{window}"] = getattr(roller, agg)()

    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.copy()


def add_diff_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: int = 1,
) -> pd.DataFrame:
    """Add first-difference features."""
    new_cols = {}
    for col in columns:
        if col not in df.columns:
            continue
        new_cols[f"{col}_diff_{periods}"] = df[col].diff(periods)

    if new_cols:
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df.copy()
