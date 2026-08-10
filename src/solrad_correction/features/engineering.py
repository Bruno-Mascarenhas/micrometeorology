"""Feature engineering: lags, rolling statistics, differences."""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of specified columns.

    Creates columns named ``{col}_lag_{n}`` for each column and lag value.
    Columns absent from *df* are skipped.
    """
    derived = {}
    for column in columns:
        if column not in df.columns:
            continue
        for lag in lags:
            derived[f"{column}_lag_{lag}"] = df[column].shift(lag)

    if derived:
        return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df.copy()


def add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    aggs: list[str] | None = None,
) -> pd.DataFrame:
    """Add rolling statistics for specified columns.

    Creates columns named ``{col}_roll_{agg}_{window}`` for each combination.
    Columns absent from *df* are skipped.
    """
    if aggs is None:
        aggs = ["mean", "std"]

    derived = {}
    for column in columns:
        if column not in df.columns:
            continue
        for window in windows:
            roller = df[column].rolling(window, min_periods=1)
            for agg in aggs:
                derived[f"{column}_roll_{agg}_{window}"] = getattr(roller, agg)()

    if derived:
        return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df.copy()


def add_diff_features(
    df: pd.DataFrame,
    columns: list[str],
    periods: int = 1,
) -> pd.DataFrame:
    """Add first-difference features.

    Creates columns named ``{col}_diff_{periods}``; columns absent from *df*
    are skipped.
    """
    derived = {}
    for column in columns:
        if column not in df.columns:
            continue
        derived[f"{column}_diff_{periods}"] = df[column].diff(periods)

    if derived:
        return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df.copy()
