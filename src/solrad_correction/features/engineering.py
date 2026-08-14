"""Feature engineering: lags, rolling statistics, differences."""

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Add lagged versions of specified columns.

    Parameters
    ----------
    df:
        Frame indexed in chronological order; lags are taken along that order.
    columns:
        Base columns to lag. Names absent from ``df`` are skipped.
    lags:
        Positive step counts. A lag of ``n`` reads the value ``n`` rows back, so
        only past rows are visible and no future value leaks into a feature.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with one ``{col}_lag_{n}`` column per (column, lag)
        pair, in the units of the source column. The first ``n`` rows of each
        added column are NaN.
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

    Parameters
    ----------
    df:
        Frame indexed in chronological order; windows are taken along that order.
    columns:
        Base columns to aggregate. Names absent from ``df`` are skipped.
    windows:
        Window widths in rows.
    aggs:
        ``pandas.core.window.Rolling`` method names; defaults to
        ``["mean", "std"]``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with one ``{col}_roll_{agg}_{window}`` column per
        (column, window, agg) combination. The windows are trailing and use
        ``min_periods=1``, so a partial window at the start of the series is
        still aggregated (``std`` is the exception: its first row is NaN, having
        one observation) and every value includes the current row — which is why
        rolling the target column onto itself is rejected by
        ``ExperimentConfig.validate``.
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

    Parameters
    ----------
    df:
        Frame indexed in chronological order; differences are taken along that
        order.
    columns:
        Base columns to difference. Names absent from ``df`` are skipped.
    periods:
        Row distance of the backward difference.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with one ``{col}_diff_{periods}`` column per input
        column, in the units of the source column per ``periods`` rows. The
        first ``periods`` rows of each added column are NaN, and the difference
        reads the current row — which is why differencing the target column is
        rejected by ``ExperimentConfig.validate``.
    """
    derived = {}
    for column in columns:
        if column not in df.columns:
            continue
        derived[f"{column}_diff_{periods}"] = df[column].diff(periods)

    if derived:
        return pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df.copy()
