"""Climatological groupings of station time series: diurnal, monthly, seasonal.

These are descriptive-statistics helpers over a :class:`pandas.DataFrame` whose
index is a :class:`pandas.DatetimeIndex` (typically the hourly export of
``labmim-sensor-process``). They collapse a long record into the average day,
the average year, or per-season subsets — the summaries the site and reports
reuse. All functions preserve the public names operational consumers import.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "daily_totals",
    "diurnal_cycle",
    "monthly_means",
    "seasonal_groups",
]

# Meteorological seasons on the Southern-Hemisphere convention (the LabMiM
# station is at ~13 S): summer is DJF, winter is JJA.
_SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def _datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Return ``df``'s index narrowed to a :class:`~pandas.DatetimeIndex`.

    Narrowing at the boundary lets the grouping helpers use the calendar
    accessors (``.hour``, ``.month``) with a statically known type instead of
    silencing the checker.

    Raises
    ------
    TypeError
        If ``df`` is not indexed by a ``DatetimeIndex``.
    """
    index = df.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"climatology helpers require a DatetimeIndex, got {type(index).__name__}")
    return index


def _select_columns(df: pd.DataFrame, columns: list[str] | None) -> list[str]:
    """Return the requested columns that actually exist, preserving their order.

    A ``None`` request selects every column. Names absent from ``df`` are
    dropped silently so a logger change (missing sensor) never raises here.
    """
    requested = columns if columns is not None else list(df.columns)
    return [c for c in requested if c in df.columns]


def diurnal_cycle(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Mean diurnal cycle: the average value at each hour of day.

    Groups rows by ``index.hour`` and averages, giving the "average day" of the
    record — every 00:00 sample averaged together, every 01:00 together, etc.

    Parameters
    ----------
    df:
        Frame indexed by a :class:`~pandas.DatetimeIndex`.
    columns:
        Subset to summarise; ``None`` uses every column. Missing names are
        ignored.

    Returns
    -------
    pandas.DataFrame
        Indexed by hour ``0..23`` with one column per selected variable. NaNs
        are skipped per hour (``groupby`` mean semantics); an hour with no valid
        sample yields NaN.
    """
    index = _datetime_index(df)
    cols = _select_columns(df, columns)
    return df[cols].groupby(index.hour).mean()


def monthly_means(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Mean value in each calendar month, pooled across all years.

    Parameters
    ----------
    df:
        Frame indexed by a :class:`~pandas.DatetimeIndex`.
    columns:
        Subset to summarise; ``None`` uses every column. Missing names are
        ignored.

    Returns
    -------
    pandas.DataFrame
        Indexed by month ``1..12`` with one column per selected variable. NaNs
        are skipped per month; a month with no valid sample yields NaN.
    """
    index = _datetime_index(df)
    cols = _select_columns(df, columns)
    return df[cols].groupby(index.month).mean()


def seasonal_groups(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split a frame into the four meteorological seasons (Southern Hemisphere).

    Rows are assigned by calendar month: ``DJF`` (Dec-Jan-Feb, austral summer),
    ``MAM``, ``JJA`` (austral winter), ``SON``. Every input row lands in exactly
    one season, so the four subsets partition ``df`` without overlap.

    Parameters
    ----------
    df:
        Frame indexed by a :class:`~pandas.DatetimeIndex`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys ``"DJF"``, ``"MAM"``, ``"JJA"``, ``"SON"``; each value is the
        matching row subset (possibly empty), columns unchanged.
    """
    index = _datetime_index(df)
    month = index.month
    return {name: df.loc[month.isin(months)] for name, months in _SEASON_MONTHS.items()}


def daily_totals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """Resample to daily resolution with a sum or mean aggregation.

    Use ``agg="sum"`` for accumulating quantities (precipitation) and
    ``agg="mean"`` for state variables (temperature, pressure).

    Parameters
    ----------
    df:
        Frame indexed by a :class:`~pandas.DatetimeIndex`.
    columns:
        Subset to aggregate; ``None`` uses every column. Missing names are
        ignored.
    agg:
        ``"sum"`` (default) or ``"mean"``. Any other value falls back to mean.

    Returns
    -------
    pandas.DataFrame
        One row per calendar day, one column per selected variable.
    """
    cols = _select_columns(df, columns)
    grouped = df[cols].resample("1D")
    if agg == "sum":
        return grouped.sum()
    return grouped.mean()
