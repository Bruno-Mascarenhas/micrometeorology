"""Climatological analysis: diurnal, monthly, and seasonal groupings."""

from __future__ import annotations

import pandas as pd  # noqa: TC002 — used at runtime


def diurnal_cycle(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute the mean diurnal cycle (hourly averages across all days).

    Returns a DataFrame indexed by hour (0–23).
    """
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    return df[cols].groupby(df.index.hour).mean()


def monthly_means(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute monthly means.

    Returns a DataFrame indexed by month (1–12).
    """
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    return df[cols].groupby(df.index.month).mean()


def seasonal_groups(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Split a DataFrame into meteorological seasons (Southern Hemisphere).

    Returns a dict with keys ``'DJF'``, ``'MAM'``, ``'JJA'``, ``'SON'``.
    """
    seasons = {
        "DJF": [12, 1, 2],
        "MAM": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
    }
    return {name: df[df.index.month.isin(months)] for name, months in seasons.items()}


def daily_totals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """Aggregate to daily resolution using the specified aggregation (sum or mean)."""
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    grouped = df[cols].resample("1D")
    if agg == "sum":
        return grouped.sum()
    return grouped.mean()
