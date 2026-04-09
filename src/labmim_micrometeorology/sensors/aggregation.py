"""Temporal aggregation of high-frequency sensor data.

Replaces the manual ``while`` loop + ``timedelta(hours=1)`` pattern used
in controle_old.py, graficos1_UFBA_v5.py, and graficos3_UFBA_v1.py with
a clean ``pandas.resample()``-based approach.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from labmim_micrometeorology.sensors.wind import vector_mean_direction, wind_components

logger = logging.getLogger(__name__)


def aggregate_to_hourly(
    df: pd.DataFrame,
    *,
    min_samples: int = 6,
    sum_columns: list[str] | None = None,
    wind_dir_columns: list[str] | None = None,
    wind_speed_column_map: dict[str, str] | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    """Aggregate a high-frequency DataFrame to hourly resolution.

    Parameters
    ----------
    df:
        Input DataFrame with a DatetimeIndex (e.g. 5-minute data).
    min_samples:
        Minimum number of valid (non-NaN) samples required per window.
        Windows with fewer samples produce NaN.
    sum_columns:
        Column names that should be *summed* (e.g. precipitation).
    wind_dir_columns:
        Column names containing wind direction (degrees) that need
        vector-mean averaging.
    wind_speed_column_map:
        Mapping of ``{wind_dir_col: wind_speed_col}`` so that the correct
        speed column is used to compute U/V components for direction averaging.
        If not provided, direction columns use their own raw values.
    freq:
        Resampling frequency (default ``"1h"``).

    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame at the requested frequency.
    """
    sum_columns = set(sum_columns or [])
    wind_dir_columns = set(wind_dir_columns or [])
    wind_speed_column_map = wind_speed_column_map or {}

    # Identify standard-mean columns (everything not in sum or wind_dir)
    all_cols = set(df.columns)
    mean_columns = all_cols - sum_columns - wind_dir_columns

    results: dict[str, pd.Series] = {}

    resampler = df.resample(freq)

    # Mean columns
    for col in sorted(mean_columns):
        if col not in df.columns:
            continue
        grouped = resampler[col]
        counts = grouped.apply(lambda x: x.dropna().shape[0])
        means = grouped.mean()
        means[counts < min_samples] = np.nan
        results[col] = means

    # Sum columns (precipitation)
    for col in sorted(sum_columns):
        if col not in df.columns:
            continue
        grouped = resampler[col]
        counts = grouped.apply(lambda x: x.dropna().shape[0])
        sums = grouped.sum(min_count=1)
        sums[counts < min_samples] = np.nan
        results[col] = sums

    # Wind direction columns (vector mean)
    for dir_col in sorted(wind_dir_columns):
        if dir_col not in df.columns:
            continue
        speed_col = wind_speed_column_map.get(dir_col)
        if speed_col and speed_col in df.columns:
            u, v = wind_components(df[speed_col].values, df[dir_col].values)
        else:
            # Use unit speed if no speed column available
            u, v = wind_components(np.ones(len(df)), df[dir_col].values)

        df_uv = pd.DataFrame({"u": u, "v": v}, index=df.index)
        grouped_uv = df_uv.resample(freq)

        directions: list[float] = []
        for _, group in grouped_uv:
            valid = group.dropna()
            if len(valid) >= min_samples:
                directions.append(vector_mean_direction(valid["u"].values, valid["v"].values))
            else:
                directions.append(np.nan)

        results[dir_col] = pd.Series(directions, index=resampler.mean().index)

    out = pd.DataFrame(results)
    out.index.name = None
    logger.info("Aggregated %d rows → %d rows (%s)", len(df), len(out), freq)
    return out
