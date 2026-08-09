"""Temporal aggregation of high-frequency sensor data.

Replaces the manual ``while`` loop + ``timedelta(hours=1)`` pattern used
in controle_old.py, graficos1_UFBA_v5.py, and graficos3_UFBA_v1.py with
a clean ``pandas.resample()``-based approach.
"""

import logging

import numpy as np
import pandas as pd

from micrometeorology.sensors.wind import (
    wind_components,
    wind_direction_from_components,
)

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
    sum_cols = set(sum_columns or [])
    dir_cols = set(wind_dir_columns or [])
    speed_column_map = wind_speed_column_map or {}

    # Identify standard-mean columns (everything not in sum or wind_dir).
    # Non-numeric columns are excluded rather than attempted: the datalogger's
    # per-row quality flag is text ("OK" / "Unknown Fault"), and pandas 3 raises
    # on a mean over a string dtype instead of quietly producing NaN. Convert
    # such a flag to a number before calling this (e.g. the fraction of samples
    # reading OK) if you want it in the hourly frame.
    numeric_columns = set(df.select_dtypes(include="number").columns)
    all_cols = set(df.columns)
    mean_columns = (all_cols & numeric_columns) - sum_cols - dir_cols
    skipped = all_cols - numeric_columns - sum_cols - dir_cols
    if skipped:
        logger.debug("skipping %d non-numeric column(s): %s", len(skipped), sorted(skipped))

    results: dict[str, pd.Series] = {}

    resampler = df.resample(freq)

    # Mean columns — use vectorized .count() instead of apply(lambda)
    for col in sorted(mean_columns):
        if col not in df.columns:
            continue
        grouped = resampler[col]
        counts = grouped.count()
        means = grouped.mean()
        means[counts < min_samples] = np.nan
        results[col] = means

    # Sum columns (precipitation)
    for col in sorted(sum_cols):
        if col not in df.columns:
            continue
        grouped = resampler[col]
        counts = grouped.count()
        sums = grouped.sum(min_count=1)
        sums[counts < min_samples] = np.nan
        results[col] = sums

    # Wind direction columns (vector mean) — vectorized resample on u/v
    for dir_col in sorted(dir_cols):
        if dir_col not in df.columns:
            continue
        speed_col = speed_column_map.get(dir_col)
        if speed_col and speed_col in df.columns:
            u, v = wind_components(df[speed_col].to_numpy(), df[dir_col].to_numpy())
        else:
            # Use unit speed if no speed column available
            u, v = wind_components(np.ones(len(df)), df[dir_col].to_numpy())

        df_uv = pd.DataFrame({"u": u, "v": v}, index=df.index)
        uv_resampled = df_uv.resample(freq)
        uv_counts = uv_resampled["u"].count()
        uv_means = uv_resampled.mean()

        # Vectorized direction from mean components
        dirs = wind_direction_from_components(uv_means["u"].to_numpy(), uv_means["v"].to_numpy())
        dir_series = pd.Series(dirs, index=uv_means.index)
        dir_series[uv_counts < min_samples] = np.nan
        results[dir_col] = dir_series

    out = pd.DataFrame(results)
    out.index.name = None
    logger.info("Aggregated %d rows -> %d rows (%s)", len(df), len(out), freq)
    return out
