"""Temporal aggregation of high-frequency sensor data.

Each column is reduced by the rule its quantity demands: mean for state
variables, sum for accumulations such as precipitation, and a vector mean for
wind direction (which has no scalar average across the 0/360 wrap).

Timestamps are naive station-local throughout, as they arrive from the
datalogger's own clock (see :mod:`micrometeorology.sensors.ingestion`); the
window edges are therefore local wall-clock hours, not UTC ones.
"""

import logging

import numpy as np
import pandas as pd
from numpy.typing import NDArray

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
        Input DataFrame with a naive station-local ``DatetimeIndex`` (e.g.
        5-minute data). Non-numeric columns are excluded from the result.
    min_samples:
        Minimum number of valid (non-NaN) samples required per window.
        Windows with fewer samples produce NaN. This is a completeness gate, not
        a physical one: it decides whether an hour is representable at all, so a
        partly-clouded hour reported from four samples is withheld rather than
        published as if it summarised twelve. It counts SAMPLES, not a fraction
        of the window, so it has to be chosen together with *freq* and the
        cadence of *df*: the default 6 assumes the twelve five-minute samples of
        an hour, and 15-minute input under that default empties every column
        without reporting anything.
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
        Aggregated frame at the requested frequency, indexed by window start in
        the same naive station-local clock as *df*. It holds every numeric
        column plus any declared sum or direction column, in the units they
        arrived in; direction stays in degrees ``[0, 360)``. Windows failing
        *min_samples* hold ``NaN``, and so do windows the index never covered,
        since resampling fills the gaps.
    """
    sum_cols = set(sum_columns or [])
    dir_cols = set(wind_dir_columns or [])
    speed_column_map = wind_speed_column_map or {}

    numeric_columns = set(df.select_dtypes(include="number").columns)
    all_cols = set(df.columns)
    mean_columns = numeric_columns - sum_cols - dir_cols
    skipped = all_cols - numeric_columns - sum_cols - dir_cols
    if skipped:
        logger.debug("skipping %d non-numeric column(s): %s", len(skipped), sorted(skipped))

    results: dict[str, pd.Series] = {}

    resampler = df.resample(freq)

    for col in sorted(mean_columns):
        grouped = resampler[col]
        counts = grouped.count()
        means = grouped.mean()
        means[counts < min_samples] = np.nan
        results[col] = means

    for col in sorted(sum_cols):
        if col not in df.columns:
            continue
        grouped = resampler[col]
        counts = grouped.count()
        sums = grouped.sum(min_count=1)
        sums[counts < min_samples] = np.nan
        results[col] = sums

    for dir_col in sorted(dir_cols):
        if dir_col not in df.columns:
            continue
        speed_col = speed_column_map.get(dir_col)

        def directional_mean(weights: NDArray, column: str = dir_col) -> pd.DataFrame:
            u, v = wind_components(weights, df[column].to_numpy())
            return pd.DataFrame({"u": u, "v": v}, index=df.index).resample(freq).mean()

        if speed_col is not None and speed_col in df.columns:
            speed = df[speed_col].to_numpy(dtype=float)
            # Pinned by test_a_sample_whose_speed_alone_is_missing_still_bears_on_the_direction.
            uv_means = directional_mean(np.where(np.isfinite(speed), speed, 1.0))
            # Where the speed is too sparse to weight with, fall back to the
            # unit-weight directional mean: losing the weight beats losing the hour.
            sparse = (resampler[speed_col].count() < min_samples).reindex(
                uv_means.index, fill_value=True
            )
            if sparse.any():
                uv_means = uv_means.mask(sparse, directional_mean(np.ones(len(df))))
        else:
            if speed_col is not None:
                logger.warning(
                    "%r is mapped to speed column %r, absent from the frame: its %s "
                    "direction is an UNWEIGHTED vector mean, not the speed-weighted "
                    "one the mapping declares",
                    dir_col,
                    speed_col,
                    freq,
                )
            uv_means = directional_mean(np.ones(len(df)))

        # Pinned by test_a_complete_direction_hour_survives_a_dead_anemometer.
        dir_counts = resampler[dir_col].count()

        dirs = wind_direction_from_components(uv_means["u"].to_numpy(), uv_means["v"].to_numpy())
        dir_series = pd.Series(dirs, index=uv_means.index)
        dir_series[dir_counts.reindex(dir_series.index) < min_samples] = np.nan
        results[dir_col] = dir_series

    out = pd.DataFrame(results)
    out.index.name = None
    logger.info("Aggregated %d rows -> %d rows (%s)", len(df), len(out), freq)
    return out
