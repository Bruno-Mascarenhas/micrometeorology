"""Temporal alignment of sensor and WRF time series."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def align_sensor_wrf(
    sensor_df: pd.DataFrame,
    wrf_df: pd.DataFrame,
    *,
    tolerance: str = "30min",
    method: str = "nearest",
) -> pd.DataFrame:
    """Align sensor and WRF DataFrames temporally.

    Parameters
    ----------
    sensor_df:
        Sensor observations with DatetimeIndex.
    wrf_df:
        WRF model output with DatetimeIndex.
    tolerance:
        Maximum time difference for matching.
    method:
        Alignment method: ``"nearest"`` or ``"exact"``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with ``_obs`` and ``_wrf`` suffixes where names collide.
    """
    if method == "nearest":
        merged = pd.merge_asof(
            sensor_df.sort_index(),
            wrf_df.sort_index(),
            left_index=True,
            right_index=True,
            tolerance=pd.Timedelta(tolerance),
            direction="nearest",
            suffixes=("_obs", "_wrf"),
        )
    else:
        merged = sensor_df.join(wrf_df, how="inner", lsuffix="_obs", rsuffix="_wrf")

    logger.info("Aligned: %d rows (from %d obs, %d wrf)", len(merged), len(sensor_df), len(wrf_df))
    return merged


def select_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select feature columns and target column from a DataFrame.

    Raises ``KeyError`` if the target or any feature column is missing.
    """
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_features:
        raise KeyError(f"Missing feature columns: {missing_features}")
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found")

    features = df[feature_columns].copy()
    target = df[target_column].copy()
    return features, target
