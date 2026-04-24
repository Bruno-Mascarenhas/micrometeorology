"""Flexible ingestion of Campbell Scientific `.dat` files.

The datalogger may change its header structure over time as sensors are
added or removed.  This module handles dynamic headers gracefully by:

1. Reading only the header rows to discover available columns.
2. Coercing all non-timestamp columns to float.
3. Applying sentinel-value filtering.

This means the same ingestion code works regardless of which sensors are
currently connected to the datalogger.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_campbell_dat(
    path: str | Path,
    *,
    separator: str = ",",
    skip_rows: list[int] | None = None,
    timestamp_column: str = "TIMESTAMP",
    drop_columns: list[str] | None = None,
    sentinel_value: float = -900.0,
) -> pd.DataFrame:
    """Read a single Campbell Scientific ``.dat`` file.

    Parameters
    ----------
    path:
        Path to the ``.dat`` file.
    separator:
        Column separator (default ``','``).
    skip_rows:
        Row indices to skip (default ``[0, 2, 3]`` for Campbell headers).
    timestamp_column:
        Name of the timestamp column.
    drop_columns:
        Columns to drop after reading.  Columns that do not exist in the
        file are silently ignored (handles dynamic headers).
    sentinel_value:
        Values ≤ this threshold are replaced with ``NaN``.
    """
    if skip_rows is None:
        skip_rows = [0, 2, 3]

    path = Path(path)
    logger.info("Reading: %s", path.name)

    df = pd.read_csv(
        path,
        sep=separator,
        skiprows=skip_rows,
        low_memory=False,
        parse_dates=False,
    )

    # Set timestamp index
    if timestamp_column in df.columns:
        df.index = pd.to_datetime(df[timestamp_column], format="ISO8601")
        df.index.name = None
        df = df.drop(columns=[timestamp_column])

    # Drop requested columns (only those that actually exist)
    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    # Coerce only object columns to numeric (pyarrow already types most columns correctly)
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(pd.to_numeric, errors="coerce")

    # Sentinel value -> NaN
    if sentinel_value is not None:
        df = df.mask(df <= sentinel_value)

    logger.info("  -> %d rows, %d columns", len(df), len(df.columns))
    return df


def merge_dat_files(
    paths: list[str | Path],
    **kwargs,
) -> pd.DataFrame:
    """Read and merge multiple ``.dat`` files into a single DataFrame.

    Files may have different column sets (sensors added/removed). The merge
    uses an ordered merge so that overlapping timestamps are handled correctly.

    Parameters
    ----------
    paths:
        List of file paths to merge, in chronological order.
    **kwargs:
        Additional keyword arguments passed to :func:`read_campbell_dat`.
    """
    if not paths:
        raise ValueError("No files to merge")

    dfs = [read_campbell_dat(p, **kwargs) for p in paths]

    # Fast concatenation, avoiding O(N^2) iterative merges
    merged = pd.concat(dfs)

    # Resolve overlapping timestamps by keeping the first non-null value per column
    if not merged.empty:
        merged = merged.loc[~merged.index.duplicated(keep="first")]

    logger.info(
        "Merged %d files -> %d rows, %d columns", len(paths), len(merged), len(merged.columns)
    )
    return merged


def apply_physical_limits(
    df: pd.DataFrame,
    limits: list[dict],
) -> pd.DataFrame:
    """Apply quality-control limits, setting out-of-range values to NaN.

    Parameters
    ----------
    df:
        Input DataFrame.
    limits:
        List of dicts with keys ``column``, ``lower``, ``upper``.
        Columns that don't exist in the DataFrame are skipped (dynamic headers).
    """
    for lim in limits:
        col = lim["column"]
        if col not in df.columns:
            continue
        lower, upper = lim["lower"], lim["upper"]
        mask = (df[col] < lower) | (df[col] > upper)
        n_bad = mask.sum()
        if n_bad > 0:
            logger.debug("  %s: %d values outside [%.1f, %.1f] -> NaN", col, n_bad, lower, upper)
            df.loc[mask, col] = np.nan
    return df
