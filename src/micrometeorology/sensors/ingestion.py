"""Flexible ingestion of Campbell Scientific `.dat` files.

The datalogger may change its header structure over time as sensors are
added or removed.  This module handles dynamic headers gracefully by:

1. Reading only the header rows to discover available columns.
2. Coercing all non-timestamp columns to float.
3. Applying sentinel-value filtering.

This means the same ingestion code works regardless of which sensors are
currently connected to the datalogger.

Timestamps are **naive station-local** end to end, by construction: they are
stamped by the datalogger's own clock, which runs on local time and knows
nothing of a zone. Re-labelling them UTC would invent a precision the
acquisition does not have -- that clock has slipped by an hour at least twice in
the record (see :mod:`micrometeorology.sensors.archive`). The UTC boundary
belongs to the layers that publish, which apply the site's pinned offset.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CampbellReadOptions(TypedDict, total=False):
    """The keyword options :func:`read_campbell_dat` accepts, for forwarding.

    Every key is optional and falls back to that function's own default, so a
    forwarder never restates a default and the two cannot drift apart.
    """

    separator: str
    skip_rows: list[int] | None
    timestamp_column: str
    drop_columns: list[str] | None
    sentinel_value: float | None
    text_columns: Sequence[str] | None


def read_campbell_dat(
    path: str | Path,
    *,
    separator: str = ",",
    skip_rows: list[int] | None = None,
    timestamp_column: str = "TIMESTAMP",
    drop_columns: list[str] | None = None,
    sentinel_value: float | None = -900.0,
    text_columns: Sequence[str] | None = None,
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
        Values ≤ this threshold are replaced with ``NaN``. Pass ``None`` to
        disable the rule entirely: the -900 default catches nothing in the
        LabMiM archive, whose sentinels are ``1000``, ``999``, ``-46.8``,
        ``-273.1``, ``-7999``, ``-6673`` and a windowed ``0``, so leaving it on
        only creates the impression that missing data has been handled.
    text_columns:
        Columns to leave as text instead of coercing to numeric. The datalogger's
        per-row quality flag (``MetSENS1_Status``, values ``"OK"`` /
        ``"Unknown Fault"``) is the only such column in the archive; the blanket
        coercion below would turn it into an all-NaN column that still looks
        populated. Names absent from the file are ignored.

    Returns
    -------
    pd.DataFrame
        The table indexed by its naive station-local ``TIMESTAMP``, that column
        dropped and duplicate stamps reduced to their first occurrence. Values
        keep the logger's own units, uncalibrated.

    Raises
    ------
    ValueError
        If *timestamp_column* is absent after *skip_rows*. Raised rather than
        returned with a ``RangeIndex``: the column names would come from the
        first surviving data row, and the caller's ``sort_index`` then dies on
        mixed int/Timestamp labels with a ``TypeError`` naming neither the file
        nor the cause. The archive holds one such table -- a headerless CSV
        whose TOA5 metadata line is missing -- which the manifest path stages
        and repairs.
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
        # The logger writes a bare NAN token for a missing sample. Declaring it
        # keeps such a column numeric on read instead of arriving as text and
        # becoming indistinguishable from the text quality flag below.
        na_values=["NAN"],
        keep_default_na=True,
    )

    if timestamp_column not in df.columns:
        raise ValueError(
            f"{path.name}: no {timestamp_column!r} column after skiprows={skip_rows}. "
            "A non-TOA5 table needs staging before it can be read (see sensors.archive)."
        )
    df.index = pd.to_datetime(df[timestamp_column], format="ISO8601")
    df.index.name = None
    df = df.drop(columns=[timestamp_column])
    df = df.loc[~df.index.duplicated(keep="first")]

    if drop_columns:
        existing = [c for c in drop_columns if c in df.columns]
        if existing:
            df = df.drop(columns=existing)

    # Catches tokens ``na_values`` above does not cover. Both dtype names are
    # needed: pandas 3 reads text as ``str`` dtype, and ``include=["object"]``
    # matches it only through a shim scheduled for removal.
    preserved = [c for c in (text_columns or ()) if c in df.columns]
    text_to_coerce = [
        c for c in df.select_dtypes(include=["object", "str"]).columns if c not in preserved
    ]
    if text_to_coerce:
        df[text_to_coerce] = df[text_to_coerce].apply(pd.to_numeric, errors="coerce")

    # Restricted to the numeric columns so a preserved text flag is not compared
    # against a float, which raises in pandas 3.
    if sentinel_value is not None:
        numeric = df.select_dtypes(include="number").columns
        df[numeric] = df[numeric].mask(df[numeric] <= sentinel_value)

    logger.info("  -> %d rows, %d columns", len(df), len(df.columns))
    return df


def merge_dat_files(
    paths: Sequence[str | Path],
    **kwargs: Unpack[CampbellReadOptions],
) -> pd.DataFrame:
    """Read and merge multiple ``.dat`` files into a single DataFrame.

    Files may have different column sets (sensors added/removed). Overlapping
    timestamps are resolved **per column**: the first non-null value in
    chronological file order wins. A column that only exists in a later file is
    therefore preserved even when an earlier file shares the same timestamp
    (that earlier row simply contributes ``NaN`` for the absent column), while
    a column present in both keeps the earlier file's value on a conflict.

    Parameters
    ----------
    paths:
        File paths to merge, in chronological order. Any sequence: callers
        collect them as ``list[Path]`` or ``list[str]``, and an invariant
        ``list[str | Path]`` would reject both.
    **kwargs:
        Forwarded to :func:`read_campbell_dat`; see
        :class:`CampbellReadOptions` for the keys it accepts.

    Returns
    -------
    pd.DataFrame
        One frame sorted by its naive station-local index, holding the union of
        every file's columns; a file that lacks a column contributes ``NaN``
        there rather than shortening the frame.

    Raises
    ------
    ValueError
        If *paths* is empty.
    """
    if not paths:
        raise ValueError("No files to merge")

    dfs = [read_campbell_dat(p, **kwargs) for p in paths]

    # Fast concatenation, avoiding O(N^2) iterative merges. Files are stacked in
    # chronological order, so within any duplicated-timestamp group the earlier
    # file's row precedes the later file's row.
    merged = pd.concat(dfs)

    if not merged.empty:
        duplicated = merged.index.duplicated(keep=False)
        if duplicated.any():
            # Collapse only the duplicated timestamps per column (first non-null
            # wins). Restricting groupby to the overlap keeps the common
            # no-overlap path cheap instead of grouping the whole frame.
            unique_part = merged.loc[~duplicated]
            collapsed = merged.loc[duplicated].groupby(level=0, sort=False).first()
            merged = pd.concat([unique_part, collapsed])
        merged = merged.sort_index()

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
        List of dicts with keys ``column``, ``lower``, ``upper``. The bounds are
        in the RAW logger units the gate was written for, several of them
        millivolts, so they are applied before any calibration factor.
        Columns that don't exist in the DataFrame are skipped (dynamic headers).

    Returns
    -------
    pd.DataFrame
        The same object, mutated in place, with out-of-range samples set to
        ``NaN``. Nothing is clipped to the bound: a rejected sample becomes
        missing rather than becoming the threshold.
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


def values_outside_declared_limits(df: pd.DataFrame, limits: list[dict]) -> dict[str, int]:
    """Columns still holding values their own declared gate rejects.

    The gates run on the RAW signal, before the instrument factors, so a value
    at the boundary crosses it once calibrated: ``CM3Up_Wm2_Avg`` capped at
    exactly 1500 W/m2 by its gate reaches the published artifact at 1508.65,
    which is 1500 x its post-2019 factor. Re-checking after calibration makes
    "the published column obeys its declared range" verifiable instead of
    assumed. The gate itself stays where it is: its thresholds are written in
    the logger's units and several name millivolts.

    Returns
    -------
    dict
        ``{column: sample count}`` for the columns still holding an offending
        value, omitting the columns that are clean. Empty means every declared
        gate holds on the frame as published.
    """
    outside: dict[str, int] = {}
    for lim in limits:
        col = lim["column"]
        if col not in df.columns:
            continue
        values = df[col]
        count = int(((values < lim["lower"]) | (values > lim["upper"])).sum())
        if count:
            outside[str(col)] = count
    return outside
