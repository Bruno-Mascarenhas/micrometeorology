"""Date-precise instrument calibration corrections.

Calibration records are loaded from ``configs/calibrations.yaml``.
Each record specifies a column, a date range, and a multiplicative factor.
Records are **immutable historical facts** — new calibrations must be
appended, never overwriting existing entries.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_calibrations(config_path: str | Path) -> list[dict[str, Any]]:
    """Load calibration records from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.warning("Calibration config not found: %s", path)
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("calibrations", [])  # type: ignore


def _resolve_inclusive_end(value: Any, fallback: pd.Timestamp) -> pd.Timestamp:
    """Resolve an ``end_date`` config value to an *inclusive* upper bound.

    A **date-only** boundary such as ``"2018-12-31"`` parses to midnight
    (``2018-12-31 00:00:00``). Compared with ``df.index <= end`` that would
    exclude every sample of the boundary day after ``00:00`` — the whole day
    is silently dropped from the calibration (and, for ``factor: null``
    records, left un-NaN'd). Because ``end_date`` is documented as the *last
    day* the record applies (inclusive), a midnight-resolution timestamp is
    extended to the final nanosecond of that day (``+ 1 day - 1 ns``) so the
    entire day is covered.

    A value that carries an explicit non-midnight time keeps exact
    ``<= end`` semantics. A falsy value (``None``/empty) returns ``fallback``
    unchanged (meaning "until the end of the dataset").
    """
    if not value:
        return fallback
    end = pd.Timestamp(value)
    if end == end.normalize():
        # Date-only boundary → inclusive of the whole day.
        return end + pd.Timedelta(days=1) - pd.Timedelta(1, "ns")
    return end


# A resolved date range for one record, tagged with a human-readable label used
# only in overlap-error messages: ``(label, inclusive_start, inclusive_end)``.
_ResolvedRange = tuple[str, pd.Timestamp, pd.Timestamp]


def _describe_record(record: dict[str, Any]) -> str:
    """Return a one-line identity for a calibration/mapping record.

    Names the entry by column, its configured date range, and (when present)
    its description, so overlap errors point at the exact offending config rows.
    """
    start = record.get("start_date") or "dataset-start"
    end = record.get("end_date") or "dataset-end"
    parts = [str(record["column"]), f"{start} -> {end}"]
    description = record.get("description")
    if description:
        parts.append(str(description))
    return " | ".join(parts)


def _resolve_record_range(
    record: dict[str, Any], df: pd.DataFrame
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve a record's inclusive ``[start, end]`` bounds against *df*'s index.

    Mirrors the resolution used when the record is applied: an absent
    ``start_date`` defaults to the first index timestamp and ``end_date`` is
    resolved to an inclusive upper bound (see :func:`_resolve_inclusive_end`).
    """
    start = (
        pd.Timestamp(record["start_date"])
        if record.get("start_date")
        else pd.Timestamp(df.index.min())
    )
    end = _resolve_inclusive_end(record.get("end_date"), df.index.max())
    return start, end


def _find_overlapping_pair(
    ranges: list[_ResolvedRange],
) -> tuple[_ResolvedRange, _ResolvedRange, pd.Timestamp, pd.Timestamp] | None:
    """Return the first overlapping pair of inclusive ranges, or ``None``.

    Each range is ``(label, start, end)`` with inclusive ``[start, end]``.
    Sorting by ``start`` makes a single adjacent-pair scan sufficient: if any
    two ranges overlap, two consecutive ones (in start order) also overlap.
    """
    ordered = sorted(ranges, key=lambda item: item[1])
    for earlier, later in itertools.pairwise(ordered):
        # earlier.start <= later.start by the sort; the ranges overlap iff
        # later.start falls on or before earlier's inclusive end.
        if later[1] <= earlier[2]:
            overlap_start = later[1]
            overlap_end = min(earlier[2], later[2])
            return earlier, later, overlap_start, overlap_end
    return None


def _overlap_error(
    kind: str,
    group: str,
    earlier: _ResolvedRange,
    later: _ResolvedRange,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> ValueError:
    """Build a clear ``ValueError`` for an overlapping-range configuration error."""
    day_lo = overlap_start.date()
    day_hi = overlap_end.date()
    day = str(day_lo) if day_lo == day_hi else f"{day_lo}..{day_hi}"
    return ValueError(
        f"Overlapping {kind} for {group!r} on {day}: [{earlier[0]}] and "
        f"[{later[0]}] both cover {day}. Date ranges are inclusive of the whole "
        f"end day, so consecutive records must abut on the NEXT day "
        f"(e.g. end_date 2018-12-31 then start_date 2019-01-01), not the same day."
    )


def apply_calibrations(
    df: pd.DataFrame,
    calibrations: list[dict[str, Any]],
) -> pd.DataFrame:
    """Apply calibration corrections to a DataFrame in-place.

    Parameters
    ----------
    df:
        DataFrame with a DatetimeIndex.
    calibrations:
        List of calibration records, each with keys:
        ``column``, ``start_date``, ``end_date``, ``factor``, ``description``.

    Notes
    -----
    Both ``start_date`` and ``end_date`` are **inclusive at day granularity**.
    A date-only ``end_date`` (e.g. ``"2018-12-31"``) resolves to midnight but
    is treated as the last nanosecond of that day, so every sub-daily sample of
    the boundary day is calibrated (and, for ``factor: null`` records, NaN'd).
    An ``end_date`` carrying an explicit time keeps exact ``<= end`` semantics.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with corrections applied.
    """
    ranges_by_column: dict[str, list[_ResolvedRange]] = {}
    for cal in calibrations:
        column = cal["column"]
        if column not in df.columns:
            continue
        start, end = _resolve_record_range(cal, df)
        ranges_by_column.setdefault(column, []).append((_describe_record(cal), start, end))
    for column, ranges in ranges_by_column.items():
        overlap = _find_overlapping_pair(ranges)
        if overlap is not None:
            raise _overlap_error("calibrations", column, *overlap)

    for cal in calibrations:
        col = cal["column"]
        if col not in df.columns:
            logger.debug("Skipping calibration for missing column: %s", col)
            continue

        start = pd.Timestamp(cal["start_date"]) if cal.get("start_date") else df.index.min()
        end = _resolve_inclusive_end(cal.get("end_date"), df.index.max())
        factor = cal.get("factor")
        desc = cal.get("description", "")

        mask = (df.index >= start) & (df.index <= end)

        if factor is None:
            # Null factor means the data is invalid for this period
            df.loc[mask, col] = np.nan
            logger.info("  %s [%s -> %s]: set to NaN (%s)", col, start.date(), end.date(), desc)
        else:
            df.loc[mask, col] *= factor
            logger.info(
                "  %s [%s -> %s]: x %.10f (%s)", col, start.date(), end.date(), factor, desc
            )

    return df


def load_sensor_switches(config_path: str | Path) -> list[dict[str, Any]]:
    """Load sensor-switch definitions from the calibrations YAML.

    Sensor switches define which raw column maps to a unified variable
    during specific date ranges (e.g. PSP1 → CM3Up after 2018-11).
    """
    path = Path(config_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("sensor_switches", [])  # type: ignore


def unify_sensor_columns(
    df: pd.DataFrame,
    switches: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create unified columns from sensor-switch definitions.

    For each switch, creates a new column with the ``unified_name`` that
    concatenates data from different raw columns based on date ranges.

    Notes
    -----
    Mapping date ranges are **inclusive at day granularity**. A date-only
    ``end_date`` covers the whole boundary day (extended to its last
    nanosecond), so consecutive mappings that abut on a day boundary leave no
    unfilled hole for that day; an explicit time keeps exact ``<= end``
    semantics.
    """
    for switch in switches:
        unified_name = switch["unified_name"]
        mapping_ranges = [
            (_describe_record(mapping), *_resolve_record_range(mapping, df))
            for mapping in switch["mappings"]
            if mapping["column"] in df.columns
        ]
        overlap = _find_overlapping_pair(mapping_ranges)
        if overlap is not None:
            raise _overlap_error("sensor-switch mappings", unified_name, *overlap)

        series_parts: list[pd.Series] = []

        for mapping in switch["mappings"]:
            col = mapping["column"]
            if col not in df.columns:
                logger.warning("Column %s not found for unified variable %s", col, unified_name)
                continue

            start = (
                pd.Timestamp(mapping["start_date"]) if mapping.get("start_date") else df.index.min()
            )
            end = _resolve_inclusive_end(mapping.get("end_date"), df.index.max())

            mask = (df.index >= start) & (df.index <= end)
            part = df.loc[mask, col].rename(unified_name)
            series_parts.append(part)

        if series_parts:
            df[unified_name] = pd.concat(series_parts).reindex(df.index)
            logger.info("Created unified column: %s", unified_name)

    return df
