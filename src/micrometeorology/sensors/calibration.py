"""Date-precise instrument calibration corrections.

Calibration records are loaded from ``configs/calibrations.yaml``.
Each record specifies a column, a date range, and a multiplicative factor.
Records are **immutable historical facts** — new calibrations must be
appended, never overwriting existing entries.
"""

from __future__ import annotations

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
