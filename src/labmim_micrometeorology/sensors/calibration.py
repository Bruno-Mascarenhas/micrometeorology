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
    return data.get("calibrations", [])


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
        end = pd.Timestamp(cal["end_date"]) if cal.get("end_date") else df.index.max()
        factor = cal.get("factor")
        desc = cal.get("description", "")

        mask = (df.index >= start) & (df.index <= end)

        if factor is None:
            # Null factor means the data is invalid for this period
            df.loc[mask, col] = np.nan
            logger.info("  %s [%s → %s]: set to NaN (%s)", col, start.date(), end.date(), desc)
        else:
            df.loc[mask, col] *= factor
            logger.info("  %s [%s → %s]: × %.10f (%s)", col, start.date(), end.date(), factor, desc)

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
    return data.get("sensor_switches", [])


def unify_sensor_columns(
    df: pd.DataFrame,
    switches: list[dict[str, Any]],
) -> pd.DataFrame:
    """Create unified columns from sensor-switch definitions.

    For each switch, creates a new column with the ``unified_name`` that
    concatenates data from different raw columns based on date ranges.
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
            end = pd.Timestamp(mapping["end_date"]) if mapping.get("end_date") else df.index.max()

            mask = (df.index >= start) & (df.index <= end)
            part = df.loc[mask, col].rename(unified_name)
            series_parts.append(part)

        if series_parts:
            df[unified_name] = pd.concat(series_parts).reindex(df.index)
            logger.info("Created unified column: %s", unified_name)

    return df
