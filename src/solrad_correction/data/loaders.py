"""Data loaders — wrappers around micrometeorology for TCC use.

These functions provide a clean interface for loading sensor and WRF data,
delegating the actual I/O to the existing micrometeorology package.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_sensor_raw(
    data_dir: str | Path,
    *,
    pattern: str = "*.dat",
    calibrations_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw sensor data using the micrometeorology ingestion pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``.dat`` files.
    pattern:
        Glob pattern for file selection.
    calibrations_path:
        Path to calibrations YAML.  If provided, calibrations are applied.
    """
    from micrometeorology.common.paths import find_files
    from micrometeorology.sensors.ingestion import merge_dat_files

    files = find_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    df = merge_dat_files(files)  # type: ignore

    if calibrations_path and Path(calibrations_path).exists():
        from micrometeorology.sensors.calibration import (
            apply_calibrations,
            load_calibrations,
        )

        cals = load_calibrations(calibrations_path)
        df = apply_calibrations(df, cals)

    logger.info("Loaded raw sensor data: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_sensor_hourly(path: str | Path) -> pd.DataFrame:
    """Load pre-processed hourly sensor CSV.

    The CSV must have a datetime index in the first column.
    """
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    logger.info("Loaded hourly data: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_wrf_series(
    wrf_files: list[str | Path],
    lat: float,
    lon: float,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Extract WRF point time-series at (lat, lon).

    Delegates to ``micrometeorology.wrf.series.extract_point_series``.
    """
    from micrometeorology.wrf.series import extract_point_series

    paths = [Path(f) for f in wrf_files]
    df = extract_point_series(paths, lat, lon, variables)
    logger.info("Loaded WRF series: %d rows at (%.4f, %.4f)", len(df), lat, lon)
    return df
