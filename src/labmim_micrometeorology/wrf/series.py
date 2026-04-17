"""Time-series extraction from WRF NetCDF files.

Provides utilities to extract point time-series from gridded WRF output
at specific lat/lon coordinates (e.g. for comparison with observations).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def find_nearest_indices(
    lat_grid: NDArray,
    lon_grid: NDArray,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int]:
    """Find the (row, col) indices of the nearest grid point.

    Uses Euclidean distance on the lat/lon arrays.
    """
    dist = np.hypot(lat_grid - target_lat, lon_grid - target_lon)
    idx = np.unravel_index(np.argmin(dist), dist.shape)
    return int(idx[0]), int(idx[1])


def extract_point_series(
    files: list[Path],
    target_lat: float,
    target_lon: float,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Extract time-series at a single point from a list of WRF files.

    Parameters
    ----------
    files:
        Sorted list of NetCDF file paths.
    target_lat, target_lon:
        Coordinates of the target point.
    variables:
        List of NetCDF variable names to extract.  If ``None``, a default
        set of surface variables is used.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by time, with one column per variable.
    """
    if variables is None:
        variables = ["T2", "PSFC", "U10", "V10", "Q2", "SWDOWN", "HFX", "LH"]

    all_records: list[dict] = []

    for fpath in files:
        logger.info("Extracting from %s", fpath.name)
        with xr.open_dataset(str(fpath)) as ds:
            # Grid coordinates (first time step)
            lat_grid = ds["XLAT"].isel(Time=0).values
            lon_grid = ds["XLONG"].isel(Time=0).values
            row, col = find_nearest_indices(lat_grid, lon_grid, target_lat, target_lon)
            logger.debug(
                "Nearest grid point: row=%d, col=%d (lat=%.4f, lon=%.4f)",
                row,
                col,
                float(lat_grid[row, col]),
                float(lon_grid[row, col]),
            )

            # Parse times
            times_raw = ds["Times"].values
            try:
                if isinstance(times_raw[0], np.ndarray):
                    times_str = [b"".join(t).decode("UTF-8").replace("_", " ") for t in times_raw]
                else:
                    times_str = [str(t).replace("_", " ") for t in times_raw]
            except Exception:
                times_str = [str(t).replace("_", " ") for t in times_raw]

            time_idx = pd.to_datetime(times_str, errors="coerce")

            # Extract spatial slice for all times and convert to DataFrame
            # Filter variables that exist in the dataset
            valid_vars = [v for v in variables if v in ds]
            if not valid_vars:
                continue

            # Handle variables that have spatial dims and those that don't
            extracted = {}
            for vname in valid_vars:
                val = ds[vname]
                if "south_north" in val.dims and "west_east" in val.dims:
                    extracted[vname] = val.isel(south_north=row, west_east=col).values
                else:
                    extracted[vname] = val.values

            # Combine into DataFrame
            df_part = pd.DataFrame(extracted, index=time_idx)
            all_records.append(df_part)

    if not all_records:
        return pd.DataFrame()

    df = pd.concat(all_records)
    df.index.name = "time"
    # Drop rows with NaT index which might happen on failed parses
    df = df[df.index.notnull()]
    return df.sort_index()
