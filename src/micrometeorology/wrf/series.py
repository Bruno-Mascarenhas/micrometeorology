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

from micrometeorology.wrf.reader import _decode_wrf_time_strings
from micrometeorology.wrf.safety import assert_reasonable_array_size

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
            lat_grid = ds["XLAT"].isel(Time=0).to_numpy()
            lon_grid = ds["XLONG"].isel(Time=0).to_numpy()
            row, col = find_nearest_indices(lat_grid, lon_grid, target_lat, target_lon)
            logger.debug(
                "Nearest grid point: row=%d, col=%d (lat=%.4f, lon=%.4f)",
                row,
                col,
                float(lat_grid[row, col]),
                float(lon_grid[row, col]),
            )

            # Parse times
            times_raw = ds["Times"].to_numpy()
            times_str = [ts.replace("_", " ") for ts in _decode_wrf_time_strings(times_raw)]
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
                    point = val.isel(south_north=row, west_east=col)
                else:
                    point = val

                if point.dims != ("Time",):
                    logger.warning(
                        "Skipping %s in point series: expected a Time-only selection, got dims=%s",
                        vname,
                        point.dims,
                    )
                    continue
                assert_reasonable_array_size(
                    point.shape,
                    point.dtype,
                    context=f"point-series extraction for {vname}",
                )
                extracted[vname] = point.to_numpy()

            if not extracted:
                continue

            # Combine into DataFrame
            df_part = pd.DataFrame(extracted, index=time_idx)
            all_records.append(df_part)  # type: ignore

    if not all_records:
        return pd.DataFrame()

    df = pd.concat(all_records)  # type: ignore
    df.index.name = "time"
    # Drop rows with NaT index which might happen on failed parses
    df = df[df.index.notna()]
    return df.sort_index()
