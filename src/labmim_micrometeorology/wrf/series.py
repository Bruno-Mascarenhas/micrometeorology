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
    dist = np.sqrt((lat_grid - target_lat) ** 2 + (lon_grid - target_lon) ** 2)
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
        ds = xr.open_dataset(str(fpath))

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
        for t_idx in range(len(times_raw)):
            time_bytes = times_raw[t_idx]
            if isinstance(time_bytes, np.ndarray):
                time_str = b"".join(time_bytes).decode("UTF-8")
            else:
                time_str = str(time_bytes)
            try:
                dt = pd.Timestamp(time_str.replace("_", " "))
            except Exception:
                continue

            record: dict = {"time": dt}
            for vname in variables:
                if vname in ds:
                    val = ds[vname].isel(Time=t_idx)
                    # Handle different dimensionalities
                    if "south_north" in val.dims and "west_east" in val.dims:
                        record[vname] = float(val.isel(south_north=row, west_east=col).values)
                    else:
                        record[vname] = float(val.values)
            all_records.append(record)

        ds.close()

    df = pd.DataFrame(all_records)
    if "time" in df.columns:
        df = df.set_index("time").sort_index()
    return df
