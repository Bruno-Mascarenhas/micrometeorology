"""WRF NetCDF file reading and grid extraction.

Provides a thin wrapper around ``netCDF4.Dataset`` to standardize
grid coordinate extraction, time parsing, and metadata access.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import netCDF4
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from micrometeorology.common.types import WEEKDAY_PT, GridLevel
from micrometeorology.wrf.safety import assert_reasonable_array_size

logger = logging.getLogger(__name__)


def _decode_wrf_time_strings(times_raw: Any) -> list[str]:
    """Decode WRF ``Times`` values from netCDF char arrays or xarray byte arrays."""
    arr = np.asarray(times_raw)
    if arr.ndim == 1 and arr.dtype.kind in {"S", "U", "O"}:
        return [ts.decode("ascii") if isinstance(ts, bytes | np.bytes_) else str(ts) for ts in arr]
    return [str(ts) for ts in netCDF4.chartostring(arr)]


class WRFDataset:
    """Thin wrapper around a WRF ``netCDF4.Dataset``.

    Parameters
    ----------
    path:
        Path to a ``wrfout_*`` NetCDF file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._ds = netCDF4.Dataset(str(self.path), mode="r")
        self._ds.set_auto_mask(False)  # Return plain ndarray, not MaskedArray
        self._grid_level = self._detect_grid_level()
        self._grid_cache: tuple[NDArray, NDArray] | None = None
        self._time_cache: list[datetime] | None = None
        logger.info("Opened WRF dataset: %s (grid %s)", self.path.name, self._grid_level)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> netCDF4.Dataset:
        return self._ds

    @property
    def grid_level(self) -> GridLevel:
        return self._grid_level

    @property
    def dx(self) -> float:
        """Grid spacing in x-direction (meters)."""
        return float(self._ds.getncattr("DX"))

    @property
    def dy(self) -> float:
        """Grid spacing in y-direction (meters)."""
        return float(self._ds.getncattr("DY"))

    # ------------------------------------------------------------------
    # Grid coordinates
    # ------------------------------------------------------------------

    def read_grid(self) -> tuple[NDArray, NDArray]:
        """Return ``(lon, lat)`` 2-D arrays for the first time step."""
        if self._grid_cache is None:
            lon = np.asarray(self._ds.variables["XLONG"][0, :, :])
            lat = np.asarray(self._ds.variables["XLAT"][0, :, :])
            self._grid_cache = (lon, lat)
        return self._grid_cache

    def grid_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(lon_min, lon_max, lat_min, lat_max)``."""
        lon, lat = self.read_grid()
        return (
            float(np.amin(lon)),
            float(np.amax(lon)),
            float(np.amin(lat)),
            float(np.amax(lat)),
        )

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------

    def parse_times(self) -> list[datetime]:
        """Parse the ``Times`` variable into a list of UTC ``datetime`` objects."""
        if self._time_cache is not None:
            return self._time_cache
        times_var = self._ds.variables["Times"]
        time_strings = _decode_wrf_time_strings(times_var[:])
        result: list[datetime] = []
        for ts in time_strings:
            dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S")
            dt = dt.replace(tzinfo=UTC)
            result.append(dt)
        self._time_cache = result
        return result

    def build_date_metadata(
        self,
        skip_first_n: int = 0,
    ) -> list[dict]:
        """Build metadata dicts for each valid time step.

        Returns a list of dicts with keys:
        ``index``, ``datetime_utc``, ``datetime_local``, ``label``, ``name_suffix``.
        """
        times = self.parse_times()
        grade = self._grid_level.value
        entries: list[dict] = []
        start_label = ""

        for i, dt_utc in enumerate(times):
            dt_local = dt_utc.astimezone(tz=None)
            if i == 0:
                start_label = dt_utc.strftime("%d/%m/%Y %H") + " (UTC)"

            label = (
                f"\nInício Análise: {start_label}\n"
                f"Previsão: {dt_local.strftime('%d/%m/%Y %H')}HL "
                f"({WEEKDAY_PT.get(dt_local.isoweekday(), '')})"
            )
            suffix = f"{grade}_{i:03d}"

            if i < skip_first_n:
                entries.append(
                    {
                        "index": i,
                        "datetime_utc": dt_utc,
                        "datetime_local": dt_local,
                        "label": label,
                        "name_suffix": suffix,
                        "skip": True,
                    }
                )
            else:
                entries.append(
                    {
                        "index": i,
                        "datetime_utc": dt_utc,
                        "datetime_local": dt_local,
                        "label": label,
                        "name_suffix": suffix,
                        "skip": False,
                    }
                )
        return entries

    # ------------------------------------------------------------------
    # Variable access
    # ------------------------------------------------------------------

    def get_variable(self, name: str) -> NDArray:
        """Read a variable from the dataset, squeezed.

        All singleton axes are squeezed EXCEPT axis 0 (``Time``), so a
        single-timestep file keeps its time axis and downstream per-step
        slicing/bounds logic keeps working.
        """
        var = self._ds.variables[name]
        shape = tuple(int(size) for size in var.shape)
        dtype = np.dtype(var.dtype)
        assert_reasonable_array_size(shape, dtype, context=f"eager read of WRF variable {name}")
        arr = np.asarray(var[:])
        squeeze_axes = tuple(i for i, size in enumerate(arr.shape) if size == 1 and i != 0)
        if not squeeze_axes:
            return arr
        return arr.squeeze(axis=squeeze_axes)

    @property
    def n_time_steps(self) -> int:
        """Number of entries along the ``Time`` dimension."""
        return len(self._ds.dimensions["Time"])

    def get_variable_block(self, name: str, t_start: int, t_stop: int) -> NDArray:
        """Read a ``[t_start:t_stop]`` time block of a variable, unsqueezed.

        Blocks always span the full spatial extent so each compressed HDF5
        chunk is decompressed exactly once per streaming pass.
        """
        if t_start < 0 or t_stop <= t_start:
            raise ValueError(f"Invalid time block [{t_start}:{t_stop}] for variable {name}")
        var = self._ds.variables[name]
        n_times = int(var.shape[0])
        t_stop = min(t_stop, n_times)
        shape = (t_stop - t_start, *(int(size) for size in var.shape[1:]))
        assert_reasonable_array_size(
            shape,
            np.dtype(var.dtype),
            context=f"block read of WRF variable {name}",
        )
        return np.asarray(var[t_start:t_stop])

    def has_variable(self, name: str) -> bool:
        return name in self._ds.variables

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_grid_level(self) -> GridLevel:
        """Infer the grid level from the file name (e.g. ``wrfout_d01_…``)."""
        name = self.path.name.lower()
        for level in GridLevel:
            if level.value.lower() in name:
                return level
        logger.warning("Could not detect grid level from %s; defaulting to D01", name)
        return GridLevel.D01

    def close(self) -> None:
        self._ds.close()

    def __enter__(self) -> WRFDataset:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resolve_wrfout_paths(
    wrf_dir: str | Path,
    date: str,
    domains: tuple[int, ...] | None = None,
) -> list[Path]:
    """Resolve WRF output file paths using robust glob matching.

    Handles any filename suffix convention — colons (``00:00:00``),
    underscores (``00_00_00``), and non-standard trailing suffixes
    (e.g. ``wrfout_d01_2013-07-01_01_00_00-003_``).

    Parameters
    ----------
    wrf_dir:
        Directory containing ``wrfout_*`` files.
    date:
        Simulation date in ``YYYYMMDD`` format.
    domains:
        Exact domain numbers to search (no range widening: ``(1, 4)``
        matches only d01 and d04). Defaults to ``(1, 2, 3, 4)``.

    Returns
    -------
    list[Path]
        Sorted list of matching paths.
    """
    year, month, day = date[:4], date[4:6], date[6:8]
    selected = sorted(set(domains)) if domains else [1, 2, 3, 4]

    paths: list[Path] = []
    base = Path(wrf_dir)
    for d in selected:
        pattern = f"wrfout_d{d:02d}_{year}-{month}-{day}*"
        matches = sorted(base.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            logger.warning("No wrfout match for pattern %s in %s", pattern, wrf_dir)
    return paths
