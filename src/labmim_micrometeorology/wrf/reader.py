"""WRF NetCDF file reading and grid extraction.

Provides a thin wrapper around ``netCDF4.Dataset`` to standardize
grid coordinate extraction, time parsing, and metadata access.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from labmim_micrometeorology.common.types import WEEKDAY_PT, GridLevel

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WRFDataset:
    """Thin wrapper around a WRF ``netCDF4.Dataset``.

    Parameters
    ----------
    path:
        Path to a ``wrfout_*`` NetCDF file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._ds = netCDF4.Dataset(str(self.path))
        self._grid_level = self._detect_grid_level()
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
        xlat = self._ds.variables["XLAT"][:, :, :]
        xlong = self._ds.variables["XLONG"][:, :, :]
        lon = np.squeeze(xlong[:1, :, :])
        lat = np.squeeze(xlat[:1, :, :])
        return lon, lat

    def grid_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(lon_min, lon_max, lat_min, lat_max)``."""
        xlat = self._ds.variables["XLAT"][:, :, :]
        xlong = self._ds.variables["XLONG"][:, :, :]
        return (
            float(np.amin(xlong)),
            float(np.amax(xlong)),
            float(np.amin(xlat)),
            float(np.amax(xlat)),
        )

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------

    def parse_times(self) -> list[datetime]:
        """Parse the ``Times`` variable into a list of UTC ``datetime`` objects."""
        times_array = self._ds.variables["Times"][:]
        result: list[datetime] = []
        for time_chars in times_array:
            time_str = b"".join(time_chars.tolist()).decode("UTF-8")
            dt = datetime.strptime(time_str, "%Y-%m-%d_%H:%M:%S")
            dt = dt.replace(tzinfo=UTC)
            result.append(dt)
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
                entries.append({
                    "index": i,
                    "datetime_utc": dt_utc,
                    "datetime_local": dt_local,
                    "label": label,
                    "name_suffix": suffix,
                    "skip": True,
                })
            else:
                entries.append({
                    "index": i,
                    "datetime_utc": dt_utc,
                    "datetime_local": dt_local,
                    "label": label,
                    "name_suffix": suffix,
                    "skip": False,
                })
        return entries

    # ------------------------------------------------------------------
    # Variable access
    # ------------------------------------------------------------------

    def get_variable(self, name: str) -> NDArray:
        """Read a variable from the dataset, squeezed."""
        return np.asarray(self._ds.variables[name][:]).squeeze()

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
