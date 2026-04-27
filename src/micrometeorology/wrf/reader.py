"""WRF NetCDF file reading and grid extraction.

Provides a thin wrapper around ``netCDF4.Dataset`` to standardize
grid coordinate extraction, time parsing, and metadata access.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import netCDF4
import numpy as np
import xarray as xr
from numpy.typing import NDArray

from micrometeorology.common.types import WEEKDAY_PT, GridLevel

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

ReaderMode = Literal["eager", "lazy"]
ChunkSpec = dict[str, int] | str | None
WRFArray = NDArray | xr.DataArray


class WRFReader(Protocol):
    """Shared reader surface used by WRF variable extraction and CLIs."""

    path: Path

    @property
    def dataset(self) -> Any: ...

    @property
    def grid_level(self) -> GridLevel: ...

    @property
    def dx(self) -> float: ...

    @property
    def dy(self) -> float: ...

    def read_grid(self) -> tuple[NDArray, NDArray]: ...

    def grid_bounds(self) -> tuple[float, float, float, float]: ...

    def parse_times(self) -> list[datetime]: ...

    def build_date_metadata(self, skip_first_n: int = 0) -> list[dict]: ...

    def get_variable(self, name: str) -> WRFArray: ...

    def has_variable(self, name: str) -> bool: ...


def parse_chunks(chunks: str | None) -> ChunkSpec:
    """Parse CLI chunk specifications for xarray-backed lazy reading.

    Accepted values:
    - ``None`` / ``""`` / ``"none"``: no xarray chunking.
    - ``"auto"``: pass xarray's automatic chunking through.
    - ``"Time=1,south_north=256,west_east=256"``: explicit chunk sizes.
    """
    if chunks is None or chunks.strip().lower() in {"", "none"}:
        return None
    if chunks.strip().lower() == "auto":
        return "auto"

    parsed: dict[str, int] = {}
    for item in chunks.split(","):
        if "=" not in item:
            raise ValueError(
                "Chunk settings must be 'auto', 'none', or comma-separated dim=size pairs"
            )
        dim, value = item.split("=", 1)
        dim = dim.strip()
        try:
            size = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid chunk size for {dim!r}: {value!r}") from exc
        if not dim or size <= 0:
            raise ValueError(f"Invalid chunk entry: {item!r}")
        parsed[dim] = size
    return parsed


def _decode_wrf_time_strings(times_raw: Any) -> list[str]:
    """Decode WRF ``Times`` values from netCDF char arrays or xarray byte arrays."""
    arr = np.asarray(times_raw)
    if arr.ndim == 1 and arr.dtype.kind in {"S", "U", "O"}:
        return [
            ts.decode("ascii") if isinstance(ts, bytes | np.bytes_) else str(ts)
            for ts in arr
        ]
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


class LazyWRFDataset:
    """Optional xarray-backed WRF reader for lazy variable selection.

    This class is intentionally separate from ``WRFDataset`` so the existing
    NetCDF4 eager path remains unchanged.  Without ``chunks`` xarray still
    defers array reads until values are requested; with ``chunks`` callers can
    opt into dask-backed chunking when dask is available in their environment.
    """

    def __init__(self, path: str | Path, *, chunks: ChunkSpec = None) -> None:
        self.path = Path(path)
        self.chunks = chunks
        self._ds: xr.Dataset | None = None
        self._grid_level = self._detect_grid_level()
        self._grid_cache: tuple[NDArray, NDArray] | None = None
        self._time_cache: list[datetime] | None = None

    @property
    def dataset(self) -> xr.Dataset:
        if self._ds is None:
            raise RuntimeError("Dataset is not open. Use as a context manager.")
        return self._ds

    def __enter__(self) -> LazyWRFDataset:
        kwargs: dict[str, Any] = {}
        if self.chunks is not None:
            kwargs["chunks"] = self.chunks
        self._ds = xr.open_dataset(self.path, **kwargs)
        return self

    def __exit__(self, *exc) -> None:
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def has_variable(self, name: str) -> bool:
        return name in self.dataset

    @property
    def grid_level(self) -> GridLevel:
        return self._grid_level

    @property
    def dx(self) -> float:
        return float(self.dataset.attrs["DX"])

    @property
    def dy(self) -> float:
        return float(self.dataset.attrs["DY"])

    def get_variable(self, name: str) -> xr.DataArray:
        """Select a variable lazily and return a squeezed xarray object."""
        return self.dataset[name].squeeze()

    def read_grid(self) -> tuple[NDArray, NDArray]:
        if self._grid_cache is None:
            lon = self.dataset["XLONG"].isel(Time=0).to_numpy()
            lat = self.dataset["XLAT"].isel(Time=0).to_numpy()
            self._grid_cache = (lon, lat)
        return self._grid_cache

    def grid_bounds(self) -> tuple[float, float, float, float]:
        lon, lat = self.read_grid()
        return (
            float(np.amin(lon)),
            float(np.amax(lon)),
            float(np.amin(lat)),
            float(np.amax(lat)),
        )

    def parse_times(self) -> list[datetime]:
        if self._time_cache is not None:
            return self._time_cache
        times_raw = self.dataset["Times"].to_numpy()
        time_strings = _decode_wrf_time_strings(times_raw)
        result: list[datetime] = []
        for ts in time_strings:
            dt = datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S")
            result.append(dt.replace(tzinfo=UTC))
        self._time_cache = result
        return result

    def build_date_metadata(self, skip_first_n: int = 0) -> list[dict]:
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
            entries.append(
                {
                    "index": i,
                    "datetime_utc": dt_utc,
                    "datetime_local": dt_local,
                    "label": label,
                    "name_suffix": suffix,
                    "skip": i < skip_first_n,
                }
            )
        return entries

    def _detect_grid_level(self) -> GridLevel:
        name = self.path.name.lower()
        for level in GridLevel:
            if level.value.lower() in name:
                return level
        logger.warning("Could not detect grid level from %s; defaulting to D01", name)
        return GridLevel.D01


@contextmanager
def open_wrf_dataset(
    path: str | Path,
    *,
    reader: ReaderMode = "eager",
    chunks: ChunkSpec = None,
) -> Iterator[WRFDataset | LazyWRFDataset]:
    """Open a WRF file using the selected reader backend."""
    if reader == "eager":
        with WRFDataset(path) as ds:
            yield ds
    elif reader == "lazy":
        with LazyWRFDataset(path, chunks=chunks) as ds:
            yield ds
    else:
        raise ValueError(f"Unknown WRF reader backend: {reader}")
