"""WRF NetCDF file reading and grid extraction.

Provides a thin wrapper around ``netCDF4.Dataset`` to standardize
grid coordinate extraction, time parsing, and metadata access.
"""

import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

import netCDF4
import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.types import WEEKDAY_PT, GridLevel
from micrometeorology.wrf.safety import assert_reasonable_array_size

logger = logging.getLogger(__name__)

# Every date_time string in the exported JSONs is local, so the product timezone
# is pinned here rather than taken from the host's TZ: a UTC-configured container
# would shift the whole forecast by 3 hours while the site keeps labelling values
# as UTC-03:00. Prefer fixed-offset zones — the site renders timeline labels with
# flat one-hour-per-index arithmetic from the run anchor, so a DST transition
# inside a run would make labels disagree with the per-file date_time strings
# (America/Bahia observes no DST).
LABMIM_TIMEZONE_ENV = "LABMIM_TIMEZONE"
DEFAULT_TIMEZONE = "America/Bahia"


@lru_cache(maxsize=1)
def _cached_timezone(name: str) -> tzinfo:
    return ZoneInfo(name)


def product_timezone() -> tzinfo:
    """The timezone all exported local datetimes are expressed in."""
    return _cached_timezone(os.environ.get(LABMIM_TIMEZONE_ENV) or DEFAULT_TIMEZONE)


def _decode_wrf_time_strings(times_raw: Any) -> list[str]:
    """Decode WRF ``Times`` values from netCDF char arrays or xarray byte arrays."""
    arr = np.asarray(times_raw)
    if arr.ndim == 1 and arr.dtype.kind in {"S", "U", "O"}:
        return [ts.decode("ascii") if isinstance(ts, bytes | np.bytes_) else str(ts) for ts in arr]
    return [str(ts) for ts in netCDF4.chartostring(arr)]


def detect_grid_level(path: str | Path) -> GridLevel | None:
    """The domain a wrfout file name carries (``wrfout_d01_…`` → ``D01``).

    Purely name-based, so callers can group files by domain without opening a
    single NetCDF. ``None`` when the name carries no ``D01``..``D05`` token —
    the caller decides whether that is fatal (:class:`WRFDataset` refuses to
    guess) or merely not groupable.
    """
    name = Path(path).name.lower()
    for level in GridLevel:
        if level.value.lower() in name:
            return level
    return None


def assert_one_file_per_domain(paths: Sequence[str | Path]) -> None:
    """Raise ``ValueError`` when two files would publish under the same domain.

    Every WRF product name holds exactly one slot per domain —
    ``{D}_{VAR}_{nnn}.json``, ``{D}_{VAR}.series.bin``, ``{D}_{VAR}.summary.json``,
    ``{D}.geojson``, ``{D}.grid.json`` — so a run covering two files of the same
    domain can only overwrite, and because the units run concurrently on one
    pool the surviving mix of per-step JSONs, series matrix and summary is
    whichever unit finished last. Names with no recognizable domain are left
    out: :class:`WRFDataset` fails those units individually rather than letting
    them publish under a guessed domain.
    """
    names_by_domain: dict[GridLevel, list[str]] = {}
    for path in paths:
        level = detect_grid_level(path)
        if level is not None:
            names_by_domain.setdefault(level, []).append(Path(path).name)
    collisions = [
        f"{level.value}: {', '.join(sorted(names))}"
        for level, names in sorted(names_by_domain.items())
        if len(names) > 1
    ]
    if collisions:
        raise ValueError(
            "A run publishes one set of files per domain, but these files map to "
            f"the same domain and would overwrite each other — {'; '.join(collisions)}. "
            "Cover at most one file per domain: narrow the selection with "
            "-d/--dataset or -D/--domains, or give each file its own -o/-g "
            "output directories."
        )


class WRFDataset:
    """Thin wrapper around a WRF ``netCDF4.Dataset``.

    The file is opened with auto-masking off, so every read returns a plain
    ``ndarray`` rather than a ``MaskedArray``. The grid level is detected from
    the file name before the open, because that detection needs only the name
    and raising after the open would strand an HDF5 handle no ``__exit__`` ever
    closes. Grid coordinates and parsed times are cached for the object's life.

    Parameters
    ----------
    path:
        Path to a ``wrfout_*`` NetCDF file.

    Raises
    ------
    ValueError
        When the file name carries no ``D01``..``D05`` domain token.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._grid_level = self._detect_grid_level()
        self._ds = netCDF4.Dataset(str(self.path), mode="r")
        self._ds.set_auto_mask(False)
        self._grid_cache: tuple[NDArray, NDArray] | None = None
        self._time_cache: list[datetime] | None = None
        logger.info("Opened WRF dataset: %s (grid %s)", self.path.name, self._grid_level)

    @property
    def dataset(self) -> netCDF4.Dataset:
        """The underlying open ``netCDF4.Dataset`` (auto-masking disabled)."""
        return self._ds

    @property
    def grid_level(self) -> GridLevel:
        """Domain grid level inferred from the file name (D01/D02/...)."""
        return self._grid_level

    @property
    def dx(self) -> float:
        """Grid spacing in x-direction (meters)."""
        return float(self._ds.getncattr("DX"))

    @property
    def dy(self) -> float:
        """Grid spacing in y-direction (meters)."""
        return float(self._ds.getncattr("DY"))

    def read_grid(self) -> tuple[NDArray, NDArray]:
        """Return the cell-centre coordinates of the domain.

        Returns
        -------
        tuple[NDArray, NDArray]
            ``(lon, lat)``, each ``(ny, nx)`` in the file's own dtype (float32
            in operational wrfout), degrees east and degrees north on WGS84.
            Read from the first time step, which the grid never leaves.
        """
        if self._grid_cache is None:
            lon = np.asarray(self._ds.variables["XLONG"][0, :, :])
            lat = np.asarray(self._ds.variables["XLAT"][0, :, :])
            self._grid_cache = (lon, lat)
        return self._grid_cache

    def grid_bounds(self) -> tuple[float, float, float, float]:
        """Return ``(lon_min, lon_max, lat_min, lat_max)`` in degrees.

        The extent of the cell CENTRES, so it is half a cell inside the extent
        of the cell rectangles the GeoJSON writers publish.
        """
        lon, lat = self.read_grid()
        return (
            float(np.amin(lon)),
            float(np.amax(lon)),
            float(np.amin(lat)),
            float(np.amax(lat)),
        )

    def parse_times(self) -> list[datetime]:
        """Parse the ``Times`` variable into a list of UTC ``datetime`` objects.

        WRF writes ``Times`` as ``YYYY-MM-DD_HH:MM:SS`` without an offset and the
        values are UTC by definition, so the tzinfo attached here states the
        file's own convention rather than assuming one.
        """
        if self._time_cache is not None:
            return self._time_cache
        times_var = self._ds.variables["Times"]
        time_strings = _decode_wrf_time_strings(times_var[:])
        result: list[datetime] = [
            datetime.strptime(ts, "%Y-%m-%d_%H:%M:%S").replace(tzinfo=UTC) for ts in time_strings
        ]
        self._time_cache = result
        return result

    def build_date_metadata(
        self,
        skip_first_n: int = 0,
    ) -> list[dict]:
        """Build one metadata dict per time step of the file.

        Parameters
        ----------
        skip_first_n:
            How many leading steps to mark ``skip``; the entries are still
            returned, so indices keep matching the file's time axis.

        Returns
        -------
        list[dict]
            One entry per step, with keys ``index`` (position on the file's time
            axis), ``datetime_utc``, ``datetime_local`` (in
            :func:`product_timezone`), ``label`` (the Portuguese figure caption),
            ``name_suffix`` (``{domain}_{index:03d}``) and ``skip``.
        """
        times = self.parse_times()
        grid = self._grid_level.value
        entries: list[dict] = []
        start_label = ""

        tz = product_timezone()
        for i, dt_utc in enumerate(times):
            dt_local = dt_utc.astimezone(tz)
            if i == 0:
                start_label = dt_utc.strftime("%d/%m/%Y %H") + " (UTC)"

            label = (
                f"\nInício Análise: {start_label}\n"
                f"Previsão: {dt_local.strftime('%d/%m/%Y %H')}HL "
                f"({WEEKDAY_PT.get(dt_local.isoweekday(), '')})"
            )
            entries.append(
                {
                    "index": i,
                    "datetime_utc": dt_utc,
                    "datetime_local": dt_local,
                    "label": label,
                    "name_suffix": f"{grid}_{i:03d}",
                    "skip": i < skip_first_n,
                }
            )
        return entries

    def get_variable(self, name: str) -> NDArray:
        """Read a whole variable eagerly, squeezed, in the file's own dtype.

        All singleton axes are squeezed EXCEPT axis 0 (``Time``), so a
        single-timestep file keeps its time axis and downstream per-step
        slicing/bounds logic keeps working. A surface field therefore comes back
        ``(T, ny, nx)``.

        Raises
        ------
        KeyError
            When the file carries no variable *name* — callers that treat an
            absent field as skippable check :meth:`has_variable` first.
        MemoryError
            When the full read would exceed the array-size ceiling.
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
        chunk is decompressed exactly once per streaming pass. *t_stop* is
        clamped to the file's step count, so the last block of a pass may be
        shorter than requested.

        Raises
        ------
        ValueError
            When the block is empty or starts before the first step.
        MemoryError
            When the block would exceed the array-size ceiling.
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
        """Whether ``name`` is present among the file's NetCDF variables."""
        return name in self._ds.variables

    def _detect_grid_level(self) -> GridLevel:
        """Infer the grid level from the file name (e.g. ``wrfout_d01_…``).

        Refuses to guess: every product name is built from this value, so
        defaulting an untokenized (or out-of-range, e.g. ``d06``) file to
        ``D01`` republishes its grid and values over the real D01 products.
        """
        level = detect_grid_level(self.path)
        if level is None:
            raise ValueError(
                f"Could not detect grid level ({', '.join(g.value for g in GridLevel)}) "
                f"from WRF filename {self.path.name!r}"
            )
        return level

    def close(self) -> None:
        """Close the underlying NetCDF file handle."""
        self._ds.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def normalize_run_date(date: str) -> str:
    """Return the digit run :func:`resolve_wrfout_paths` slices, rejecting typos.

    ``2026-07-27`` and ``2026/07/27`` are accepted — the wrfout names are
    themselves ISO, so those are the natural typo — and a longer
    ``YYYYMMDDHH`` prefix keeps working. Anything that is not at least a full
    8-digit day raises ``ValueError``: silently slicing it would report a
    mistyped date as a day on which WRF produced no files.
    """
    digits = date.replace("-", "").replace("/", "")
    if not digits.isdigit() or len(digits) < 8:
        raise ValueError(f"--date must be YYYYMMDD (got {date!r})")
    return digits


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
        Simulation date in ``YYYYMMDD`` format; separators are tolerated and a
        longer ``YYYYMMDDHH`` run prefix is accepted. Validated here so every
        CLI reaching for wrfout files rejects the same typos —
        see :func:`normalize_run_date`.
    domains:
        Exact domain numbers to search (no range widening: ``(1, 4)``
        matches only d01 and d04). Defaults to ``(1, 2, 3, 4)``.

    Returns
    -------
    list[Path]
        Sorted list of matching paths.

    Raises
    ------
    ValueError
        When *date* is not at least an 8-digit day.
    """
    date = normalize_run_date(date)
    year, month, day = date[:4], date[4:6], date[6:8]
    selected = sorted(set(domains)) if domains else [1, 2, 3, 4]

    paths: list[Path] = []
    base = Path(wrf_dir)
    for domain in selected:
        pattern = f"wrfout_d{domain:02d}_{year}-{month}-{day}*"
        matches = sorted(base.glob(pattern))
        if matches:
            paths.extend(matches)
        else:
            logger.warning("No wrfout match for pattern %s in %s", pattern, wrf_dir)
    return paths
