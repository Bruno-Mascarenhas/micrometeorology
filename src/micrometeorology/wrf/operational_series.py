"""Turning one WRF run into one station's block of the operational record.

This is the netCDF half of the operational series: it finds the grid cell that
serves a coordinate, reads that cell's hourly series, and hands
:mod:`micrometeorology.wrf.operational_record` a frame to render. What the
record IS -- its schema, its rows, its repair -- lives there and needs no NetCDF
stack, so the CLIs that only read the record do not pay for one.

**Timestamps are naive station-local (UTC-03).** WRF's ``Times`` are UTC by
construction and :meth:`~micrometeorology.wrf.reader.WRFDataset.parse_times`
returns them aware; the conversion to local happens once, here, using the pinned
:data:`~micrometeorology.common.site.STATION_UTC_OFFSET_HOURS` rather than the
host's zone. Hour 21 local is 00 UTC, each run's initialisation step, where the
physics has not been called yet and every flux column is no-value.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from micrometeorology.common.site import STATION_UTC_OFFSET_HOURS
from micrometeorology.wrf.operational_record import (
    OPERATIONAL_CATALOG,
    PointSample,
    SeriesColumn,
    Station,
)
from micrometeorology.wrf.reader import WRFDataset
from micrometeorology.wrf.series import find_nearest_indices

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HOURS",
    "DomainAssignment",
    "OperationalBlock",
    "assign_domains",
    "extract_operational_block",
]


#: Hours of each run the operational file keeps: 00 UTC (21 local, the
#: initialisation step) through 23 UTC. Every one of the 1087 blocks the v1
#: extraction appended is exactly this long.
DEFAULT_HOURS = 24

#: How far a step's ``Times`` stamp may sit from its nominal hour. WRF writes the
#: model clock, which lands a couple of seconds late when the adaptive time step
#: does not divide the hour; anything larger is a non-hourly output interval and
#: this extraction has no defensible way to stamp it.
MAX_STEP_DRIFT = pd.Timedelta(minutes=5)

#: One degree of arc at the equator, km, used as the scale factor of the
#: equirectangular approximation that reports how far the serving cell centre
#: sits from the requested coordinate. It is a log line, not a computed
#: quantity: over a nest a few hundred km wide the approximation is well under
#: the cell size it is describing.
KM_PER_DEGREE = 111.32


@dataclass(frozen=True, slots=True)
class OperationalBlock:
    """One run's contribution to the file, and where it came from.

    Attributes
    ----------
    frame:
        ``(H, C)`` of published values indexed by naive station-local hour; the
        columns are the requested :class:`SeriesColumn` names, without the time
        fields, which the index carries.
    row:
        Grid row of the serving cell on the ``south_north`` axis.
    col:
        Grid column on the ``west_east`` axis.
    latitude:
        Cell-centre latitude actually served, degrees north.
    longitude:
        Cell-centre longitude actually served, degrees east.
    distance_km:
        Distance from the requested coordinate to that cell centre.
    """

    frame: pd.DataFrame
    row: int
    col: int
    latitude: float
    longitude: float
    distance_km: float


def _local_index(ds: WRFDataset, start_step: int, hours: int) -> pd.DatetimeIndex:
    """Naive station-local stamps for ``[start_step, start_step + hours)``.

    Raises
    ------
    ValueError
        When the file holds fewer steps than requested, or when the window is
        not on a one-hour grid. Both are silent corruption otherwise: a short
        window publishes a partial day as a whole one, and a non-hourly file
        would be stamped as if it were hourly.
    """
    times = ds.parse_times()
    stop = start_step + hours
    if start_step < 0 or hours < 1:
        raise ValueError(f"invalid window [{start_step}:{stop}]")
    if stop > len(times):
        raise ValueError(
            f"{ds.path.name} holds {len(times)} steps; the requested window "
            f"[{start_step}:{stop}] needs {stop}"
        )

    utc = pd.DatetimeIndex(times[start_step:stop]).tz_convert(None)
    hourly = utc.round("h")
    drift = np.abs(utc - hourly).max()
    if drift > MAX_STEP_DRIFT:
        raise ValueError(
            f"{ds.path.name}: step stamps sit {drift} from the hour; this "
            "extraction publishes whole local hours and cannot stamp a "
            "non-hourly output interval"
        )
    steps = hourly.to_series().diff().dropna().unique()
    if len(steps) > 1 or (len(steps) == 1 and steps[0] != pd.Timedelta(hours=1)):
        raise ValueError(f"{ds.path.name}: window is not contiguous hourly, steps={list(steps)}")

    return hourly + pd.Timedelta(hours=STATION_UTC_OFFSET_HOURS)


def _read_point_series(
    ds: WRFDataset, name: str, row: int, col: int, start_step: int, stop_step: int
) -> NDArray:
    """One variable's ``(stop - start,)`` series at a single cell.

    Reads through the NetCDF variable's own slicing, so a 76-step surface field
    costs one cell rather than the 76 x ny x nx the eager reader would load.
    """
    variable = ds.dataset.variables[name]
    if variable.ndim != 3:
        raise ValueError(
            f"{name} has dims {variable.dimensions}; the operational series "
            "carries surface fields only (Time, south_north, west_east)"
        )
    return np.asarray(variable[start_step:stop_step, row, col], dtype=np.float64)


def _collect_sample(
    ds: WRFDataset,
    columns: Sequence[SeriesColumn],
    row: int,
    col: int,
    start_step: int,
    hours: int,
) -> PointSample:
    stop_step = start_step + hours
    wanted = {source for column in columns for source in column.sources}
    wanted |= {source for column in columns for source in column.optional_sources}
    # GLW marks the steps that precede WRF's first radiation call, so a block
    # with any cold-start-sensitive column needs it even when nothing publishes
    # it -- see PointSample.uninitialised.
    if any(column.cold_start_blank for column in columns):
        wanted.add("GLW")

    fields: dict[str, NDArray] = {}
    for name in sorted(wanted):
        if ds.has_variable(name):
            fields[name] = _read_point_series(ds, name, row, col, start_step, stop_step)

    increments: dict[str, NDArray] = {}
    for name in sorted({source for column in columns for source in column.increments}):
        if not ds.has_variable(name):
            continue
        # One step before the window, so a window that does not start at the run's
        # own first step still differences against a real predecessor rather than
        # losing its first hour.
        first = max(start_step - 1, 0)
        accumulated = _read_point_series(ds, name, row, col, first, stop_step)
        step = np.diff(accumulated)
        increments[name] = step if first < start_step else np.insert(step, 0, np.nan)

    return PointSample(steps=hours, fields=fields, increments=increments)


def extract_operational_block(
    ds: WRFDataset,
    latitude: float,
    longitude: float,
    hours: int = DEFAULT_HOURS,
    start_step: int = 0,
    columns: Sequence[SeriesColumn] = OPERATIONAL_CATALOG,
) -> OperationalBlock:
    """Extract one run's block at the grid cell nearest a coordinate.

    Parameters
    ----------
    ds:
        An open wrfout.
    latitude, longitude:
        Target coordinate, degrees north and degrees east. Must lie inside the
        domain's cell-centre extent.
    hours:
        Steps to take, counted from *start_step*. Each is one local hour.
    start_step:
        First step of the window on the file's time axis; ``0`` is the run's
        initialisation, which the record keeps as its hour-21 local row and
        where every physics column is no-value.
    columns:
        The columns to compute. Any column whose wrfout sources this file lacks
        comes back all-NaN rather than absent, so the frame's shape does not
        depend on the model configuration.

    Returns
    -------
    OperationalBlock
        The frame plus the cell that served it.

    Raises
    ------
    ValueError
        When the coordinate falls outside the domain, when the file is shorter
        than the window, or when the window is not contiguous hourly.
    """
    lon_grid, lat_grid = ds.read_grid()
    lon_min, lon_max, lat_min, lat_max = ds.grid_bounds()
    if not (lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max):
        raise ValueError(
            f"({latitude}, {longitude}) is outside {ds.path.name}'s cell centres "
            f"(lat {lat_min:.4f}..{lat_max:.4f}, lon {lon_min:.4f}..{lon_max:.4f}); "
            "the nearest cell would be an edge of the domain, not the site"
        )

    row, col = find_nearest_indices(lat_grid, lon_grid, latitude, longitude)
    cell_lat = float(lat_grid[row, col])
    cell_lon = float(lon_grid[row, col])
    distance_km = KM_PER_DEGREE * float(
        np.hypot(
            cell_lat - latitude,
            (cell_lon - longitude) * np.cos(np.radians(latitude)),
        )
    )

    index = _local_index(ds, start_step, hours)
    sample = _collect_sample(ds, columns, row, col, start_step, hours)
    frame = pd.DataFrame(
        {column.name: column.evaluate(sample) for column in columns},
        index=index,
    )
    logger.info(
        "%s: cell (%d, %d) at %.4f, %.4f is %.2f km from the target; "
        "%d hours from %s, %d before the first radiation call",
        ds.path.name,
        row,
        col,
        cell_lat,
        cell_lon,
        distance_km,
        len(frame),
        frame.index[0],
        int(sample.uninitialised().sum()),
    )
    return OperationalBlock(
        frame=frame,
        row=row,
        col=col,
        latitude=cell_lat,
        longitude=cell_lon,
        distance_km=distance_km,
    )


@dataclass(frozen=True, slots=True)
class DomainAssignment:
    """Which wrfout serves one station, and why that one.

    Attributes
    ----------
    station:
        The point.
    path:
        The wrfout that will be read for it.
    dx_m:
        Grid spacing of that domain, metres.
    """

    station: Station
    path: Path
    dx_m: float


def assign_domains(
    stations: Sequence[Station], paths: Sequence[Path]
) -> tuple[tuple[DomainAssignment, ...], tuple[Station, ...]]:
    """Give every station the FINEST domain whose grid contains it.

    A nested WRF run covers the same site at several resolutions, and the point
    of the nest is that the innermost domain is the best answer wherever it
    reaches. So a station is served by the smallest ``DX`` among the domains
    that contain it — the tower falls in the 1 km nest, a site in the south of
    Bahia only in the 27 km parent — and a station no domain contains is
    returned unassigned rather than snapped to the nearest edge cell of one.

    Parameters
    ----------
    stations:
        The points to serve.
    paths:
        Candidate wrfout files, normally one per domain of the same run.

    Returns
    -------
    tuple[tuple[DomainAssignment, ...], tuple[Station, ...]]
        The assignments, in the order the stations were given, and the stations
        no domain covers.
    """
    extents: list[tuple[Path, float, tuple[float, float, float, float]]] = []
    for path in paths:
        with WRFDataset(path) as ds:
            extents.append((path, ds.dx, ds.grid_bounds()))

    assignments: list[DomainAssignment] = []
    uncovered: list[Station] = []
    for station in stations:
        covering = [
            (dx, path)
            for path, dx, (lon_min, lon_max, lat_min, lat_max) in extents
            if lat_min <= station.latitude <= lat_max and lon_min <= station.longitude <= lon_max
        ]
        if not covering:
            uncovered.append(station)
            continue
        dx, path = min(covering, key=lambda item: (item[0], item[1].name))
        assignments.append(DomainAssignment(station=station, path=path, dx_m=dx))
        logger.info(
            "%s (%.4f, %.4f): served by %s at %.0f m",
            station.name,
            station.latitude,
            station.longitude,
            path.name,
            dx,
        )
    return tuple(assignments), tuple(uncovered)
