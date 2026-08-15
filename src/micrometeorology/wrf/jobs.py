"""Coarse-grained work units for WRF JSON/GeoJSON generation.

Each unit covers one (wrfout file, variable) pair — or the grid GeoJSON /
standalone wind vectors for one file — and is executed by a worker process
that opens the NetCDF itself, derives its variable EAGERLY (a lazy dask graph
crossing the pool costs ~180x) and writes every timestep JSON in-process. Unit
payloads are plain strings and ints: no arrays, datasets or handles ever cross
the process boundary, and one persistent pool executes all units of a CLI run.

Outputs are written to a temporary name and ``os.replace``d into place, so a
killed worker can never leave a truncated JSON visible to consumers.
"""

import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.types import VARIABLE_NETCDF_MAP, WRFVariable
from micrometeorology.wrf import geojson, variables
from micrometeorology.wrf.batch import _max_tasks_per_child
from micrometeorology.wrf.geojson import create_wind_vectors_json, write_values_json_stream
from micrometeorology.wrf.reader import (
    WRFDataset,
    assert_one_file_per_domain,
    detect_grid_level,
    product_timezone,
)
from micrometeorology.wrf.value_source import build_value_frame_source, publishes_step

logger = logging.getLogger(__name__)

UnitKind = Literal["values_json", "poteolico", "wind_vectors", "grid_geojson"]

HDF5_LOCKING_ENV = "LABMIM_HDF5_FILE_LOCKING"

POTEOLICO_ALL_HEIGHTS: tuple[int, ...] = (50, 100, 150)

WIND_VECTORS_VARIABLE = "wind_vectors"

#: Casefolded request spelling -> the canonical name every downstream lookup
#: compares against.  Both the JSON export and the figure renderer fold through
#: this before choosing a daylight gate, a colormap or an output filename, so a
#: mis-cased ``-v swdown`` cannot reach one of them canonical and the other raw.
CANONICAL_VARIABLES: dict[str, str] = {
    **{variable.value.casefold(): variable.value for variable in WRFVariable},
    **{f"poteolico{height}": f"poteolico{height}" for height in POTEOLICO_ALL_HEIGHTS},
    WIND_VECTORS_VARIABLE: WIND_VECTORS_VARIABLE,
}

WIND_VECTORS_OUTPUT_ID = "WIND_VECTORS"


def values_output_id(variable: str) -> str:
    """The ``{D}_{ID}_{NNN}.json`` id one values variable publishes under.

    The single definition the writer, the manifest seeding and the failed-unit
    report all spell it with: a request the enum does not name still publishes
    under its uppercased self, and a second spelling of that rule would let the
    manifest advertise an id no file was ever written under.
    """
    return VARIABLE_NETCDF_MAP.get(variable, variable.upper())


def poteolico_output_id(height: int) -> str:
    """The ``{D}_{ID}_{NNN}.json`` id one wind-potential target height publishes under."""
    return f"POT_EOLICO_{height}M"


def unit_kind_for(variable: str) -> UnitKind:
    """The pipeline a requested variable name runs through.

    The single definition of the rule, so the manifest cannot classify a name
    one way while the unit that publishes it ran the other.
    """
    if variable.startswith(WRFVariable.WIND_POTENTIAL):
        return "poteolico"
    if variable == WIND_VECTORS_VARIABLE:
        return "wind_vectors"
    return "values_json"


def parse_poteolico_heights(variable: str) -> tuple[int, ...]:
    """Parse a poteolico variable name into the target heights it requests.

    ``"poteolico"`` means all of ``(50, 100, 150)``; ``"poteolico<nn>"`` means
    ``(<nn>,)`` for those same heights. Anything else raises ``ValueError``.
    """
    if variable == "poteolico":
        return POTEOLICO_ALL_HEIGHTS
    if variable.startswith("poteolico"):
        suffix = variable[len("poteolico") :]
        if suffix.isdigit() and int(suffix) in POTEOLICO_ALL_HEIGHTS:
            return (int(suffix),)
    raise ValueError(
        f"Unknown wind-potential variable {variable!r}: expected 'poteolico' or "
        f"one of {', '.join(f'poteolico{h}' for h in POTEOLICO_ALL_HEIGHTS)}"
    )


@dataclass(frozen=True, slots=True)
class WorkUnit:
    """Picklable description of one unit of work. Strings, ints and bools only.

    Attributes
    ----------
    kind:
        Which pipeline the worker runs for this unit.
    wrf_path:
        The wrfout the worker opens itself; the domain every output is named
        after is read back out of this name.
    variable:
        Exported variable id, empty for the grid GeoJSON unit.
    json_dir, geojson_dir:
        Output directories for the values/series artifacts and for the grid
        files respectively.
    skip_first:
        Leading time steps to leave unpublished.
    site_artifacts:
        Whether to also accumulate the consolidated ``.series.bin`` and
        ``.summary.json`` the site front-end ingests.
    """

    kind: UnitKind
    wrf_path: str
    variable: str
    json_dir: str
    geojson_dir: str
    skip_first: int = 0
    site_artifacts: bool = True

    @property
    def label(self) -> str:
        """Short ``filename:variable`` tag used in logs and the run manifest."""
        name = Path(self.wrf_path).name
        return f"{name}:{self.variable or self.kind}"


@dataclass(frozen=True, slots=True)
class UnitResult:
    """Outcome of one work unit, including the manifest of files written.

    Attributes
    ----------
    label:
        ``filename:variable`` tag of the unit that produced this result.
    kind:
        The unit's pipeline, carried through so the manifest can group results.
    files:
        Paths of every file the unit wrote.
    seconds:
        Wall-clock time the unit took, including the NetCDF open.
    warnings:
        Non-fatal messages the CLI surfaces after the run.
    error:
        ``None`` on success, otherwise ``"ExcType: message"``. A unit reports its
        failure rather than raising, so one bad variable cannot sink the run.
    start_local:
        Run metadata for the manifest: the run's first local datetime, formatted
        exactly like every JSON ``date_time``.
    n_steps:
        The file's time-step count, which the cell-series byte offsets depend on.
    domain:
        The domain this unit publishes under, derived from its filename so it is
        known even when the unit never managed to open the file.
    missing_variables:
        Output variable ids the unit published nothing under — because the
        wrfout does not carry the variable, or because the unit failed — so the
        manifest can say "no steps this run" instead of leaving the previous
        run's files silently vouched for under the freshly bumped version.
    """

    label: str
    kind: UnitKind
    files: tuple[str, ...] = ()
    seconds: float = 0.0
    warnings: tuple[str, ...] = field(default=())
    error: str | None = None
    start_local: str | None = None
    n_steps: int | None = None
    domain: str | None = None
    missing_variables: tuple[str, ...] = ()


def apply_hdf5_locking_policy() -> None:
    """Propagate the opt-in HDF5 locking policy BEFORE any pool is created.

    Forkserver children capture the environment when the forkserver starts,
    so this must run before the first executor. ``BEST_EFFORT`` is the
    documented opt-in for network filesystems; nothing is set by default.
    """
    policy = os.environ.get(HDF5_LOCKING_ENV)
    if policy:
        os.environ["HDF5_USE_FILE_LOCKING"] = policy


def _format_datetime(timestamp: datetime | None) -> str:
    """Format a datetime for the JSON ``date_time`` field the site parses.

    ``None`` yields ``"N/A"``; a timestamp object that cannot be formatted falls
    back to ``str()`` rather than failing the work unit.
    """
    if timestamp is None:
        return "N/A"
    try:
        formatted: str = timestamp.replace(minute=0, second=0, microsecond=0, tzinfo=None).strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    except Exception:  # noqa: BLE001 - formatting must never fail the work unit
        return str(timestamp)
    return formatted


def _atomic_values_json(
    output_path: Path,
    data: NDArray,
    vmin: float,
    vmax: float,
    date_str: str,
    wind_data: dict | None,
) -> str:
    tmp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    write_values_json_stream(tmp, data, vmin, vmax, date_str, wind_data)
    os.replace(tmp, output_path)
    return str(output_path)


def _atomic_json_dump(output_path: Path, payload: dict) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        # allow_nan=False: a NaN must fail the unit here, not ship as an
        # invalid-JSON token that only breaks in the browser. dumps-then-write,
        # because json.dump(obj, fp) never reaches the C encoder and costs 3x
        # for the same bytes on a wind-vector payload.
        f.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
    os.replace(tmp, output_path)
    return str(output_path)


# Fixed-point int32 encoding for the cell-series matrix: ``rint(value * 100)``
# keeps two decimals, SERIES_MISSING (int32 min) is the reserved no-value
# sentinel, and _SERIES_INT_MAX is the bound past which encoding raises.
SERIES_MISSING = -(2**31)
SERIES_SCALE = 100
_SERIES_INT_MAX = 2**31 - 1


class _SiteArtifactAccumulator:
    """Collect one (domain, variable)'s frames into the site's two consolidated artifacts.

    - ``{stem}.series.bin`` — row-major (cells x steps) little-endian int32
      matrix of ``rint(round(value, 2) * 100)``, :data:`SERIES_MISSING` where a
      step has no value. Fixed-size records let the front-end pull ONE cell's
      whole series with a single HTTP Range request.
    - ``{stem}.summary.json`` — per-step domain mean/min/max, for the preview
      panel (one request instead of one per step).

    Both round to two decimals exactly like the per-step JSONs, so the views of
    the data always agree.
    """

    def __init__(self, n_steps: int) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps
        self._matrix: NDArray | None = None
        self.indices: list[int] = []
        self.finite_cells: list[int] = []
        self.date_times: list[str] = []
        self.means: list[float] = []
        self.mins: list[float] = []
        self.maxs: list[float] = []

    def add(self, index: int, values: NDArray, date_str: str) -> None:
        """Quantize one time step's frame into column ``index`` of the matrix.

        Values are rounded to two decimals and stored as scaled int32; masked
        or NaN cells become :data:`SERIES_MISSING`. A step with at least one
        finite cell also contributes a mean/min/max row to the summary.

        Parameters
        ----------
        index:
            Column of the matrix, the step's position on the file's time axis.
        values:
            The step's ``(ny, nx)`` frame in the variable's published unit;
            ravelled row-major, so cell ids match the grid's ``linear_index``.
        date_str:
            The step's local datetime as the per-step JSONs write it.

        Raises
        ------
        ValueError
            When a value is too large for the int32 hundredths encoding.
        """
        arr = values.filled(np.nan) if isinstance(values, np.ma.MaskedArray) else values
        flat = np.round(np.ravel(np.asarray(arr)).astype(np.float64, copy=False), 2)
        if self._matrix is None:
            self._matrix = np.full((flat.size, self.n_steps), SERIES_MISSING, dtype="<i4")
        finite = np.isfinite(flat)
        scaled = np.rint(flat[finite] * SERIES_SCALE)
        # Refused, not clipped: int32 hundredths cannot represent past
        # +-21,474,836.47, and clipping would publish the ceiling in the series
        # while the per-step JSON and the summary publish the true number.
        if scaled.size and (scaled.min() <= SERIES_MISSING or scaled.max() > _SERIES_INT_MAX):
            extreme = flat[finite][np.argmax(np.abs(scaled))]
            raise ValueError(
                f"step {index}: {extreme:g} exceeds what the int32 series can carry "
                f"(+-{_SERIES_INT_MAX / SERIES_SCALE:,.2f}); the field is broken, not extreme"
            )
        column = np.full(flat.size, SERIES_MISSING, dtype="<i4")
        column[finite] = scaled.astype("<i4")
        self._matrix[:, index] = column
        if finite.any():
            valid = flat[finite]
            self.indices.append(index)
            self.date_times.append(date_str)
            self.finite_cells.append(int(finite.sum()))
            self.means.append(round(float(valid.mean()), 2))
            self.mins.append(round(float(valid.min()), 2))
            self.maxs.append(round(float(valid.max()), 2))

    def write(self, json_dir: str, stem: str, domain: str, variable: str) -> list[str]:
        """Atomically flush the ``.series.bin`` and ``.summary.json`` artifacts.

        Returns the paths written, empty when no finite step was accumulated.
        """
        if self._matrix is None or not self.indices:
            return []
        series_path = Path(json_dir) / f"{stem}.series.bin"
        tmp = series_path.with_name(f".{series_path.name}.tmp-{os.getpid()}")
        tmp.write_bytes(self._matrix.tobytes())
        os.replace(tmp, series_path)

        summary = {
            "format": "domain-summary-v1",
            "domain": domain,
            "variable": variable,
            "indices": self.indices,
            "date_times": self.date_times,
            "mean": self.means,
            "min": self.mins,
            "max": self.maxs,
            # The denominator of every mean above — not the cell count, and not
            # constant: the clearness index masks cells by solar geometry, so a
            # sunrise mean covers a fraction of the grid and reads like a
            # physical trend when it is really the shrinking sample.
            "finite_cells": self.finite_cells,
            "cells": self._matrix.shape[0] if self._matrix is not None else 0,
        }
        summary_path = _atomic_json_dump(Path(json_dir) / f"{stem}.summary.json", summary)
        return [str(series_path), summary_path]


def _run_values_unit(unit: WorkUnit, dataset: WRFDataset) -> tuple[list[str], list[str], list[str]]:
    """Write every timestep JSON of one values variable.

    Returns the files written, the warnings, and the output variable ids this
    wrfout does not carry, and any variable that published no step at all — both
    leave the previous run's fixed output names newest on disk, which is what
    the manifest must not vouch for.  An all-night SWDOWN window and
    ``--skip-first`` reach the second case: legitimately empty for this run, and
    still not evidence for last run's matrix.
    """
    written_files: list[str] = []
    unit_warnings: list[str] = []
    grid_level = dataset.grid_level.value
    output_variable_id = values_output_id(unit.variable)
    time_step_metadata = dataset.build_date_metadata(skip_first_n=unit.skip_first)

    frame_source = build_value_frame_source(dataset, unit.variable)
    if frame_source is None:
        unit_warnings.append(f"Variable {unit.variable.upper()} not found — skipped")
        return written_files, unit_warnings, [output_variable_id]

    site_artifact_accumulator = (
        _SiteArtifactAccumulator(dataset.n_time_steps) if unit.site_artifacts else None
    )
    for step_metadata in time_step_metadata:
        if step_metadata.get("skip"):
            continue
        if not publishes_step(unit.variable, step_metadata):
            continue
        time_step_index = step_metadata["index"]
        frame_values = frame_source.frame_for_step(time_step_index)
        # An all-non-finite step is not published: the clearness index is null
        # wherever cos(z) is below its cutoff, so sunrise and sunset steps are
        # empty even inside the daylight window. Writing them would desync the
        # run's two artifacts — `availability` derives from the files WRITTEN,
        # while the summary records only steps with a finite cell.
        if not np.isfinite(np.asarray(frame_values, dtype=float)).any():
            continue
        formatted_local_time = _format_datetime(step_metadata["datetime_local"])
        output_path = (
            Path(unit.json_dir) / f"{grid_level}_{output_variable_id}_{time_step_index:03d}.json"
        )
        written_files.append(
            _atomic_values_json(
                output_path,
                frame_values,
                frame_source.scale_min,
                frame_source.scale_max,
                formatted_local_time,
                None,
            )
        )
        if site_artifact_accumulator is not None:
            site_artifact_accumulator.add(time_step_index, frame_values, formatted_local_time)
    if site_artifact_accumulator is not None:
        written_files.extend(
            site_artifact_accumulator.write(
                unit.json_dir,
                f"{grid_level}_{output_variable_id}",
                grid_level,
                output_variable_id,
            )
        )
    if not written_files:
        # Found the variable and still published nothing: every step was skipped,
        # gated out or all-non-finite (a wrfout window lying entirely outside the
        # daylight hours does this to SWDOWN/SWUP/SWNET/clearness_index). The
        # previous run's fixed output names are then the newest on disk, which is
        # the same state a variable the wrfout does not carry leaves behind — so
        # it is reported the same way, or the directory-wide series descriptor
        # would vouch for that matrix under THIS run's step count.
        unit_warnings.append(
            f"Variable {unit.variable.upper()} published no step in this window — "
            "its previous artifacts are not vouched for"
        )
        return written_files, unit_warnings, [output_variable_id]
    return written_files, unit_warnings, []


def _run_poteolico_unit(unit: WorkUnit, dataset: WRFDataset) -> tuple[list[str], list[str]]:
    """Write every timestep JSON of the wind-potential heights the unit requests.

    Each height publishes under its own ``POT_EOLICO_{h}M`` output id, with the
    arrow overlay embedded in the same file.
    """
    files: list[str] = []
    grid_level = dataset.grid_level.value
    time_step_metadata = dataset.build_date_metadata(skip_first_n=unit.skip_first)
    targets = parse_poteolico_heights(unit.variable)
    series = variables.stream_wind_at_heights(dataset, targets)

    for height_series in series:
        suffix = poteolico_output_id(height_series.target)
        accumulator = (
            _SiteArtifactAccumulator(dataset.n_time_steps) if unit.site_artifacts else None
        )
        for step_metadata in time_step_metadata:
            if step_metadata.get("skip"):
                continue
            step_index = step_metadata["index"]
            date_str = _format_datetime(step_metadata["datetime_local"])
            output_path = Path(unit.json_dir) / f"{grid_level}_{suffix}_{step_index:03d}.json"
            files.append(
                _atomic_values_json(
                    output_path,
                    height_series.speed_steps[step_index],
                    height_series.vmin,
                    height_series.vmax,
                    date_str,
                    height_series.wind_vectors[step_index],
                )
            )
            if accumulator is not None:
                accumulator.add(step_index, height_series.speed_steps[step_index], date_str)
        if accumulator is not None:
            files.extend(
                accumulator.write(unit.json_dir, f"{grid_level}_{suffix}", grid_level, suffix)
            )
    return files, []


def _run_wind_vectors_unit(unit: WorkUnit, dataset: WRFDataset) -> tuple[list[str], list[str]]:
    """Write the standalone 10-m arrow overlay for every timestep.

    These files carry no grid values, so the site can draw wind arrows over ANY
    variable without every variable's JSON embedding them.
    """
    files: list[str] = []
    grid_level = dataset.grid_level.value
    time_step_metadata = dataset.build_date_metadata(skip_first_n=unit.skip_first)
    u10, v10, _vmin, _vmax = variables.extract_wind(dataset)

    for step_metadata in time_step_metadata:
        if step_metadata.get("skip"):
            continue
        step_index = step_metadata["index"]
        u = variables.materialize_2d(u10[step_index : step_index + 1])
        v = variables.materialize_2d(v10[step_index : step_index + 1])
        payload = create_wind_vectors_json(
            u, v, date_time=step_metadata["datetime_local"], downsampling=4
        )
        output_path = (
            Path(unit.json_dir) / f"{grid_level}_{WIND_VECTORS_OUTPUT_ID}_{step_index:03d}.json"
        )
        files.append(_atomic_json_dump(output_path, payload))
    return files, []


def _run_grid_geojson_unit(unit: WorkUnit, dataset: WRFDataset) -> tuple[list[str], list[str]]:
    """Write the domain's cell rectangles, once as legacy GeoJSON and once compact."""
    lon, lat = dataset.read_grid()
    grid_level = dataset.grid_level.value
    output_path = Path(unit.geojson_dir) / f"{grid_level}.geojson"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    geojson.write_grid_geojson_stream(tmp, lon, lat, dataset.dx, dataset.dy)
    os.replace(tmp, output_path)
    logger.info("Saved GeoJSON: %s", output_path)

    # Compact companion preferred by the site front-end (which falls back to
    # the legacy .geojson above); a fraction of the size on the wire.
    compact = Path(unit.geojson_dir) / f"{grid_level}.grid.json"
    compact_tmp = compact.with_name(f".{compact.name}.tmp-{os.getpid()}")
    geojson.write_grid_compact_json_stream(compact_tmp, lon, lat, dataset.dx, dataset.dy)
    os.replace(compact_tmp, compact)
    return [str(output_path), str(compact)], []


def _run_time_metadata(dataset: WRFDataset) -> tuple[str | None, int | None]:
    """The run's first local datetime (JSON date_time format) and step count."""
    try:
        times = dataset.parse_times()
        if not times:
            return None, None
        return _format_datetime(times[0].astimezone(product_timezone())), len(times)
    except Exception:  # noqa: BLE001 - metadata is best-effort; units must not fail over it
        return None, None


# `{i:03d}` is a MINIMUM width — step 1000 is written `_1000.json` — so the
# scanner must accept three OR MORE digits, or those frames reach neither the
# timeline nor `availability`.
_TIMESTEP_FILE_RE = re.compile(r"^(D\d{2})_([A-Z0-9_]+)_(\d{3,})\.json$")


def _compress_index_ranges(indices: set[int]) -> list[list[int]]:
    """Compress a set of indices into sorted inclusive ``[start, end]`` runs."""
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def write_run_manifest(
    json_dir: str | Path,
    results: Sequence[UnitResult],
    requested_variables: Sequence[str] = (),
    covers_every_variable: bool = True,
) -> Path | None:
    """Write ``manifest.json`` into *json_dir* after a generation run.

    The site fetches it with ``Cache-Control: no-cache`` and appends
    ``?v=<version>`` to every data URL, so the fixed-name data files stay
    cacheable yet fresh; without the manifest the front-end uses unversioned
    URLs.

    The version bumps whenever the run executed ANY unit, failed runs included:
    a crashed unit may already have replaced files under unchanged names
    (``files`` is empty on error, so the count proves nothing), and a stale
    version would pin long-cached clients to outdated data — an extra bump only
    costs a cache miss.

    Parameters
    ----------
    json_dir:
        Directory the manifest is written into, alongside the data files it
        describes.
    results:
        Every unit result of the run, failures included.
    requested_variables:
        The variable ids the run was asked for, so ``availability`` can carry an
        empty range for one that wrote nothing at all.
    covers_every_variable:
        Whether the run covered the full variable set. The ``features``
        templates are directory-wide wildcards, so a partial run must not vouch
        for files it did not rewrite.

    Returns
    -------
    Path | None
        Path of the written ``manifest.json``, or ``None`` when *results* is
        empty and there is nothing to advertise.
    """
    if not results:
        return None
    written = sum(len(result.files) for result in results)
    # The same filename->domain rule the writers name their files with, so the
    # advertised domains can never disagree with the published `{D}_*` files.
    domains = sorted({result.domain for result in results if result.domain})
    payload: dict = {
        "version": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "domains": domains,
        "files": written,
    }

    # v2 fields (additive; the site falls back to hardcoded defaults without
    # them), derived from the files ACTUALLY written this run, never from
    # arithmetic that could drift from the writers. Keyed by (domain, variable),
    # never by variable alone: `availability` has no domain axis, and domains do
    # publish different step sets for the same hour (the clearness index masks by
    # cos(z) geographically), so a union sends the site after a file this run
    # never wrote — and fixed names replaced in place serve the PREVIOUS run's
    # frame under this run's version stamp.
    var_indices: dict[tuple[str, str], set[int]] = {}
    domain_indices: dict[str, set[int]] = {}
    have_summary = False
    have_series = False
    for result in results:
        # A variable the wrfout does not carry wrote no file, yet its fixed
        # output names still hold the previous run's data: seed an empty index
        # set so availability says "no steps this run" instead of omitting the
        # key, which the front-end reads as "full range".
        for missing_variable in result.missing_variables:
            if result.domain:
                var_indices.setdefault((result.domain, missing_variable), set())
        for file_path in result.files:
            name = Path(file_path).name
            match = _TIMESTEP_FILE_RE.match(name)
            if match:
                index = int(match.group(3))
                var_indices.setdefault((match.group(1), match.group(2)), set()).add(index)
                domain_indices.setdefault(match.group(1), set()).add(index)
            elif name.endswith(".summary.json"):
                have_summary = True
            elif name.endswith(".series.bin"):
                have_series = True

    # ONE timeline is rendered across all domains, so advertise the intersection
    # of the per-domain ranges: a mixed-length run must never label a frame some
    # domain lacks as available.
    if domain_indices:
        index_min = max(min(indices) for indices in domain_indices.values())
        index_max = min(max(indices) for indices in domain_indices.values())
    else:
        index_min, index_max = 0, -1
    if index_min <= index_max:
        payload["format"] = "labmim-data-manifest-v2"
        payload["timezone"] = str(product_timezone())
        payload["index_min"] = index_min
        payload["index_max"] = index_max
        # The anchor pairs with index 0 (the file's first time step), NOT with
        # index_min: the client must anchor `initialIndex=0` whatever index_min
        # says. Results arrive in completion order, so advertise it only when
        # every unit that reported one agrees.
        start_locals = {r.start_local for r in results if r.start_local}
        if len(start_locals) == 1:
            payload["start_local"] = start_locals.pop()

        full_range = set(range(index_min, index_max + 1))
        # INTERSECTION across domains, for the same reason index_min/max are:
        # advertise an index only when every domain wrote that variable's frame
        # for it, since the other direction serves a stale frame as a current one.
        per_variable: dict[str, list[set[int]]] = {}
        for (_domain, var), indices in var_indices.items():
            per_variable.setdefault(var, []).append(indices)
        # A requested variable that wrote nothing gets an empty set: an omitted
        # key reads as "full range" to the page, which then serves the PREVIOUS
        # run's frames (fixed names, still on disk) under the new version.
        # Only the values pipeline, whose request maps to ONE output id. Wind
        # potential fans out over target heights the run may not all have been
        # asked for, and the vector overlay writes its own stems, so seeding
        # from their requested name advertises ids nothing ever wrote.
        for requested in requested_variables:
            if unit_kind_for(requested) == "values_json":
                per_variable.setdefault(values_output_id(requested), [set()])
        availability = {
            var: _compress_index_ranges(shared)
            for var, shared in sorted(
                (var, set.intersection(*per_domain) & full_range)
                for var, per_domain in per_variable.items()
            )
            if not full_range <= shared
        }
        if availability:
            payload["availability"] = availability

        # These descriptors are a byte-offset contract: a failed unit, or one
        # whose variable the wrfout no longer carries, leaves LAST run's
        # {D}_{VAR}.series.bin/.summary.json in place, and a stale matrix of a
        # different step count is read at the WRONG offsets, not merely stale.
        # So vouch only when every unit succeeded and found its variable, and
        # only when the run covered every variable — the templates below are
        # wildcards over the whole directory. (The site falls back to the
        # per-step JSONs otherwise.)
        run_clean = covers_every_variable and not any(
            r.error or r.missing_variables for r in results
        )
        features: dict = {}
        if have_summary and run_clean:
            features["domain_summary"] = {
                "format": "domain-summary-v1",
                "template": "JSON/{domain}_{variable}.summary.json",
            }
        # Matrices span columns 0..n_steps-1 whatever was written (skip-first
        # and night gaps are MISSING columns), so the offset contract needs the
        # step count, and a run whose files disagree on it would corrupt offsets.
        step_counts = {r.n_steps for r in results if r.n_steps}
        if have_series and run_clean and len(step_counts) == 1:
            n_steps = step_counts.pop()
            features["cell_series"] = {
                "format": "cell-series-int32-le-v1",
                "template": "JSON/{domain}_{variable}.series.bin",
                "dtype": "int32",
                "byte_order": "little",
                "scale": 0.01,
                "missing": SERIES_MISSING,
                "index_min": 0,
                "index_max": n_steps - 1,
            }
        if features:
            payload["features"] = features

    return Path(_atomic_json_dump(Path(json_dir) / "manifest.json", payload))


_TEMP_FILE_PATTERN = re.compile(r"\.tmp-(\d+)$")


def _sweep_stale_temp_files(
    dirs: Sequence[str], *, sweep_pids: frozenset[int] = frozenset()
) -> int:
    """Remove orphaned ``.tmp-<pid>`` files whose owning process is dead.

    Only files whose embedded PID no longer exists are removed, so concurrent
    runs writing into the same directories are never disturbed. ``sweep_pids``
    forces pids whose debris is known-orphaned although the process is alive —
    the serial path writes with the parent's own pid, so its failed-unit
    leftovers would otherwise survive every end-of-run sweep.
    """
    removed = 0
    for directory in dict.fromkeys(dirs):
        for path in Path(directory).glob(".*.tmp-*"):
            match = _TEMP_FILE_PATTERN.search(path.name)
            if match is None:
                continue
            pid = int(match.group(1))
            if pid in sweep_pids:
                path.unlink(missing_ok=True)
                removed += 1
                continue
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                path.unlink(missing_ok=True)
                removed += 1
            except PermissionError:  # pragma: no cover - pid exists, other user
                continue
    return removed


def _unit_domain(unit: WorkUnit) -> str | None:
    """The domain a unit publishes under, from its filename alone."""
    level = detect_grid_level(unit.wrf_path)
    return level.value if level is not None else None


def _unit_output_ids(unit: WorkUnit) -> tuple[str, ...]:
    """The output file ids a unit publishes under, from the unit alone.

    Never raises: this runs on the failure path, where the unit's own variable
    name may be what raised. A name no height parses out of owns no id, and
    guessing one advertises a variable nothing ever wrote.
    """
    if unit.kind == "values_json":
        return (values_output_id(unit.variable),)
    if unit.kind == "poteolico":
        try:
            heights = parse_poteolico_heights(unit.variable)
        except ValueError:
            return ()
        return tuple(poteolico_output_id(height) for height in heights)
    if unit.kind == "wind_vectors":
        return (WIND_VECTORS_OUTPUT_ID,)
    return ()


def process_unit(unit: WorkUnit) -> UnitResult:
    """Execute one work unit. Runs in a worker process; never raises.

    A failure is reported in the result's ``error`` instead, so one bad variable
    cannot take the pool's other units with it. Setting ``LABMIM_TEST_CRASH_UNIT``
    to a variable name makes that unit exit as an OOM-killed worker would, which
    is how the pool-break recovery path is exercised.
    """
    if os.environ.get("LABMIM_TEST_CRASH_UNIT") == unit.variable:
        os._exit(137)
    t0 = time.perf_counter()
    missing_variables: list[str] = []
    try:
        with WRFDataset(unit.wrf_path) as dataset:
            if unit.kind == "values_json":
                files, warnings, missing_variables = _run_values_unit(unit, dataset)
            elif unit.kind == "poteolico":
                files, warnings = _run_poteolico_unit(unit, dataset)
            elif unit.kind == "wind_vectors":
                files, warnings = _run_wind_vectors_unit(unit, dataset)
            elif unit.kind == "grid_geojson":
                files, warnings = _run_grid_geojson_unit(unit, dataset)
            else:
                raise ValueError(f"Unknown work unit kind: {unit.kind}")
            start_local, n_steps = _run_time_metadata(dataset)
        return UnitResult(
            label=unit.label,
            kind=unit.kind,
            files=tuple(files),
            seconds=time.perf_counter() - t0,
            warnings=tuple(warnings),
            start_local=start_local,
            n_steps=n_steps,
            domain=_unit_domain(unit),
            missing_variables=tuple(missing_variables),
        )
    except Exception as exc:
        logger.exception("Work unit failed: %s", unit.label)
        return UnitResult(
            label=unit.label,
            kind=unit.kind,
            seconds=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
            domain=_unit_domain(unit),
            missing_variables=_unit_output_ids(unit),
        )


_KIND_COST_RANK = {"poteolico": 0, "values_json": 1, "wind_vectors": 1, "grid_geojson": 2}


def _init_worker_logging(level: int) -> None:
    """Give a fresh worker the logging config forkserver children never inherit.

    Without it every worker record falls through to ``logging.lastResort``,
    unattributable across 44 concurrent workers. Records carry the pid and go
    to stderr, leaving stdout to the CLI's progress lines.

    Held at WARNING or above whatever the parent's level: the streaming writers
    log their ``.tmp-<pid>`` argument rather than the final path, so worker INFO
    would name files that never exist.
    """
    logging.basicConfig(
        level=max(level, logging.WARNING),
        format="%(asctime)s | %(process)d | %(name)-40s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


def build_units(
    wrf_paths: Sequence[str | Path],
    variables: Sequence[str],
    json_dir: str | Path,
    geojson_dir: str | Path,
    skip_first: int = 0,
    site_artifacts: bool = True,
) -> list[WorkUnit]:
    """Expand (files x variables) into work units, one grid GeoJSON per file.

    The unit kind follows from the variable name: anything starting with the
    wind-potential prefix streams heights, ``wind_vectors`` writes the standalone
    arrow overlay, everything else is a values variable.

    Returns
    -------
    list[WorkUnit]
        Units in build order; :func:`execute_units` reorders them by cost.

    Raises
    ------
    ValueError
        Before any unit is built, when two of *wrf_paths* belong to the same
        domain: see
        :func:`~micrometeorology.wrf.reader.assert_one_file_per_domain`.
    """
    assert_one_file_per_domain(wrf_paths)
    units: list[WorkUnit] = []
    for path in wrf_paths:
        units.append(
            WorkUnit(
                kind="grid_geojson",
                wrf_path=str(path),
                variable="",
                json_dir=str(json_dir),
                geojson_dir=str(geojson_dir),
                skip_first=skip_first,
                site_artifacts=site_artifacts,
            )
        )
        units.extend(
            WorkUnit(
                kind=unit_kind_for(var_name),
                wrf_path=str(path),
                variable=var_name,
                json_dir=str(json_dir),
                geojson_dir=str(geojson_dir),
                skip_first=skip_first,
                site_artifacts=site_artifacts,
            )
            for var_name in variables
        )
    return units


def execute_units(
    units: Sequence[WorkUnit],
    workers: int,
    *,
    echo: Callable[[str], None] = logger.info,
) -> list[UnitResult]:
    """Execute all units on ONE persistent process pool, heaviest first.

    Unit failures are isolated in the unit's result. If the pool itself breaks
    (an OOM-killed worker), the incomplete units are retried one at a time in
    single-worker pools, so a unit that keeps killing its worker fails alone —
    with an error result, so callers can exit non-zero — instead of dooming the
    rest.

    Parameters
    ----------
    units:
        Units to run; a single resolved worker runs them in this process.
    workers:
        Upper bound on pool size, clamped to the unit count.
    echo:
        Progress sink, one line per completed unit.

    Returns
    -------
    list[UnitResult]
        One result per unit, in COMPLETION order, not in *units* order.
    """
    if not units:
        return []
    apply_hdf5_locking_policy()

    ordered = sorted(units, key=lambda u: _KIND_COST_RANK.get(u.kind, 1))
    n_workers = max(1, min(workers, len(ordered)))
    t0 = time.perf_counter()

    output_dirs = [d for unit in ordered for d in (unit.json_dir, unit.geojson_dir)]

    if n_workers == 1:
        serial_results = [process_unit(u) for u in ordered]
        # All units are done: our own pid's leftovers are orphans too.
        _sweep_stale_temp_files(output_dirs, sweep_pids=frozenset({os.getpid()}))
        echo(f"✓ {len(serial_results)} work units in {time.perf_counter() - t0:.1f}s (serial)")
        return serial_results

    results: list[UnitResult] = []
    pending: list[WorkUnit] = list(ordered)
    completed: set[int] = set()
    worker_log_level = logging.getLogger().getEffectiveLevel()
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            max_tasks_per_child=_max_tasks_per_child(n_workers),
            initializer=_init_worker_logging,
            initargs=(worker_log_level,),
        ) as pool:
            futures = {pool.submit(process_unit, unit): idx for idx, unit in enumerate(pending)}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                completed.add(idx)
                results.append(result)
                status = f"✗ {result.error}" if result.error else f"{len(result.files)} files"
                echo(f"  [{len(results)}/{len(ordered)}] {result.label}: {status}")
        pending = []
    except BrokenProcessPool:
        pending = [u for idx, u in enumerate(pending) if idx not in completed]
        echo(
            f"⚠ Worker pool broke (possible OOM kill); retrying "
            f"{len(pending)} incomplete units in isolation"
        )

    if pending:
        swept = _sweep_stale_temp_files(output_dirs)
        if swept:
            echo(f"  swept {swept} stale temp file(s) left by killed workers")
    for unit in pending:
        try:
            with ProcessPoolExecutor(
                max_workers=1,
                initializer=_init_worker_logging,
                initargs=(worker_log_level,),
            ) as retry_pool:
                result = retry_pool.submit(process_unit, unit).result()
        except BrokenProcessPool:
            result = UnitResult(
                label=unit.label,
                kind=unit.kind,
                error="worker crashed while processing this unit (possible OOM kill)",
                domain=_unit_domain(unit),
                missing_variables=_unit_output_ids(unit),
            )
        results.append(result)
        status = f"✗ {result.error}" if result.error else f"{len(result.files)} files"
        echo(f"  [{len(results)}/{len(ordered)}] {result.label}: {status}")

    # Always sweep: a unit that failed mid-write WITHOUT breaking the pool (the
    # common case, since process_unit catches everything) leaves its temp file
    # behind, and the debris must not wait for a future failing run.
    swept = _sweep_stale_temp_files(output_dirs, sweep_pids=frozenset({os.getpid()}))
    if swept:
        echo(f"  swept {swept} stale temp file(s)")

    echo(f"✓ {len(results)} work units in {time.perf_counter() - t0:.1f}s")
    return results
