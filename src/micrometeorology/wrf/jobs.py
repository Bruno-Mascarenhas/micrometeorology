"""Coarse-grained work units for WRF JSON/GeoJSON generation.

Each unit covers one (wrfout file, variable) pair — or the grid GeoJSON /
standalone wind vectors for one file — and is executed by a worker process
that opens the NetCDF itself, derives its variable eagerly, computes scale
bounds, and writes every timestep JSON in-process. Unit payloads are plain
strings and ints: no arrays, datasets, or handles ever cross the process
boundary, and one persistent pool executes all units for a CLI run.

Output files are written to a temporary name and ``os.replace``d into place
so a killed worker can never leave a truncated JSON visible to consumers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from micrometeorology.common.types import VARIABLE_NETCDF_MAP, WRFVariable
from micrometeorology.wrf import geojson
from micrometeorology.wrf import variables as vmod
from micrometeorology.wrf.batch import _max_tasks_per_child
from micrometeorology.wrf.geojson import create_wind_vectors_json, write_values_json_stream
from micrometeorology.wrf.reader import WRFDataset, product_timezone

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

UnitKind = Literal["values_json", "poteolico", "wind_vectors", "grid_geojson"]

HDF5_LOCKING_ENV = "LABMIM_HDF5_FILE_LOCKING"

POTEOLICO_ALL_HEIGHTS: tuple[int, ...] = (50, 100, 150)


def parse_poteolico_heights(variable: str) -> tuple[int, ...]:
    """Parse a poteolico variable name into the target heights it requests.

    ``"poteolico"`` requests all heights ``(50, 100, 150)``;
    ``"poteolico<nn>"`` requests ``(<nn>,)`` for ``nn`` in ``{50, 100, 150}``.
    Anything else raises ``ValueError`` with a CLI-friendly message.
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
    """Picklable description of one unit of work. Strings, ints and bools only."""

    kind: UnitKind
    wrf_path: str
    variable: str
    json_dir: str
    geojson_dir: str
    skip_first: int = 0
    site_artifacts: bool = True

    @property
    def label(self) -> str:
        name = Path(self.wrf_path).name
        return f"{name}:{self.variable or self.kind}"


@dataclass(frozen=True, slots=True)
class UnitResult:
    """Outcome of one work unit, including the manifest of files written."""

    label: str
    kind: UnitKind
    files: tuple[str, ...] = ()
    seconds: float = 0.0
    warnings: tuple[str, ...] = field(default=())
    error: str | None = None
    # Run metadata for the manifest: the run's first local datetime (formatted
    # like every JSON date_time) and the file's time-step count.
    start_local: str | None = None
    n_steps: int | None = None


def apply_hdf5_locking_policy() -> None:
    """Propagate the opt-in HDF5 locking policy BEFORE any pool is created.

    Forkserver children capture the environment when the forkserver starts,
    so this must run before the first executor. ``BEST_EFFORT`` is the
    documented opt-in for network filesystems; nothing is set by default.
    """
    policy = os.environ.get(HDF5_LOCKING_ENV)
    if policy:
        os.environ["HDF5_USE_FILE_LOCKING"] = policy


def _format_datetime(dt) -> str:
    """Format a datetime for JSON output (identical to the legacy CLI helper)."""
    if dt is None:
        return "N/A"
    try:
        formatted: str = dt.replace(minute=0, second=0, microsecond=0, tzinfo=None).strftime(
            "%d/%m/%Y %H:%M:%S"
        )
    except Exception:
        return str(dt)
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
        # allow_nan=False: a NaN that slips into a payload must fail the work
        # unit here, not ship as an invalid-JSON token that only breaks in
        # every visitor's browser.
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    os.replace(tmp, output_path)
    return str(output_path)


SERIES_MISSING = -(2**31)
SERIES_SCALE = 100
_SERIES_INT_MAX = 2**31 - 1


class _SiteArtifactAccumulator:
    """Collects the per-step frames of one (domain, variable) into the two
    consolidated artifacts the site front-end ingests.

    - ``{stem}.series.bin`` — row-major (cells x steps) little-endian int32
      matrix: ``rint(round(value, 2) * 100)``, :data:`SERIES_MISSING` where a
      step has no value (never written, masked, or NaN). Fixed-size records
      let the front-end fetch ONE cell's full series with a single HTTP Range
      request instead of downloading every per-step JSON of the domain.
    - ``{stem}.summary.json`` — per-step domain mean/min/max over the same
      rounded values, for the lightweight domain-preview panel (one request
      instead of one per step).

    Values match the per-step JSONs (same round-to-2-decimals), so both views
    of the data always agree.
    """

    def __init__(self, n_steps: int) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.n_steps = n_steps
        self._matrix: NDArray | None = None
        self.indices: list[int] = []
        self.date_times: list[str] = []
        self.means: list[float] = []
        self.mins: list[float] = []
        self.maxs: list[float] = []

    def add(self, index: int, values: NDArray, date_str: str) -> None:
        arr = values.filled(np.nan) if isinstance(values, np.ma.MaskedArray) else values
        flat = np.round(np.ravel(np.asarray(arr)).astype(np.float64, copy=False), 2)
        if self._matrix is None:
            self._matrix = np.full((flat.size, self.n_steps), SERIES_MISSING, dtype="<i4")
        finite = np.isfinite(flat)
        column = np.full(flat.size, SERIES_MISSING, dtype="<i4")
        column[finite] = np.clip(
            np.rint(flat[finite] * SERIES_SCALE), SERIES_MISSING + 1, _SERIES_INT_MAX
        ).astype("<i4")
        self._matrix[:, index] = column
        if finite.any():
            valid = flat[finite]
            self.indices.append(index)
            self.date_times.append(date_str)
            self.means.append(round(float(valid.mean()), 2))
            self.mins.append(round(float(valid.min()), 2))
            self.maxs.append(round(float(valid.max()), 2))

    def write(self, json_dir: str, stem: str, domain: str, variable: str) -> list[str]:
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
        }
        summary_path = _atomic_json_dump(Path(json_dir) / f"{stem}.summary.json", summary)
        return [str(series_path), summary_path]


def _values_unit_frames(
    ds: WRFDataset,
    variable: str,
) -> tuple[Callable[[int], NDArray], float, float] | None:
    """Return ``(frame(i), vmin, vmax)`` for a plain values-JSON variable.

    Mirrors the per-variable branches of the legacy task builder exactly —
    same extractors, same per-step arithmetic — so output bytes match.
    Returns ``None`` when the variable is absent from the file.
    """
    if variable == WRFVariable.TEMPERATURE:
        t2, vmin, vmax = vmod.extract_temperature(ds)
        return (
            lambda i: vmod.materialize_2d(vmod.extract_temperature_step(t2[i : i + 1, :, :])),
            vmin,
            vmax,
        )
    if variable == WRFVariable.SKIN_TEMPERATURE:
        tsk, vmin, vmax = vmod.extract_skin_temperature(ds)
        return (
            lambda i: vmod.materialize_2d(vmod.extract_temperature_step(tsk[i : i + 1, :, :])),
            vmin,
            vmax,
        )
    if variable == WRFVariable.RELATIVE_HUMIDITY:
        rh, vmin, vmax = vmod.extract_relative_humidity(ds)
        return lambda i: vmod.materialize_2d(rh[i : i + 1, :, :]), vmin, vmax
    if variable == WRFVariable.RAIN:
        total, vmin, vmax = vmod.extract_rain(ds)
        return lambda i: vmod.materialize_2d(vmod.extract_rain_step(total, i)), vmin, vmax
    if variable == WRFVariable.WIND:
        u10, v10, vmin, vmax = vmod.extract_wind(ds)

        def wind_frame(i: int) -> NDArray:
            u = vmod.materialize_2d(u10[i : i + 1])
            v = vmod.materialize_2d(v10[i : i + 1])
            speed: NDArray = np.hypot(u, v)
            return speed

        return wind_frame, vmin, vmax
    if variable == WRFVariable.WIND_POWER_DENSITY_10M:
        power_density, vmin, vmax = vmod.extract_wind_power_density_10m(ds)
        return lambda i: vmod.materialize_2d(power_density[i : i + 1, :, :]), vmin, vmax
    if variable == WRFVariable.PRESSURE:
        var_data, vmin, vmax = vmod.extract_pressure(ds)
    elif variable == WRFVariable.VAPOR:
        var_data, vmin, vmax = vmod.extract_vapor(ds)
    else:
        nc_var = variable.upper()
        if not ds.has_variable(nc_var):
            return None
        var_data, vmin, vmax = vmod.extract_scalar(ds, nc_var)
    return lambda i: vmod.materialize_2d(var_data[i : i + 1, :, :]), vmin, vmax


def _run_values_unit(unit: WorkUnit, ds: WRFDataset) -> tuple[list[str], list[str]]:
    files: list[str] = []
    warnings: list[str] = []
    grid = ds.grid_level.value
    nc_suffix = VARIABLE_NETCDF_MAP.get(unit.variable, unit.variable.upper())
    time_meta = ds.build_date_metadata(skip_first_n=unit.skip_first)

    frames = _values_unit_frames(ds, unit.variable)
    if frames is None:
        warnings.append(f"Variable {unit.variable.upper()} not found — skipped")
        return files, warnings
    frame, vmin, vmax = frames

    acc = _SiteArtifactAccumulator(ds.n_time_steps) if unit.site_artifacts else None
    for meta in time_meta:
        if meta.get("skip"):
            continue
        if unit.variable == WRFVariable.SWDOWN:
            local_hour = meta["datetime_local"].hour
            if local_hour < 6 or local_hour > 18:
                continue
        i = meta["index"]
        data = frame(i)
        date_str = _format_datetime(meta["datetime_local"])
        out = Path(unit.json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"
        files.append(_atomic_values_json(out, data, vmin, vmax, date_str, None))
        if acc is not None:
            acc.add(i, data, date_str)
    if acc is not None:
        files.extend(acc.write(unit.json_dir, f"{grid}_{nc_suffix}", grid, nc_suffix))
    return files, warnings


def _run_poteolico_unit(unit: WorkUnit, ds: WRFDataset) -> tuple[list[str], list[str]]:
    files: list[str] = []
    grid = ds.grid_level.value
    time_meta = ds.build_date_metadata(skip_first_n=unit.skip_first)
    targets = parse_poteolico_heights(unit.variable)
    series = vmod.stream_wind_at_heights(ds, targets)

    for s in series:
        suffix = f"POT_EOLICO_{s.target}M"
        acc = _SiteArtifactAccumulator(ds.n_time_steps) if unit.site_artifacts else None
        for meta in time_meta:
            if meta.get("skip"):
                continue
            i = meta["index"]
            date_str = _format_datetime(meta["datetime_local"])
            out = Path(unit.json_dir) / f"{grid}_{suffix}_{i:03d}.json"
            files.append(
                _atomic_values_json(
                    out,
                    s.speed_steps[i],
                    s.vmin,
                    s.vmax,
                    date_str,
                    s.wind_vectors[i],
                )
            )
            if acc is not None:
                acc.add(i, s.speed_steps[i], date_str)
        if acc is not None:
            files.extend(acc.write(unit.json_dir, f"{grid}_{suffix}", grid, suffix))
    return files, []


def _run_wind_vectors_unit(unit: WorkUnit, ds: WRFDataset) -> tuple[list[str], list[str]]:
    files: list[str] = []
    grid = ds.grid_level.value
    time_meta = ds.build_date_metadata(skip_first_n=unit.skip_first)
    u10, v10, _vmin, _vmax = vmod.extract_wind(ds)

    for meta in time_meta:
        if meta.get("skip"):
            continue
        i = meta["index"]
        u = vmod.materialize_2d(u10[i : i + 1])
        v = vmod.materialize_2d(v10[i : i + 1])
        payload = create_wind_vectors_json(u, v, date_time=meta["datetime_local"], downsampling=4)
        out = Path(unit.json_dir) / f"{grid}_WIND_VECTORS_{i:03d}.json"
        files.append(_atomic_json_dump(out, payload))
    return files, []


def _run_grid_geojson_unit(unit: WorkUnit, ds: WRFDataset) -> tuple[list[str], list[str]]:
    lon, lat = ds.read_grid()
    grid = ds.grid_level.value
    out = Path(unit.geojson_dir) / f"{grid}.geojson"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    geojson.write_grid_geojson_stream(tmp, lon, lat, ds.dx, ds.dy)
    os.replace(tmp, out)
    logger.info("Saved GeoJSON: %s", out)

    # Compact companion preferred by the site front-end (which falls back to
    # the legacy .geojson above); a fraction of the size on the wire.
    compact = Path(unit.geojson_dir) / f"{grid}.grid.json"
    compact_tmp = compact.with_name(f".{compact.name}.tmp-{os.getpid()}")
    geojson.write_grid_compact_json_stream(compact_tmp, lon, lat, ds.dx, ds.dy)
    os.replace(compact_tmp, compact)
    return [str(out), str(compact)], []


def _run_time_metadata(ds: WRFDataset) -> tuple[str | None, int | None]:
    """The run's first local datetime (JSON date_time format) and step count."""
    try:
        times = ds.parse_times()
        if not times:
            return None, None
        return _format_datetime(times[0].astimezone(product_timezone())), len(times)
    except Exception:  # metadata is best-effort; units must not fail over it
        return None, None


_TIMESTEP_FILE_RE = re.compile(r"^(D\d{2})_([A-Z0-9_]+)_(\d{3})\.json$")


def _compress_index_ranges(indices: set[int]) -> list[list[int]]:
    """Compress a set of indices into sorted inclusive ``[start, end]`` runs."""
    runs: list[list[int]] = []
    for i in sorted(indices):
        if runs and i == runs[-1][1] + 1:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return runs


def write_run_manifest(json_dir: str | Path, results: Sequence[UnitResult]) -> Path | None:
    """Write ``manifest.json`` into *json_dir* after a generation run.

    The site front-end fetches this tiny file with ``Cache-Control: no-cache``
    and appends ``?v=<version>`` to every data URL, which lets the fixed-name
    data files be cached long-term while staying fresh across runs. Absence of
    the manifest simply keeps the front-end on unversioned URLs.

    The version is bumped whenever the run executed ANY unit — including
    fully failed runs: a unit that crashed mid-way may already have atomically
    replaced some files under unchanged filenames (its ``files`` manifest is
    empty on error, so the file count cannot be trusted), and keeping the
    previous version alive would let long-cached clients pin outdated data
    under the old ``?v=`` URLs. An extra bump is only ever a cache miss.
    """
    if not results:
        return None
    written = sum(len(result.files) for result in results)
    domains = sorted(
        {
            match.group(1).upper()
            for result in results
            if (match := re.search(r"_(d\d+)_", result.label)) is not None
        }
    )
    payload: dict = {
        "version": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "generated_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "domains": domains,
        "files": written,
    }

    # v2 fields (additive; the site falls back to hardcoded defaults without
    # them): timeline range and per-variable availability derived from the
    # files ACTUALLY written this run — never re-derived arithmetic that could
    # drift from the writers — plus feature descriptors for the consolidated
    # summary/series artifacts.
    var_indices: dict[str, set[int]] = {}
    domain_indices: dict[str, set[int]] = {}
    have_summary = False
    have_series = False
    for result in results:
        for file_path in result.files:
            name = Path(file_path).name
            match = _TIMESTEP_FILE_RE.match(name)
            if match:
                index = int(match.group(3))
                var_indices.setdefault(match.group(2), set()).add(index)
                domain_indices.setdefault(match.group(1), set()).add(index)
            elif name.endswith(".summary.json"):
                have_summary = True
            elif name.endswith(".series.bin"):
                have_series = True

    # The site renders ONE timeline across all domains, so advertise the
    # intersection of the per-domain ranges: never an index some domain
    # lacks entirely (a mixed-length run would otherwise label missing
    # frames as available).
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
        # The anchor pairs with index 0 (the file's first time step) — the
        # client must anchor initialIndex=0 regardless of index_min. Results
        # arrive in completion order, so only advertise the anchor when every
        # unit that reported one agrees.
        start_locals = {r.start_local for r in results if r.start_local}
        if len(start_locals) == 1:
            payload["start_local"] = start_locals.pop()

        full_range = set(range(index_min, index_max + 1))
        availability = {
            var: _compress_index_ranges(indices & full_range)
            for var, indices in sorted(var_indices.items())
            if not full_range <= indices
        }
        if availability:
            payload["availability"] = availability

        # Consolidated-artifact descriptors are a byte-offset contract: a
        # failed unit leaves LAST run's {D}_{VAR}.series.bin/.summary.json in
        # place under this run's version, so only vouch for the artifacts
        # when every unit succeeded (the site falls back to the per-step
        # JSONs otherwise).
        run_clean = not any(r.error for r in results)
        features: dict = {}
        if have_summary and run_clean:
            features["domain_summary"] = {
                "format": "domain-summary-v1",
                "template": "JSON/{domain}_{variable}.summary.json",
            }
        # The series matrices span columns 0..n_steps-1 regardless of which
        # steps were written (skip-first / night gaps are MISSING columns), so
        # the byte-offset contract needs the step count — advertised only when
        # every file agrees on it (a mixed-length run would corrupt offsets).
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

    A worker killed mid-write (OOM kill, broken pool teardown) can leave its
    private temp file behind; the final outputs are never affected because of
    the atomic rename, but the debris should not accumulate. Only files whose
    embedded PID no longer exists are removed, so concurrent runs writing into
    the same directories are never disturbed. ``sweep_pids`` marks pids whose
    debris is known-orphaned even though the process is alive — the serial
    path writes with the parent's own pid, so its failed-unit leftovers would
    otherwise survive every end-of-run sweep.
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


def process_unit(unit: WorkUnit) -> UnitResult:
    """Execute one work unit. Runs in a worker process; never raises."""
    if os.environ.get("LABMIM_TEST_CRASH_UNIT") == unit.variable:
        # Test hook: simulate an OOM-killed worker (exercises pool-break recovery).
        os._exit(137)
    t0 = time.perf_counter()
    try:
        with WRFDataset(unit.wrf_path) as ds:
            if unit.kind == "values_json":
                files, warnings = _run_values_unit(unit, ds)
            elif unit.kind == "poteolico":
                files, warnings = _run_poteolico_unit(unit, ds)
            elif unit.kind == "wind_vectors":
                files, warnings = _run_wind_vectors_unit(unit, ds)
            elif unit.kind == "grid_geojson":
                files, warnings = _run_grid_geojson_unit(unit, ds)
            else:
                raise ValueError(f"Unknown work unit kind: {unit.kind}")
            start_local, n_steps = _run_time_metadata(ds)
        return UnitResult(
            label=unit.label,
            kind=unit.kind,
            files=tuple(files),
            seconds=time.perf_counter() - t0,
            warnings=tuple(warnings),
            start_local=start_local,
            n_steps=n_steps,
        )
    except Exception as exc:
        logger.exception("Work unit failed: %s", unit.label)
        return UnitResult(
            label=unit.label,
            kind=unit.kind,
            seconds=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


_KIND_COST_RANK = {"poteolico": 0, "values_json": 1, "wind_vectors": 1, "grid_geojson": 2}


def build_units(
    wrf_paths: Sequence[str | Path],
    variables: Sequence[str],
    json_dir: str | Path,
    geojson_dir: str | Path,
    skip_first: int = 0,
    site_artifacts: bool = True,
) -> list[WorkUnit]:
    """Expand (files x variables) into work units, one grid GeoJSON per file."""
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
        for var_name in variables:
            if var_name.startswith(WRFVariable.WIND_POTENTIAL):
                kind: UnitKind = "poteolico"
            elif var_name == "wind_vectors":
                kind = "wind_vectors"
            else:
                kind = "values_json"
            units.append(
                WorkUnit(
                    kind=kind,
                    wrf_path=str(path),
                    variable=var_name,
                    json_dir=str(json_dir),
                    geojson_dir=str(geojson_dir),
                    skip_first=skip_first,
                    site_artifacts=site_artifacts,
                )
            )
    return units


def execute_units(
    units: Sequence[WorkUnit],
    workers: int,
    *,
    echo: Callable[[str], None] = logger.info,
) -> list[UnitResult]:
    """Execute all units on ONE persistent process pool, heaviest first.

    Ordinary unit failures are isolated (reported in the unit's result). If
    the pool itself breaks — e.g. a worker is OOM-killed — units that never
    completed are retried one at a time in isolated single-worker pools, so a
    unit that keeps killing its worker fails alone instead of dooming the
    other pending units; it gets an error result so callers can exit non-zero.
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
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            max_tasks_per_child=_max_tasks_per_child(n_workers),
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
            with ProcessPoolExecutor(max_workers=1) as retry_pool:
                result = retry_pool.submit(process_unit, unit).result()
        except BrokenProcessPool:
            result = UnitResult(
                label=unit.label,
                kind=unit.kind,
                error="worker crashed while processing this unit (possible OOM kill)",
            )
        results.append(result)
        status = f"✗ {result.error}" if result.error else f"{len(result.files)} files"
        echo(f"  [{len(results)}/{len(ordered)}] {result.label}: {status}")

    # Always sweep at the end: a unit that failed mid-write WITHOUT breaking
    # the pool (the common case — process_unit catches everything) leaves its
    # dead-pid temp file behind, and debris must not wait for a future run
    # that happens to crash before being cleaned up.
    swept = _sweep_stale_temp_files(output_dirs, sweep_pids=frozenset({os.getpid()}))
    if swept:
        echo(f"  swept {swept} stale temp file(s)")

    echo(f"✓ {len(results)} work units in {time.perf_counter() - t0:.1f}s")
    return results
