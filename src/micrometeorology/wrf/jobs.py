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
from micrometeorology.wrf.reader import WRFDataset

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
    """Picklable description of one unit of work. Strings and ints only."""

    kind: UnitKind
    wrf_path: str
    variable: str
    json_dir: str
    geojson_dir: str
    skip_first: int = 0

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
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, output_path)
    return str(output_path)


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
        t2, _psfc, vmin, vmax = vmod.extract_temperature(ds)
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

    for meta in time_meta:
        if meta.get("skip"):
            continue
        if unit.variable == WRFVariable.SWDOWN:
            local_hour = meta["datetime_local"].hour
            if local_hour < 6 or local_hour > 18:
                continue
        i = meta["index"]
        out = Path(unit.json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"
        files.append(
            _atomic_values_json(
                out, frame(i), vmin, vmax, _format_datetime(meta["datetime_local"]), None
            )
        )
    return files, warnings


def _run_poteolico_unit(unit: WorkUnit, ds: WRFDataset) -> tuple[list[str], list[str]]:
    files: list[str] = []
    grid = ds.grid_level.value
    time_meta = ds.build_date_metadata(skip_first_n=unit.skip_first)
    targets = parse_poteolico_heights(unit.variable)
    series = vmod.stream_wind_at_heights(ds, targets)

    for s in series:
        suffix = f"POT_EOLICO_{s.target}M"
        for meta in time_meta:
            if meta.get("skip"):
                continue
            i = meta["index"]
            out = Path(unit.json_dir) / f"{grid}_{suffix}_{i:03d}.json"
            files.append(
                _atomic_values_json(
                    out,
                    s.speed_steps[i],
                    s.vmin,
                    s.vmax,
                    _format_datetime(meta["datetime_local"]),
                    s.wind_vectors[i],
                )
            )
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
    return [str(out)], []


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
        return UnitResult(
            label=unit.label,
            kind=unit.kind,
            files=tuple(files),
            seconds=time.perf_counter() - t0,
            warnings=tuple(warnings),
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

    if n_workers == 1:
        serial_results = [process_unit(u) for u in ordered]
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

    echo(f"✓ {len(results)} work units in {time.perf_counter() - t0:.1f}s")
    return results
