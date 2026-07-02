"""High-performance parallel WRF figure and JSON generation.

This module is the core optimisation layer. It:

1. Loads each NetCDF domain file **once**, extracts all variable data into
   memory as NumPy arrays.
2. Builds a flat list of lightweight ``RenderTask`` tuples — one per frame.
3. Dispatches all tasks to a ``ProcessPoolExecutor`` (default: ``cpu_count - 4``).
4. Each worker runs with the ``Agg`` backend (no GUI, no GIL lock).

Typical speed-up on a 48-core workstation: **~30x** vs serial rendering.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import numpy as np

from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
MAX_TASKS_PER_CHILD = int(os.environ.get("LABMIM_MAX_TASKS_PER_CHILD", "64"))

WorkerBackend = Literal["auto", "serial", "memmap"]
JsonWorkerBackend = WorkerBackend


# ---------------------------------------------------------------------------
# Configuration structures (frozen, picklable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MapConfig:
    """Invariant per-domain map configuration, passed to every worker."""

    grid_level: str  # "D01", "D02", etc. (str for pickling)
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    coast_width: int
    state_width: int
    draw_municipalities: bool
    shapes_dir: str | None


class FigureTask(NamedTuple):
    """Lightweight, picklable description of a single frame to render."""

    # Data (pre-sliced 2D arrays → small, picklable)
    lon: NDArray
    lat: NDArray
    data: NDArray
    vmin: float
    vmax: float
    cmap_name: str

    # Overlay (optional pressure contours for temperature)
    overlay_data: NDArray | None
    overlay_levels: list[float] | None

    # Wind-specific (optional)
    u: NDArray | None
    v: NDArray | None

    # Labels
    title: str
    output_path: str

    # Map config
    map_config: MapConfig

    # Rendering options
    dpi: int
    saturation: float


class JsonTask(NamedTuple):
    """Lightweight description of a JSON file to write."""

    data: NDArray
    scale_min: float
    scale_max: float
    date_str: str
    output_path: str
    wind_data: dict | None


class JsonMemmapTask(NamedTuple):
    """JSON task with array data stored in a temporary ``.npy`` file."""

    data_path: str
    scale_min: float
    scale_max: float
    date_str: str
    output_path: str
    wind_data: dict | None


class FigureMemmapTask(NamedTuple):
    """Figure task with array payloads stored in temporary ``.npy`` files."""

    lon_path: str
    lat_path: str
    data_path: str
    overlay_data_path: str | None
    u_path: str | None
    v_path: str | None
    vmin: float
    vmax: float
    cmap_name: str
    overlay_levels: list[float] | None
    title: str
    output_path: str
    map_config: MapConfig
    dpi: int
    saturation: float


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickling)
# ---------------------------------------------------------------------------


def _render_figure(task: FigureTask) -> str:
    """Render a single map figure. Runs in a worker process."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    from micrometeorology.wrf.plotting import saturated_cmap

    mc = task.map_config

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([mc.lon_min, mc.lon_max, mc.lat_min, mc.lat_max], crs=ccrs.PlateCarree())

    # Map features
    ax.coastlines(resolution="10m", linewidth=mc.coast_width)
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "10m"),
        linewidth=mc.state_width,
        edgecolor="black",
        facecolor="none",
    )

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    transform = ccrs.PlateCarree()
    cmap = saturated_cmap(task.cmap_name, task.saturation)

    if task.u is not None and task.v is not None:
        # Wind field
        speed = task.data
        mesh = ax.pcolormesh(
            task.lon,
            task.lat,
            speed,
            alpha=0.4,
            cmap=cmap,
            vmin=task.vmin,
            vmax=task.vmax,
            transform=transform,
            shading="auto",
        )
        cb = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.04)
        cb.ax.tick_params(labelsize=10)

        # Quiver (sub-sampled)
        stride_map = {"D01": 6, "D02": 3, "D03": 4, "D04": 4, "D05": 4}
        stride = stride_map.get(mc.grid_level, 4)
        ax.quiver(
            task.lon[::stride, ::stride],
            task.lat[::stride, ::stride],
            task.u[::stride, ::stride],
            task.v[::stride, ::stride],
            transform=transform,
            scale=50,
            width=0.003,
        )
    else:
        # Scalar field — single pcolormesh (no double contourf+pcolor)
        mesh = ax.pcolormesh(
            task.lon,
            task.lat,
            task.data,
            alpha=0.4,
            cmap=cmap,
            vmin=task.vmin,
            vmax=task.vmax,
            transform=transform,
            shading="auto",
        )
        cb = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.04)
        cb.ax.tick_params(labelsize=10)

    # Pressure contour overlay
    if task.overlay_data is not None:
        levels = task.overlay_levels or [880, 900, 950, 1000, 1013]
        cs = ax.contour(
            task.lon,
            task.lat,
            task.overlay_data,
            levels=levels,
            linewidths=0.8,
            colors="black",
            transform=transform,
        )
        ax.clabel(cs, colors="black", fmt="%.0f")

    ax.set_title(task.title, fontsize=9)

    out = Path(task.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=task.dpi)
    plt.close(fig)

    return str(out)


def _write_json_payload(
    arr: NDArray,
    *,
    scale_min: float,
    scale_max: float,
    date_str: str,
    output_path: str,
    wind_data: dict | None,
) -> str:
    """Write a single values JSON payload from an ndarray-like object."""
    from micrometeorology.wrf.geojson import write_values_json_stream

    arr = arr.filled(np.nan) if hasattr(arr, "filled") else np.asarray(arr, dtype=float)
    assert_reasonable_array_size(
        arr.shape,
        arr.dtype,
        context=f"JSON payload materialization for {output_path}",
        multiplier=4.0,
    )

    out = Path(output_path)
    write_values_json_stream(
        out,
        arr,
        scale_min=scale_min,
        scale_max=scale_max,
        date_str=date_str,
        wind_data=wind_data,
    )
    return str(out)


def _load_memmap_array(path: str | None) -> NDArray | None:
    if path is None:
        return None
    return cast("NDArray", np.load(path, mmap_mode="r"))


def _render_figure_memmap(task: FigureMemmapTask) -> str:
    """Render a figure from memmap-backed arrays."""
    return _render_figure(
        FigureTask(
            lon=np.load(task.lon_path, mmap_mode="r"),
            lat=np.load(task.lat_path, mmap_mode="r"),
            data=np.load(task.data_path, mmap_mode="r"),
            vmin=task.vmin,
            vmax=task.vmax,
            cmap_name=task.cmap_name,
            overlay_data=_load_memmap_array(task.overlay_data_path),
            overlay_levels=task.overlay_levels,
            u=_load_memmap_array(task.u_path),
            v=_load_memmap_array(task.v_path),
            title=task.title,
            output_path=task.output_path,
            map_config=task.map_config,
            dpi=task.dpi,
            saturation=task.saturation,
        )
    )


def _write_json(task: JsonTask) -> str:
    """Write a single JSON file. Runs in a worker process."""
    return _write_json_payload(
        task.data,
        scale_min=task.scale_min,
        scale_max=task.scale_max,
        date_str=task.date_str,
        output_path=task.output_path,
        wind_data=task.wind_data,
    )


def _write_json_memmap(task: JsonMemmapTask) -> str:
    """Write a JSON file from a memmap-backed task. Runs in a worker process."""
    arr = np.load(task.data_path, mmap_mode="r")
    return _write_json_payload(
        arr,
        scale_min=task.scale_min,
        scale_max=task.scale_max,
        date_str=task.date_str,
        output_path=task.output_path,
        wind_data=task.wind_data,
    )


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


def build_map_config(
    grid_level: str,
    bounds: tuple[float, float, float, float],
    shapes_dir: str | None = None,
) -> MapConfig:
    """Build a frozen ``MapConfig`` from domain metadata."""
    lon_min, lon_max, lat_min, lat_max = bounds
    coast_map = {"D03": 2, "D04": 3, "D05": 3}
    state_map = {"D03": 2, "D04": 2, "D05": 2}
    muni_set = {"D03", "D04", "D05"}

    return MapConfig(
        grid_level=grid_level,
        lon_min=lon_min,
        lon_max=lon_max,
        lat_min=lat_min,
        lat_max=lat_max,
        coast_width=coast_map.get(grid_level, 1),
        state_width=state_map.get(grid_level, 1),
        draw_municipalities=grid_level in muni_set,
        shapes_dir=shapes_dir,
    )


def default_workers() -> int:
    """Return the default number of parallel workers."""
    n = os.cpu_count() or 4
    return max(1, n - 4)


def _max_tasks_per_child(n_workers: int) -> int | None:
    if n_workers <= 1 or MAX_TASKS_PER_CHILD <= 0:
        return None
    return MAX_TASKS_PER_CHILD


def run_figure_tasks(
    tasks: list[FigureTask],
    workers: int | None = None,
    *,
    backend: WorkerBackend = "auto",
    tmp_dir: str | Path | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[str]:
    """Execute figure rendering tasks in parallel.

    Parameters
    ----------
    tasks:
        List of ``FigureTask`` to render.
    workers:
        Number of parallel workers. Defaults to ``cpu_count - 4``.
    executor:
        Optional caller-owned process pool. When provided and the resolved
        backend is ``"memmap"`` with more than one worker, tasks are
        submitted to it instead of creating a fresh pool per call; the
        executor is used as-is (no worker clamping) and never shut down
        here. The serial backend ignores it.

    Returns
    -------
    list[str]
        Paths of generated PNG files.
    """
    n_workers = workers or default_workers()
    if executor is None:
        n_workers = min(n_workers, len(tasks)) if tasks else 1
    total = len(tasks)

    if backend not in {"auto", "serial", "memmap"}:
        raise ValueError(f"Unknown figure worker backend: {backend}")
    resolved_backend: Literal["serial", "memmap"] = (
        "serial" if backend == "serial" or (backend == "auto" and n_workers == 1) else "memmap"
    )

    logger.info(
        "Rendering %d figures with %d workers (%s backend)",
        total,
        n_workers,
        resolved_backend,
    )
    t0 = time.perf_counter()

    paths: list[str] = []
    if not tasks:
        return paths

    if resolved_backend == "serial":
        paths = [_render_figure(task) for task in tasks]
        elapsed = time.perf_counter() - t0
        logger.info(
            "✓ Rendered %d figures in %.1fs (%.1f img/s)",
            len(paths),
            elapsed,
            len(paths) / elapsed if elapsed > 0 else 0,
        )
        return paths

    if resolved_backend == "memmap":
        return _run_figure_tasks_memmap(tasks, n_workers, tmp_dir, t0, executor=executor)

    raise RuntimeError("unreachable figure backend resolution")


def run_json_tasks(
    tasks: list[JsonTask],
    workers: int | None = None,
    *,
    backend: JsonWorkerBackend = "auto",
    tmp_dir: str | Path | None = None,
    executor: ProcessPoolExecutor | None = None,
) -> list[str]:
    """Execute JSON writing tasks in parallel.

    Parameters
    ----------
    tasks:
        List of ``JsonTask`` to write.
    workers:
        Number of parallel workers. Defaults to ``cpu_count - 4``.
    backend:
        ``"serial"`` writes in-process without creating worker processes.
        ``"memmap"`` stores arrays in temporary ``.npy`` files and sends
        lightweight file references, reducing process-pool IPC payload size.
    tmp_dir:
        Parent directory for temporary memmap payloads when backend is
        ``"memmap"``. A per-run subdirectory is created and removed.
    executor:
        Optional caller-owned process pool. When provided and the resolved
        backend is ``"memmap"`` with more than one worker, tasks are
        submitted to it instead of creating a fresh pool per call; the
        executor is used as-is (no worker clamping) and never shut down
        here. The serial backend ignores it.

    Returns
    -------
    list[str]
        Paths of generated JSON files.
    """
    n_workers = workers or default_workers()
    if executor is None:
        n_workers = min(n_workers, len(tasks)) if tasks else 1
    total = len(tasks)

    if backend not in {"auto", "serial", "memmap"}:
        raise ValueError(f"Unknown JSON worker backend: {backend}")
    resolved_backend: Literal["serial", "memmap"] = (
        "serial" if backend == "serial" or (backend == "auto" and n_workers == 1) else "memmap"
    )

    logger.info(
        "Writing %d JSON files with %d workers (%s backend)",
        total,
        n_workers,
        resolved_backend,
    )
    t0 = time.perf_counter()

    paths: list[str] = []
    if not tasks:
        return paths

    if resolved_backend == "serial":
        paths = [_write_json(task) for task in tasks]
        elapsed = time.perf_counter() - t0
        logger.info("✓ Wrote %d JSON files in %.1fs", len(paths), elapsed)
        return paths

    if resolved_backend == "memmap":
        return _run_json_tasks_memmap(tasks, n_workers, tmp_dir, t0, executor=executor)

    raise RuntimeError("unreachable JSON backend resolution")


def _save_memmap_payload(
    run_dir: Path,
    name: str,
    arr: NDArray | None,
    cache: dict[int, str] | None = None,
) -> str | None:
    if arr is None:
        return None
    cache_key = id(arr)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    data_path = run_dir / f"{name}.npy"
    assert_reasonable_array_size(arr.shape, arr.dtype, context=f"memmap payload {name}")
    np.save(data_path, np.asarray(arr), allow_pickle=False)
    path_str = str(data_path)
    if cache is not None:
        cache[cache_key] = path_str
    return path_str


def _collect_pool_paths[TaskT](
    pool: ProcessPoolExecutor,
    worker: Callable[[TaskT], str],
    tasks: list[TaskT],
    task_kind: str,
) -> list[str]:
    """Submit memmap tasks to *pool* and collect result paths as they complete."""
    paths: list[str] = []
    futures = {pool.submit(worker, task): i for i, task in enumerate(tasks)}
    for future in as_completed(futures):
        try:
            paths.append(future.result())
        except BrokenProcessPool:
            # The pool itself died (e.g. OOM-killed worker): every remaining
            # task is doomed, so surface the failure instead of logging it away.
            raise
        except Exception:
            idx = futures[future]
            logger.exception("Failed to %s task %d", task_kind, idx)
    return paths


def _run_figure_tasks_memmap(
    tasks: list[FigureTask],
    n_workers: int,
    tmp_dir: str | Path | None,
    t0: float,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> list[str]:
    """Materialize figure task arrays to temporary .npy files and process by reference."""
    parent: Path | None = Path(tmp_dir) if tmp_dir is not None else None
    if parent is None:
        run_dir_ctx = tempfile.TemporaryDirectory(prefix="labmim-figure-memmap-")
        run_dir = Path(run_dir_ctx.name)
    else:
        parent.mkdir(parents=True, exist_ok=True)
        run_dir_ctx = None
        run_dir = parent / f"labmim-figure-memmap-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)

    paths: list[str] = []
    try:
        grid_cache: dict[int, str] = {}
        memmap_tasks: list[FigureMemmapTask] = []
        for idx, task in enumerate(tasks):
            prefix = f"task_{idx:06d}"
            lon_path = _save_memmap_payload(run_dir, f"{prefix}_lon", task.lon, grid_cache)
            lat_path = _save_memmap_payload(run_dir, f"{prefix}_lat", task.lat, grid_cache)
            data_path = _save_memmap_payload(run_dir, f"{prefix}_data", task.data)
            if lon_path is None or lat_path is None or data_path is None:
                raise ValueError("Figure memmap task requires lon, lat, and data arrays")
            memmap_tasks.append(
                FigureMemmapTask(
                    lon_path=lon_path,
                    lat_path=lat_path,
                    data_path=data_path,
                    overlay_data_path=_save_memmap_payload(
                        run_dir, f"{prefix}_overlay", task.overlay_data
                    ),
                    u_path=_save_memmap_payload(run_dir, f"{prefix}_u", task.u),
                    v_path=_save_memmap_payload(run_dir, f"{prefix}_v", task.v),
                    vmin=task.vmin,
                    vmax=task.vmax,
                    cmap_name=task.cmap_name,
                    overlay_levels=task.overlay_levels,
                    title=task.title,
                    output_path=task.output_path,
                    map_config=task.map_config,
                    dpi=task.dpi,
                    saturation=task.saturation,
                )
            )

        if n_workers == 1:
            paths = [_render_figure_memmap(task) for task in memmap_tasks]
        elif executor is not None:
            paths = _collect_pool_paths(
                executor, _render_figure_memmap, memmap_tasks, "render memmap figure"
            )
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                max_tasks_per_child=_max_tasks_per_child(n_workers),
            ) as pool:
                paths = _collect_pool_paths(
                    pool, _render_figure_memmap, memmap_tasks, "render memmap figure"
                )
    finally:
        if run_dir_ctx is not None:
            run_dir_ctx.cleanup()
        else:
            shutil.rmtree(run_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        "✓ Rendered %d figures in %.1fs (%.1f img/s)",
        len(paths),
        elapsed,
        len(paths) / elapsed if elapsed > 0 else 0,
    )
    return paths


def _run_json_tasks_memmap(
    tasks: list[JsonTask],
    n_workers: int,
    tmp_dir: str | Path | None,
    t0: float,
    *,
    executor: ProcessPoolExecutor | None = None,
) -> list[str]:
    """Materialize JSON task arrays to temporary .npy files and process by reference."""
    parent: Path | None = Path(tmp_dir) if tmp_dir is not None else None
    if parent is None:
        run_dir_ctx = tempfile.TemporaryDirectory(prefix="labmim-json-memmap-")
        run_dir = Path(run_dir_ctx.name)
    else:
        parent.mkdir(parents=True, exist_ok=True)
        run_dir_ctx = None
        run_dir = parent / f"labmim-json-memmap-{uuid.uuid4().hex}"
        run_dir.mkdir(parents=True, exist_ok=False)

    paths: list[str] = []
    try:
        memmap_tasks: list[JsonMemmapTask] = []
        for idx, task in enumerate(tasks):
            data_path = run_dir / f"task_{idx:06d}.npy"
            data = task.data.filled(np.nan) if hasattr(task.data, "filled") else task.data
            assert_reasonable_array_size(
                data.shape,
                data.dtype,
                context=f"JSON memmap payload task {idx}",
            )
            np.save(data_path, np.asarray(data), allow_pickle=False)
            memmap_tasks.append(
                JsonMemmapTask(
                    data_path=str(data_path),
                    scale_min=task.scale_min,
                    scale_max=task.scale_max,
                    date_str=task.date_str,
                    output_path=task.output_path,
                    wind_data=task.wind_data,
                )
            )

        if n_workers == 1:
            paths = [_write_json_memmap(task) for task in memmap_tasks]
        elif executor is not None:
            paths = _collect_pool_paths(
                executor, _write_json_memmap, memmap_tasks, "write memmap JSON"
            )
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                max_tasks_per_child=_max_tasks_per_child(n_workers),
            ) as pool:
                paths = _collect_pool_paths(
                    pool, _write_json_memmap, memmap_tasks, "write memmap JSON"
                )
    finally:
        if run_dir_ctx is not None:
            run_dir_ctx.cleanup()
        else:
            shutil.rmtree(run_dir, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    logger.info("✓ Wrote %d JSON files in %.1fs", len(paths), elapsed)
    return paths
