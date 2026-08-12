"""Parallel WRF figure rendering.

Builds lightweight ``FigureTask`` tuples (one per frame), spills their array
payloads to temporary ``.npy`` files, and dispatches them to a process pool
(one persistent pool per CLI run; workers render with the ``Agg`` backend).

JSON generation does not live here: see ``micrometeorology.wrf.jobs`` for the
work-unit pipeline where each worker reads the NetCDF itself.

Cartopy Data Requirements
-------------------------
Cartopy needs Natural Earth data for coastlines and borders.  On systems
without internet access, pre-download the data::

    python -c "import cartopy; cartopy.config['data_dir'] = '/path/to/data'"

See https://scitools.org.uk/cartopy/docs/latest/installing.html#data
"""

import functools
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
from typing import Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
)

logger = logging.getLogger(__name__)
MAX_TASKS_PER_CHILD = int(os.environ.get("LABMIM_MAX_TASKS_PER_CHILD", "64"))

WorkerBackend = Literal["auto", "serial", "memmap"]


@dataclass(frozen=True)
class MapConfig:
    """Invariant per-domain map configuration, passed to every worker.

    Attributes
    ----------
    grid_level:
        Domain id as a plain string (``"D01"``, ``"D02"``, ...) rather than the
        enum, so the config pickles across the pool boundary.
    lon_min, lon_max, lat_min, lat_max:
        Map extent in degrees east and degrees north, set with PlateCarree.
    coast_width, state_width:
        Line widths for the coastline and the state borders, in points; the
        inner domains carry heavier lines because they cover less ground.
    draw_municipalities:
        Whether to overlay the IBGE municipality mesh.
    shapes_dir:
        Directory holding that shapefile, or ``None`` when it was not supplied.
    """

    grid_level: str
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    coast_width: int
    state_width: int
    draw_municipalities: bool
    shapes_dir: str | None


class FigureTask(NamedTuple):
    """Lightweight, picklable description of a single frame to render.

    Attributes
    ----------
    lon, lat:
        ``(ny, nx)`` cell-centre coordinates in degrees east and degrees north.
    data:
        ``(ny, nx)`` field to colour, in its published unit. Pre-sliced to one
        time step, which is what keeps the task small enough to pickle across the
        pool boundary.
    vmin, vmax:
        Colour-scale bounds in the unit of *data*.
    cmap_name:
        Matplotlib colormap name, saturated by
        :func:`~micrometeorology.wrf.plotting.saturated_cmap`.
    overlay_data, overlay_levels:
        Optional ``(ny, nx)`` field drawn as labelled contours over the mesh —
        surface pressure in hPa over temperature — and the levels to draw.
    u, v:
        Optional ``(ny, nx)`` wind components in m/s, drawn as a quiver over the
        mesh; both or neither.
    title:
        Figure caption.
    output_path:
        Destination PNG; parent directories are created by the worker.
    map_config:
        The domain's invariant map configuration.
    dpi:
        Raster resolution of the saved PNG.
    saturation:
        Colour saturation multiplier applied to the colormap.
    """

    lon: NDArray
    lat: NDArray
    data: NDArray
    vmin: float
    vmax: float
    cmap_name: str

    overlay_data: NDArray | None
    overlay_levels: list[float] | None

    u: NDArray | None
    v: NDArray | None

    title: str
    output_path: str
    map_config: MapConfig
    dpi: int
    saturation: float


class FigureMemmapTask(NamedTuple):
    """Figure task with array payloads stored in temporary ``.npy`` files.

    Field for field a :class:`FigureTask` with every array replaced by the path
    of the ``.npy`` it was spilled to, so a frame crosses the pool boundary as a
    filename and the worker maps it read-only instead of unpickling a copy. The
    ``None`` paths mean the same as the ``None`` arrays there. Scalar fields keep
    their meaning, shape, dtype and unit unchanged.
    """

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


@functools.lru_cache(maxsize=4)
def _municipality_geometries(shp_path: str) -> tuple[object, ...]:
    """Read the IBGE municipality mesh once per worker process.

    ``BRMUE250GC_SIR`` carries every Brazilian municipality (~5.5k polygons);
    re-parsing it per frame would dominate the render. Returns an empty tuple
    (warning once per process) when the shapefile is not where ``--shapes-dir``
    says it is.
    """
    from cartopy.io import shapereader

    if not Path(shp_path).exists():
        logger.warning("Municipality shapefile not found: %s", shp_path)
        return ()
    return tuple(shapereader.Reader(shp_path).geometries())


def _render_figure(task: FigureTask) -> str:
    """Render a single map figure. Runs in a worker process."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    from micrometeorology.wrf.plotting import saturated_cmap

    map_config = task.map_config

    fig = plt.figure(figsize=(8, 6))
    try:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
        ax.set_extent(
            [map_config.lon_min, map_config.lon_max, map_config.lat_min, map_config.lat_max],
            crs=ccrs.PlateCarree(),
        )

        ax.coastlines(resolution="10m", linewidth=map_config.coast_width)
        ax.add_feature(
            cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "10m"),
            linewidth=map_config.state_width,
            edgecolor="black",
            facecolor="none",
        )

        if map_config.draw_municipalities and map_config.shapes_dir:
            geometries = _municipality_geometries(
                str(Path(map_config.shapes_dir) / "BRMUE250GC_SIR.shp")
            )
            if geometries:
                ax.add_geometries(
                    geometries,
                    ccrs.PlateCarree(),
                    facecolor="none",
                    edgecolor="gray",
                    linewidth=0.5,
                )

        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False

        transform = ccrs.PlateCarree()
        cmap = saturated_cmap(task.cmap_name, task.saturation)

        if task.u is not None and task.v is not None:
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

            # Sub-sampled so the arrows stay readable at each domain's grid pitch.
            stride_map = {"D01": 6, "D02": 3, "D03": 4, "D04": 4, "D05": 4}
            stride = stride_map.get(map_config.grid_level, 4)
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
    finally:
        # pyplot's global registry keeps a strong reference: without this the
        # figure survives every failure path until the worker is recycled.
        plt.close(fig)

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


def build_map_config(
    grid_level: str,
    bounds: tuple[float, float, float, float],
    shapes_dir: str | None = None,
) -> MapConfig:
    """Build a frozen ``MapConfig`` from domain metadata.

    Parameters
    ----------
    grid_level:
        Domain id (``"D01"``..``"D05"``); it selects the line widths and whether
        the municipality mesh is drawn, which only the inner domains resolve.
    bounds:
        ``(lon_min, lon_max, lat_min, lat_max)`` in degrees, as
        :meth:`~micrometeorology.wrf.reader.WRFDataset.grid_bounds` returns them.
    shapes_dir:
        Directory holding the IBGE municipality shapefile.

    Returns
    -------
    MapConfig
        Configuration every worker of this domain renders with.
    """
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
    """Number of parallel workers to use when the caller names none.

    Four cores are left to the parent process and the operating system, so a
    full-machine render does not starve the run's own I/O; a machine reporting no
    core count is treated as a 4-core one, which yields a single worker.
    """
    n = os.cpu_count() or 4
    return max(1, n - 4)


def _max_tasks_per_child(n_workers: int) -> int | None:
    """How many tasks a pool worker runs before being replaced.

    ``None`` (no recycling) for a single worker, since there is no other process
    to take over, and whenever ``LABMIM_MAX_TASKS_PER_CHILD`` is set to zero or
    less. Recycling bounds the memory a long-lived worker accumulates.
    """
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
    backend:
        ``"serial"`` renders in-process; ``"memmap"`` spills each task's arrays
        to ``.npy`` files and passes them by path; ``"auto"`` picks serial only
        when a single worker was resolved.
    tmp_dir:
        Parent directory for the memmap spill directory. ``None`` uses a
        self-cleaning :class:`tempfile.TemporaryDirectory`.
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


def _collect_pool_paths(
    pool: ProcessPoolExecutor,
    tasks: list[FigureMemmapTask],
) -> list[str]:
    """Submit memmap tasks to *pool* and collect result paths as they complete."""
    paths: list[str] = []
    futures = {pool.submit(_render_figure_memmap, task): task for task in tasks}
    for future in as_completed(futures):
        try:
            paths.append(future.result())
        except BrokenProcessPool:
            # The pool itself died (e.g. OOM-killed worker): every remaining
            # task is doomed, so surface the failure instead of logging it away.
            raise
        except Exception:
            failed_task = futures[future]
            logger.exception(
                "Failed to render figure %s (%s)", failed_task.output_path, failed_task.title
            )
    dropped = len(tasks) - len(paths)
    if dropped:
        logger.error("%d of %d figure tasks failed to render", dropped, len(tasks))
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
            paths = _collect_pool_paths(executor, memmap_tasks)
        else:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                max_tasks_per_child=_max_tasks_per_child(n_workers),
            ) as pool:
                paths = _collect_pool_paths(pool, memmap_tasks)
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
