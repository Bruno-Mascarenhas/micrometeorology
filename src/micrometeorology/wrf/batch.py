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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.physics import PASCAL_PER_HECTOPASCAL
from micrometeorology.common.types import (
    VARIABLE_COLORMAPS,
    VARIABLE_NETCDF_MAP,
    WRFVariable,
)
from micrometeorology.wrf import reader
from micrometeorology.wrf import variables as vmod
from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
)
from micrometeorology.wrf.value_source import build_value_frame_source, publishes_step

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

        if task.u is not None and task.v is not None:
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

        # On the figure, not on the axes: a cartopy GeoAxes carrying an extent
        # and drawn gridlines reports a layout box that leaves ax.set_title with
        # nowhere to render, and the published maps shipped with no title at all
        # — no forecast hour anywhere on the image but the file name's index.
        fig.suptitle(task.title, fontsize=9, y=0.98)

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


VARIABLES_WITHOUT_FIGURE_RENDERER = {"poteolico", "weibull"}

# The phrase each variable's figure title opens with. A variable absent from
# this table titles from its NetCDF output suffix (HFX, PRES, VAPOR, GLW, LH),
# which is what the published PNGs carry.
FIGURE_TITLES: dict[str, str] = {
    WRFVariable.TEMPERATURE: "Temperature (°C)",
    WRFVariable.SKIN_TEMPERATURE: "Skin Temperature (°C)",
    WRFVariable.RELATIVE_HUMIDITY: "Relative Humidity 2m (%)",
    WRFVariable.WIND: "Wind 10m (m/s)",
    WRFVariable.RAIN: "Rain (mm)",
    WRFVariable.SWDOWN: "SWDOWN (W/m²)",
    WRFVariable.WIND_POWER_DENSITY_10M: "Wind Power Density 10m (W/m²)",
    WRFVariable.LWUP: "Upwelling Longwave (W/m²)",
    WRFVariable.SWUP: "Reflected Shortwave (W/m²)",
    WRFVariable.LWNET: "Net Longwave (W/m²)",
    WRFVariable.SWNET: "Net Shortwave (W/m²)",
    WRFVariable.RNET: "Net Radiation (W/m²)",
    WRFVariable.SKY_EMISSIVITY: "Effective Sky Emissivity (-)",
    WRFVariable.CLEARNESS_INDEX: "Clearness Index kt (-)",
}

# Surface-pressure contours drawn over the temperature field, in hPa.
PRESSURE_CONTOUR_LEVELS: list[float] = [880, 900, 950, 1000, 1013]


def build_tasks_for_domain(
    ds: reader.WRFDataset,
    var_list: list[str],
    output_dir: Path | str,
    shapes_dir: Path | str | None,
    skip_first: int,
    dpi: int,
    task_sink: Callable[[list[FigureTask], str], None] | None = None,
    task_batch_size: int = 16,
    warn: Callable[[str], None] = logger.warning,
) -> list[FigureTask]:
    """Build all FigureTasks for a single domain file.

    Values and colour-scale bounds come from
    :func:`~micrometeorology.wrf.value_source.build_value_frame_source`, the
    same dispatcher the values-JSON work units use, so a variable is renderable
    here exactly when it is exportable there. Only the figure decoration —
    title phrase, colormap, pressure contours and wind quiver — is decided here.

    Parameters
    ----------
    warn:
        Called with a one-line message for each variable skipped — one with no
        figure renderer, one absent from the dataset. Neither is fatal, so the
        run continues; the three WRF commands pass ``typer.echo``, because a
        silently skipped variable is a missing figure the operator has to be
        able to see on the console.
    """
    lon, lat = ds.read_grid()
    bounds = (
        float(np.amin(lon)),
        float(np.amax(lon)),
        float(np.amin(lat)),
        float(np.amax(lat)),
    )
    grid = ds.grid_level.value
    map_config = build_map_config(grid, bounds, str(shapes_dir) if shapes_dir else None)
    time_meta = ds.build_date_metadata(skip_first_n=skip_first)

    tasks: list[FigureTask] = []
    scheduled_output_paths: set[str] = set()

    for var_name in var_list:

        def add_task(task: FigureTask, label: str = var_name) -> None:
            # Two requests for the same frame would race on the non-atomic
            # savefig and duplicate the frame in the WebM.
            if task.output_path in scheduled_output_paths:
                return
            scheduled_output_paths.add(task.output_path)
            tasks.append(task)
            if task_sink is not None and len(tasks) >= task_batch_size:
                task_sink(tasks, label)
                tasks.clear()

        if var_name in VARIABLES_WITHOUT_FIGURE_RENDERER:
            warn(f"  ⚠ Skipping {var_name} (no figure renderer)")
            continue
        frame_source = build_value_frame_source(ds, var_name)
        if frame_source is None:
            warn(f"  ⚠ Variable {var_name.upper()} not found in dataset — skipping")
            continue

        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())
        title_prefix = FIGURE_TITLES.get(var_name, nc_suffix)
        cmap = VARIABLE_COLORMAPS.get(var_name, "viridis")
        # Hoisted out of the step loop: the whole PSFC time axis is one eager
        # read either way. Presence-checked because this is the contour OVERLAY
        # and not the mapped field — a wrfout carrying T2 without PSFC still has
        # a temperature figure to draw, and `get_variable` is a bare dict lookup
        # whose KeyError would take every later variable and domain with it.
        surface_pressure_hpa = (
            ds.get_variable("PSFC") / PASCAL_PER_HECTOPASCAL
            if var_name == WRFVariable.TEMPERATURE and ds.has_variable("PSFC")
            else None
        )

        for meta in time_meta:
            if meta.get("skip"):
                continue
            if not publishes_step(var_name, meta):
                continue
            step = meta["index"]
            u: NDArray | None
            v: NDArray | None
            if frame_source.vector_for_step is not None:
                u, v = frame_source.vector_for_step(step)
                data = np.hypot(u, v)
            else:
                u = v = None
                data = frame_source.frame_for_step(step)
            overlay_data = (
                vmod.materialize_2d(surface_pressure_hpa[step : step + 1, :, :])
                if surface_pressure_hpa is not None
                else None
            )
            add_task(
                FigureTask(
                    lon=lon,
                    lat=lat,
                    data=data,
                    vmin=frame_source.scale_min,
                    vmax=frame_source.scale_max,
                    cmap_name=cmap,
                    overlay_data=overlay_data,
                    overlay_levels=PRESSURE_CONTOUR_LEVELS if overlay_data is not None else None,
                    u=u,
                    v=v,
                    title=f"{title_prefix}{meta['label']}",
                    output_path=str(Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"),
                    map_config=map_config,
                    dpi=dpi,
                    saturation=2.0,
                )
            )

        if task_sink is not None and tasks:
            task_sink(tasks, var_name)
            tasks.clear()

    return tasks


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
