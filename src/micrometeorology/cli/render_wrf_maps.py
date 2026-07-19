"""CLI: Generate WRF map figures with parallel rendering.

Supports multiple domains in a single run. Each domain file is loaded
once; all time steps x variables are dispatched to a worker pool.

Usage::

    # Single domain
    labmim-wrf-figures -d wrfout_d03_2024-01-01 -o output/figures -v temperature wind

    # Multiple domains (auto-detected from directory)
    labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \\
        --domains 1,4 -v temperature,wind,rain,SWDOWN -o output/figures --workers 44

    # All variables, generate WebM videos too
    labmim-wrf-figures --wrf-dir /path/to/ --date 20240101 \\
        -D 1 -D 4 -o output/ --also-video
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import numpy as np
import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.types import (
    VARIABLE_COLORMAPS,
    VARIABLE_NETCDF_MAP,
    WRFVariable,
)
from micrometeorology.wrf import reader
from micrometeorology.wrf import variables as vmod
from micrometeorology.wrf.batch import (
    FigureTask,
    _max_tasks_per_child,
    build_map_config,
    default_workers,
    run_figure_tasks,
)
from micrometeorology.wrf.reader import resolve_wrfout_paths

if TYPE_CHECKING:
    from collections.abc import Callable

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

# Default variables when none specified
DEFAULT_VARS = [
    "temperature",
    "pressure",
    "wind",
    "rain",
    "vapor",
    "skin_temperature",
    "relative_humidity",
    "HFX",
    "LH",
    "SWDOWN",
    "GLW",
    "wind_power_density_10m",
]

# Variables that exist in the pipeline but don't have figure renderers yet.
# We skip these silently rather than showing confusing "not found" warnings.
_SKIP_FOR_FIGURES = {"poteolico", "weibull"}


def _normalize_var_list(var_list: list[str]) -> list[str]:
    """Normalize legacy variable names.

    Collapses ``poteolico50``, ``poteolico100``, ``poteolico150`` into
    a single ``poteolico`` entry (deduplicating).
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for v in var_list:
        if v.startswith("poteolico") and v != "poteolico":
            v = "poteolico"
        if v not in seen:
            normalized.append(v)
            seen.add(v)
    return normalized


def _resolve_wrfout_paths(
    wrf_dir: Path | str | None,
    date: str | None,
    domains: tuple[int, ...],
    dataset: Path | str | None,
) -> list[Path]:
    """Resolve WRF output file paths.

    Delegates to :func:`micrometeorology.wrf.reader.resolve_wrfout_paths`
    for robust glob-based matching of any wrfout filename convention.
    """
    if dataset:
        return [Path(dataset)]

    if not wrf_dir or not date:
        raise typer.BadParameter("Provide either --dataset or --wrf-dir + --date")

    paths = resolve_wrfout_paths(wrf_dir, date, domains or None)
    if not paths:
        typer.echo(f"  ⚠ No wrfout files found for date {date} in {wrf_dir}")
    return paths


def _build_tasks_for_domain(
    ds: reader.WRFDataset,
    var_list: list[str],
    output_dir: Path | str,
    shapes_dir: Path | str | None,
    skip_first: int,
    dpi: int,
    task_sink: Callable[[list[FigureTask], str], None] | None = None,
    task_batch_size: int = 16,
) -> list[FigureTask]:
    """Build all FigureTasks for a single domain file."""
    lon, lat = ds.read_grid()
    bounds = (
        float(np.amin(lon)),
        float(np.amax(lon)),
        float(np.amin(lat)),
        float(np.amax(lat)),
    )
    grid = ds.grid_level.value
    mc = build_map_config(grid, bounds, str(shapes_dir) if shapes_dir else None)
    time_meta = ds.build_date_metadata(skip_first_n=skip_first)

    tasks: list[FigureTask] = []

    for var_name in var_list:

        def add_task(task: FigureTask, label: str = var_name) -> None:
            tasks.append(task)
            if task_sink is not None and len(tasks) >= task_batch_size:
                task_sink(tasks, label)
                tasks.clear()

        if var_name in _SKIP_FOR_FIGURES:
            typer.echo(f"  ⚠ Skipping {var_name} (no figure renderer)")
            continue
        cmap = VARIABLE_COLORMAPS.get(var_name, "viridis")
        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())

        if var_name == WRFVariable.TEMPERATURE:
            t2, vmin, vmax = vmod.extract_temperature(ds)
            psfc = ds.get_variable("PSFC") / 100.0
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_temperature_step(t2[i : i + 1, :, :])
                pressure = vmod.materialize_2d(psfc[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=vmod.materialize_2d(data),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=pressure,
                        overlay_levels=[880, 900, 950, 1000, 1013],
                        u=None,
                        v=None,
                        title=f"Temperature (°C){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.SKIN_TEMPERATURE:
            tsk, vmin, vmax = vmod.extract_skin_temperature(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_temperature_step(tsk[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=vmod.materialize_2d(data),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"Skin Temperature (°C){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.RELATIVE_HUMIDITY:
            rh, vmin, vmax = vmod.extract_relative_humidity(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(rh[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"Relative Humidity 2m (%){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.WIND:
            u10, v10, vmin, vmax = vmod.extract_wind(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                u = vmod.materialize_2d(u10[i : i + 1])
                v = vmod.materialize_2d(v10[i : i + 1])
                speed = np.hypot(u, v)
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=speed,
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=u,
                        v=v,
                        title=f"Wind 10m (m/s){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.RAIN:
            total, vmin, vmax = vmod.extract_rain(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_rain_step(total, i)
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=vmod.materialize_2d(data),
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"Rain (mm){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.SWDOWN:
            # Solar radiation — skip nighttime (local hours 0-5 and 19-23)
            var_data, vmin, vmax = vmod.extract_scalar(ds, "SWDOWN")
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                local_hour = meta["datetime_local"].hour
                if local_hour < 6 or local_hour > 18:
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(var_data[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"SWDOWN (W/m²){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        elif var_name == WRFVariable.WIND_POWER_DENSITY_10M:
            power_density, vmin, vmax = vmod.extract_wind_power_density_10m(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(power_density[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"Wind Power Density 10m (W/m²){meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        else:
            # Generic scalar (HFX, LH, pressure, vapor)
            nc_var = var_name.upper()
            if var_name == WRFVariable.PRESSURE:
                var_data, vmin, vmax = vmod.extract_pressure(ds)
            elif var_name == WRFVariable.VAPOR:
                var_data, vmin, vmax = vmod.extract_vapor(ds)
            elif ds.has_variable(nc_var):
                var_data, vmin, vmax = vmod.extract_scalar(ds, nc_var)
            else:
                typer.echo(f"  ⚠ Variable {nc_var} not found in dataset — skipping")
                continue

            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(var_data[i : i + 1, :, :])
                add_task(
                    FigureTask(
                        lon=lon,
                        lat=lat,
                        data=data,
                        vmin=vmin,
                        vmax=vmax,
                        cmap_name=cmap,
                        overlay_data=None,
                        overlay_levels=None,
                        u=None,
                        v=None,
                        title=f"{nc_suffix}{meta['label']}",
                        output_path=str(
                            Path(output_dir) / f"{nc_suffix}_{meta['name_suffix']}.png"
                        ),
                        map_config=mc,
                        dpi=dpi,
                        saturation=2.0,
                    )
                )

        if task_sink is not None and tasks:
            task_sink(tasks, var_name)
            tasks.clear()

    return tasks


def _parse_csv(raw: str | list[str] | None) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    res: list[str] = []
    for item in raw:
        res.extend(x.strip() for x in item.split(",") if x.strip())
    return tuple(res)


def _parse_int_csv(raw: str | list[str] | None) -> tuple[int, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    res: list[int] = []
    for item in raw:
        res.extend(int(x.strip()) for x in item.split(",") if x.strip())
    return tuple(res)


@app.command()
def run(
    dataset: Annotated[
        Path | None, typer.Option("-d", "--dataset", help="Single WRF file.")
    ] = None,
    wrf_dir: Annotated[Path | None, typer.Option(help="Directory with wrfout files.")] = None,
    date: Annotated[str | None, typer.Option(help="Simulation date YYYYMMDD.")] = None,
    domains: Annotated[
        list[str] | None,
        typer.Option("-D", "--domains", help="Domain numbers. Can be repeated or comma-separated."),
    ] = None,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output dir.")] = Path(
        "output/figures"
    ),
    variables: Annotated[
        list[str] | None,
        typer.Option(
            "-v", "--variables", help="Variables to process. Can be repeated or comma-separated."
        ),
    ] = None,
    shapes_dir: Annotated[Path | None, typer.Option(help="Municipality shapefiles dir.")] = None,
    skip_first: Annotated[int, typer.Option(help="Time steps to skip.")] = 0,
    workers: Annotated[
        int | None,
        typer.Option("-w", "--workers", help=f"Parallel workers (default: {default_workers()})."),
    ] = None,
    dpi: Annotated[int, typer.Option(help="Image DPI.")] = 100,
    also_video: Annotated[
        bool, typer.Option("--also-video", help="Also generate WebM videos.")
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate WRF map figures with parallel rendering."""
    setup_logging(log_level)

    var_list = list(_parse_csv(variables)) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, _parse_int_csv(domains), dataset)
    if not paths:
        typer.echo("No WRF files found.")
        return

    resolved_workers = workers or default_workers()
    if resolved_workers < 1:
        raise typer.BadParameter("--workers must be >= 1")

    typer.echo(f"Files: {[p.name for p in paths]}")
    typer.echo(f"Variables: {var_list}")
    typer.echo(f"Output: {output}")
    typer.echo(f"Workers: {resolved_workers}")

    # Build and render tasks per domain/variable to avoid retaining all frames in RAM.
    # One process pool is hoisted over the whole run so each 16-task batch reuses
    # warm workers instead of paying pool spawn overhead per flush.
    png_paths: list[str] = []
    pool_ctx: ProcessPoolExecutor | nullcontext[None] = (
        ProcessPoolExecutor(
            max_workers=resolved_workers,
            max_tasks_per_child=_max_tasks_per_child(resolved_workers),
        )
        if resolved_workers > 1
        else nullcontext()
    )
    with pool_ctx as pool:

        def render_task_batch(tasks: list[FigureTask], label: str) -> None:
            rendered = run_figure_tasks(
                tasks,
                resolved_workers,
                backend="auto",
                executor=pool,
            )
            png_paths.extend(rendered)
            typer.echo(f"  -> {len(rendered)} figures generated for {label}")

        for wrf_path in paths:
            typer.echo(f"\nLoading {wrf_path.name}...")

            with reader.WRFDataset(wrf_path) as ds:
                _build_tasks_for_domain(
                    ds,
                    var_list,
                    output,
                    shapes_dir,
                    skip_first,
                    dpi,
                    task_sink=render_task_batch,
                )

    typer.echo(f"\n✓ Generated {len(png_paths)} figures")

    # Phase 3: WebM (optional)
    if also_video and png_paths:
        typer.echo("\nGenerating WebM videos...")
        from micrometeorology.wrf.animation import batch_create_webm

        # Group PNGs by variable+domain prefix (e.g. "TEMP_D03")
        grouped: dict[str, list[str]] = defaultdict(list)
        for p in sorted(png_paths):
            stem = Path(p).stem  # e.g. "TEMP_D03_001"
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                grouped[parts[0]].append(p)
            else:
                grouped[stem].append(p)

        webm_paths = batch_create_webm(grouped, output, fps=2, workers=resolved_workers)
        typer.echo(f"✓ Generated {len(webm_paths)} videos")

    typer.echo("\n✓ Done")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
