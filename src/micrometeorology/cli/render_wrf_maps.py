"""CLI: Generate WRF map figures with parallel rendering.

Supports multiple domains in a single run. Each domain file is loaded
once; all time steps x variables are dispatched to a worker pool.

Usage::

    # Single domain
    labmim-wrf-figures -d wrfout_d03_2024-01-01 -o output/figures -v temperature wind

    # Multiple domains (auto-detected from directory)
    labmim-wrf-figures --wrf-dir /path/to/wrfout/ --date 20240101 \\
        --domains 1 4 -v temperature wind rain SWDOWN -o output/figures --workers 44

    # All variables, generate WebM videos too
    labmim-wrf-figures --wrf-dir /path/to/ --date 20240101 \\
        --domains 1 4 -o output/ --also-video

    # Auto mode chooses eager for tiny files and lazy for large/chunked inputs
    labmim-wrf-figures --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/figures

    # Force old eager behavior
    labmim-wrf-figures --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/figures --reader eager --chunks none --worker-backend pickle

    # Force xarray-backed lazy reader
    labmim-wrf-figures --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/figures --reader lazy --chunks auto --worker-backend memmap
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import cast

import click
import numpy as np

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
    build_map_config,
    default_workers,
    run_figure_tasks,
)
from micrometeorology.wrf.execution import (
    JsonWorkerRequest,
    ReaderRequest,
    estimate_figure_payload_bytes,
    format_wrf_execution_plan,
    resolve_wrf_execution_plan,
)

# Default variables when none specified
DEFAULT_VARS = [
    "temperature",
    "pressure",
    "wind",
    "rain",
    "vapor",
    "HFX",
    "LH",
    "SWDOWN",
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
    wrf_dir: str | None,
    date: str | None,
    domains: tuple[int, ...],
    dataset: str | None,
) -> list[Path]:
    """Resolve WRF output file paths from arguments."""
    if dataset:
        return [Path(dataset)]

    if not wrf_dir or not date:
        raise click.UsageError("Provide either --dataset or --wrf-dir + --date")

    year, month, day = date[:4], date[4:6], date[6:8]
    dom_start = min(domains) if domains else 1
    dom_end = max(domains) if domains else 4

    paths: list[Path] = []
    for d in range(dom_start, dom_end + 1):
        p = Path(wrf_dir) / f"wrfout_d{d:02d}_{year}-{month}-{day}_00:00:00"
        if p.exists():
            paths.append(p)
        else:
            click.echo(f"  ⚠ Not found: {p}")
    return paths


def _build_tasks_for_domain(
    ds: reader.WRFReader,
    var_list: list[str],
    output_dir: str,
    shapes_dir: str | None,
    skip_first: int,
    dpi: int,
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
    mc = build_map_config(grid, bounds, shapes_dir)
    time_meta = ds.build_date_metadata(skip_first_n=skip_first)

    tasks: list[FigureTask] = []

    for var_name in var_list:
        if var_name in _SKIP_FOR_FIGURES:
            click.echo(f"  ⚠ Skipping {var_name} (no figure renderer)")
            continue
        cmap = VARIABLE_COLORMAPS.get(var_name, "viridis")
        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())

        if var_name == WRFVariable.TEMPERATURE:
            t2, psfc, vmin, vmax = vmod.extract_temperature(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_temperature_step(t2[i : i + 1, :, :])
                pressure = vmod.materialize_2d(psfc[i : i + 1, :, :])
                tasks.append(
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

        elif var_name == WRFVariable.WIND:
            u10, v10, vmin, vmax = vmod.extract_wind(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                u = vmod.materialize_2d(u10[i : i + 1])
                v = vmod.materialize_2d(v10[i : i + 1])
                speed = np.hypot(u, v)
                tasks.append(
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
                tasks.append(
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
                tasks.append(
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
                click.echo(f"  ⚠ Variable {nc_var} not found in dataset — skipping")
                continue

            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(var_data[i : i + 1, :, :])
                tasks.append(
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

    return tasks


@click.command()
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Single WRF NetCDF file.")
@click.option("--wrf-dir", default=None, type=click.Path(), help="Directory with wrfout files.")
@click.option("--date", default=None, help="Simulation date YYYYMMDD.")
@click.option(
    "--domains",
    "-D",
    type=int,
    multiple=True,
    default=None,
    help="Domain numbers (e.g. -D 1 -D 4).",
)
@click.option("--output", "-o", default="output/figures", type=click.Path(), help="Output dir.")
@click.option("--variables", "-v", multiple=True, default=None, help="Variables to plot.")
@click.option("--shapes-dir", default=None, help="Municipality shapefiles dir.")
@click.option("--skip-first", default=0, type=int, help="Time steps to skip.")
@click.option(
    "--reader",
    "reader_backend",
    default="auto",
    type=click.Choice(["auto", "eager", "lazy"]),
    show_default=True,
    help="WRF reader backend. Auto chooses eager for small inputs and lazy for large/chunked inputs.",
)
@click.option(
    "--chunks",
    default="auto",
    show_default=True,
    help="Lazy-reader chunks: 'auto', 'none', or comma-separated dim=size pairs.",
)
@click.option(
    "--workers",
    "-w",
    default=None,
    type=int,
    help=f"Parallel workers (default: {default_workers()}).",
)
@click.option(
    "--worker-backend",
    default="auto",
    type=click.Choice(["auto", "serial", "pickle", "memmap"]),
    show_default=True,
    help="Figure worker payload backend. Auto uses serial for single-worker work and memmap for large multi-worker payloads.",
)
@click.option(
    "--tmp-dir",
    "tmp_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Parent directory for temporary memmap payloads when --worker-backend memmap.",
)
@click.option("--dpi", default=100, type=int, help="Image DPI.")
@click.option("--also-video", is_flag=True, help="Also generate WebM videos.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    dataset: str | None,
    wrf_dir: str | None,
    date: str | None,
    domains: tuple[int, ...],
    output: str,
    variables: tuple[str, ...],
    shapes_dir: str | None,
    skip_first: int,
    reader_backend: str,
    chunks: str,
    workers: int | None,
    worker_backend: str,
    tmp_dir: str | None,
    dpi: int,
    also_video: bool,
    log_level: str,
) -> None:
    """Generate WRF map figures with parallel rendering."""
    setup_logging(log_level)

    var_list = list(variables) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, domains, dataset)
    try:
        plan = resolve_wrf_execution_plan(
            paths=paths,
            workflow="figures",
            reader_request=cast("ReaderRequest", reader_backend),
            chunks_request=chunks,
            json_worker_request=cast("JsonWorkerRequest", worker_backend),
            workers=workers,
            tmp_dir=tmp_dir,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc

    if not paths:
        click.echo("No WRF files found.")
        return

    click.echo(f"Files: {[p.name for p in paths]}")
    click.echo(f"Variables: {var_list}")
    click.echo(f"Output: {output}")
    click.echo(format_wrf_execution_plan(plan))

    # Phase 1: Build all tasks (serial — fast, I/O-bound NetCDF reads)
    all_tasks: list[FigureTask] = []
    for wrf_path in paths:
        click.echo(f"\nLoading {wrf_path.name}...")
        with reader.open_wrf_dataset(
            wrf_path,
            reader=plan.reader,
            chunks=plan.chunks,
        ) as ds:
            tasks = _build_tasks_for_domain(ds, var_list, output, shapes_dir, skip_first, dpi)
            all_tasks.extend(tasks)
            click.echo(f"  → {len(tasks)} frames queued")

    click.echo(f"\nTotal frames: {len(all_tasks)}")
    try:
        final_plan = resolve_wrf_execution_plan(
            paths=paths,
            workflow="figures",
            reader_request=cast("ReaderRequest", plan.reader),
            chunks_request=chunks,
            json_worker_request=cast("JsonWorkerRequest", worker_backend),
            workers=plan.workers,
            tmp_dir=tmp_dir,
            estimated_json_payload_bytes=estimate_figure_payload_bytes(all_tasks),
            json_task_count=len(all_tasks),
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if final_plan != plan:
        click.echo(format_wrf_execution_plan(final_plan))

    # Phase 2: Parallel rendering
    png_paths = run_figure_tasks(
        all_tasks,
        workers=final_plan.workers,
        backend=final_plan.json_worker_backend,
        tmp_dir=final_plan.tmp_dir,
    )

    click.echo(f"\n✓ Generated {len(png_paths)} figures")

    # Phase 3: WebM (optional)
    if also_video and png_paths:
        click.echo("\nGenerating WebM videos...")
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

        webm_paths = batch_create_webm(grouped, output, fps=2, workers=final_plan.workers)
        click.echo(f"✓ Generated {len(webm_paths)} videos")

    click.echo("\n✓ Done")


if __name__ == "__main__":
    main()
