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
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from numpy.typing import NDArray

from micrometeorology.common.cli_options import parse_csv, parse_int_csv
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
from micrometeorology.wrf.value_source import build_value_frame_source, publishes_step

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
    "lwup",
    "swup",
    "lwnet",
    "swnet",
    "rnet",
    "sky_emissivity",
    "clearness_index",
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
    for robust glob-based matching of any wrfout filename convention, which
    also rejects a mistyped ``--date`` rather than matching nothing.
    """
    if dataset:
        return [Path(dataset)]

    if not wrf_dir or not date:
        raise typer.BadParameter("Provide either --dataset or --wrf-dir + --date")

    try:
        paths = resolve_wrfout_paths(wrf_dir, date, domains or None)
    except ValueError as invalid_date:
        raise typer.BadParameter(str(invalid_date)) from invalid_date
    if not paths:
        typer.echo(f"  ⚠ No wrfout files found for date {date} in {wrf_dir}")
    return paths


# The phrase each variable's figure title opens with. Everything absent from
# this table titles from its NetCDF output suffix (HFX, PRES, VAPOR, GLW, LH),
# which is what the published PNGs already carry.
_FIGURE_TITLES: dict[str, str] = {
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
_PRESSURE_CONTOUR_LEVELS: list[float] = [880, 900, 950, 1000, 1013]


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
    """Build all FigureTasks for a single domain file.

    Values and colour-scale bounds come from
    :func:`~micrometeorology.wrf.value_source.build_value_frame_source`, the
    same dispatcher the values-JSON work units use, so a variable is renderable
    here exactly when it is exportable there. Only the figure decoration —
    title phrase, colormap, the temperature pressure contours and the wind
    quiver — is decided in this module.
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

        if var_name in _SKIP_FOR_FIGURES:
            typer.echo(f"  ⚠ Skipping {var_name} (no figure renderer)")
            continue
        frame_source = build_value_frame_source(ds, var_name)
        if frame_source is None:
            typer.echo(f"  ⚠ Variable {var_name.upper()} not found in dataset — skipping")
            continue

        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())
        title_prefix = _FIGURE_TITLES.get(var_name, nc_suffix)
        cmap = VARIABLE_COLORMAPS.get(var_name, "viridis")
        # Read once per variable, not once per step: the whole time axis of
        # PSFC is one eager read either way.
        surface_pressure_hpa = (
            ds.get_variable("PSFC") / 100.0 if var_name == WRFVariable.TEMPERATURE else None
        )

        for meta in time_meta:
            if meta.get("skip"):
                continue
            if not publishes_step(var_name, meta):
                continue
            i = meta["index"]
            u: NDArray | None
            v: NDArray | None
            if frame_source.vector_for_step is not None:
                u, v = frame_source.vector_for_step(i)
                data = np.hypot(u, v)
            else:
                u = v = None
                data = frame_source.frame_for_step(i)
            overlay_data = (
                vmod.materialize_2d(surface_pressure_hpa[i : i + 1, :, :])
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
                    overlay_levels=_PRESSURE_CONTOUR_LEVELS if overlay_data is not None else None,
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

    var_list = list(parse_csv(variables)) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, parse_int_csv(domains), dataset)
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
    failed_figures = 0
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
            nonlocal failed_figures
            rendered = run_figure_tasks(
                tasks,
                resolved_workers,
                backend="auto",
                executor=pool,
            )
            png_paths.extend(rendered)
            failed_figures += len(tasks) - len(rendered)
            typer.echo(f"  -> {len(rendered)}/{len(tasks)} figures generated for {label}")

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
    failed_videos = 0
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
        failed_videos = len(grouped) - len(webm_paths)
        typer.echo(f"✓ Generated {len(webm_paths)} videos")

    typer.echo("\n✓ Done")

    # Cron chains on the exit status: a run that dropped frames is not a success,
    # matching labmim-wrf-geojson. Videos and successful PNGs are already final.
    if failed_figures or failed_videos:
        typer.echo(f"✗ {failed_figures} figures and {failed_videos} videos failed (see log)")
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-wrf-figures``)."""
    app()


if __name__ == "__main__":
    main()
