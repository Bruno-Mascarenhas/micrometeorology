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

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.cli.wrfout_selection import glob_wrfout_day, reject_output_id_variables
from micrometeorology.common.cli_options import parse_csv, parse_int_csv
from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf import jobs, reader
from micrometeorology.wrf.batch import (
    FigureTask,
    _max_tasks_per_child,
    build_tasks_for_domain,
    default_workers,
    run_figure_tasks,
)
from micrometeorology.wrf.reader import assert_one_file_per_domain

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

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


def resolve_selection(
    wrf_dir: Path | str | None,
    date: str | None,
    domains: tuple[int, ...],
    dataset: Path | str | None,
) -> list[Path]:
    """Resolve WRF output file paths.

    Delegates to :func:`micrometeorology.wrf.reader.resolve_wrfout_paths`
    for robust glob-based matching of any wrfout filename convention, which
    also rejects a mistyped ``--date`` rather than matching nothing.

    A day carrying two forecast cycles resolves two files for one domain, and a
    figure name is ``{VAR}_{D}_{nnn}.png`` — domain and step index, no token of
    the source file — so the second cycle overwrites the first cycle's PNGs and
    ``--also-video`` mixes both into one WebM.  The guard the JSON path already
    applies in :func:`micrometeorology.wrf.jobs.build_units` is applied here,
    where both the figure and the JSON path read their selection.

    Raises
    ------
    typer.BadParameter
        When two resolved files map to the same domain, so both CLIs report it
        as the usage error it is instead of a traceback.
    """
    if dataset:
        return [Path(dataset)]

    if not wrf_dir or not date:
        raise typer.BadParameter("Provide either --dataset or --wrf-dir + --date")

    paths = glob_wrfout_day(wrf_dir, date, domains)
    if not paths:
        typer.echo(f"  ⚠ No wrfout files found for date {date} in {wrf_dir}")
    try:
        assert_one_file_per_domain(paths)
    except ValueError as invalid_selection:
        raise typer.BadParameter(str(invalid_selection)) from invalid_selection
    return paths


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
    if variables is not None and not var_list:
        raise typer.BadParameter(f"--variables names no variable (got {variables!r})")
    var_list = jobs.normalize_var_list(var_list, collapse_heights=True)
    # Output file ids are not input variables: `-v TSK` would reach the raw-NetCDF
    # passthrough and publish unconverted KELVIN into TSK_D0X_nnn.png, the exact
    # filenames skin_temperature publishes in °C.
    reject_output_id_variables(var_list)
    resolved_workers = default_workers() if workers is None else workers
    if resolved_workers < 1:
        raise typer.BadParameter("--workers must be >= 1")
    paths = resolve_selection(wrf_dir, date, parse_int_csv(domains), dataset)
    if not paths:
        typer.echo("No WRF files found.")
        return

    typer.echo(f"Files: {[p.name for p in paths]}")
    typer.echo(f"Variables: {var_list}")
    typer.echo(f"Output: {output}")
    typer.echo(f"Workers: {resolved_workers}")

    # Tasks are built and rendered per domain/variable so the run never retains
    # every frame in RAM, and one process pool is hoisted over the whole run so
    # each batch reuses warm workers instead of paying pool spawn per flush.
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
                build_tasks_for_domain(
                    ds,
                    var_list,
                    output,
                    shapes_dir,
                    skip_first,
                    dpi,
                    task_sink=render_task_batch,
                    warn=typer.echo,
                )

    typer.echo(f"\n✓ Generated {len(png_paths)} figures")

    failed_videos = 0
    if also_video and png_paths:
        typer.echo("\nGenerating WebM videos...")
        from micrometeorology.wrf.animation import batch_create_webm

        # One WebM per variable+domain: frames are named "<VAR>_<DOMAIN>_<step>.png".
        frames_by_video: dict[str, list[str]] = defaultdict(list)
        for png_path in sorted(png_paths):
            stem = Path(png_path).stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                frames_by_video[parts[0]].append(png_path)
            else:
                frames_by_video[stem].append(png_path)

        webm_paths = batch_create_webm(frames_by_video, output, fps=2, workers=resolved_workers)
        failed_videos = len(frames_by_video) - len(webm_paths)
        typer.echo(f"✓ Generated {len(webm_paths)} videos")

    typer.echo("\n✓ Done")

    # Cron chains on the exit status: a run that dropped frames is not a success.
    # The videos and PNGs that did render are already final.
    if failed_figures or failed_videos:
        typer.echo(f"✗ {failed_figures} figures and {failed_videos} videos failed (see log)")
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-wrf-figures``)."""
    app()


if __name__ == "__main__":
    main()
