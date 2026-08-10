"""Local WRF processing — figures + GeoJSON + WebM in a single command.

This script is designed for local testing and development. It combines
the figure generation, GeoJSON/JSON export, and WebM video creation
pipelines into one convenient command.

Usage::

    python -m micrometeorology.cli.run_wrf_pipeline \\
        --wrf-dir /path/to/wrfout_files/ \\
        --date 20260409 \\
        --domains 1,4 \\
        --variables temperature,wind,rain,SWDOWN \\
        --output output/wrf_local/ \\
        --workers 8

    # Quick single-domain test with videos
    python -m micrometeorology.cli.run_wrf_pipeline \\
        --dataset wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/ \\
        --also-video
"""

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.common.cli_options import parse_csv, parse_int_csv
from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf.batch import FigureTask, _max_tasks_per_child, default_workers

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

DEFAULT_VARS = [
    "temperature",
    "pressure",
    "wind",
    "rain",
    "vapor",
    "relative_humidity",
    "skin_temperature",
    "HFX",
    "LH",
    "SWDOWN",
    "poteolico",
    "GLW",
    "wind_power_density_10m",
]


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
    output: Annotated[Path, typer.Option("-o", "--output", help="Base output dir.")] = Path(
        "output/wrf_local"
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
        int | None, typer.Option("-w", "--workers", help=f"Workers (default: {default_workers()}).")
    ] = None,
    dpi: Annotated[int, typer.Option(help="Image DPI.")] = 100,
    no_figures: Annotated[
        bool, typer.Option("--no-figures", help="Skip figure generation.")
    ] = False,
    no_geojson: Annotated[
        bool, typer.Option("--no-geojson", help="Skip GeoJSON/JSON generation.")
    ] = False,
    also_video: Annotated[
        bool, typer.Option("--also-video", help="Generate WebM videos from figures.")
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Run WRF processing locally: figures + GeoJSON + WebM."""
    setup_logging(log_level)
    from micrometeorology.cli.export_wrf_geojson import _reject_output_id_variables
    from micrometeorology.cli.render_wrf_maps import _resolve_wrfout_paths
    from micrometeorology.wrf import reader as wrf_reader

    base_out = Path(output)
    figures_dir = base_out / "figures"
    json_dir = base_out / "JSON"
    geojson_dir = base_out / "GeoJSON"
    video_dir = base_out / "videos"

    resolved_workers = workers or default_workers()
    if resolved_workers < 1:
        raise typer.BadParameter("--workers must be >= 1")
    # Phase 3 encodes the PNGs Phase 1 rendered, so --no-figures leaves it
    # nothing to do. The gate below is `also_video and png_paths`, which the
    # combination makes unconditionally false: the run printed "Complete" and
    # exited 0 having produced none of the videos it was asked for.
    if also_video and no_figures:
        raise typer.BadParameter("--also-video encodes the rendered figures; drop --no-figures")

    var_list = list(parse_csv(variables)) if variables else DEFAULT_VARS
    # An output file id (-v TSK) reaches the raw-NetCDF passthrough in BOTH
    # phases and publishes unconverted Kelvin into the PNGs and JSONs that
    # skin_temperature owns. Refuse before a single frame is rendered.
    _reject_output_id_variables(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, parse_int_csv(domains), dataset)
    if not paths:
        typer.echo("No WRF files found.")
        return

    t0 = time.perf_counter()

    typer.echo("=" * 70)
    typer.echo("  WRF Local Processing Pipeline")
    typer.echo("=" * 70)
    typer.echo(f"  Files: {[p.name for p in paths]}")
    typer.echo(f"  Workers: {resolved_workers}")

    # Phase 1: Figures
    failed_figures = 0
    if not no_figures:
        typer.echo("\n── Phase 1: Figure Generation ──")
        from micrometeorology.cli.render_wrf_maps import _build_tasks_for_domain
        from micrometeorology.cli.render_wrf_maps import (
            _normalize_var_list as _collapse_poteolico_heights,
        )
        from micrometeorology.wrf.batch import run_figure_tasks

        # The figure renderer has no per-height poteolico arm, so heights collapse
        # here only. Phase 2 must keep them: they select which JSON files exist.
        figure_var_list = _collapse_poteolico_heights(var_list)

        # One process pool is hoisted over the figure stage so each 16-task
        # batch reuses warm workers instead of paying pool spawn overhead.
        png_paths: list[str] = []
        figure_pool_ctx: ProcessPoolExecutor | nullcontext[None] = (
            ProcessPoolExecutor(
                max_workers=resolved_workers,
                max_tasks_per_child=_max_tasks_per_child(resolved_workers),
            )
            if resolved_workers > 1
            else nullcontext()
        )
        with figure_pool_ctx as figure_pool:

            def render_task_batch(tasks: list[FigureTask], label: str) -> None:
                nonlocal failed_figures
                rendered = run_figure_tasks(
                    tasks,
                    resolved_workers,
                    backend="auto",
                    executor=figure_pool,
                )
                png_paths.extend(rendered)
                failed_figures += len(tasks) - len(rendered)
                typer.echo(f"    -> {len(rendered)}/{len(tasks)} figures generated for {label}")

            for wrf_path in paths:
                typer.echo(f"  Loading {wrf_path.name}...")

                with wrf_reader.WRFDataset(wrf_path) as ds:
                    _build_tasks_for_domain(
                        ds,
                        figure_var_list,
                        str(figures_dir),
                        shapes_dir,
                        skip_first,
                        dpi,
                        task_sink=render_task_batch,
                    )
        typer.echo(f"  ✓ {len(png_paths)} figures generated")
    else:
        png_paths = []

    # Phase 2: GeoJSON / JSON
    if not no_geojson:
        typer.echo("\n── Phase 2: GeoJSON & JSON Generation ──")
        from micrometeorology.cli.export_wrf_geojson import _normalize_var_list
        from micrometeorology.wrf import jobs

        json_var_list = _normalize_var_list(var_list)
        try:
            units = jobs.build_units(paths, json_var_list, json_dir, geojson_dir, skip_first)
        except ValueError as invalid_selection:
            # A selection covering one domain twice is an operator mistake about
            # -d/-D/-o, so it reads as a usage error rather than a traceback.
            raise typer.BadParameter(str(invalid_selection)) from invalid_selection
        results = jobs.execute_units(units, resolved_workers, echo=typer.echo)

        for result in results:
            for warning in result.warnings:
                typer.echo(f"  ⚠ {warning}")
        manifest_path = jobs.write_run_manifest(
            json_dir,
            results,
            json_var_list,
            covers_every_variable=set(json_var_list) >= set(DEFAULT_VARS),
        )
        if manifest_path:
            typer.echo(f"  ✓ Manifest: {manifest_path}")
        generated_json_count = sum(
            len(result.files) for result in results if result.kind in {"values_json", "poteolico"}
        )
        failed = [result for result in results if result.error]
        typer.echo(f"  ✓ {generated_json_count} JSON files generated")
        if failed:
            typer.echo(f"  ✗ {len(failed)} work units failed:")
            for result in failed:
                typer.echo(f"    - {result.label}: {result.error}")
            raise typer.Exit(code=1)

    # Phase 3: WebM Videos
    failed_videos = 0
    if also_video and png_paths:
        typer.echo("\n── Phase 3: WebM Video Generation ──")
        from micrometeorology.wrf.animation import batch_create_webm

        grouped: dict[str, list[str]] = defaultdict(list)
        for p in sorted(png_paths):
            stem = Path(p).stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                grouped[parts[0]].append(p)
            else:
                grouped[stem].append(p)

        webm_paths = batch_create_webm(grouped, str(video_dir), fps=2, workers=resolved_workers)
        failed_videos = len(grouped) - len(webm_paths)
        typer.echo(f"  ✓ {len(webm_paths)} videos generated")

    elapsed = time.perf_counter() - t0
    typer.echo("\n" + "=" * 70)
    typer.echo(f"  ✓ Complete in {elapsed:.1f}s")
    typer.echo(f"  Output: {base_out.resolve()}")
    typer.echo("=" * 70)

    # Deferred past Phase 2/3 on purpose: dropped frames must not cost the run
    # its front-end JSON/GeoJSON, but they must still be a non-zero exit.
    if failed_figures or failed_videos:
        typer.echo(f"  ✗ {failed_figures} figures and {failed_videos} videos failed (see log)")
        raise typer.Exit(code=1)


def main() -> None:
    """Module entry point (``python -m micrometeorology.cli.run_wrf_pipeline``)."""
    app()


if __name__ == "__main__":
    main()
