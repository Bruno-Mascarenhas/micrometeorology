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

    # Auto mode chooses serial for single-worker work and memmap for multi-worker work
    python -m micrometeorology.cli.run_wrf_pipeline \\
        --dataset /path/to/wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/

    # Force lazy reader plus memmap figure and JSON worker payloads
    python -m micrometeorology.cli.run_wrf_pipeline \\
        --dataset /path/to/wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/ --reader lazy \\
        --figure-worker-backend memmap --json-worker-backend memmap
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast

import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf.batch import _max_tasks_per_child, default_workers
from micrometeorology.wrf.execution import (
    JsonWorkerRequest,
    ReaderRequest,
    format_wrf_execution_plan,
    resolve_wrf_execution_plan,
)

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


def _parse_int_csv(raw: str | list[str] | None) -> tuple[int, ...]:
    """Parse comma-separated or repeated integers."""
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    res: list[int] = []
    for item in raw:
        res.extend(int(x.strip()) for x in item.split(",") if x.strip())
    return tuple(res)


def _parse_csv(raw: str | list[str] | None) -> tuple[str, ...]:
    """Parse comma-separated or repeated strings."""
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    res: list[str] = []
    for item in raw:
        res.extend(x.strip() for x in item.split(",") if x.strip())
    return tuple(res)


class ReaderChoice(StrEnum):
    auto = "auto"
    eager = "eager"
    lazy = "lazy"


class WorkerChoice(StrEnum):
    auto = "auto"
    serial = "serial"
    memmap = "memmap"


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
    reader_backend: Annotated[
        ReaderChoice, typer.Option("--reader", help="WRF reader backend.")
    ] = ReaderChoice.auto,
    chunks: Annotated[
        str, typer.Option(help="Lazy-reader chunks: 'auto', 'none', or dim=size pairs.")
    ] = "auto",
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
    figure_worker_backend: Annotated[
        WorkerChoice, typer.Option("--figure-worker-backend", help="Figure worker backend.")
    ] = WorkerChoice.auto,
    figure_tmp_dir: Annotated[
        Path | None, typer.Option("--figure-tmp-dir", help="Temp dir for figure memmap payloads.")
    ] = None,
    json_worker_backend: Annotated[
        WorkerChoice, typer.Option("--json-worker-backend", help="JSON worker backend.")
    ] = WorkerChoice.auto,
    json_tmp_dir: Annotated[
        Path | None,
        typer.Option("--json-tmp-dir", "--tmp-dir", help="Temp dir for JSON memmap payloads."),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Run WRF processing locally: figures + GeoJSON + WebM."""
    setup_logging(log_level)
    from micrometeorology.wrf import reader as wrf_reader

    base_out = Path(output)
    figures_dir = base_out / "figures"
    json_dir = base_out / "JSON"
    geojson_dir = base_out / "GeoJSON"
    video_dir = base_out / "videos"

    t0 = time.perf_counter()

    typer.echo("=" * 70)
    typer.echo("  WRF Local Processing Pipeline")
    typer.echo("=" * 70)

    video_workers = workers

    # Phase 1: Figures
    if not no_figures:
        typer.echo("\n── Phase 1: Figure Generation ──")
        from micrometeorology.cli.render_wrf_maps import (
            _build_tasks_for_domain,
            _resolve_wrfout_paths,
        )
        from micrometeorology.wrf.batch import run_figure_tasks

        default_vars = [
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
        var_list = list(_parse_csv(variables)) if variables else default_vars
        paths = _resolve_wrfout_paths(wrf_dir, date, _parse_int_csv(domains), dataset)

        if not paths:
            typer.echo("No WRF files found.")
            return
        try:
            figure_plan = resolve_wrf_execution_plan(
                paths=paths,
                workflow="figures",
                reader_request=cast("ReaderRequest", reader_backend),
                chunks_request=chunks,
                json_worker_request=cast("JsonWorkerRequest", figure_worker_backend),
                workers=workers,
                tmp_dir=figure_tmp_dir,
                requested_variables=var_list,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(format_wrf_execution_plan(figure_plan))

        # One process pool is hoisted over the figure stage so each 16-task
        # batch reuses warm workers instead of paying pool spawn overhead.
        png_paths: list[str] = []
        figure_pool_ctx: ProcessPoolExecutor | nullcontext[None] = (
            ProcessPoolExecutor(
                max_workers=figure_plan.workers,
                max_tasks_per_child=_max_tasks_per_child(figure_plan.workers),
            )
            if figure_plan.workers > 1 and figure_plan.json_worker_backend != "serial"
            else nullcontext()
        )
        with figure_pool_ctx as figure_pool:

            def render_task_batch(tasks, label):  # type: ignore[no-untyped-def]
                rendered = run_figure_tasks(
                    tasks,
                    figure_plan.workers,
                    backend=figure_plan.json_worker_backend,
                    tmp_dir=figure_plan.tmp_dir,
                    executor=figure_pool,
                )
                png_paths.extend(rendered)
                typer.echo(f"    -> {len(rendered)} figures generated for {label}")

            for wrf_path in paths:
                typer.echo(f"  Loading {wrf_path.name}...")

                with wrf_reader.open_wrf_dataset(
                    wrf_path,
                    reader=figure_plan.reader,
                    chunks=figure_plan.chunks,
                ) as ds:
                    _build_tasks_for_domain(
                        ds,
                        var_list,
                        str(figures_dir),
                        shapes_dir,
                        skip_first,
                        dpi,
                        task_sink=render_task_batch,
                    )
        video_workers = figure_plan.workers
        typer.echo(f"  ✓ {len(png_paths)} figures generated")
    else:
        png_paths = []

    # Phase 2: GeoJSON / JSON
    if not no_geojson:
        typer.echo("\n── Phase 2: GeoJSON & JSON Generation ──")
        from micrometeorology.cli.export_wrf_geojson import (
            _build_json_tasks_for_domain,
            _normalize_var_list,
        )
        from micrometeorology.cli.export_wrf_geojson import (
            _resolve_wrfout_paths as _resolve_geo,
        )
        from micrometeorology.wrf.batch import run_json_tasks

        default_vars = [
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
        var_list = _normalize_var_list(list(_parse_csv(variables)) if variables else default_vars)
        paths = _resolve_geo(wrf_dir, date, _parse_int_csv(domains), dataset)
        try:
            json_plan = resolve_wrf_execution_plan(
                paths=paths,
                workflow="json",
                reader_request=cast("ReaderRequest", reader_backend),
                chunks_request=chunks,
                json_worker_request=cast("JsonWorkerRequest", json_worker_backend),
                workers=workers,
                tmp_dir=json_tmp_dir,
                requested_variables=var_list,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        typer.echo(format_wrf_execution_plan(json_plan))

        # Same hoist as the figure stage: one pool for the whole JSON stage.
        generated_json_count = 0
        json_pool_ctx: ProcessPoolExecutor | nullcontext[None] = (
            ProcessPoolExecutor(
                max_workers=json_plan.workers,
                max_tasks_per_child=_max_tasks_per_child(json_plan.workers),
            )
            if json_plan.workers > 1 and json_plan.json_worker_backend != "serial"
            else nullcontext()
        )
        with json_pool_ctx as json_pool:

            def write_task_batch(tasks, label):  # type: ignore[no-untyped-def]
                nonlocal generated_json_count
                json_paths = run_json_tasks(
                    tasks,
                    json_plan.workers,
                    backend=json_plan.json_worker_backend,
                    tmp_dir=json_plan.tmp_dir,
                    executor=json_pool,
                )
                generated_json_count += len(json_paths)
                typer.echo(f"    -> {len(json_paths)} JSON files generated for {label}")

            for wrf_path in paths:
                typer.echo(f"  Loading {wrf_path.name}...")

                with wrf_reader.open_wrf_dataset(
                    wrf_path,
                    reader=json_plan.reader,
                    chunks=json_plan.chunks,
                ) as ds:
                    _build_json_tasks_for_domain(
                        ds,
                        var_list,
                        str(json_dir),
                        str(geojson_dir),
                        skip_first,
                        task_sink=write_task_batch,
                    )
        typer.echo(f"  ✓ {generated_json_count} JSON files generated")

    # Phase 3: WebM Videos
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

        webm_paths = batch_create_webm(grouped, str(video_dir), fps=2, workers=video_workers)
        typer.echo(f"  ✓ {len(webm_paths)} videos generated")

    elapsed = time.perf_counter() - t0
    typer.echo("\n" + "=" * 70)
    typer.echo(f"  ✓ Complete in {elapsed:.1f}s")
    typer.echo(f"  Output: {base_out.resolve()}")
    typer.echo("=" * 70)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
