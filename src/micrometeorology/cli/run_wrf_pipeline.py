"""Local WRF processing — figures + GeoJSON + WebM in a single command.

This script is designed for local testing and development. It combines
the figure generation, GeoJSON/JSON export, and WebM video creation
pipelines into one convenient command.

Usage::

    python scripts/micromet/run_wrf_local.py \\
        --wrf-dir /path/to/wrfout_files/ \\
        --date 20260409 \\
        --domains 1 4 \\
        --variables temperature wind rain SWDOWN \\
        --output output/wrf_local/ \\
        --workers 8

    # Quick single-domain test with videos
    python scripts/micromet/run_wrf_local.py \\
        --dataset wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/ \\
        --also-video

    # Auto mode chooses eager/pickle for tiny work and lazy/memmap for large work
    python scripts/micromet/run_wrf_local.py \\
        --dataset /path/to/wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/

    # Force old eager/pickle behavior
    python scripts/micromet/run_wrf_local.py \\
        --dataset /path/to/wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/ --reader eager --chunks none --json-worker-backend pickle

    # Force lazy reader plus memmap figure and JSON worker payloads
    python scripts/micromet/run_wrf_local.py \\
        --dataset /path/to/wrfout_d03_2026-04-09_00:00:00 \\
        --output output/test/ --reader lazy \\
        --figure-worker-backend memmap --json-worker-backend memmap
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import cast

import click

from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf.batch import default_workers
from micrometeorology.wrf.execution import (
    JsonWorkerRequest,
    ReaderRequest,
    estimate_figure_payload_bytes,
    estimate_json_payload_bytes,
    format_wrf_execution_plan,
    resolve_wrf_execution_plan,
)


@click.command()
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Single WRF file.")
@click.option("--wrf-dir", default=None, type=click.Path(), help="Directory with wrfout files.")
@click.option("--date", default=None, help="Simulation date YYYYMMDD.")
@click.option("--domains", "-D", type=int, multiple=True, default=None, help="Domain numbers.")
@click.option(
    "--output", "-o", default="output/wrf_local", type=click.Path(), help="Base output dir."
)
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
    "--workers", "-w", default=None, type=int, help=f"Workers (default: {default_workers()})."
)
@click.option("--dpi", default=100, type=int, help="Image DPI.")
@click.option("--no-figures", is_flag=True, help="Skip figure generation (only JSON/GeoJSON).")
@click.option("--no-geojson", is_flag=True, help="Skip GeoJSON/JSON generation.")
@click.option("--also-video", is_flag=True, help="Generate WebM videos from figures.")
@click.option(
    "--figure-worker-backend",
    default="auto",
    type=click.Choice(["auto", "serial", "pickle", "memmap"]),
    show_default=True,
    help="Figure worker payload backend.",
)
@click.option(
    "--figure-tmp-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Parent directory for temporary figure memmap payloads.",
)
@click.option(
    "--json-worker-backend",
    default="auto",
    type=click.Choice(["auto", "serial", "pickle", "memmap"]),
    show_default=True,
    help="JSON worker payload backend. Auto uses serial for single-worker work and memmap for large multi-worker exports.",
)
@click.option(
    "--json-tmp-dir",
    "--tmp-dir",
    "json_tmp_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Parent directory for temporary memmap payloads in JSON phase.",
)
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
    dpi: int,
    no_figures: bool,
    no_geojson: bool,
    also_video: bool,
    figure_worker_backend: str,
    figure_tmp_dir: str | None,
    json_worker_backend: str,
    json_tmp_dir: str | None,
    log_level: str,
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

    click.echo("=" * 70)
    click.echo("  WRF Local Processing Pipeline")
    click.echo("=" * 70)

    video_workers = workers

    # Phase 1: Figures
    if not no_figures:
        click.echo("\n── Phase 1: Figure Generation ──")
        from micrometeorology.cli.render_wrf_maps import (
            _build_tasks_for_domain,
            _resolve_wrfout_paths,
        )
        from micrometeorology.wrf.batch import FigureTask, run_figure_tasks

        default_vars = [
            "temperature",
            "pressure",
            "wind",
            "rain",
            "vapor",
            "HFX",
            "LH",
            "SWDOWN",
        ]
        var_list = list(variables) if variables else default_vars
        paths = _resolve_wrfout_paths(wrf_dir, date, domains, dataset)

        if not paths:
            click.echo("No WRF files found.")
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
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        click.echo(format_wrf_execution_plan(figure_plan))

        all_fig_tasks: list[FigureTask] = []
        for wrf_path in paths:
            click.echo(f"  Loading {wrf_path.name}...")
            with wrf_reader.open_wrf_dataset(
                wrf_path,
                reader=figure_plan.reader,
                chunks=figure_plan.chunks,
            ) as ds:
                figure_tasks = _build_tasks_for_domain(
                    ds,
                    var_list,
                    str(figures_dir),
                    shapes_dir,
                    skip_first,
                    dpi,
                )
                all_fig_tasks.extend(figure_tasks)
                click.echo(f"    → {len(figure_tasks)} frames queued")

        click.echo(f"  Total: {len(all_fig_tasks)} frames")
        try:
            final_figure_plan = resolve_wrf_execution_plan(
                paths=paths,
                workflow="figures",
                reader_request=cast("ReaderRequest", figure_plan.reader),
                chunks_request=chunks,
                json_worker_request=cast("JsonWorkerRequest", figure_worker_backend),
                workers=figure_plan.workers,
                tmp_dir=figure_tmp_dir,
                estimated_json_payload_bytes=estimate_figure_payload_bytes(all_fig_tasks),
                json_task_count=len(all_fig_tasks),
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if final_figure_plan != figure_plan:
            click.echo(format_wrf_execution_plan(final_figure_plan))
        png_paths = run_figure_tasks(
            all_fig_tasks,
            workers=final_figure_plan.workers,
            backend=final_figure_plan.json_worker_backend,
            tmp_dir=final_figure_plan.tmp_dir,
        )
        video_workers = final_figure_plan.workers
        click.echo(f"  ✓ {len(png_paths)} figures generated")
    else:
        png_paths = []

    # Phase 2: GeoJSON / JSON
    if not no_geojson:
        click.echo("\n── Phase 2: GeoJSON & JSON Generation ──")
        from micrometeorology.cli.export_wrf_geojson import (
            _build_json_tasks_for_domain,
        )
        from micrometeorology.cli.export_wrf_geojson import (
            _resolve_wrfout_paths as _resolve_geo,
        )
        from micrometeorology.wrf.batch import JsonTask, run_json_tasks

        default_vars = [
            "temperature",
            "pressure",
            "wind",
            "rain",
            "vapor",
            "HFX",
            "LH",
            "SWDOWN",
        ]
        var_list = list(variables) if variables else default_vars
        paths = _resolve_geo(wrf_dir, date, domains, dataset)
        try:
            json_plan = resolve_wrf_execution_plan(
                paths=paths,
                workflow="json",
                reader_request=cast("ReaderRequest", reader_backend),
                chunks_request=chunks,
                json_worker_request=cast("JsonWorkerRequest", json_worker_backend),
                workers=workers,
                tmp_dir=json_tmp_dir,
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        click.echo(format_wrf_execution_plan(json_plan))

        all_json_tasks: list[JsonTask] = []
        for wrf_path in paths:
            click.echo(f"  Loading {wrf_path.name}...")
            with wrf_reader.open_wrf_dataset(
                wrf_path,
                reader=json_plan.reader,
                chunks=json_plan.chunks,
            ) as ds:
                json_tasks = _build_json_tasks_for_domain(
                    ds,
                    var_list,
                    str(json_dir),
                    str(geojson_dir),
                    skip_first,
                )
                all_json_tasks.extend(json_tasks)
                click.echo(f"    → {len(json_tasks)} JSON files queued")

        click.echo(f"  Total: {len(all_json_tasks)} JSON files")
        try:
            final_json_plan = resolve_wrf_execution_plan(
                paths=paths,
                workflow="json",
                reader_request=cast("ReaderRequest", json_plan.reader),
                chunks_request=chunks,
                json_worker_request=cast("JsonWorkerRequest", json_worker_backend),
                workers=json_plan.workers,
                tmp_dir=json_tmp_dir,
                estimated_json_payload_bytes=estimate_json_payload_bytes(all_json_tasks),
                json_task_count=len(all_json_tasks),
            )
        except ValueError as exc:
            raise click.UsageError(str(exc)) from exc
        if final_json_plan != json_plan:
            click.echo(format_wrf_execution_plan(final_json_plan))
        json_paths = run_json_tasks(
            all_json_tasks,
            workers=final_json_plan.workers,
            backend=final_json_plan.json_worker_backend,
            tmp_dir=final_json_plan.tmp_dir,
        )
        click.echo(f"  ✓ {len(json_paths)} JSON files generated")

    # Phase 3: WebM Videos
    if also_video and png_paths:
        click.echo("\n── Phase 3: WebM Video Generation ──")
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
        click.echo(f"  ✓ {len(webm_paths)} videos generated")

    elapsed = time.perf_counter() - t0
    click.echo("\n" + "=" * 70)
    click.echo(f"  ✓ Complete in {elapsed:.1f}s")
    click.echo(f"  Output: {base_out.resolve()}")
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
