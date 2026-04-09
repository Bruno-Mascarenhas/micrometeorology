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
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import click

from labmim_micrometeorology.common.logging import setup_logging
from labmim_micrometeorology.wrf.batch import default_workers


@click.command()
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Single WRF file.")
@click.option("--wrf-dir", default=None, type=click.Path(), help="Directory with wrfout files.")
@click.option("--date", default=None, help="Simulation date YYYYMMDD.")
@click.option("--domains", "-D", type=int, multiple=True, default=None, help="Domain numbers.")
@click.option("--output", "-o", default="output/wrf_local", type=click.Path(), help="Base output dir.")
@click.option("--variables", "-v", multiple=True, default=None, help="Variables to plot.")
@click.option("--shapes-dir", default=None, help="Municipality shapefiles dir.")
@click.option("--skip-first", default=0, type=int, help="Time steps to skip.")
@click.option("--workers", "-w", default=None, type=int, help=f"Workers (default: {default_workers()}).")
@click.option("--dpi", default=100, type=int, help="Image DPI.")
@click.option("--no-figures", is_flag=True, help="Skip figure generation (only JSON/GeoJSON).")
@click.option("--no-geojson", is_flag=True, help="Skip GeoJSON/JSON generation.")
@click.option("--also-video", is_flag=True, help="Generate WebM videos from figures.")
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
    workers: int | None,
    dpi: int,
    no_figures: bool,
    no_geojson: bool,
    also_video: bool,
    log_level: str,
) -> None:
    """Run WRF processing locally: figures + GeoJSON + WebM."""
    setup_logging(log_level)

    base_out = Path(output)
    figures_dir = base_out / "figures"
    json_dir = base_out / "JSON"
    geojson_dir = base_out / "GeoJSON"
    video_dir = base_out / "videos"

    t0 = time.perf_counter()

    click.echo("=" * 70)
    click.echo("  WRF Local Processing Pipeline")
    click.echo("=" * 70)

    # Phase 1: Figures
    if not no_figures:
        click.echo("\n── Phase 1: Figure Generation ──")
        from scripts.micromet.process_wrf_figures import (
            _build_tasks_for_domain,
            _resolve_wrfout_paths,
        )

        from labmim_micrometeorology.wrf import reader
        from labmim_micrometeorology.wrf.batch import FigureTask, run_figure_tasks

        default_vars = [
            "temperature", "pressure", "wind", "rain",
            "vapor", "HFX", "LH", "SWDOWN",
        ]
        var_list = list(variables) if variables else default_vars
        paths = _resolve_wrfout_paths(wrf_dir, date, domains, dataset)

        if not paths:
            click.echo("No WRF files found.")
            return

        all_fig_tasks: list[FigureTask] = []
        for wrf_path in paths:
            click.echo(f"  Loading {wrf_path.name}...")
            with reader.WRFDataset(wrf_path) as ds:
                tasks = _build_tasks_for_domain(
                    ds, var_list, str(figures_dir), shapes_dir, skip_first, dpi,
                )
                all_fig_tasks.extend(tasks)
                click.echo(f"    → {len(tasks)} frames queued")

        click.echo(f"  Total: {len(all_fig_tasks)} frames")
        png_paths = run_figure_tasks(all_fig_tasks, workers=workers)
        click.echo(f"  ✓ {len(png_paths)} figures generated")
    else:
        png_paths = []

    # Phase 2: GeoJSON / JSON
    if not no_geojson:
        click.echo("\n── Phase 2: GeoJSON & JSON Generation ──")
        from scripts.micromet.process_wrf_geojson import (
            _build_json_tasks_for_domain,
        )
        from scripts.micromet.process_wrf_geojson import (
            _resolve_wrfout_paths as _resolve_geo,
        )

        from labmim_micrometeorology.wrf import reader as reader2
        from labmim_micrometeorology.wrf.batch import JsonTask, run_json_tasks

        default_vars = [
            "temperature", "pressure", "wind", "rain",
            "vapor", "HFX", "LH", "SWDOWN",
        ]
        var_list = list(variables) if variables else default_vars
        paths = _resolve_geo(wrf_dir, date, domains, dataset)

        all_json_tasks: list[JsonTask] = []
        for wrf_path in paths:
            click.echo(f"  Loading {wrf_path.name}...")
            with reader2.WRFDataset(wrf_path) as ds:
                tasks = _build_json_tasks_for_domain(
                    ds, var_list, str(json_dir), str(geojson_dir), skip_first,
                )
                all_json_tasks.extend(tasks)
                click.echo(f"    → {len(tasks)} JSON files queued")

        click.echo(f"  Total: {len(all_json_tasks)} JSON files")
        json_paths = run_json_tasks(all_json_tasks, workers=workers)
        click.echo(f"  ✓ {len(json_paths)} JSON files generated")

    # Phase 3: WebM Videos
    if also_video and png_paths:
        click.echo("\n── Phase 3: WebM Video Generation ──")
        from labmim_micrometeorology.wrf.animation import batch_create_webm

        grouped: dict[str, list[str]] = defaultdict(list)
        for p in sorted(png_paths):
            stem = Path(p).stem
            parts = stem.rsplit("_", 1)
            if len(parts) == 2:
                grouped[parts[0]].append(p)
            else:
                grouped[stem].append(p)

        webm_paths = batch_create_webm(grouped, str(video_dir), fps=2, workers=workers)
        click.echo(f"  ✓ {len(webm_paths)} videos generated")

    elapsed = time.perf_counter() - t0
    click.echo("\n" + "=" * 70)
    click.echo(f"  ✓ Complete in {elapsed:.1f}s")
    click.echo(f"  Output: {base_out.resolve()}")
    click.echo("=" * 70)


if __name__ == "__main__":
    main()
