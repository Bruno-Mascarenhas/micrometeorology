"""CLI: Generate GeoJSON/JSON files from WRF output.

Supports parallel JSON writing for high-throughput data generation.

Usage::

    # Single domain
    labmim-wrf-geojson -d wrfout_d03_2024-01-01 \\
        -o output/JSON -g output/GeoJSON -v temperature wind rain

    # Multiple domains
    labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \\
        --domains 1 4 -o output/JSON -g output/GeoJSON --workers 44

    # Auto mode chooses eager/pickle for tiny work and lazy/memmap for large work
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON

    # Force old eager/pickle behavior
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --reader eager --chunks none --worker-backend pickle

    # Force xarray-backed lazy reader
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --reader lazy --chunks none

    # Force memmap worker references for large JSON exports
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --worker-backend memmap --tmp-dir scratch/wrf-json
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import click
import numpy as np

from micrometeorology.common.logging import setup_logging
from micrometeorology.common.types import VARIABLE_NETCDF_MAP, WRFVariable
from micrometeorology.wrf import geojson, reader
from micrometeorology.wrf import variables as vmod
from micrometeorology.wrf.batch import (
    JsonTask,
    default_workers,
    run_json_tasks,
)
from micrometeorology.wrf.execution import (
    JsonWorkerRequest,
    ReaderRequest,
    estimate_json_payload_bytes,
    format_wrf_execution_plan,
    resolve_wrf_execution_plan,
)
from micrometeorology.wrf.geojson import create_wind_vectors_json
from micrometeorology.wrf.interpolation import (
    compute_wind_vectors_at_height,
    interpolate_speed_to_height,
)

DEFAULT_VARS = [
    "temperature",
    "pressure",
    "wind",
    "rain",
    "vapor",
    "HFX",
    "LH",
    "SWDOWN",
    "poteolico",
    "wind_vectors",
]


def _normalize_var_list(var_list: list[str]) -> list[str]:
    """Normalize legacy variable names to new names.

    The legacy system passes ``poteolico50``, ``poteolico100``, ``poteolico150``
    as separate variables.  The new pipeline handles all three heights from a
    single ``poteolico`` entry, so we collapse them and deduplicate.
    """
    normalized: list[str] = []
    seen: set[str] = set()
    for v in var_list:
        # poteolico50 / poteolico100 / poteolico150 → poteolico
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
    """Resolve WRF output file paths."""
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


def _format_datetime(dt) -> str:
    """Format a datetime for JSON output."""
    if dt is None:
        return "N/A"
    try:
        return dt.replace(minute=0, second=0, microsecond=0, tzinfo=None).strftime(  # type: ignore
            "%d/%m/%Y %H:%M:%S"
        )
    except Exception:
        return str(dt)


def _build_json_tasks_for_domain(
    ds: reader.WRFReader,
    var_list: list[str],
    json_dir: str,
    geojson_dir: str,
    skip_first: int,
) -> list[JsonTask]:
    """Build all JSON tasks for a single domain, saving GeoJSON grids along the way."""
    lon, lat = ds.read_grid()
    grid = ds.grid_level.value
    time_meta = ds.build_date_metadata(skip_first_n=skip_first)

    # Save grid GeoJSON ONCE per domain (geometry is identical for all variables)
    geojson.save_geojson(geojson_dir, grid, lon, lat, ds.dx, ds.dy)

    tasks: list[JsonTask] = []

    for var_name in var_list:
        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())

        if var_name == WRFVariable.TEMPERATURE:
            t2, _psfc, vmin, vmax = vmod.extract_temperature(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_temperature_step(t2[i : i + 1, :, :])
                tasks.append(
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
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
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

        elif var_name == WRFVariable.WIND:
            u10, v10, vmin, vmax = vmod.extract_wind(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                u = np.squeeze(u10[i : i + 1])
                v = np.squeeze(v10[i : i + 1])
                speed = np.hypot(u, v)
                tasks.append(
                    JsonTask(
                        data=speed,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

        elif var_name == WRFVariable.SWDOWN:
            var_data, vmin, vmax = vmod.extract_scalar(ds, "SWDOWN")
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                local_hour = meta["datetime_local"].hour
                if local_hour < 6 or local_hour > 18:
                    continue
                i = meta["index"]
                data = np.squeeze(var_data[i : i + 1, :, :])
                tasks.append(
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

        elif var_name == WRFVariable.WIND_POTENTIAL:
            # Wind potential: interpolate wind speed to 50m, 100m, 150m
            click.echo("  Computing adjusted heights for wind potential...")
            u_central, v_central, height_adjusted, speed_4d = vmod.compute_adjusted_heights(ds)

            for target_height, suffix in [
                (50, "POT_EOLICO_50M"),
                (100, "POT_EOLICO_100M"),
                (150, "POT_EOLICO_150M"),
            ]:
                click.echo(f"    -> Interpolating to {target_height}m ({suffix})...")
                speed_3d = interpolate_speed_to_height(speed_4d, height_adjusted, target_height)

                vmin = float(np.nanmin(speed_3d))
                vmax = float(np.nanmax(speed_3d))

                for meta in time_meta:
                    if meta.get("skip"):
                        continue
                    i = meta["index"]
                    data = np.squeeze(speed_3d[i : i + 1, :, :])

                    # Compute wind vectors per timestep (matches legacy behavior)
                    try:
                        wind_vectors = compute_wind_vectors_at_height(
                            u_central[i : i + 1],
                            v_central[i : i + 1],
                            height_adjusted[i : i + 1],
                            target_height,
                            downsampling=4,
                        )
                    except Exception:
                        wind_vectors = None

                    tasks.append(
                        JsonTask(
                            data=data,
                            scale_min=vmin,
                            scale_max=vmax,
                            date_str=_format_datetime(meta["datetime_local"]),
                            output_path=str(Path(json_dir) / f"{grid}_{suffix}_{i:03d}.json"),
                            wind_data=wind_vectors,
                        )
                    )

        elif var_name == "wind_vectors":
            # Standalone wind vector overlay files (surface U10/V10)
            click.echo("  Computing standalone wind vectors (U10/V10)...")
            u10, v10, _vmin, _vmax = vmod.extract_wind(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                u = np.squeeze(u10[i : i + 1])
                v = np.squeeze(v10[i : i + 1])
                wv_json = create_wind_vectors_json(
                    u,
                    v,
                    date_time=meta["datetime_local"],
                    downsampling=4,
                )
                # Write directly via save_values_json (same format)
                name = f"{grid}_WIND_VECTORS_{i:03d}"
                out_path = Path(json_dir) / f"{name}.json"
                import json as _json

                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    _json.dump(wv_json, f, separators=(",", ":"), ensure_ascii=False)

        else:
            nc_var = var_name.upper()
            if var_name == WRFVariable.PRESSURE:
                var_data, vmin, vmax = vmod.extract_pressure(ds)
            elif var_name == WRFVariable.VAPOR:
                var_data, vmin, vmax = vmod.extract_vapor(ds)
            elif ds.has_variable(nc_var):
                var_data, vmin, vmax = vmod.extract_scalar(ds, nc_var)
            else:
                click.echo(f"  ⚠ Variable {nc_var} not found — skipping")
                continue

            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = np.squeeze(var_data[i : i + 1, :, :])
                tasks.append(
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

    return tasks


@click.command()
@click.option("--dataset", "-d", default=None, type=click.Path(), help="Single WRF file.")
@click.option("--wrf-dir", default=None, type=click.Path(), help="Directory with wrfout files.")
@click.option("--date", default=None, help="Simulation date YYYYMMDD.")
@click.option("--domains", "-D", type=int, multiple=True, default=None, help="Domain numbers.")
@click.option("--output-dir", "-o", required=True, help="Output dir for value JSON files.")
@click.option("--geojson-dir", "-g", required=True, help="Output dir for GeoJSON grid files.")
@click.option("--variables", "-v", multiple=True, default=None, help="Variables to process.")
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
    "worker_backend",
    default="auto",
    type=click.Choice(["auto", "pickle", "memmap"]),
    show_default=True,
    help="JSON worker payload backend. Auto uses pickle for small/serial work and memmap for large multi-worker exports.",
)
@click.option(
    "--tmp-dir",
    "tmp_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Parent directory for temporary memmap payloads when --worker-backend memmap.",
)
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    dataset: str | None,
    wrf_dir: str | None,
    date: str | None,
    domains: tuple[int, ...],
    output_dir: str,
    geojson_dir: str,
    variables: tuple[str, ...],
    skip_first: int,
    reader_backend: str,
    chunks: str,
    workers: int | None,
    worker_backend: str,
    tmp_dir: str | None,
    log_level: str,
) -> None:
    """Generate GeoJSON and value JSON files with parallel writing."""
    setup_logging(log_level)

    var_list = list(variables) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, domains, dataset)
    try:
        initial_plan = resolve_wrf_execution_plan(
            paths=paths,
            workflow="json",
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
    click.echo(format_wrf_execution_plan(initial_plan))

    # Build all tasks
    all_tasks: list[JsonTask] = []
    for wrf_path in paths:
        click.echo(f"\nLoading {wrf_path.name}...")
        with reader.open_wrf_dataset(
            wrf_path,
            reader=initial_plan.reader,
            chunks=initial_plan.chunks,
        ) as ds:
            tasks = _build_json_tasks_for_domain(ds, var_list, output_dir, geojson_dir, skip_first)
            all_tasks.extend(tasks)
            click.echo(f"  → {len(tasks)} JSON files queued")

    click.echo(f"\nTotal JSON files: {len(all_tasks)}")
    try:
        final_plan = resolve_wrf_execution_plan(
            paths=paths,
            workflow="json",
            reader_request=cast("ReaderRequest", initial_plan.reader),
            chunks_request=chunks,
            json_worker_request=cast("JsonWorkerRequest", worker_backend),
            workers=initial_plan.workers,
            tmp_dir=tmp_dir,
            estimated_json_payload_bytes=estimate_json_payload_bytes(all_tasks),
            json_task_count=len(all_tasks),
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    if final_plan != initial_plan:
        click.echo(format_wrf_execution_plan(final_plan))

    # Parallel JSON writing
    json_paths = run_json_tasks(
        all_tasks,
        workers=final_plan.workers,
        backend=final_plan.json_worker_backend,
        tmp_dir=final_plan.tmp_dir,
    )
    click.echo(f"\n✓ Generated {len(json_paths)} JSON files")
    click.echo("✓ Done")


if __name__ == "__main__":
    main()
