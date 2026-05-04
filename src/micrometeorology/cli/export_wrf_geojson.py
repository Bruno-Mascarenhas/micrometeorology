"""CLI: Generate GeoJSON/JSON files from WRF output.

Supports parallel JSON writing for high-throughput data generation.

Usage::

    # Single domain
    labmim-wrf-geojson -d wrfout_d03_2024-01-01 \\
        -o output/JSON -g output/GeoJSON -v temperature wind rain

    # Multiple domains
    labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \\
        --domains 1,4 -o output/JSON -g output/GeoJSON --workers 44

    # Auto mode chooses serial for single-worker work and memmap for multi-worker work
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON

    # Force xarray-backed lazy reader
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --reader lazy --chunks none

    # Force serial writes or memmap worker references
    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --worker-backend serial

    labmim-wrf-geojson --dataset /path/to/wrfout_d03_2024-01-01_00:00:00 \\
        -o output/JSON -g output/GeoJSON --worker-backend memmap --tmp-dir scratch/wrf-json
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import numpy as np
import typer

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
from micrometeorology.wrf.reader import resolve_wrfout_paths

if TYPE_CHECKING:
    from collections.abc import Callable

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
    "poteolico",
    "wind_power_density_10m",
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
    json_dir: Path | str,
    geojson_dir: Path | str,
    skip_first: int,
    task_sink: Callable[[list[JsonTask], str], None] | None = None,
    task_batch_size: int = 16,
) -> list[JsonTask]:
    """Build all JSON tasks for a single domain, saving GeoJSON grids along the way."""
    lon, lat = ds.read_grid()
    grid = ds.grid_level.value
    time_meta = ds.build_date_metadata(skip_first_n=skip_first)

    # Save grid GeoJSON ONCE per domain (geometry is identical for all variables)
    geojson.save_geojson(geojson_dir, grid, lon, lat, ds.dx, ds.dy)

    tasks: list[JsonTask] = []

    for var_name in var_list:

        def add_task(task: JsonTask, label: str = var_name) -> None:
            tasks.append(task)
            if task_sink is not None and len(tasks) >= task_batch_size:
                task_sink(tasks, label)
                tasks.clear()

        nc_suffix = VARIABLE_NETCDF_MAP.get(var_name, var_name.upper())

        if var_name == WRFVariable.TEMPERATURE:
            t2, _psfc, vmin, vmax = vmod.extract_temperature(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.extract_temperature_step(t2[i : i + 1, :, :])
                add_task(
                    JsonTask(
                        data=vmod.materialize_2d(data),
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
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
                    JsonTask(
                        data=vmod.materialize_2d(data),
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
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
                add_task(
                    JsonTask(
                        data=vmod.materialize_2d(data),
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
                u = vmod.materialize_2d(u10[i : i + 1])
                v = vmod.materialize_2d(v10[i : i + 1])
                speed = np.hypot(u, v)
                add_task(
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
                data = vmod.materialize_2d(var_data[i : i + 1, :, :])
                add_task(
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
            typer.echo("  Computing adjusted heights for wind potential...")
            u_central, v_central, height_adjusted, speed_4d = vmod.compute_adjusted_heights(ds)

            for target_height, suffix in [
                (50, "POT_EOLICO_50M"),
                (100, "POT_EOLICO_100M"),
                (150, "POT_EOLICO_150M"),
            ]:
                typer.echo(f"    -> Interpolating to {target_height}m ({suffix})...")
                speed_3d = interpolate_speed_to_height(speed_4d, height_adjusted, target_height)

                vmin = float(np.nanmin(speed_3d))
                vmax = float(np.nanmax(speed_3d))

                for meta in time_meta:
                    if meta.get("skip"):
                        continue
                    i = meta["index"]
                    data = vmod.materialize_2d(speed_3d[i : i + 1, :, :])

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

                    add_task(
                        JsonTask(
                            data=data,
                            scale_min=vmin,
                            scale_max=vmax,
                            date_str=_format_datetime(meta["datetime_local"]),
                            output_path=str(Path(json_dir) / f"{grid}_{suffix}_{i:03d}.json"),
                            wind_data=wind_vectors,
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
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

        elif var_name == "wind_vectors":
            # Standalone wind vector overlay files (surface U10/V10)
            typer.echo("  Computing standalone wind vectors (U10/V10)...")
            u10, v10, _vmin, _vmax = vmod.extract_wind(ds)
            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                u = vmod.materialize_2d(u10[i : i + 1])
                v = vmod.materialize_2d(v10[i : i + 1])
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
                typer.echo(f"  ⚠ Variable {nc_var} not found — skipping")
                continue

            for meta in time_meta:
                if meta.get("skip"):
                    continue
                i = meta["index"]
                data = vmod.materialize_2d(var_data[i : i + 1, :, :])
                add_task(
                    JsonTask(
                        data=data,
                        scale_min=vmin,
                        scale_max=vmax,
                        date_str=_format_datetime(meta["datetime_local"]),
                        output_path=str(Path(json_dir) / f"{grid}_{nc_suffix}_{i:03d}.json"),
                        wind_data=None,
                    )
                )

        if task_sink is not None and tasks:
            task_sink(tasks, var_name)
            tasks.clear()

    return tasks


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
    output_dir: Annotated[
        Path, typer.Option("-o", "--output-dir", help="Output dir for value JSON files.")
    ] = ...,  # type: ignore[assignment]
    geojson_dir: Annotated[
        Path, typer.Option("-g", "--geojson-dir", help="Output dir for GeoJSON grid files.")
    ] = ...,  # type: ignore[assignment]
    variables: Annotated[
        list[str] | None,
        typer.Option(
            "-v", "--variables", help="Variables to process. Can be repeated or comma-separated."
        ),
    ] = None,
    skip_first: Annotated[int, typer.Option(help="Time steps to skip.")] = 0,
    reader_backend: Annotated[
        ReaderChoice, typer.Option("--reader", help="WRF reader backend.")
    ] = ReaderChoice.auto,
    chunks: Annotated[
        str, typer.Option(help="Lazy-reader chunks: 'auto', 'none', or dim=size pairs.")
    ] = "auto",
    workers: Annotated[
        int | None,
        typer.Option("-w", "--workers", help=f"Parallel workers (default: {default_workers()})."),
    ] = None,
    worker_backend: Annotated[
        WorkerChoice, typer.Option("--worker-backend", help="JSON worker backend.")
    ] = WorkerChoice.auto,
    tmp_dir: Annotated[
        Path | None, typer.Option(help="Temp directory for memmap payloads.")
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate GeoJSON and value JSON files with parallel writing."""
    setup_logging(log_level)

    var_list = list(_parse_csv(variables)) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_wrfout_paths(wrf_dir, date, _parse_int_csv(domains), dataset)
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
        raise typer.BadParameter(str(exc)) from exc

    if not paths:
        typer.echo("No WRF files found.")
        return

    typer.echo(f"Files: {[p.name for p in paths]}")
    typer.echo(f"Variables: {var_list}")
    typer.echo(format_wrf_execution_plan(initial_plan))

    # Build and write tasks per domain/variable to avoid keeping all frames in RAM.
    generated_count = 0
    printed_plans: set[str] = set()
    for wrf_path in paths:
        typer.echo(f"\nLoading {wrf_path.name}...")

        def write_task_batch(
            tasks: list[JsonTask],
            label: str,
            current_wrf_path: Path = wrf_path,
        ) -> None:
            nonlocal generated_count
            try:
                batch_plan = resolve_wrf_execution_plan(
                    paths=[current_wrf_path],
                    workflow="json",
                    reader_request=cast("ReaderRequest", initial_plan.reader),
                    chunks_request=chunks,
                    json_worker_request=cast("JsonWorkerRequest", worker_backend),
                    workers=initial_plan.workers,
                    tmp_dir=tmp_dir,
                    estimated_json_payload_bytes=estimate_json_payload_bytes(tasks),
                    json_task_count=len(tasks),
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            plan_text = format_wrf_execution_plan(batch_plan)
            if batch_plan != initial_plan and plan_text not in printed_plans:
                typer.echo(plan_text)
                printed_plans.add(plan_text)
            json_paths = run_json_tasks(
                tasks,
                workers=batch_plan.workers,
                backend=batch_plan.json_worker_backend,
                tmp_dir=batch_plan.tmp_dir,
            )
            generated_count += len(json_paths)
            typer.echo(f"  -> {len(json_paths)} JSON files generated for {label}")

        with reader.open_wrf_dataset(
            wrf_path,
            reader=initial_plan.reader,
            chunks=initial_plan.chunks,
        ) as ds:
            _build_json_tasks_for_domain(
                ds,
                var_list,
                output_dir,
                geojson_dir,
                skip_first,
                task_sink=write_task_batch,
            )

    typer.echo(f"\n✓ Generated {generated_count} JSON files")
    typer.echo("✓ Done")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
