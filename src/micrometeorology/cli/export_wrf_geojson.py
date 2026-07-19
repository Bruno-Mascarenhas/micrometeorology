"""CLI: Generate GeoJSON/JSON files from WRF output.

Runs coarse (file, variable) work units on one persistent process pool:
each worker opens the NetCDF itself, derives its variable eagerly, computes
scale bounds, and writes every timestep JSON in-process (atomic renames, no
array IPC). See ``micrometeorology.wrf.jobs``.

Usage::

    # Single domain
    labmim-wrf-geojson -d wrfout_d03_2024-01-01 \\
        -o output/JSON -g output/GeoJSON -v temperature wind rain

    # Multiple domains
    labmim-wrf-geojson --wrf-dir /path/to/wrfout/ --date 20240101 \\
        --domains 1,4 -o output/JSON -g output/GeoJSON --workers 44
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf import jobs
from micrometeorology.wrf.batch import default_workers
from micrometeorology.wrf.reader import resolve_wrfout_paths

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
    """Deduplicate variables; a bare ``poteolico`` supersedes height-specific requests."""
    if "poteolico" in var_list:
        var_list = [v for v in var_list if not (v.startswith("poteolico") and v != "poteolico")]
    normalized: list[str] = []
    seen: set[str] = set()
    for v in var_list:
        if v not in seen:
            normalized.append(v)
            seen.add(v)
    return normalized


def _resolve_paths(
    wrf_dir: Path | str | None,
    date: str | None,
    domains: tuple[int, ...],
    dataset: Path | str | None,
) -> list[Path]:
    if dataset:
        return [Path(dataset)]
    if not wrf_dir:
        raise typer.BadParameter("Provide either --dataset or --wrf-dir (optionally with --date)")
    if not date:
        # No date: batch mode — every wrfout file in the directory.
        paths = sorted(Path(wrf_dir).glob("wrfout*"))
        if not paths:
            typer.echo(f"  ⚠ No wrfout files found in {wrf_dir}")
        return paths
    paths = resolve_wrfout_paths(wrf_dir, date, domains or None)
    if not paths:
        typer.echo(f"  ⚠ No wrfout files found for date {date} in {wrf_dir}")
    return paths


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


@app.command()
def run(
    dataset: Annotated[
        Path | None, typer.Option("-d", "--dataset", help="Single WRF file.")
    ] = None,
    wrf_dir: Annotated[Path | None, typer.Option(help="Directory with wrfout files.")] = None,
    date: Annotated[
        str | None,
        typer.Option(help="Simulation date YYYYMMDD. Omit to process every wrfout in --wrf-dir."),
    ] = None,
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
    site_artifacts: Annotated[
        bool,
        typer.Option(
            "--site-artifacts/--no-site-artifacts",
            help=(
                "Also write the consolidated site artifacts per domain/variable: "
                "{D}_{VAR}.series.bin (cell time-series via HTTP Range) and "
                "{D}_{VAR}.summary.json (per-step domain stats), plus the v2 "
                "manifest fields describing them."
            ),
        ),
    ] = True,
    workers: Annotated[
        int | None,
        typer.Option("-w", "--workers", help=f"Parallel workers (default: {default_workers()})."),
    ] = None,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate GeoJSON and value JSON files with parallel work units."""
    setup_logging(log_level)

    var_list = list(_parse_csv(variables)) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    paths = _resolve_paths(wrf_dir, date, _parse_int_csv(domains), dataset)
    if not paths:
        typer.echo("No WRF files found.")
        return

    resolved_workers = workers or default_workers()
    if resolved_workers < 1:
        raise typer.BadParameter("--workers must be >= 1")

    typer.echo(f"Files: {[p.name for p in paths]}")
    typer.echo(f"Variables: {var_list}")
    typer.echo(f"Workers: {resolved_workers}")

    units = jobs.build_units(
        paths, var_list, output_dir, geojson_dir, skip_first, site_artifacts=site_artifacts
    )
    results = jobs.execute_units(units, resolved_workers, echo=typer.echo)

    for result in results:
        for warning in result.warnings:
            typer.echo(f"  ⚠ {warning}")
    manifest_path = jobs.write_run_manifest(output_dir, results)
    if manifest_path:
        typer.echo(f"✓ Manifest: {manifest_path}")
    step_count = 0
    artifact_count = 0
    for result in results:
        if result.kind not in {"values_json", "poteolico"}:
            continue
        for file_path in result.files:
            if file_path.endswith((".series.bin", ".summary.json")):
                artifact_count += 1
            else:
                step_count += 1
    failed = [result for result in results if result.error]
    typer.echo(f"\n✓ Generated {step_count} JSON files")
    if artifact_count:
        typer.echo(f"✓ Generated {artifact_count} consolidated site artifacts (series/summary)")
    if failed:
        typer.echo(f"✗ {len(failed)} work units failed:")
        for result in failed:
            typer.echo(f"  - {result.label}: {result.error}")
        raise typer.Exit(code=1)
    typer.echo("✓ Done")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-wrf-geojson``)."""
    app()


if __name__ == "__main__":
    main()
