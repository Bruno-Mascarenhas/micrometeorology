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

import re
from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.common.cli_options import parse_csv, parse_int_csv
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.types import VARIABLE_NETCDF_MAP, WRFVariable
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


_WRFOUT_DOMAIN_RE = re.compile(r"^wrfout_(d\d+)_", re.IGNORECASE)

_CANONICAL_VARIABLES: dict[str, str] = {
    **{variable.value.casefold(): variable.value for variable in WRFVariable},
    **{f"poteolico{height}": f"poteolico{height}" for height in jobs.POTEOLICO_ALL_HEIGHTS},
    "wind_vectors": "wind_vectors",
}

# Output file ids whose input spelling is a DIFFERENT word. Passing one of
# these as ``-v`` reaches the raw-NetCDF passthrough and publishes unconverted
# values into the files the derived variable owns.
_OUTPUT_ID_OWNERS: dict[str, str] = {
    netcdf_id.upper(): str(variable)
    for variable, netcdf_id in VARIABLE_NETCDF_MAP.items()
    if str(variable).casefold() != netcdf_id.casefold()
}


def _normalize_var_list(var_list: list[str]) -> list[str]:
    """Canonicalize spelling and deduplicate; a bare ``poteolico`` supersedes height requests.

    Case is folded to the canonical spelling first, because every downstream
    branch compares against ``WRFVariable`` values with ``==``: a mis-cased
    ``-v swdown`` used to miss its own handling (including the daylight gate)
    and fall through to the raw-NetCDF passthrough. Tokens that name no known
    variable are left untouched — raw NetCDF fields such as ``T2`` are a
    supported passthrough.
    """
    var_list = [_CANONICAL_VARIABLES.get(v.casefold(), v) for v in var_list]
    if "poteolico" in var_list:
        var_list = [v for v in var_list if not (v.startswith("poteolico") and v != "poteolico")]
    normalized: list[str] = []
    seen: set[str] = set()
    for v in var_list:
        if v not in seen:
            normalized.append(v)
            seen.add(v)
    return normalized


def _missing_domains(paths: list[Path], domains: tuple[int, ...]) -> tuple[int, ...]:
    """Explicitly requested domains that no selected file provides."""
    found = {
        match.group(1).lower()
        for match in (_WRFOUT_DOMAIN_RE.match(p.name) for p in paths)
        if match is not None
    }
    return tuple(d for d in sorted(set(domains)) if f"d{d:02d}" not in found)


def _matching_wrfout_paths(wrf_dir: Path | str, date: str, domains: tuple[int, ...]) -> list[Path]:
    """Glob the requested day, reporting a mistyped ``--date`` as a usage error."""
    try:
        return resolve_wrfout_paths(wrf_dir, date, domains or None)
    except ValueError as invalid_date:
        raise typer.BadParameter(str(invalid_date)) from invalid_date


def _resolve_paths(
    wrf_dir: Path | str | None,
    date: str | None,
    domains: tuple[int, ...],
    dataset: Path | str | None,
) -> tuple[list[Path], tuple[int, ...]]:
    """Return the wrfout files to process and the requested domains that are missing."""
    if dataset:
        return [Path(dataset)], ()
    if not wrf_dir:
        raise typer.BadParameter("Provide either --dataset or --wrf-dir (optionally with --date)")
    if not date:
        # No date: batch mode — every wrfout FILE in the directory, restricted
        # to --domains when the operator named them.
        candidates = [p for p in sorted(Path(wrf_dir).glob("wrfout*")) if p.is_file()]
        if domains:
            wanted = {f"d{d:02d}" for d in domains}
            paths = [
                p
                for p in candidates
                if (match := _WRFOUT_DOMAIN_RE.match(p.name)) and match.group(1).lower() in wanted
            ]
            if len(paths) != len(candidates):
                typer.echo(
                    f"  --domains {sorted(set(domains))} selected "
                    f"{len(paths)} of {len(candidates)} wrfout files"
                )
        else:
            paths = candidates
        if not paths:
            typer.echo(f"  ⚠ No wrfout files found in {wrf_dir}")
    else:
        paths = _matching_wrfout_paths(wrf_dir, date, domains)
        if not paths:
            typer.echo(f"  ⚠ No wrfout files found for date {date} in {wrf_dir}")
    return paths, _missing_domains(paths, domains)


def _reject_output_id_variables(var_list: list[str]) -> None:
    """Reject tokens that name an OUTPUT file id instead of an input variable.

    ``-v TSK`` falls through to the raw-NetCDF passthrough and writes Kelvin
    into the very ``D0X_TSK_*.json`` files ``skin_temperature`` publishes in
    °C. These ids cannot produce correct site bytes, so they fail loudly
    instead of silently mislabelling the map.
    """
    for variable in var_list:
        owner = _OUTPUT_ID_OWNERS.get(variable.upper())
        if owner:
            raise typer.BadParameter(
                f"{variable} is the output file id of {owner}; pass -v {owner}"
            )


@app.command()
def run(
    dataset: Annotated[
        Path | None, typer.Option("-d", "--dataset", help="Single WRF file.")
    ] = None,
    wrf_dir: Annotated[Path | None, typer.Option(help="Directory with wrfout files.")] = None,
    date: Annotated[
        str | None,
        typer.Option(
            help=(
                "Simulation date YYYYMMDD. Omit to batch every wrfout in --wrf-dir "
                "(restricted to --domains when given)."
            )
        ),
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
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit non-zero when no wrfout is selected or a requested --domains is missing.",
        ),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Generate GeoJSON and value JSON files with parallel work units.

    A day whose selection is empty or partial logs a warning and still exits 0
    -- unless ``--strict`` is given, which turns both into a non-zero exit
    before anything is written.
    """
    setup_logging(log_level)

    var_list = list(parse_csv(variables)) if variables else DEFAULT_VARS
    var_list = _normalize_var_list(var_list)
    _reject_output_id_variables(var_list)
    paths, missing_domains = _resolve_paths(wrf_dir, date, parse_int_csv(domains), dataset)
    if not paths:
        typer.echo("No WRF files found.")
        raise typer.Exit(code=1 if strict else 0)
    for domain in missing_domains:
        typer.echo(f"  ⚠ No wrfout file for requested domain d{domain:02d}")
    if strict and missing_domains:
        raise typer.Exit(code=1)

    resolved_workers = workers or default_workers()
    if resolved_workers < 1:
        raise typer.BadParameter("--workers must be >= 1")

    typer.echo(f"Files: {[p.name for p in paths]}")
    typer.echo(f"Variables: {var_list}")
    typer.echo(f"Workers: {resolved_workers}")

    try:
        units = jobs.build_units(
            paths, var_list, output_dir, geojson_dir, skip_first, site_artifacts=site_artifacts
        )
    except ValueError as invalid_selection:
        # A selection covering one domain twice is an operator mistake about
        # -d/-D/-o, so it reads as a usage error rather than a traceback.
        raise typer.BadParameter(str(invalid_selection)) from invalid_selection
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
