"""CLI: append one WRF run's point series to each station's operational record.

The server simulates one day at a time. This is the producer that turns each of
those runs into 24 more rows per station -- the files ``labmim-climatology``,
``labmim-monitoring`` and ``labmim-site-graphs`` read as their model layer. The
extraction had no producer in this repository until now: it lived in a script
(``dailyColect.generateOperationalSeries``) that was never committed, and the
format had to be recovered from the 26,087 rows it left behind.

One call, several points, one domain each
-----------------------------------------
A run covers the same region at four resolutions, so which wrfout answers for a
station depends on where the station is: the tower falls inside the 1 km nest,
a site in the south of Bahia only inside the 27 km parent. Pass the whole run
and every station, and each station is served by the FINEST domain that contains
it, writing to its own ``{name}_series_operacional.dat``.

What the append guarantees
--------------------------
The file's own header is the schema. Each row is rendered against the header the
file declares, field by field, with ``nan`` wherever this run produced no value.
So a wrfout that stops carrying a variable empties one column instead of shifting
every column after it, and a column added by a later version of this extraction
is appended to the header while every row already written keeps its meaning.

Examples
--------
The daily operational line -- first 24 hours of today's run, at the tower::

    labmim-wrf-series run --wrf-dir /data/wrf/20260815/wrf01 --date 20260815 \\
        -o data/series

Several stations in the same call, each from whichever domain reaches it::

    labmim-wrf-series run --wrf-dir /data/wrf/20260815/wrf01 --date 20260815 \\
        -o data/series -s labmim:-13.0055:-38.5089 -s ilheus:-14.7889:-39.0339

The same from a list::

    labmim-wrf-series run --wrf-dir ... --date ... -o data/series \\
        --stations estacoes.csv

See what a run would write, without touching the records::

    labmim-wrf-series run -d <wrfout> -o data/series --dry-run

Bring the historical file onto the v2 schema (once, before the first append)::

    labmim-wrf-series migrate -i data/series_operacional.dat
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.common.cli_options import parse_csv, parse_int_csv
from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf.operational_record import (
    DEFAULT_HEADER,
    DEFAULT_STATION,
    Station,
    append_block,
    build_columns,
    duplicated,
    extend_header,
    legacy_spellings,
    migrate_to_v2,
    parse_station,
    read_header,
    read_stations,
    render_rows,
)
from micrometeorology.wrf.operational_series import (
    DEFAULT_HOURS,
    DomainAssignment,
    assign_domains,
    extract_operational_block,
)
from micrometeorology.wrf.reader import WRFDataset, resolve_wrfout_paths

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

logger = logging.getLogger(__name__)


def _resolve_datasets(
    datasets: list[Path] | None,
    wrf_dir: Path | None,
    date: str | None,
    domains: tuple[int, ...],
) -> list[Path]:
    """The wrfout files this run offers, normally one per domain."""
    if datasets:
        return list(datasets)
    if wrf_dir is None or date is None:
        raise typer.BadParameter("give either -d/--dataset, or --wrf-dir together with --date")
    try:
        paths = resolve_wrfout_paths(wrf_dir, date, domains or None)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None
    if not paths:
        raise typer.BadParameter(f"no wrfout for {date} in {wrf_dir}")
    return paths


def _resolve_stations(tokens: tuple[str, ...], listing: Path | None) -> tuple[Station, ...]:
    """The points to publish: the options, the list, or the tower by default."""
    stations: list[Station] = []
    try:
        stations.extend(parse_station(token) for token in tokens)
        if listing is not None:
            stations.extend(read_stations(listing))
    except ValueError as error:
        raise typer.BadParameter(str(error)) from None

    if not stations:
        return (DEFAULT_STATION,)
    duplicates = duplicated(station.name for station in stations)
    if duplicates:
        raise typer.BadParameter(
            f"{', '.join(duplicates)} named more than once; two stations of one "
            "name would append to the same file"
        )
    return tuple(stations)


def _publish(
    assignment: DomainAssignment,
    ds: WRFDataset,
    output_dir: Path,
    *,
    hours: int,
    start_step: int,
    requested: list[str] | None,
    extend: bool,
    force: bool,
    dry_run: bool,
) -> None:
    """Extract one station's block and append it to that station's own file."""
    station = assignment.station
    columns = build_columns(requested, ds.has_variable)
    block = extract_operational_block(
        ds,
        latitude=station.latitude,
        longitude=station.longitude,
        hours=hours,
        start_step=start_step,
        columns=columns,
    )

    target = output_dir / station.filename
    header = read_header(target) if target.exists() else DEFAULT_HEADER
    legacy = legacy_spellings(header)
    if legacy:
        raise typer.BadParameter(
            f"{target.name} is still on the v1 schema ({', '.join(legacy)}); run "
            f"`labmim-wrf-series migrate -i {target}` before appending to it"
        )
    extra = tuple(name for name in block.frame.columns if name not in header)
    full_header = header + extra
    empty = [name for name in block.frame.columns if block.frame[name].isna().all()]

    typer.echo(
        f"  {station.name}: {assignment.path.name} (dx {assignment.dx_m:.0f} m), "
        f"celula ({block.row}, {block.col}) em {block.latitude:.4f}, {block.longitude:.4f} "
        f"a {block.distance_km:.2f} km"
    )
    typer.echo(
        f"    {block.frame.index[0]} .. {block.frame.index[-1]} (hora local, UTC-03), "
        f"{len(block.frame.columns)} colunas, {len(empty)} sem valor"
    )
    if empty:
        typer.echo(f"    sem valor: {', '.join(empty)}")

    if extra and not extend:
        raise typer.BadParameter(
            f"{target.name} has no column for {', '.join(extra)}; rerun with "
            "--extend-header to append them to its header"
        )
    if extra:
        typer.echo(f"    cabecalho: acrescentando {', '.join(extra)}")

    if dry_run:
        for row in render_rows(block.frame, full_header):
            typer.echo(row)
        typer.echo(f"    --dry-run: {len(block.frame)} linhas NAO gravadas em {target}")
        return

    if extra and target.exists():
        extend_header(target, extra)
    written = append_block(target, block.frame, full_header, force=force)
    if written:
        typer.echo(f"    >> {target} (+{written} linhas)")
    else:
        typer.echo(f"    >> {target} inalterado: bloco ja presente (use --force)")


@app.command()
def run(
    output_dir: Annotated[
        Path,
        typer.Option("-o", "--output", help="Directory holding one .dat per station."),
    ],
    datasets: Annotated[
        list[Path] | None,
        typer.Option("-d", "--dataset", help="wrfout file; repeat for several domains."),
    ] = None,
    wrf_dir: Annotated[
        Path | None, typer.Option("--wrf-dir", help="Directory holding the run's wrfout files.")
    ] = None,
    date: Annotated[
        str | None, typer.Option("--date", help="Simulation date YYYYMMDD, with --wrf-dir.")
    ] = None,
    domains: Annotated[
        list[str] | None,
        typer.Option(
            "-D", "--domains", help="Domains to offer (default: every domain of the run)."
        ),
    ] = None,
    stations: Annotated[
        list[str] | None,
        typer.Option("-s", "--station", help="`name:lat:lon`, repeatable."),
    ] = None,
    station_list: Annotated[
        Path | None,
        typer.Option("--stations", help="CSV of `name,lat,lon`, header optional."),
    ] = None,
    hours: Annotated[
        int, typer.Option("--hours", help="Hours to take from the run.")
    ] = DEFAULT_HOURS,
    start_step: Annotated[
        int, typer.Option("--start-step", help="First time step of the window (0 = run start).")
    ] = 0,
    variables: Annotated[
        list[str] | None,
        typer.Option(
            "-v",
            "--variables",
            help=(
                "Columns to extract, repeated or comma-separated. A name the "
                "catalogue does not define is taken as a raw wrfout variable and "
                "becomes a new column. Default: the whole catalogue."
            ),
        ),
    ] = None,
    extend: Annotated[
        bool,
        typer.Option(
            "--extend-header/--no-extend-header",
            help="Append columns the file's header does not carry yet.",
        ),
    ] = True,
    force: Annotated[
        bool, typer.Option("--force", help="Append even if every hour is already in the file.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the blocks and write nothing.")
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Extract each station's point series from one run and append it.

    Every station is served by the finest domain whose grid contains it, so one
    call covers a nest and its parents at once. A station no domain reaches is
    reported and the others are still written, but the command exits non-zero:
    a series that silently stops being published is the failure worth catching.

    The window defaults to the first 24 hours, which is the block the daily
    operational line has appended since 2022: step 0 is 00 UTC, written as hour
    21 of the previous local day, through 23 UTC as hour 20.
    """
    setup_logging(log_level)

    paths = _resolve_datasets(datasets, wrf_dir, date, parse_int_csv(domains))
    wanted = _resolve_stations(tuple(parse_csv(stations)), station_list)
    requested = list(parse_csv(variables)) or None

    typer.echo(f"Rodada:    {len(paths)} dominio(s) — {', '.join(p.name for p in paths)}")
    try:
        assignments, uncovered = assign_domains(wanted, paths)
    except OSError as error:
        # Reading the grids is the first thing that touches the files, so a run
        # interrupted mid-transfer surfaces here rather than at extraction.
        raise typer.BadParameter(f"could not be read as NetCDF: {error}") from error
    typer.echo(f"Estacoes:  {len(wanted)} pedida(s), {len(assignments)} atendida(s)")

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    by_path: dict[Path, list[DomainAssignment]] = {}
    for assignment in assignments:
        by_path.setdefault(assignment.path, []).append(assignment)

    for path, group in by_path.items():
        with WRFDataset(path) as ds:
            for assignment in group:
                _publish(
                    assignment,
                    ds,
                    output_dir,
                    hours=hours,
                    start_step=start_step,
                    requested=requested,
                    extend=extend,
                    force=force,
                    dry_run=dry_run,
                )

    for station in uncovered:
        typer.echo(
            f"  [--] {station.name}: ({station.latitude:.4f}, {station.longitude:.4f}) "
            "fica fora de todos os dominios desta rodada — nada gravado"
        )
    if uncovered:
        raise typer.Exit(code=1)


@app.command()
def migrate(
    input_path: Annotated[
        Path, typer.Option("-i", "--input", help="Operational .dat to migrate.", exists=True)
    ],
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Rewrite a v1 operational file onto the v2 schema, keeping a `.bak`.

    The v1 record names its columns without units, carries four columns computed
    from wrong formulas, and publishes WRF's cold-start step as if it were a
    measurement. This repairs all of it in place; the module docstring of
    `micrometeorology.wrf.operational_series` states each defect and the
    arithmetic that inverts it. Cells nothing touched are passed through
    verbatim, so a diff against the `.bak` shows only what changed.
    """
    setup_logging(log_level)

    before = read_header(input_path)
    report = migrate_to_v2(input_path)
    after = read_header(input_path)

    typer.echo(f"Cabecalho: {len(before)} -> {len(after)} colunas (v1 -> v2)")
    typer.echo(f"Linhas:    {report.rows} reescritas")
    typer.echo(f"           {report.recovered} valores recuperados do fim da linha")
    typer.echo("Corrigidas:")
    for name, count in sorted(report.repaired.items()):
        typer.echo(f"           {name:20s} {count:7d} celulas")
    typer.echo("Sem valor (passo anterior a primeira chamada de radiacao):")
    for name, count in sorted(report.blanked.items()):
        typer.echo(f"           {name:20s} {count:7d} celulas")
    typer.echo(f"\n>> {input_path} (copia em {input_path.name}.bak)")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-wrf-series``)."""
    app()


if __name__ == "__main__":
    main()
