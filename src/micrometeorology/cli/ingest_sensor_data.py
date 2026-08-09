"""CLI: Process raw sensor .dat files into aggregated hourly data.

The QC limits, the sum-aggregated columns, the vector-averaged wind-direction
columns and the default sample floor all come from ``get_settings()``, so the
full ``default.yaml`` -> ``LABMIM_ENV`` -> ``LABMIM_CONFIG_PATH`` -> ``LABMIM_*``
layering applies. Earlier revisions re-parsed ``configs_dir/default.yaml`` here
and therefore ignored every layer above the shipped defaults.

Examples
--------
Process raw sensor data with default settings:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv

Process with custom calibrations:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv --calibrations configs/calibrations.yaml
"""

from pathlib import Path
from typing import Annotated

import typer

from micrometeorology.common.config import get_settings
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import find_files
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.calibration import apply_calibrations, load_calibrations
from micrometeorology.sensors.export import export_csv
from micrometeorology.sensors.ingestion import apply_physical_limits, merge_dat_files

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    input_dir: Annotated[
        Path,
        typer.Option(
            "-i", "--input", help="Directory with raw .dat files.", exists=True, dir_okay=True
        ),
    ],
    output_path: Annotated[Path, typer.Option("-o", "--output", help="Output CSV file path.")],
    calibrations: Annotated[
        Path | None,
        typer.Option(
            help="Path to calibrations.yaml.", exists=True, file_okay=True, dir_okay=False
        ),
    ] = None,
    pattern: Annotated[
        str,
        typer.Option(
            help=(
                "File glob. The default now also matches the logger's `.backup` "
                "rotation files: in the LabMiM archive three of them are the ONLY "
                "source of a whole austral winter, and a bare `*.dat` dropped them "
                "silently. Never point this at a directory holding more than one "
                "station or sampling rate."
            )
        ),
    ] = "*.dat*",
    freq: Annotated[str, typer.Option(help="Aggregation frequency.")] = "1h",
    min_samples: Annotated[
        int | None,
        typer.Option(help="Min samples per window. Defaults to sensor_min_samples_per_hour."),
    ] = None,
    datetime_columns: Annotated[
        bool,
        typer.Option(
            "--datetime-columns/--no-datetime-columns", help="Include year/month/day/hour columns."
        ),
    ] = False,
    log_level: Annotated[str, typer.Option(help="Logging level.")] = "INFO",
) -> None:
    """Process raw sensor files: read -> merge -> QC -> calibrate -> aggregate -> export."""
    settings = get_settings()
    setup_logging(log_level)

    files = find_files(input_dir, pattern)
    if not files:
        typer.echo(f"No files matching '{pattern}' found in {input_dir}")
        return

    typer.echo(f"Found {len(files)} files")

    df = merge_dat_files(files)

    if settings.sensor_limits:
        df = apply_physical_limits(df, settings.sensor_limits)
    else:
        typer.echo("  ⚠ No sensor_limits configured: skipping QC limits")

    cal_path = calibrations or settings.configs_dir / "calibrations.yaml"
    if cal_path.is_file():
        df = apply_calibrations(df, load_calibrations(cal_path))
    else:
        typer.echo(f"  ⚠ No calibrations at {cal_path}: exporting uncalibrated values")

    df_hourly = aggregate_to_hourly(
        df,
        min_samples=(
            min_samples if min_samples is not None else settings.sensor_min_samples_per_hour
        ),
        sum_columns=settings.sensor_sum_columns,
        wind_dir_columns=settings.sensor_wind_dir_columns,
        # aggregate_to_hourly has accepted this pairing since it was written, but
        # nothing ever passed it: every direction was vector-averaged with unit
        # weight, which puts about one hourly bearing in six more than 5 deg off.
        wind_speed_column_map=settings.sensor_wind_speed_column_map,
        freq=freq,
    )

    export_csv(df_hourly, output_path, include_datetime_columns=datetime_columns)
    typer.echo(f"\n>> Exported {len(df_hourly)} rows to {output_path}")


def main() -> None:
    """Console-script entry point (pyproject: ``labmim-sensor-process``)."""
    app()


if __name__ == "__main__":
    main()
