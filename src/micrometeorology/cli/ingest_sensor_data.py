"""CLI: Process raw sensor .dat files into aggregated hourly data.

The QC limits, the sum-aggregated columns, the vector-averaged wind-direction
columns and the default sample floor all come from ``get_settings()``, so the
full ``default.yaml`` -> ``LABMIM_ENV`` -> ``LABMIM_CONFIG_PATH`` -> ``LABMIM_*``
layering applies.

Examples
--------
Process raw sensor data with default settings:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv

Process with custom calibrations:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv --calibrations configs/calibrations.yaml
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Tick

from micrometeorology.common.config import get_settings
from micrometeorology.common.logging import setup_logging
from micrometeorology.common.paths import find_files
from micrometeorology.sensors.aggregation import aggregate_to_hourly
from micrometeorology.sensors.calibration import apply_calibrations, load_calibrations
from micrometeorology.sensors.export import export_csv
from micrometeorology.sensors.ingestion import apply_physical_limits, merge_dat_files

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)

DATETIME_COLUMN_RESOLUTION = pd.Timedelta(hours=1)


def _opens_windows_off_the_hour(freq: str) -> bool:
    """Whether ``freq``'s windows can start away from the hour the date columns resolve.

    Only fixed-length (tick) offsets can misalign: every calendar offset pandas
    builds -- day, week, month and up -- opens on an hour boundary. A tick
    shorter than an hour misaligns, and so does any tick that is not a whole
    multiple of one: ``90min`` clears a "shorter than an hour" test and still
    opens windows at 01:30, which the four integer columns cannot carry.
    """
    offset = to_offset(freq)
    if not isinstance(offset, Tick):
        return False
    return pd.Timedelta(offset) % DATETIME_COLUMN_RESOLUTION != pd.Timedelta(0)


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
    if datetime_columns and _opens_windows_off_the_hour(freq):
        raise typer.BadParameter(
            f"--datetime-columns writes only year/month/day/hour, which cannot carry "
            f"the minute of --freq {freq}: windows that do not open on the hour would "
            f"collapse onto one key. Use a whole number of hours, or --no-datetime-columns."
        )

    settings = get_settings()
    setup_logging(log_level)

    files = find_files(input_dir, pattern)
    if not files:
        raise typer.BadParameter(
            f"no files matching '{pattern}' in {input_dir}; nothing to export"
        )

    typer.echo(f"Found {len(files)} files")

    # Passed explicitly so `sensor_sentinel_value: null` really switches the
    # guard off, rather than falling back to the reader's hard-coded default.
    df = merge_dat_files(files, sentinel_value=settings.sensor_sentinel_value)

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
        # Weights each direction by its own speed: under unit weight instead,
        # about one hourly bearing in six lands more than 5 deg off.
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
