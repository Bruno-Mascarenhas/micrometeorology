"""CLI: Process raw sensor .dat files into aggregated hourly data.

Examples
--------
Process raw sensor data with default settings:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv

Process with custom calibrations:
    labmim-sensor-process -i data/raw/ -o data/hourly/output.csv --calibrations configs/calibrations.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

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
    pattern: Annotated[str, typer.Option(help="File glob pattern.")] = "*.dat",
    freq: Annotated[str, typer.Option(help="Aggregation frequency.")] = "1h",
    min_samples: Annotated[int, typer.Option(help="Min samples per window.")] = 6,
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

    df = merge_dat_files(files)  # type: ignore

    config: dict = {}
    config_file = settings.configs_dir / "default.yaml"
    if config_file.exists():
        with open(config_file, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        limits = config.get("sensor_limits", [])
        if limits:
            df = apply_physical_limits(df, limits)

    cal_path = calibrations or str(settings.configs_dir / "calibrations.yaml")
    if Path(cal_path).exists():
        cals = load_calibrations(cal_path)
        df = apply_calibrations(df, cals)

    sum_cols = config.get("sensor_sum_columns", []) if config_file.exists() else []
    wd_cols = config.get("sensor_wind_dir_columns", []) if config_file.exists() else []

    df_hourly = aggregate_to_hourly(
        df, min_samples=min_samples, sum_columns=sum_cols, wind_dir_columns=wd_cols, freq=freq
    )

    export_csv(df_hourly, output_path, include_datetime_columns=datetime_columns)
    typer.echo(f"\n>> Exported {len(df_hourly)} rows to {output_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
