"""CLI: Process raw sensor .dat files into aggregated hourly data.

Usage::

    python scripts/process_sensor_data.py --input data/raw/ --output data/hourly/ \\
        --config configs/default.yaml --calibrations configs/calibrations.yaml
"""

from __future__ import annotations

from pathlib import Path

import click
import yaml

from labmim_micrometeorology.common.config import get_settings
from labmim_micrometeorology.common.logging import setup_logging
from labmim_micrometeorology.common.paths import find_files
from labmim_micrometeorology.sensors.aggregation import aggregate_to_hourly
from labmim_micrometeorology.sensors.calibration import apply_calibrations, load_calibrations
from labmim_micrometeorology.sensors.export import export_csv
from labmim_micrometeorology.sensors.ingestion import apply_physical_limits, merge_dat_files


@click.command()
@click.option("--input", "-i", "input_dir", required=True, help="Directory with raw .dat files.")
@click.option("--output", "-o", "output_path", required=True, help="Output CSV file path.")
@click.option("--calibrations", default=None, help="Path to calibrations.yaml.")
@click.option("--pattern", default="*.dat", help="File glob pattern.")
@click.option("--freq", default="1h", help="Aggregation frequency.")
@click.option("--min-samples", default=6, type=int, help="Min samples per window.")
@click.option("--datetime-columns", is_flag=True, help="Include year/month/day/hour columns.")
@click.option("--log-level", default="INFO", help="Logging level.")
def main(
    input_dir: str,
    output_path: str,
    calibrations: str | None,
    pattern: str,
    freq: str,
    min_samples: int,
    datetime_columns: bool,
    log_level: str,
) -> None:
    """Process raw sensor files: read → merge → QC → calibrate → aggregate → export."""
    settings = get_settings()
    setup_logging(log_level)

    # Find files
    files = find_files(input_dir, pattern)
    if not files:
        click.echo(f"No files matching '{pattern}' found in {input_dir}")
        return

    click.echo(f"Found {len(files)} files")

    # Merge
    df = merge_dat_files(files)

    # Quality control (physical limits from config)
    config: dict = {}
    config_file = settings.configs_dir / "default.yaml"
    if config_file.exists():
        with open(config_file, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        limits = config.get("sensor_limits", [])
        if limits:
            df = apply_physical_limits(df, limits)

    # Calibrations
    cal_path = calibrations or str(settings.configs_dir / "calibrations.yaml")
    if Path(cal_path).exists():
        cals = load_calibrations(cal_path)
        df = apply_calibrations(df, cals)

    # Aggregate
    sum_cols = config.get("sensor_sum_columns", []) if config_file.exists() else []
    wd_cols = config.get("sensor_wind_dir_columns", []) if config_file.exists() else []

    df_hourly = aggregate_to_hourly(
        df,
        min_samples=min_samples,
        sum_columns=sum_cols,
        wind_dir_columns=wd_cols,
        freq=freq,
    )

    # Export
    export_csv(df_hourly, output_path, include_datetime_columns=datetime_columns)
    click.echo(f"\n✓ Exported {len(df_hourly)} rows to {output_path}")


if __name__ == "__main__":
    main()
