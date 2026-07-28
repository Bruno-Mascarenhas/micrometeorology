"""End-to-end tests for the labmim-sensor-process CLI."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from micrometeorology.cli import ingest_sensor_data
from micrometeorology.common.config import Settings, get_settings
from tests.micromet.test_ingestion import _write_toa5

runner = CliRunner()


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ambient ``LABMIM_*`` overrides and the process-global settings cache."""
    monkeypatch.delenv("LABMIM_ENV", raising=False)
    monkeypatch.delenv("LABMIM_CONFIG_PATH", raising=False)
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"LABMIM_{field_name.upper()}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _write_one_hour_of_samples(directory: Path) -> None:
    """Twelve 5-minute records inside a single hour, above the 6-sample floor."""
    rows: list[tuple[str, list[float | str]]] = [
        (f"2025-06-25 12:{minute:02d}:00", [25.0, 0.5]) for minute in range(0, 60, 5)
    ]
    _write_toa5(directory / "station.dat", ["Temp1", "precip"], rows)


def test_cli_processes_raw_files_into_an_hourly_csv(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_one_hour_of_samples(raw_dir)
    output_path = tmp_path / "hourly" / "sensor_data.csv"

    result = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(raw_dir), "-o", str(output_path), "--log-level", "WARNING"],
    )

    assert result.exit_code == 0, result.output
    exported = pd.read_csv(output_path, index_col=0)
    assert len(exported) == 1
    assert exported["Temp1"].iloc[0] == pytest.approx(25.0)


def test_cli_reports_a_configs_dir_without_sensor_config_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configs_dir holding neither file must not silently drop QC and calibration."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_one_hour_of_samples(raw_dir)
    settings = Settings(configs_dir=tmp_path / "configs-without-sensor-files")
    monkeypatch.setattr(ingest_sensor_data, "get_settings", lambda: settings)

    result = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(raw_dir), "-o", str(tmp_path / "out.csv"), "--log-level", "WARNING"],
    )

    assert result.exit_code == 0, result.output
    assert "No sensor config at" in result.output
    assert "No calibrations at" in result.output
