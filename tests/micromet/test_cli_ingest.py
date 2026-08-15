"""End-to-end tests for the labmim-sensor-process CLI."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from micrometeorology.cli import ingest_sensor_data
from micrometeorology.common.config import Settings, get_settings
from micrometeorology.sensors.export import export_csv
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
    """Empty QC limits and a missing calibrations file must not pass silently."""
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
    assert "No sensor_limits configured" in result.output
    assert "No calibrations at" in result.output


def _write_config_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object
) -> None:
    """Point ``LABMIM_CONFIG_PATH`` at a YAML layer that overrides default.yaml."""
    override_path = tmp_path / "override.yaml"
    override_path.write_text(yaml.safe_dump(overrides), encoding="utf-8")
    monkeypatch.setenv("LABMIM_CONFIG_PATH", str(override_path))
    get_settings.cache_clear()


def _write_one_hour_of_wind_and_totals(directory: Path) -> None:
    """Twelve 5-minute records whose scalar and vector means disagree by 180 degrees."""
    rows: list[tuple[str, list[float | str]]] = [
        (
            f"2025-06-25 12:{minute:02d}:00",
            [25.0, 0.5, 350.0 if minute % 10 == 0 else 10.0],
        )
        for minute in range(0, 60, 5)
    ]
    _write_toa5(directory / "station.dat", ["Temp1", "custom_Tot", "WD_custom"], rows)


def test_cli_aggregation_honours_the_config_layer_not_just_default_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LABMIM_CONFIG_PATH`` must reach the QC, sum and wind-direction column lists.

    Re-parsing ``configs_dir/default.yaml`` inside the CLI bypassed the
    ``LABMIM_ENV`` / ``LABMIM_CONFIG_PATH`` layers, so an operator override was
    silently ignored: ``Temp1`` kept an out-of-range value, ``custom_Tot`` was
    averaged instead of summed and ``WD_custom`` was scalar-averaged.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_one_hour_of_wind_and_totals(raw_dir)
    _write_config_layer(
        tmp_path,
        monkeypatch,
        sensor_limits=[{"column": "Temp1", "lower": 0, "upper": 10}],
        sensor_sum_columns=["custom_Tot"],
        sensor_wind_dir_columns=["WD_custom"],
    )
    output_path = tmp_path / "out.csv"

    result = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(raw_dir), "-o", str(output_path), "--log-level", "WARNING"],
    )

    assert result.exit_code == 0, result.output
    exported = pd.read_csv(output_path, index_col=0)
    assert pd.isna(exported["Temp1"].iloc[0]), "overridden QC range must NaN 25 degC"
    assert exported["custom_Tot"].iloc[0] == pytest.approx(6.0), "12 x 0.5 summed"
    assert exported["WD_custom"].iloc[0] % 360 == pytest.approx(0.0, abs=1e-6)


class TestDatetimeColumnsGuard:
    """``--datetime-columns`` is the CLI's last stage, so its refusals must be legible.

    Both the misdiagnosed label and the end-of-pipeline traceback reach the
    operator only after read -> merge -> QC -> calibrate -> aggregate has run.
    """

    def test_a_missing_label_is_reported_as_missing_not_as_sub_hourly(self, tmp_path: Path) -> None:
        index = pd.DatetimeIndex(["2025-06-25 12:00", pd.NaT, "2025-06-25 13:00"])
        frame = pd.DataFrame({"Temp1": [25.0, 26.0, 27.0]}, index=index)

        with pytest.raises(ValueError, match="missing timestamp"):
            export_csv(frame, tmp_path / "with_nat.csv", include_datetime_columns=True)

    def test_a_sub_hourly_index_is_still_refused_for_losing_its_minute(
        self, tmp_path: Path
    ) -> None:
        index = pd.date_range("2025-06-25 12:00", periods=3, freq="30min")
        frame = pd.DataFrame({"Temp1": [25.0, 26.0, 27.0]}, index=index)

        with pytest.raises(ValueError, match="year/month/day/hour"):
            export_csv(frame, tmp_path / "sub_hourly.csv", include_datetime_columns=True)

    @pytest.mark.parametrize("freq", ["30min", "90min"])
    def test_a_freq_that_opens_windows_off_the_hour_is_rejected_before_any_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, freq: str
    ) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        _write_one_hour_of_samples(raw_dir)
        output_path = tmp_path / "out.csv"

        def _refuse_to_read(*_args: object, **_kwargs: object) -> pd.DataFrame:
            raise AssertionError("incompatible flags must be caught before ingestion")

        monkeypatch.setattr(ingest_sensor_data, "merge_dat_files", _refuse_to_read)

        result = runner.invoke(
            ingest_sensor_data.app,
            [
                "-i",
                str(raw_dir),
                "-o",
                str(output_path),
                "--freq",
                freq,
                "--datetime-columns",
                "--log-level",
                "WARNING",
            ],
        )

        assert result.exit_code == 2, result.output
        assert "--datetime-columns" in result.output
        assert "--freq" in result.output
        assert not output_path.exists()


def test_cli_min_samples_falls_back_to_the_configured_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``--min-samples`` the configured floor applies; the flag still wins."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_one_hour_of_samples(raw_dir)
    _write_config_layer(tmp_path, monkeypatch, sensor_min_samples_per_hour=20)

    from_config = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(raw_dir), "-o", str(tmp_path / "config.csv"), "--log-level", "WARNING"],
    )
    flag_argv = ["--min-samples", "6", "--log-level", "WARNING"]
    from_flag = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(raw_dir), "-o", str(tmp_path / "flag.csv"), *flag_argv],
    )

    assert from_config.exit_code == 0, from_config.output
    assert from_flag.exit_code == 0, from_flag.output
    assert pd.isna(pd.read_csv(tmp_path / "config.csv", index_col=0)["Temp1"].iloc[0])
    assert pd.read_csv(tmp_path / "flag.csv", index_col=0)["Temp1"].iloc[0] == pytest.approx(25.0)
