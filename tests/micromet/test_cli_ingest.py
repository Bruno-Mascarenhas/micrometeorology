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


def test_the_window_mode_reads_the_live_tables_instead_of_the_manifest(tmp_path) -> None:
    """The rolling window is a different question from the historical record.

    The manifest fails hard on a missing entry — every one is unique coverage or
    a documented repair — which is right for the ten-year record and wrong for
    the hourly job, whose input is whatever the datalogger is writing now.
    Measured on this archive, the window build costs 3.6 s against 18.9 s, and
    563 MB of peak against 4.7 GB, for a payload identical field for field.
    """
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    fatores = tmp_path / "teorica_2016-2030.csv"
    fatores.write_text(
        "ano_i,mes_i,dia_i,hor_i,min_i,fc\n"
        + "".join(f"2026,8,15,12,{minute},1.18\n" for minute in range(0, 60, 5)),
        encoding="utf-8",
    )
    lenta = tmp_path / "LBM_lenta_2025.dat"
    _write_toa5(
        lenta,
        ["CM3Up_Wm2_Avg"],
        [(f"2026-08-15 12:{minute:02d}:00", [500.0]) for minute in range(0, 60, 5)],
    )
    saida = tmp_path / "out"

    resultado = CliRunner().invoke(
        app, ["-d", str(tmp_path), "-o", str(saida), "--source", str(lenta)]
    )

    assert resultado.exit_code == 0, resultado.output
    assert "sem manifesto" in resultado.output
    assert (saida / "station_5min_qc.parquet").exists()
    assert (saida / "station_hourly.parquet").exists()


def test_the_window_mode_fails_strict_on_a_window_that_merged_to_nothing(tmp_path) -> None:
    """``--source`` built ``reports = []``, so ``--strict``'s only gate was always
    empty and the operational mode — the one that runs hourly — verified nothing
    at all. The audited row counts cannot judge a window, but its own shape can."""
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    fatores = tmp_path / "teorica_2016-2030.csv"
    fatores.write_text("ano_i,mes_i,dia_i,hor_i,min_i,fc\n", encoding="utf-8")
    lenta = tmp_path / "LBM_lenta_2025.dat"
    _write_toa5(lenta, ["CM3Up_Wm2_Avg"], [])
    saida = tmp_path / "out"

    resultado = CliRunner().invoke(
        app, ["-d", str(tmp_path), "-o", str(saida), "--source", str(lenta), "--strict"]
    )

    assert resultado.exit_code == 1, resultado.output
    assert "merged to no row at all" in resultado.output


@pytest.mark.parametrize("logger_net", ["NAN", 100.0])
def test_a_dead_balance_component_fails_strict_instead_of_blanking_the_chart(
    tmp_path, logger_net
) -> None:
    """``close_net_radiation`` drops every sample whose component is missing, and
    ``net_dropped`` was only ever printed. Chained onto export_monitoring, which
    OMITS an all-null series by design, one dead component made the balance chart
    disappear from the published page with zero error anywhere in the pipeline.

    Both spellings of the same window: the CR5000 computes ITS net from these
    four channels, so the realistic case has the logger's own net empty too and
    ``net_dropped`` counts nothing — keying the block on it would have missed
    exactly the case it exists for.
    """
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    fatores = tmp_path / "teorica_2016-2030.csv"
    fatores.write_text(
        "ano_i,mes_i,dia_i,hor_i,min_i,fc\n"
        + "".join(f"2026,8,15,12,{minute},1.18\n" for minute in range(0, 60, 5)),
        encoding="utf-8",
    )
    lenta = tmp_path / "LBM_lenta_2025.dat"
    _write_toa5(
        lenta,
        [
            "CM3Up_Wm2_Avg",
            "CM3Dn_Wm2_Avg",
            "CG3Up_Wm2Cr_Avg",
            "CG3Dn_Wm2Cr_Avg",
            "Net_Wm2_Avg",
        ],
        [
            (f"2026-08-15 12:{minute:02d}:00", [500.0, "NAN", 400.0, 450.0, logger_net])
            for minute in range(0, 60, 5)
        ],
    )
    saida = tmp_path / "out"

    resultado = CliRunner().invoke(
        app, ["-d", str(tmp_path), "-o", str(saida), "--source", str(lenta), "--strict"]
    )

    assert resultado.exit_code == 1, resultado.output
    assert "saldo recomposto ficou inteiramente ausente" in resultado.output
    assert "Sw_dw" in resultado.output


def test_an_archive_with_no_declared_physical_limit_fails_strict(tmp_path, monkeypatch) -> None:
    """The range gate is fail-open: no declared limit means no sample is ever
    refused, and the run publishes an ungated archive with exit 0 — silence
    indistinguishable from "every sample was inside its bounds"."""
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    _write_config_layer(tmp_path, monkeypatch, sensor_limits=[])
    fatores = tmp_path / "teorica_2016-2030.csv"
    fatores.write_text("ano_i,mes_i,dia_i,hor_i,min_i,fc\n", encoding="utf-8")
    lenta = tmp_path / "LBM_lenta_2025.dat"
    _write_toa5(
        lenta,
        ["CM3Up_Wm2_Avg"],
        [(f"2026-08-15 12:{minute:02d}:00", [500.0]) for minute in range(0, 60, 5)],
    )

    resultado = CliRunner().invoke(
        app, ["-d", str(tmp_path), "-o", str(tmp_path / "out"), "--source", str(lenta), "--strict"]
    )

    assert resultado.exit_code == 1, resultado.output
    assert "nenhum limite fisico declarado" in resultado.output


def test_an_archive_built_without_calibrations_fails_strict(tmp_path, monkeypatch) -> None:
    """No calibrations file means no instrument factor and no era-spanning column
    is unified: the archive publishes raw logger counts under the names the
    unified channels would have had, and every consumer reads them as
    calibrated."""
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    empty_configs = tmp_path / "configs-sem-calibracoes"
    empty_configs.mkdir()
    fatores = tmp_path / "teorica_2016-2030.csv"
    fatores.write_text("ano_i,mes_i,dia_i,hor_i,min_i,fc\n", encoding="utf-8")
    lenta = tmp_path / "LBM_lenta_2025.dat"
    _write_toa5(
        lenta,
        ["CM3Up_Wm2_Avg"],
        [(f"2026-08-15 12:{minute:02d}:00", [500.0]) for minute in range(0, 60, 5)],
    )
    from micrometeorology.cli import build_archive

    monkeypatch.setattr(build_archive, "get_settings", lambda: Settings(configs_dir=empty_configs))

    resultado = CliRunner().invoke(
        app,
        ["-d", str(tmp_path), "-o", str(tmp_path / "out"), "--source", str(lenta), "--strict"],
    )

    assert resultado.exit_code == 1, resultado.output
    assert "sem calibracoes" in resultado.output


def test_without_sources_the_manifest_still_refuses_a_missing_entry(tmp_path) -> None:
    """The historical build must keep failing loud: a dropped entry shortens the record."""
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    resultado = CliRunner().invoke(app, ["-d", str(tmp_path), "-o", str(tmp_path / "out")])

    assert resultado.exit_code != 0


def test_a_directory_with_no_matching_file_exits_non_zero_instead_of_reporting_success(
    tmp_path: Path,
):
    empty = tmp_path / "empty"
    empty.mkdir()
    output_path = tmp_path / "out.csv"

    result = runner.invoke(
        ingest_sensor_data.app,
        ["-i", str(empty), "-o", str(output_path), "--log-level", "WARNING"],
    )

    assert result.exit_code != 0
    assert not output_path.exists()


def test_the_csv_export_keeps_every_digit_the_parquet_keeps(tmp_path: Path) -> None:
    """``float_format="%.6g"`` wrote the logger's RECORD counter as 1.01786e+06 and
    rounded pressure to six significant digits, diverging from the Parquet of
    the same build."""
    from micrometeorology.cli.build_archive import _write

    frame = pd.DataFrame(
        {"RECORD": [1017857.0], "Pmb_WXT": [1013.2571]},
        index=pd.DatetimeIndex(["2026-08-15 12:00"], name="TIMESTAMP"),
    )

    path = _write(frame, tmp_path / "station_5min_raw", "csv")

    text = path.read_text(encoding="utf-8")
    assert "1017857.0" in text
    assert "1013.2571" in text


def test_a_source_file_matching_neither_table_is_refused_instead_of_dropped(
    tmp_path: Path,
) -> None:
    """A ``--source`` whose name carries neither ``lenta`` nor ``rain`` was silently
    left out while the summary line counted it as used."""
    from typer.testing import CliRunner

    from micrometeorology.cli.build_archive import app

    lenta = tmp_path / "LBM_lenta_2025.dat"
    slow = tmp_path / "LBM_slow_2025.dat"
    for path in (lenta, slow):
        _write_toa5(path, ["CM3Up_Wm2_Avg"], [("2026-08-15 12:00:00", [500.0])])

    resultado = CliRunner().invoke(
        app,
        [
            "-d",
            str(tmp_path),
            "-o",
            str(tmp_path / "out"),
            "--source",
            str(lenta),
            "--source",
            str(slow),
        ],
    )

    assert resultado.exit_code == 2, resultado.output
    assert "LBM_slow_2025.dat" in resultado.output
    assert not (tmp_path / "out" / "station_hourly.parquet").exists()


def test_files_are_merged_in_chronological_order_not_lexicographic(tmp_path: Path) -> None:
    """`merge_dat_files` resolves an overlapping stamp as "the first file in
    CHRONOLOGICAL order wins", and the glob sorts by name — for the station's own
    names a different order: `LBM_lenta.dat` sorts before `LBM_lenta_2019.dat`
    while covering 2019 itself, so the later file's value won the conflict."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    # Sorts SECOND by name, starts FIRST in time, and carries the shared stamp.
    _write_toa5(
        raw_dir / "z_starts_earlier.dat",
        ["Temp1"],
        [("2025-06-25 11:00:00", [20.0]), ("2025-06-25 12:00:00", [21.0])],
    )
    # Sorts FIRST by name, starts SECOND in time, same shared stamp.
    _write_toa5(raw_dir / "a_starts_later.dat", ["Temp1"], [("2025-06-25 12:00:00", [99.0])])
    output_path = tmp_path / "out.csv"

    result = runner.invoke(
        ingest_sensor_data.app,
        [
            "-i",
            str(raw_dir),
            "-o",
            str(output_path),
            "--min-samples",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    exported = pd.read_csv(output_path, index_col=0)
    noon = exported.loc[exported.index.astype(str).str.contains("12:00")]
    assert noon["Temp1"].iloc[0] == pytest.approx(21.0)
