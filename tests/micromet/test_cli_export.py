"""End-to-end tests for the labmim-wrf-geojson CLI (work-unit pipeline)."""

from __future__ import annotations

from typer.testing import CliRunner

from micrometeorology.cli.export_wrf_geojson import app
from tests.micromet.test_wrf_jobs import NT, _write_full_wrf_file

runner = CliRunner()


def test_cli_exports_values_and_grid_geojson(tmp_path):
    wrf = tmp_path / "wrfout_d02_cli_synth.nc"
    _write_full_wrf_file(wrf, seed=21)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    result = runner.invoke(
        app,
        [
            "-d",
            str(wrf),
            "-o",
            str(json_dir),
            "-g",
            str(geo_dir),
            "-v",
            "temperature,wind",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"Generated {2 * NT} JSON files" in result.output
    assert sorted(p.name for p in json_dir.glob("D02_TEMP_*.json")) == [
        f"D02_TEMP_{i:03d}.json" for i in range(NT)
    ]
    assert len(list(json_dir.glob("D02_WIND_*.json"))) == NT
    assert (geo_dir / "D02.geojson").exists()


def test_cli_single_height_poteolico_writes_only_that_height(tmp_path):
    wrf = tmp_path / "wrfout_d02_cli_pot.nc"
    _write_full_wrf_file(wrf, seed=22)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    result = runner.invoke(
        app,
        [
            "-d",
            str(wrf),
            "-o",
            str(json_dir),
            "-g",
            str(geo_dir),
            "-v",
            "poteolico100",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list(json_dir.glob("D02_POT_EOLICO_100M_*.json"))) == NT
    assert list(json_dir.glob("D02_POT_EOLICO_50M_*.json")) == []
    assert list(json_dir.glob("D02_POT_EOLICO_150M_*.json")) == []


def test_cli_unknown_poteolico_height_fails_nonzero(tmp_path):
    wrf = tmp_path / "wrfout_d02_cli_bad.nc"
    _write_full_wrf_file(wrf, seed=23)

    result = runner.invoke(
        app,
        [
            "-d",
            str(wrf),
            "-o",
            str(tmp_path / "json"),
            "-g",
            str(tmp_path / "geo"),
            "-v",
            "poteolico75",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 1
    assert "work units failed" in result.output
    assert "poteolico75" in result.output
