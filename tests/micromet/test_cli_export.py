"""End-to-end tests for the labmim-wrf-geojson CLI (work-unit pipeline)."""

import json

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

    # Compact grid companion for the site front-end.
    with open(geo_dir / "D02.grid.json", encoding="utf-8") as f:
        compact = json.load(f)
    assert compact["format"] in {"grid-edges-v1", "grid-bounds-v1"}
    assert len(compact["shape"]) == 2

    # Run manifest for front-end cache versioning.
    with open(json_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["version"]
    assert manifest["domains"] == ["D02"]
    assert manifest["files"] > 0


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


def test_cli_wrf_dir_without_date_processes_every_wrfout(tmp_path):
    for name in ("wrfout_d01_x", "wrfout_d02_y"):
        _write_full_wrf_file(tmp_path / name, seed=31)
    result = runner.invoke(
        app,
        [
            "--wrf-dir",
            str(tmp_path),
            "-o",
            str(tmp_path / "json"),
            "-g",
            str(tmp_path / "geo"),
            "-v",
            "temperature",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "json").glob("D01_TEMP_*.json"))) == NT
    assert len(list((tmp_path / "json").glob("D02_TEMP_*.json"))) == NT


def _invoke(tmp_path, *extra):
    return runner.invoke(
        app,
        [
            "-o",
            str(tmp_path / "json"),
            "-g",
            str(tmp_path / "geo"),
            "-v",
            "temperature",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
            *extra,
        ],
    )


def test_cli_batch_mode_restricts_to_requested_domains(tmp_path):
    """`--wrf-dir` without `--date` must still honour `--domains`; scanning the
    whole directory would publish domains nobody asked for."""
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    for name in ("wrfout_d01_a", "wrfout_d02_b", "wrfout_d04_c"):
        _write_full_wrf_file(wrf_dir / name, seed=41)

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--domains", "1")

    assert result.exit_code == 0, result.output
    assert "selected 1 of 3 wrfout files" in result.output
    assert len(list((tmp_path / "json").glob("D01_TEMP_*.json"))) == NT
    assert list((tmp_path / "json").glob("D02_TEMP_*.json")) == []
    assert list((tmp_path / "json").glob("D04_TEMP_*.json")) == []


def test_cli_batch_mode_ignores_a_wrfout_subdirectory(tmp_path):
    """A `wrfout*` directory must not be globbed as a file: opening it as NetCDF
    fails its work units and takes the whole run to exit 1."""
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d02_ok", seed=42)
    (wrf_dir / "wrfout_d03_scratch").mkdir()

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir))

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "json").glob("D02_TEMP_*.json"))) == NT


def test_cli_domain_filter_excludes_a_file_that_would_collide_on_d01(tmp_path):
    """`wrfout_d06_*` carries no grid level, so it must not enter a
    `--domains 1` selection and contend for the D01 output names."""
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d01_real", seed=45)
    _write_full_wrf_file(wrf_dir / "wrfout_d06_unknown", seed=46)

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--domains", "1")

    assert result.exit_code == 0, result.output
    assert "wrfout_d06_unknown" not in result.output
    assert len(list((tmp_path / "json").glob("D01_TEMP_*.json"))) == NT


def test_cli_rejects_an_output_file_id_as_a_variable(tmp_path):
    """`-v TSK` must be refused: the raw passthrough would publish Kelvin into
    the files `skin_temperature` publishes in °C."""
    wrf = tmp_path / "wrfout_d02_cli_tsk.nc"
    _write_full_wrf_file(wrf, seed=47)

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
            "TSK",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code != 0
    assert "skin_temperature" in result.output
    assert list((tmp_path / "json").glob("*.json")) == []


def test_cli_canonicalizes_variable_case(tmp_path):
    """`-v TEMPERATURE` must write the same files as `-v temperature`, not
    nothing alongside a success exit."""
    wrf = tmp_path / "wrfout_d02_cli_case.nc"
    _write_full_wrf_file(wrf, seed=48)

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
            "TEMPERATURE",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "json").glob("D02_TEMP_*.json"))) == NT


def test_cli_mis_cased_swdown_still_honours_the_daylight_gate(tmp_path):
    """`-v swdown` must hit the exact-match daylight gate; night frames are not
    in the site's availability metadata."""
    wrf = tmp_path / "wrfout_d02_cli_swdown.nc"
    _write_full_wrf_file(wrf, seed=51, nt=3, start_hour_utc=21)  # local 18, 19, 20 h

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
            "swdown",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in (tmp_path / "json").glob("D02_SWDOWN_*.json")) == [
        "D02_SWDOWN_000.json"
    ]


def test_cli_accepts_an_iso_date(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d02_2026-05-03_09:00:00", seed=49)

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--date", "2026-05-03")

    assert result.exit_code == 0, result.output
    assert len(list((tmp_path / "json").glob("D02_TEMP_*.json"))) == NT


def test_cli_rejects_a_malformed_date(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d02_2026-05-03_09:00:00", seed=49)

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--date", "may3")

    assert result.exit_code != 0
    assert "--date must be YYYYMMDD" in result.output


def test_cli_exits_zero_and_says_so_when_nothing_is_selected(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir))

    assert result.exit_code == 0, result.output
    assert "No WRF files found." in result.output


def test_cli_strict_exits_nonzero_when_nothing_is_selected(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--strict")

    assert result.exit_code == 1


def test_cli_publishes_the_domains_it_found_and_names_the_one_it_did_not(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d01_2026-05-03_09:00:00", seed=50)

    result = _invoke(tmp_path, "--wrf-dir", str(wrf_dir), "--date", "20260503", "--domains", "1,4")

    assert result.exit_code == 0, result.output
    assert "No wrfout file for requested domain d04" in result.output
    assert len(list((tmp_path / "json").glob("D01_TEMP_*.json"))) == NT


def test_cli_strict_aborts_on_a_missing_requested_domain_before_writing(tmp_path):
    wrf_dir = tmp_path / "wrf"
    wrf_dir.mkdir()
    _write_full_wrf_file(wrf_dir / "wrfout_d01_2026-05-03_09:00:00", seed=50)

    strict_dir = tmp_path / "strict"
    strict = runner.invoke(
        app,
        [
            "--wrf-dir",
            str(wrf_dir),
            "--date",
            "20260503",
            "--domains",
            "1,4",
            "-o",
            str(strict_dir / "json"),
            "-g",
            str(strict_dir / "geo"),
            "-v",
            "temperature",
            "--workers",
            "1",
            "--strict",
            "--log-level",
            "WARNING",
        ],
    )
    assert strict.exit_code == 1
    assert not strict_dir.exists()


def test_a_variables_option_naming_nothing_is_a_usage_error(tmp_path):
    """``-v ,`` used to select zero variables and still publish a manifest with a
    new version stamp, which the site reads as a complete new run."""
    result = runner.invoke(
        app, ["-o", str(tmp_path / "json"), "-g", str(tmp_path / "geo"), "-v", ","]
    )

    assert result.exit_code == 2
    assert "names no variable" in result.output
    assert list(tmp_path.rglob("manifest.json")) == []


def test_the_artifact_variables_are_the_default_request_without_the_two_overlays():
    """`features` vouches for the .series.bin/.summary.json byte offsets, which the
    wind arrows and the isobars never write."""
    from micrometeorology.cli.export_wrf_geojson import ARTIFACT_VARIABLES, DEFAULT_VARS

    assert frozenset(DEFAULT_VARS) - {"wind_vectors", "isobars"} == ARTIFACT_VARIABLES


def test_a_partial_variable_request_publishes_a_manifest_with_no_artifact_features(tmp_path):
    """Fixed names mean last run's matrices are still on disk, so a manifest that
    vouched for them would be read at the wrong byte offsets."""
    wrf = tmp_path / "wrfout_d02_cli_features.nc"
    _write_full_wrf_file(wrf, seed=23)
    json_dir = tmp_path / "json"

    result = runner.invoke(
        app,
        [
            "-d",
            str(wrf),
            "-o",
            str(json_dir),
            "-g",
            str(tmp_path / "geo"),
            "-v",
            "temperature",
            "--workers",
            "1",
            "--log-level",
            "WARNING",
        ],
    )

    assert result.exit_code == 0, result.output
    with open(json_dir / "manifest.json", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert "features" not in manifest


def test_zero_workers_is_a_usage_error_rather_than_the_default(tmp_path):
    """``--workers 0`` fell through ``workers or default_workers()`` to the default."""
    result = runner.invoke(
        app,
        ["-o", str(tmp_path / "json"), "-g", str(tmp_path / "geo"), "-v", "temperature", "-w", "0"],
    )

    assert result.exit_code == 2
    assert "--workers must be >= 1" in result.output
