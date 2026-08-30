"""Work-unit pipeline contracts: byte equivalence, isolation, crash recovery."""

import json
import os
import re
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import pytest

from micrometeorology.wrf import jobs
from micrometeorology.wrf.value_source import ValueFrameSource, build_value_frame_source
from tests.micromet import _reference

NT, NZ, NY, NX = 5, 4, 4, 5

VAR_LIST = [
    "temperature",
    "pressure",
    "wind",
    "rain",
    "vapor",
    "skin_temperature",
    "relative_humidity",
    "HFX",
    "SWDOWN",
    "poteolico",
    "wind_power_density_10m",
    "wind_vectors",
]


def _write_full_wrf_file(
    path: Path, *, seed: int = 5, nt: int = NT, start_hour_utc: int = 9
) -> None:
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", nt)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("bottom_top", NZ)
        ds.createDimension("bottom_top_stag", NZ + 1)
        ds.createDimension("south_north", NY)
        ds.createDimension("south_north_stag", NY + 1)
        ds.createDimension("west_east", NX)
        ds.createDimension("west_east_stag", NX + 1)
        ds.setncattr("DX", 1000.0)
        ds.setncattr("DY", 1000.0)

        def var2d(name: str, low: float, high: float) -> None:
            v = ds.createVariable(name, "f4", ("Time", "south_north", "west_east"))
            v[:] = rng.uniform(low, high, size=(nt, NY, NX)).astype(np.float32)

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array(
            [list(f"2026-05-03_{start_hour_utc + i:02d}:00:00") for i in range(nt)], dtype="S1"
        )
        lon = ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        lat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        lon[:] = (
            np.linspace(-38.5, -38.0, NX, dtype=np.float32)[None, None, :]
            .repeat(NY, axis=1)
            .repeat(nt, axis=0)
        )
        lat[:] = (
            np.linspace(-13.5, -13.0, NY, dtype=np.float32)[None, :, None]
            .repeat(NX, axis=2)
            .repeat(nt, axis=0)
        )

        var2d("T2", 290, 305)
        var2d("PSFC", 99000, 102000)
        var2d("TSK", 288, 310)
        var2d("Q2", 0.01, 0.02)
        var2d("U10", -8, 8)
        var2d("V10", -8, 8)
        var2d("HFX", -30, 400)
        var2d("SWDOWN", 0, 900)
        rainc = ds.createVariable("RAINC", "f4", ("Time", "south_north", "west_east"))
        rainc[:] = np.cumsum(rng.uniform(0, 2, size=(nt, NY, NX)).astype(np.float32), axis=0)
        rainnc = ds.createVariable("RAINNC", "f4", ("Time", "south_north", "west_east"))
        rainnc[:] = np.cumsum(rng.uniform(0, 3, size=(nt, NY, NX)).astype(np.float32), axis=0)

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        v = ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        ph = ds.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
        phb = ds.createVariable(
            "PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        u[:] = rng.uniform(-25, 25, size=(nt, NZ, NY, NX + 1)).astype(np.float32)
        v[:] = rng.uniform(-25, 25, size=(nt, NZ, NY + 1, NX)).astype(np.float32)
        base = np.cumsum(
            rng.uniform(300, 700, size=(nt, NZ + 1, NY, NX)).astype(np.float32), axis=1
        )
        ph[:] = (base * 0.05).astype(np.float32)
        phb[:] = (base * 9.5).astype(np.float32)
        hgt[:] = rng.uniform(0, 60, size=(nt, NY, NX)).astype(np.float32)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _run_units(wrf_path: Path, out_root: Path, workers: int) -> list[jobs.UnitResult]:
    json_dir = out_root / "json"
    geo_dir = out_root / "geo"
    json_dir.mkdir(parents=True)
    geo_dir.mkdir(parents=True)
    units = jobs.build_units([wrf_path], list(VAR_LIST), json_dir, geo_dir)
    return jobs.execute_units(units, workers)


def test_value_frame_source_exposes_named_scale_and_step_contract(tmp_path):
    wrf = tmp_path / "wrfout_d02_frame_source.nc"
    _write_full_wrf_file(wrf, seed=6)

    with jobs.WRFDataset(wrf) as dataset:
        temperature_source = build_value_frame_source(dataset, "temperature")
        assert isinstance(temperature_source, ValueFrameSource)

        temperature_kelvin, expected_min, expected_max = jobs.variables.extract_temperature(dataset)
        assert temperature_source.scale_min == expected_min
        assert temperature_source.scale_max == expected_max
        np.testing.assert_array_equal(
            temperature_source.frame_for_step(2),
            jobs.variables.extract_temperature_step(temperature_kelvin[2:3, :, :]),
        )

        wind_source = build_value_frame_source(dataset, "wind")
        assert isinstance(wind_source, ValueFrameSource)
        u10_values, v10_values, expected_min, expected_max = jobs.variables.extract_wind(dataset)
        assert wind_source.scale_min == expected_min
        assert wind_source.scale_max == expected_max
        np.testing.assert_array_equal(
            wind_source.frame_for_step(1),
            np.hypot(u10_values[1], v10_values[1]),
        )

        assert build_value_frame_source(dataset, "GLW") is None


def test_values_json_matches_reference_payload_with_int_formatting(tmp_path):
    """The values-JSON content is pinned by the frozen reference oracle.

    ``write_values_json_stream`` (used by every values unit through
    ``jobs._atomic_values_json``) must parse to exactly the reference payload
    — same metadata key order, compact separators, 2-decimal rounding,
    NaN→null, embedded wind payload. The one deliberate byte-level deviation
    from the reference is that whole floats in the *values* array serialize
    as integers (``0.0`` → ``0``), which parses to the same numbers.
    """
    arr = np.array(
        [[1.234, np.nan, 5.6789], [-3.21, 0.0, 2.5]],
        dtype=np.float32,
    )
    wind_data = {
        "downsampled_angles": [123.45678901234567, 350.0],
        "downsampled_magnitudes": [4.567890123456789, 0.25],
        "downsampled_linear_indices": [0, 3],
    }
    # Naive on purpose: both writers drop tzinfo before strftime, so the byte
    # oracle is only meaningful for the naive input path (pandas keeps it naive).
    dt = pd.Timestamp(2026, 5, 3, 12, 34, 56)
    out = tmp_path / "values.json"

    jobs._atomic_values_json(out, arr, 0.0, 5.0, jobs._format_datetime(dt), wind_data)

    expected = _reference.create_values_json(arr, 0.0, 5.0, dt, wind_data)
    text = out.read_text(encoding="utf-8")
    assert json.loads(text) == expected
    # metadata (including the embedded wind dict) keeps the exact reference
    # serialization; only the values array formatting deviates.
    expected_metadata = json.dumps(expected["metadata"], separators=(",", ":"), ensure_ascii=False)
    assert text.startswith('{"metadata":' + expected_metadata)
    assert text.endswith(',"values":[1.23,null,5.68,-3.21,0,2.5]}')
    assert expected["metadata"]["wind"] == wind_data
    assert expected["values"][1] is None


def test_units_parallel_output_matches_serial_bytes(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_par.nc"
    _write_full_wrf_file(wrf, seed=9)

    serial_results = _run_units(wrf, tmp_path / "serial", workers=1)
    parallel_results = _run_units(wrf, tmp_path / "parallel", workers=3)

    assert not [r for r in serial_results if r.error]
    assert not [r for r in parallel_results if r.error]
    assert _tree_bytes(tmp_path / "serial") == _tree_bytes(tmp_path / "parallel")


def test_no_temp_files_left_behind(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_tmp.nc"
    _write_full_wrf_file(wrf, seed=3)

    _run_units(wrf, tmp_path / "out", workers=2)

    leftovers = [p for p in (tmp_path / "out").rglob("*") if ".tmp-" in p.name]
    assert leftovers == []


def test_missing_variable_warns_and_missing_file_isolates_error(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_err.nc"
    _write_full_wrf_file(wrf, seed=7)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature", "GLW"], json_dir, geo_dir)
    units.append(
        jobs.WorkUnit(
            kind="values_json",
            wrf_path=str(tmp_path / "missing_wrfout"),
            variable="temperature",
            json_dir=str(json_dir),
            geojson_dir=str(geo_dir),
        )
    )
    results = jobs.execute_units(units, workers=1)

    by_label = {r.label: r for r in results}
    glw = by_label[f"{wrf.name}:GLW"]
    assert glw.error is None
    assert any("GLW not found" in w for w in glw.warnings)
    missing = by_label["missing_wrfout:temperature"]
    assert missing.error is not None
    ok = by_label[f"{wrf.name}:temperature"]
    assert ok.error is None
    # NT per-step JSONs plus the consolidated .series.bin and .summary.json.
    assert len(ok.files) == NT + 2


class _FakeFuture:
    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


def test_broken_pool_retries_each_incomplete_unit_in_isolation(monkeypatch, tmp_path):
    unit = jobs.WorkUnit(
        kind="values_json",
        wrf_path=str(tmp_path / "whatever"),
        variable="temperature",
        json_dir=str(tmp_path),
        geojson_dir=str(tmp_path),
    )

    attempts = []

    class _BrokenExecutor:
        def __init__(self, max_workers=None, **_kwargs):
            attempts.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *_args):
            return _FakeFuture(exc=BrokenProcessPool("worker died"))

    monkeypatch.setattr(jobs, "ProcessPoolExecutor", _BrokenExecutor)
    monkeypatch.setattr(jobs, "as_completed", lambda futures: list(futures))

    results = jobs.execute_units([unit, unit], workers=2)

    # One shared pool, then one isolated single-worker pool per incomplete unit.
    assert attempts == [2, 1, 1]
    assert len(results) == 2
    assert all("worker crashed while processing" in (r.error or "") for r in results)


def test_worker_crash_recovers_and_reports_nonzero(tmp_path):
    """Real forkserver worker killed via os._exit: pool respawn + clean output."""
    wrf = tmp_path / "wrfout_d02_jobs_crash.nc"
    _write_full_wrf_file(wrf, seed=13)
    out = tmp_path / "out"
    (out / "json").mkdir(parents=True)
    (out / "geo").mkdir(parents=True)

    script = f"""
import json
from micrometeorology.wrf import jobs
units = jobs.build_units([{str(wrf)!r}], ["temperature", "pressure", "wind"], {str(out / "json")!r}, {str(out / "geo")!r})
results = jobs.execute_units(units, workers=2, echo=lambda _msg: None)
print(json.dumps([[r.label, r.error is not None, len(r.files)] for r in results]))
"""
    # One retry: under a fully saturated CPU the helper interpreter can fail to
    # fork worker processes, failing innocent units. That is environment noise,
    # not the crash recovery under test — only the crashed unit may fail.
    for _attempt in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=os.environ | {"LABMIM_TEST_CRASH_UNIT": "pressure"},
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            continue
        rows = {label: (failed, n) for label, failed, n in __import__("json").loads(proc.stdout)}
        innocents_ok = all(
            not failed for label, (failed, _n) in rows.items() if not label.endswith(":pressure")
        )
        if innocents_ok:
            break
    assert proc.returncode == 0, proc.stderr[-2000:]

    # The crashed unit is reported failed; the survivors completed fully
    # (NT per-step JSONs + .series.bin + .summary.json each).
    assert rows[f"{wrf.name}:pressure"][0] is True
    assert rows[f"{wrf.name}:temperature"] == (False, NT + 2)
    assert rows[f"{wrf.name}:wind"] == (False, NT + 2)
    leftovers = [p for p in out.rglob("*") if ".tmp-" in p.name]
    assert leftovers == []
    for p in (out / "json").glob("*.json"):
        assert p.read_bytes().endswith(b"}")


def test_work_units_are_plain_picklable_payloads(tmp_path):
    import pickle

    units = jobs.build_units([tmp_path / "f"], ["temperature"], tmp_path, tmp_path)
    for unit in units:
        clone = pickle.loads(pickle.dumps(unit))
        assert clone == unit
        for field_value in (unit.kind, unit.wrf_path, unit.variable, unit.json_dir):
            assert isinstance(field_value, str)


def test_units_run_capped_serial_when_single_worker(tmp_path, monkeypatch):
    wrf = tmp_path / "wrfout_d02_jobs_serial.nc"
    _write_full_wrf_file(wrf, seed=1)

    def _boom(*_args, **_kwargs):
        raise AssertionError("no pool should be created for workers=1")

    monkeypatch.setattr(jobs, "ProcessPoolExecutor", _boom)
    results = _run_units(wrf, tmp_path / "out", workers=1)
    assert not [r for r in results if r.error]


def test_single_timestep_file_processes_without_errors(tmp_path):
    """Time=1 wrfout files must not crash the squeeze/bounds logic."""
    wrf = tmp_path / "wrfout_d02_jobs_single.nc"
    _write_full_wrf_file(wrf, seed=21, nt=1)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature", "wind", "rain"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)

    assert [r.error for r in results if r.error] == []
    value_files = sorted(
        os.path.basename(f) for r in results if r.kind == "values_json" for f in r.files
    )
    # Rain publishes nothing at all: its only step is the boundary one, which has
    # no previous total to difference. Zeros there would state that it rained
    # nowhere, and the cumulative total would be a downpour that never fell.
    assert value_files == sorted(
        [f"D02_{v}_000.json" for v in ("TEMP", "WIND")]
        + [f"D02_{v}.series.bin" for v in ("TEMP", "WIND")]
        + [f"D02_{v}.summary.json" for v in ("TEMP", "WIND")]
    )
    assert not (json_dir / "D02_RAIN_000.json").exists()


def test_parse_poteolico_heights_maps_names_to_targets():
    assert jobs.parse_poteolico_heights("poteolico") == (50, 100, 150)
    assert jobs.parse_poteolico_heights("poteolico50") == (50,)
    assert jobs.parse_poteolico_heights("poteolico100") == (100,)
    assert jobs.parse_poteolico_heights("poteolico150") == (150,)
    for bad in ("poteolico75", "poteolico1000", "poteolicoXY", "weibull"):
        with pytest.raises(ValueError, match="poteolico"):
            jobs.parse_poteolico_heights(bad)


def test_poteolico_single_height_writes_only_that_height(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_pot100.nc"
    _write_full_wrf_file(wrf, seed=17)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["poteolico100"], json_dir, geo_dir)
    assert [u.kind for u in units] == ["grid_geojson", "poteolico"]
    results = jobs.execute_units(units, workers=1)

    assert [r for r in results if r.error] == []
    written = sorted(p.name for p in json_dir.glob("*.json"))
    assert written == sorted(
        [f"D02_POT_EOLICO_100M_{i:03d}.json" for i in range(NT)]
        + ["D02_POT_EOLICO_100M.summary.json"]
    )


def test_poteolico_bare_name_writes_all_three_heights(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_potall.nc"
    _write_full_wrf_file(wrf, seed=19)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["poteolico"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)

    assert [r for r in results if r.error] == []
    written = sorted(p.name for p in json_dir.glob("*.json"))
    expected = sorted(
        [f"D02_POT_EOLICO_{h}M_{i:03d}.json" for h in (50, 100, 150) for i in range(NT)]
        + [f"D02_POT_EOLICO_{h}M.summary.json" for h in (50, 100, 150)]
    )
    assert written == expected


def test_poteolico_duplicates_normalize_to_all_heights_once(tmp_path):
    var_list = jobs.normalize_var_list(
        ["poteolico100", "poteolico", "poteolico100"], collapse_heights=False
    )
    assert var_list == ["poteolico"]

    wrf = tmp_path / "wrfout_d02_jobs_potdup.nc"
    _write_full_wrf_file(wrf, seed=23)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], var_list, json_dir, geo_dir)
    assert [u.variable for u in units if u.kind == "poteolico"] == ["poteolico"]
    results = jobs.execute_units(units, workers=1)

    assert [r for r in results if r.error] == []
    written = sorted(p.name for p in json_dir.glob("*.json"))
    expected = sorted(
        [f"D02_POT_EOLICO_{h}M_{i:03d}.json" for h in (50, 100, 150) for i in range(NT)]
        + [f"D02_POT_EOLICO_{h}M.summary.json" for h in (50, 100, 150)]
    )
    assert written == expected


def test_normalize_var_list_keeps_single_height_requests_distinct():
    assert jobs.normalize_var_list(["poteolico100"], collapse_heights=False) == ["poteolico100"]
    assert jobs.normalize_var_list(
        ["poteolico50", "poteolico150", "poteolico50"], collapse_heights=False
    ) == [
        "poteolico50",
        "poteolico150",
    ]
    assert jobs.normalize_var_list(
        ["temperature", "poteolico100", "poteolico"], collapse_heights=False
    ) == [
        "temperature",
        "poteolico",
    ]


# ---------------------------------------------------------------------------
# Consolidated site artifacts (series.bin / summary.json) and manifest v2
# ---------------------------------------------------------------------------


def _series_matrix(path: Path, n_steps: int) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype="<i4")
    assert raw.size % n_steps == 0
    return raw.reshape(raw.size // n_steps, n_steps)


def test_series_bin_and_summary_agree_with_per_step_jsons(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_series.nc"
    _write_full_wrf_file(wrf, seed=29)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)
    assert [r for r in results if r.error] == []

    matrix = _series_matrix(json_dir / "D02_TEMP.series.bin", NT)
    assert matrix.shape == (NY * NX, NT)

    with open(json_dir / "D02_TEMP.summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    assert summary["format"] == "domain-summary-v1"
    assert summary["indices"] == list(range(NT))
    assert len(summary["mean"]) == len(summary["date_times"]) == NT

    for i in range(NT):
        with open(json_dir / f"D02_TEMP_{i:03d}.json", encoding="utf-8") as fh:
            payload = json.load(fh)
        values = payload["values"]
        column = matrix[:, i]
        for cell, value in enumerate(values):
            if value is None:
                assert column[cell] == jobs.SERIES_MISSING
            else:
                assert column[cell] == round(value * jobs.SERIES_SCALE)
        finite = [v for v in values if v is not None]
        assert summary["mean"][i] == pytest.approx(np.mean(finite), abs=0.011)
        assert summary["min"][i] == min(finite)
        assert summary["max"][i] == max(finite)
        assert summary["date_times"][i] == payload["metadata"]["date_time"]


def test_manifest_v2_timeline_availability_and_features(tmp_path, monkeypatch):
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_jobs_manifest.nc"
    # 19..23 UTC = 16..20 local (America/Bahia): SWDOWN's 6-18h daylight gate
    # keeps only the first three steps, exercising the availability ranges.
    _write_full_wrf_file(wrf, seed=31, start_hour_utc=19)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature", "SWDOWN"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)
    assert [r for r in results if r.error] == []

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert manifest["format"] == "labmim-data-manifest-v2"
    assert manifest["timezone"] == "America/Bahia"
    assert manifest["index_min"] == 0
    assert manifest["index_max"] == NT - 1
    assert manifest["start_local"] == "03/05/2026 16:00:00"
    assert manifest["availability"] == {"SWDOWN": [[0, 2]]}
    assert manifest["features"]["domain_summary"]["format"] == "domain-summary-v1"
    series_feature = manifest["features"]["cell_series"]
    assert series_feature["format"] == "cell-series-int32-le-v1"
    assert series_feature["missing"] == jobs.SERIES_MISSING
    assert (series_feature["index_min"], series_feature["index_max"]) == (0, NT - 1)

    # The gated SWDOWN night steps are MISSING columns in a full-width matrix.
    matrix = _series_matrix(json_dir / "D02_SWDOWN.series.bin", NT)
    assert (matrix[:, 3:] == jobs.SERIES_MISSING).all()
    assert (matrix[:, 0] != jobs.SERIES_MISSING).any()


def test_no_site_artifacts_flag_writes_legacy_outputs_only(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_legacy.nc"
    _write_full_wrf_file(wrf, seed=37)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir, site_artifacts=False)
    results = jobs.execute_units(units, workers=1)
    assert [r for r in results if r.error] == []

    names = sorted(p.name for p in json_dir.iterdir())
    assert names == [f"D02_TEMP_{i:03d}.json" for i in range(NT)]

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert "features" not in manifest
    assert manifest["index_max"] == NT - 1  # timeline fields are still written


def test_values_json_rejects_non_finite_scale_bounds(tmp_path):
    from micrometeorology.wrf.geojson import write_values_json_stream

    arr = np.full((2, 2), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match=r"[Nn]on-finite scale bounds"):
        write_values_json_stream(tmp_path / "bad.json", arr, float("nan"), float("nan"), "N/A")
    assert not (tmp_path / "bad.json").exists() or (tmp_path / "bad.json").stat().st_size == 0


def test_sweep_removes_dead_pid_debris_on_healthy_run(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_sweep.nc"
    _write_full_wrf_file(wrf, seed=41)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"
    json_dir.mkdir()
    geo_dir.mkdir()
    # Debris from a previous run whose worker pid no longer exists: a healthy
    # run (no broken pool) must still sweep it.
    debris = json_dir / ".D02_TEMP_000.json.tmp-999999999"
    debris.write_text("truncated")

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=2)

    assert [r for r in results if r.error] == []
    assert not debris.exists()


def test_manifest_omits_features_when_any_unit_failed(tmp_path):
    """A failed unit can leave LAST run's consolidated artifacts in place;
    the manifest must not vouch for them (the site falls back to per-step
    JSONs), while the timeline fields — derived from actually written files —
    stay available."""
    wrf = tmp_path / "wrfout_d02_jobs_dirty.nc"
    _write_full_wrf_file(wrf, seed=43)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir)
    units.append(
        jobs.WorkUnit(
            kind="values_json",
            wrf_path=str(tmp_path / "missing_wrfout"),
            variable="pressure",
            json_dir=str(json_dir),
            geojson_dir=str(geo_dir),
        )
    )
    results = jobs.execute_units(units, workers=1)
    assert any(r.error for r in results)

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert "features" not in manifest
    assert manifest["index_min"] == 0
    assert manifest["index_max"] == NT - 1


def test_a_variable_that_publishes_no_step_withholds_the_directory_wide_features(
    tmp_path, monkeypatch
):
    """A gated-out variable leaves LAST run's .series.bin newest on disk.

    00..04 UTC is 21..01 local, entirely outside SWDOWN's 6-18h daylight gate, so
    the unit finds its variable, errors on nothing, and still writes no step. Its
    fixed output names then hold the previous run's matrix, which the
    directory-wide cell_series descriptor would vouch for under THIS run's step
    count — reading every cell at the wrong byte offset if the two runs differ in
    length.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_jobs_allnight.nc"
    _write_full_wrf_file(wrf, seed=57, start_hour_utc=0)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature", "SWDOWN"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)

    assert [r for r in results if r.error] == []
    assert any("SWDOWN" in variable for r in results for variable in r.missing_variables)

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert "features" not in manifest
    assert manifest["availability"]["SWDOWN"] == []


def test_build_units_refuses_two_files_of_the_same_domain(tmp_path):
    """Both files would write D02_TEMP_000.json, D02_TEMP.series.bin and
    D02.geojson, concurrently, and the survivor would be whichever unit
    finished last — a timeline whose frames come from two forecast days."""
    first = tmp_path / "wrfout_d02_2026-05-03_09:00:00"
    second = tmp_path / "wrfout_d02_2026-05-04_09:00:00"

    with pytest.raises(ValueError, match="would overwrite each other") as excinfo:
        jobs.build_units([first, second], ["temperature"], tmp_path, tmp_path)

    message = str(excinfo.value)
    assert "D02" in message
    assert first.name in message
    assert second.name in message


def test_build_units_accepts_one_file_per_domain(tmp_path):
    units = jobs.build_units(
        [tmp_path / "wrfout_d01_x", tmp_path / "wrfout_d02_y"], ["temperature"], tmp_path, tmp_path
    )
    assert sorted({Path(u.wrf_path).name for u in units}) == ["wrfout_d01_x", "wrfout_d02_y"]


def test_untokenized_filename_fails_the_unit_instead_of_publishing_as_d01(tmp_path):
    wrf = tmp_path / "wrfout_2026-07-27_00_00_00.nc"
    _write_full_wrf_file(wrf, seed=51)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"
    json_dir.mkdir()
    geo_dir.mkdir()

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)

    assert all("Could not detect grid level" in (r.error or "") for r in results)
    assert list(json_dir.iterdir()) == []
    assert list(geo_dir.iterdir()) == []

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        assert json.load(fh)["domains"] == []


def test_manifest_domains_follow_the_same_rule_as_the_published_filenames(tmp_path):
    """``wrfout_d03.nc`` has no trailing token underscore, so the manifest and
    the published filenames must agree on D03 from the same detection rule."""
    wrf = tmp_path / "wrfout_d03.nc"
    _write_full_wrf_file(wrf, seed=53)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)
    assert [r for r in results if r.error] == []

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["domains"] == ["D03"]
    assert (json_dir / "D03_TEMP_000.json").exists()


def test_manifest_publishes_a_variable_the_wrfout_does_not_carry_as_empty(tmp_path):
    """GLW is requested but absent: its fixed-name files from the previous run
    are still on disk, so the manifest must neither omit it (read as "full
    range") nor vouch for the stale .series.bin through ``features``."""
    wrf = tmp_path / "wrfout_d02_jobs_absent.nc"
    _write_full_wrf_file(wrf, seed=57)
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"

    units = jobs.build_units([wrf], ["temperature", "GLW"], json_dir, geo_dir)
    results = jobs.execute_units(units, workers=1)
    assert [r for r in results if r.error] == []
    assert {r.missing_variables for r in results} == {(), ("GLW",)}

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert manifest["availability"] == {"GLW": []}
    assert "features" not in manifest
    assert manifest["index_max"] == NT - 1


def test_a_domain_whose_unit_failed_is_not_covered_by_the_surviving_domain_range(tmp_path):
    """D02's temperature unit died; D01's wrote every step of the same variable.

    ``availability`` has no domain axis, so an omitted key reads as "full
    range" for every advertised domain. D02 is still advertised (its grid unit
    succeeded) and its fixed-name ``D02_TEMP_nnn.json`` still hold the PREVIOUS
    run's forecast, which the site would then fetch under this run's version.
    """
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    survived = jobs.UnitResult(
        label="wrfout_d01:temperature",
        kind="values_json",
        files=tuple(str(json_dir / f"D01_TEMP_{i:03d}.json") for i in range(NT)),
        domain="D01",
        n_steps=NT,
    )
    crashed = jobs.process_unit(
        jobs.WorkUnit(
            kind="values_json",
            wrf_path=str(tmp_path / "wrfout_d02_never_written.nc"),
            variable="temperature",
            json_dir=str(json_dir),
            geojson_dir=str(tmp_path / "geo"),
        )
    )
    grid = jobs.UnitResult(label="wrfout_d02:grid_geojson", kind="grid_geojson", domain="D02")

    manifest_path = jobs.write_run_manifest(
        json_dir, [survived, crashed, grid], ["temperature"], covers_every_variable=False
    )

    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["domains"] == ["D01", "D02"]
    assert manifest["availability"] == {"TEMP": []}


def test_a_raw_netcdf_request_that_published_no_step_is_advertised_as_empty(tmp_path):
    """``T2`` is not an enum member, yet it publishes under a fixed
    ``{D}_T2_nnn.json`` id all the same. A run whose every step was withheld
    must say so: an omitted ``availability`` key reads as "full range" and sends
    the site to the PREVIOUS run's files under this run's version stamp."""
    json_dir = tmp_path / "json"
    json_dir.mkdir()
    published = jobs.UnitResult(
        label="wrfout_d01:temperature",
        kind="values_json",
        files=tuple(str(json_dir / f"D01_TEMP_{i:03d}.json") for i in range(NT)),
        domain="D01",
        n_steps=NT,
    )
    withheld = jobs.UnitResult(label="wrfout_d01:T2", kind="values_json", domain="D01", n_steps=NT)

    manifest_path = jobs.write_run_manifest(
        json_dir, [published, withheld], ["temperature", "T2"], covers_every_variable=False
    )

    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["availability"] == {"T2": []}


@pytest.mark.parametrize(
    ("kind", "variable", "output_ids"),
    [
        ("values_json", "temperature", ("TEMP",)),
        ("values_json", "T2", ("T2",)),
        ("poteolico", "poteolico", ("POT_EOLICO_50M", "POT_EOLICO_100M", "POT_EOLICO_150M")),
        ("poteolico", "poteolico100", ("POT_EOLICO_100M",)),
        ("poteolico", "poteolico99", ()),
        ("wind_vectors", "wind_vectors", ("WIND_VECTORS",)),
        ("grid_geojson", "", ()),
    ],
)
def test_a_failed_unit_declares_the_output_ids_it_never_wrote(tmp_path, kind, variable, output_ids):
    """The unparseable height owns no id, and claiming one is worse than none."""
    unit = jobs.WorkUnit(
        kind=kind,
        wrf_path=str(tmp_path / "wrfout_d02_never_written.nc"),
        variable=variable,
        json_dir=str(tmp_path),
        geojson_dir=str(tmp_path),
    )

    result = jobs.process_unit(unit)

    assert result.error is not None
    assert result.missing_variables == output_ids


def test_every_unit_lost_to_a_broken_pool_declares_its_own_output_ids(monkeypatch, tmp_path):
    """One entry per lost unit, each carrying that unit's own ids.

    Collapsing the results into a set hides both the count and the pairing, so
    a run that lost two units while advertising one — or that handed a unit its
    neighbour's ids — would read as correct.
    """
    values_unit = jobs.WorkUnit(
        kind="values_json",
        wrf_path=str(tmp_path / "wrfout_d02_oom.nc"),
        variable="clearness_index",
        json_dir=str(tmp_path),
        geojson_dir=str(tmp_path),
    )
    poteolico_unit = jobs.WorkUnit(
        kind="poteolico",
        wrf_path=str(tmp_path / "wrfout_d02_oom.nc"),
        variable="poteolico100",
        json_dir=str(tmp_path),
        geojson_dir=str(tmp_path),
    )

    class _AlwaysBrokenExecutor:
        def __init__(self, max_workers=None, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *_args):
            return _FakeFuture(exc=BrokenProcessPool("worker died"))

    monkeypatch.setattr(jobs, "ProcessPoolExecutor", _AlwaysBrokenExecutor)
    monkeypatch.setattr(jobs, "as_completed", lambda futures: list(futures))
    results = jobs.execute_units([poteolico_unit, values_unit], workers=2, echo=lambda _msg: None)

    assert [r.missing_variables for r in results] == [("POT_EOLICO_100M",), ("KT",)]


def test_atomic_json_dump_writes_exactly_the_compact_encoder_bytes(tmp_path):
    """The atomic writer must emit exactly the compact-encoder bytes."""
    payload = {
        "format": "domain-summary-v1",
        "variable": "POT_EOLICO_150M",
        "label": "Previsão · Sábado",
        "indices": [0, 1, 2],
        "mean": [1.0, -3.25, 1e-07],
        "max": [305.0, 2.5, 0],
    }
    out = tmp_path / "payload.json"

    jobs._atomic_json_dump(out, payload)

    expected = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert out.read_text(encoding="utf-8") == expected
    assert json.loads(expected) == payload


def test_atomic_json_dump_still_refuses_non_finite_payloads(tmp_path):
    out = tmp_path / "bad.json"
    with pytest.raises(ValueError, match="not JSON compliant"):
        jobs._atomic_json_dump(out, {"mean": [float("nan")]})
    assert not out.exists()


def test_worker_logging_is_configured_with_pid_and_level_on_stderr(tmp_path):
    """Forkserver children inherit no handlers, so without explicit worker
    configuration their records fall through to ``logging.lastResort``: bare
    text, no timestamp and no pid, from every worker stream at once."""
    script = f"""
from micrometeorology.common.logging import setup_logging
from micrometeorology.wrf import jobs

setup_logging("INFO")
units = [
    jobs.WorkUnit(
        kind="values_json",
        wrf_path={str(tmp_path / "missing_wrfout_d02")!r},
        variable="temperature",
        json_dir={str(tmp_path)!r},
        geojson_dir={str(tmp_path)!r},
    ),
    jobs.WorkUnit(
        kind="values_json",
        wrf_path={str(tmp_path / "missing_wrfout_d03")!r},
        variable="temperature",
        json_dir={str(tmp_path)!r},
        geojson_dir={str(tmp_path)!r},
    ),
]
results = jobs.execute_units(units, workers=2, echo=lambda _msg: None)
assert all(r.error for r in results), results
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]

    worker_records = [line for line in proc.stderr.splitlines() if "Work unit failed" in line]
    assert worker_records, proc.stderr[-2000:]
    pattern = re.compile(
        r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d \| (\d+) \| micrometeorology\.wrf\.jobs\s+"
        r"\| ERROR\s+\| Work unit failed"
    )
    pids = set()
    for line in worker_records:
        match = pattern.match(line)
        assert match is not None, line
        pids.add(match.group(1))
    assert pids
    assert str(os.getpid()) not in pids


def test_serial_run_sweeps_its_own_failed_unit_debris(tmp_path):
    """workers=1 runs write temp files under the parent's own (live) pid; the
    end-of-run sweep must still remove them."""
    json_dir = tmp_path / "json"
    geo_dir = tmp_path / "geo"
    json_dir.mkdir()
    geo_dir.mkdir()
    debris = json_dir / f".D02_TEMP_000.json.tmp-{os.getpid()}"
    debris.write_text("truncated")

    unit = jobs.WorkUnit(
        kind="values_json",
        wrf_path=str(tmp_path / "missing_wrfout"),
        variable="temperature",
        json_dir=str(json_dir),
        geojson_dir=str(geo_dir),
    )
    results = jobs.execute_units([unit], workers=1)

    assert results[0].error is not None
    assert not debris.exists()


def test_the_timeline_scanner_matches_the_names_the_writers_actually_produce():
    """``{i:03d}`` is a MINIMUM width, so step 1000 is written ``_1000.json``.

    A scanner matching exactly three digits disagrees with the writer past that
    point: those files join neither the timeline nor ``availability``, so
    ``index_max`` is published as 999 against a true step count of 1007 and the
    site's slider hides the last frames.
    """
    for name, index in (
        ("D01_TEMP_000.json", "000"),
        ("D02_TEMP_999.json", "999"),
        ("D02_TEMP_1000.json", "1000"),
        ("D01_POT_EOLICO_50M_1007.json", "1007"),
    ):
        match = jobs._TIMESTEP_FILE_RE.match(name)
        assert match is not None, name
        assert match.group(3) == index

    assert jobs._TIMESTEP_FILE_RE.match("D01_TEMP_00.json") is None
    assert jobs._TIMESTEP_FILE_RE.match("D01_TEMP.summary.json") is None


def test_a_step_with_no_finite_cell_is_not_published_at_all(tmp_path, monkeypatch):
    """The daylight window is a clock rule; a value can still be undefined inside it.

    The clearness index is null wherever cos(z) falls below its cutoff, so every
    sunrise and sunset step is all-null. Writing it would make the run's two
    artifacts disagree about the same instant: ``write_run_manifest`` derives
    the timeline and ``availability`` from the files WRITTEN, while the
    ``.summary.json`` the preview panel reads records only steps that carried a
    finite cell. The site reads the first for its map slider and the second for
    its panel.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)
    wrf = tmp_path / "wrfout_d02_kt_sunrise.nc"
    _write_full_wrf_file(wrf, seed=7)
    with netCDF4.Dataset(str(wrf), "a") as ds:
        cos_zenith = ds.createVariable("COSZEN", "f4", ("Time", "south_north", "west_east"))
        values = np.full((NT, NY, NX), 0.6, dtype="f4")
        values[0, :, :] = 0.05  # below MIN_COSZEN_FOR_CLEARNESS: every cell null
        cos_zenith[:] = values

    json_dir, geo_dir = str(tmp_path / "json"), str(tmp_path / "geo")
    units = jobs.build_units([str(wrf)], ["clearness_index"], json_dir, geo_dir)
    results = [jobs.process_unit(unit) for unit in units]
    jobs.write_run_manifest(json_dir, results)

    steps = sorted(p.name for p in Path(json_dir).glob("D02_KT_[0-9]*.json"))
    assert "D02_KT_000.json" not in steps, "an all-null step must not be published"
    assert steps == [f"D02_KT_{i:03d}.json" for i in range(1, NT)]

    manifest = json.loads((Path(json_dir) / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((Path(json_dir) / "D02_KT.summary.json").read_text(encoding="utf-8"))
    # The one invariant that was broken: the timeline and the panel agree.
    assert manifest["index_min"] == min(summary["indices"])
    assert manifest["index_max"] == max(summary["indices"])


def test_availability_is_the_intersection_across_domains(tmp_path):
    """A step is advertised only when EVERY domain wrote that variable's frame.

    ``availability`` carries no domain axis and the site reads it for whichever
    domain is selected, so a per-variable union advertises D01's steps to D03.
    The clearness index diverges by construction: its cos(z) mask is geographic,
    so a coarse domain reaching further west keeps sunrise frames a nested
    domain drops. On the four-domain run of 2026-05-03 a union advertises KT
    over 36 indices while D03 and D04 wrote 30 and D02 wrote 32 — and because
    the writers use fixed names and replace in place, requesting one of the six
    absent frames serves the PREVIOUS run's field under this run's version
    stamp.
    """
    json_dir = tmp_path / "json"
    json_dir.mkdir()

    def unit(domain: str, indices: range) -> jobs.UnitResult:
        return jobs.UnitResult(
            label=f"{domain} KT",
            kind="values_json",
            files=tuple(str(json_dir / f"{domain}_KT_{i:03d}.json") for i in indices),
            domain=domain,
            n_steps=6,
        )

    results = [
        unit("D01", range(6)),  # the superset
        unit("D02", range(1, 6)),
        unit("D03", range(1, 5)),  # the narrowest
    ]

    manifest_path = jobs.write_run_manifest(json_dir, results)
    assert manifest_path is not None
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

    # index_min/index_max already intersect; availability must agree with them.
    assert (manifest["index_min"], manifest["index_max"]) == (1, 4)
    # An empty restriction is omitted: 1..4 is the full shared range.
    assert "availability" not in manifest

    # Now make one domain skip a step INSIDE the shared range.
    holed = [unit("D01", range(6)), unit("D02", range(1, 6)), unit("D03", range(1, 5))]
    holed[1] = jobs.UnitResult(
        label="D02 KT",
        kind="values_json",
        files=tuple(str(json_dir / f"D02_KT_{i:03d}.json") for i in (1, 2, 4, 5)),
        domain="D02",
        n_steps=6,
    )
    holed_path = jobs.write_run_manifest(json_dir, holed)
    assert holed_path is not None
    manifest = json.loads(Path(holed_path).read_text(encoding="utf-8"))

    assert manifest["availability"] == {"KT": [[1, 2], [4, 4]]}, (
        "index 3 is missing from D02 and must not be advertised for any domain"
    )


def test_a_value_past_the_int32_series_range_fails_instead_of_clipping():
    """Clipping would give an unrepresentable value a representation, silently.

    The per-step JSON and the summary carry the true number while the series the
    site Range-requests for the same cell would carry the int32 ceiling — three
    artifacts of one run disagreeing, with nothing saying which is real.
    """
    accumulator = jobs._SiteArtifactAccumulator(n_steps=2)
    beyond = (jobs._SERIES_INT_MAX / jobs.SERIES_SCALE) * 10

    with pytest.raises(ValueError, match="int32 series"):
        accumulator.add(0, np.array([1.0, beyond]), "01/01/2024 00:00:00")


def test_an_ordinary_magnitude_still_encodes():
    accumulator = jobs._SiteArtifactAccumulator(n_steps=1)

    accumulator.add(0, np.array([1.5, -2.25]), "01/01/2024 00:00:00")

    assert accumulator.means[0] == pytest.approx(-0.38, abs=0.01)


def test_reading_a_v1_operational_file_warns_which_columns_are_not_repaired(caplog):
    """`rename_v1_columns` renames; it does NOT apply the v1 formula repairs.

    A v1 file carries an albedo near -273 and a reflected shortwave in the
    -1e5 W/m2, because the extraction subtracted 273.15 from dimensionless
    values. `migrate_to_v2` inverts that. Until it runs, a consumer is reading
    those six columns wrong, and an INFO line saying the names were mapped
    reads as if the file had been handled.
    """
    import logging

    from micrometeorology.wrf.operational_record import (
        V1_UNREPAIRED_COLUMNS,
        rename_v1_columns,
    )

    v1 = pd.DataFrame({"ALBD": [-273.01], "EMISS": [-272.27], "Swup_calc": [-294069.0]})

    with caplog.at_level(logging.WARNING, logger="micrometeorology.wrf.operational_record"):
        renamed = rename_v1_columns(v1)

    assert set(renamed.columns) == {"albedo", "emissivity", "swup_w_m2"}
    assert caplog.records, "reading an unrepaired v1 file must warn"
    message = caplog.records[0].getMessage()
    assert "migrate" in message
    for column in ("albedo", "emissivity", "swup_w_m2"):
        assert column in message
    assert {"albedo", "emissivity", "swup_w_m2"} <= V1_UNREPAIRED_COLUMNS
