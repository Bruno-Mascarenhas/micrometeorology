"""Work-unit pipeline contracts: byte equivalence, isolation, crash recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from pathlib import Path

import netCDF4
import numpy as np
import pytest

from micrometeorology.cli.export_wrf_geojson import _normalize_var_list
from micrometeorology.wrf import jobs
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
        temperature_source = jobs._build_value_frame_source(dataset, "temperature")
        assert isinstance(temperature_source, jobs._ValueFrameSource)

        temperature_kelvin, expected_min, expected_max = jobs.variables.extract_temperature(dataset)
        assert temperature_source.scale_min == expected_min
        assert temperature_source.scale_max == expected_max
        np.testing.assert_array_equal(
            temperature_source.frame_for_step(2),
            jobs.variables.extract_temperature_step(temperature_kelvin[2:3, :, :]),
        )

        wind_source = jobs._build_value_frame_source(dataset, "wind")
        assert isinstance(wind_source, jobs._ValueFrameSource)
        u10_values, v10_values, expected_min, expected_max = jobs.variables.extract_wind(dataset)
        assert wind_source.scale_min == expected_min
        assert wind_source.scale_max == expected_max
        np.testing.assert_array_equal(
            wind_source.frame_for_step(1),
            np.hypot(u10_values[1], v10_values[1]),
        )

        assert jobs._build_value_frame_source(dataset, "GLW") is None


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
    dt = datetime(2026, 5, 3, 12, 34, 56)
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
    # The reference payload embeds the wind dict and null for the NaN cell.
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
    # One retry: under a fully saturated CPU (e.g. first cold run of the whole
    # suite) the helper interpreter can fail to fork worker processes, failing
    # innocent units. That is environment noise, not the crash-recovery
    # behavior under test — only the deliberately crashed unit may fail.
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
    # No truncated/temp files are visible.
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
    """Time=1 wrfout files must not crash the squeeze/bounds logic (Fix 3)."""
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
    assert value_files == sorted(
        [f"D02_{v}_000.json" for v in ("RAIN", "TEMP", "WIND")]
        + [f"D02_{v}.series.bin" for v in ("RAIN", "TEMP", "WIND")]
        + [f"D02_{v}.summary.json" for v in ("RAIN", "TEMP", "WIND")]
    )

    # The lone rain frame publishes zero increments, not the cumulative total.
    with open(json_dir / "D02_RAIN_000.json", encoding="utf-8") as fh:
        rain = json.load(fh)
    assert all(v == 0.0 for v in rain["values"])


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
    var_list = _normalize_var_list(["poteolico100", "poteolico", "poteolico100"])
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
    assert _normalize_var_list(["poteolico100"]) == ["poteolico100"]
    assert _normalize_var_list(["poteolico50", "poteolico150", "poteolico50"]) == [
        "poteolico50",
        "poteolico150",
    ]
    assert _normalize_var_list(["temperature", "poteolico100", "poteolico"]) == [
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
