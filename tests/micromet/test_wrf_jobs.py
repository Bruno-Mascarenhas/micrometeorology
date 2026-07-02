"""Work-unit pipeline contracts: byte equivalence, isolation, crash recovery."""

from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from micrometeorology.cli.export_wrf_geojson import _build_json_tasks_for_domain
from micrometeorology.wrf import jobs
from micrometeorology.wrf.batch import run_json_tasks
from micrometeorology.wrf.reader import open_wrf_dataset

if TYPE_CHECKING:
    from pathlib import Path

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


def _write_full_wrf_file(path: Path, *, seed: int = 5) -> None:
    rng = np.random.default_rng(seed)
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", NT)
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
            v[:] = rng.uniform(low, high, size=(NT, NY, NX)).astype(np.float32)

        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[:] = np.array([list(f"2026-05-03_{9 + i:02d}:00:00") for i in range(NT)], dtype="S1")
        lon = ds.createVariable("XLONG", "f4", ("Time", "south_north", "west_east"))
        lat = ds.createVariable("XLAT", "f4", ("Time", "south_north", "west_east"))
        lon[:] = (
            np.linspace(-38.5, -38.0, NX, dtype=np.float32)[None, None, :]
            .repeat(NY, axis=1)
            .repeat(NT, axis=0)
        )
        lat[:] = (
            np.linspace(-13.5, -13.0, NY, dtype=np.float32)[None, :, None]
            .repeat(NX, axis=2)
            .repeat(NT, axis=0)
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
        rainc[:] = np.cumsum(rng.uniform(0, 2, size=(NT, NY, NX)).astype(np.float32), axis=0)
        rainnc = ds.createVariable("RAINNC", "f4", ("Time", "south_north", "west_east"))
        rainnc[:] = np.cumsum(rng.uniform(0, 3, size=(NT, NY, NX)).astype(np.float32), axis=0)

        u = ds.createVariable("U", "f4", ("Time", "bottom_top", "south_north", "west_east_stag"))
        v = ds.createVariable("V", "f4", ("Time", "bottom_top", "south_north_stag", "west_east"))
        ph = ds.createVariable("PH", "f4", ("Time", "bottom_top_stag", "south_north", "west_east"))
        phb = ds.createVariable(
            "PHB", "f4", ("Time", "bottom_top_stag", "south_north", "west_east")
        )
        hgt = ds.createVariable("HGT", "f4", ("Time", "south_north", "west_east"))
        u[:] = rng.uniform(-25, 25, size=(NT, NZ, NY, NX + 1)).astype(np.float32)
        v[:] = rng.uniform(-25, 25, size=(NT, NZ, NY + 1, NX)).astype(np.float32)
        base = np.cumsum(
            rng.uniform(300, 700, size=(NT, NZ + 1, NY, NX)).astype(np.float32), axis=1
        )
        ph[:] = (base * 0.05).astype(np.float32)
        phb[:] = (base * 9.5).astype(np.float32)
        hgt[:] = rng.uniform(0, 60, size=(NT, NY, NX)).astype(np.float32)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


def _run_legacy_serial(wrf_path: Path, out_root: Path) -> None:
    json_dir = out_root / "json"
    geo_dir = out_root / "geo"
    json_dir.mkdir(parents=True)
    geo_dir.mkdir(parents=True)
    with open_wrf_dataset(wrf_path, reader="eager") as ds:
        tasks = _build_json_tasks_for_domain(ds, list(VAR_LIST), json_dir, geo_dir, 0)
    run_json_tasks(tasks, workers=1, backend="serial")


def _run_units(wrf_path: Path, out_root: Path, workers: int) -> list[jobs.UnitResult]:
    json_dir = out_root / "json"
    geo_dir = out_root / "geo"
    json_dir.mkdir(parents=True)
    geo_dir.mkdir(parents=True)
    units = jobs.build_units([wrf_path], list(VAR_LIST), json_dir, geo_dir)
    return jobs.execute_units(units, workers)


def test_units_serial_output_matches_legacy_serial_bytes(tmp_path):
    wrf = tmp_path / "wrfout_d02_jobs_synth.nc"
    _write_full_wrf_file(wrf)

    _run_legacy_serial(wrf, tmp_path / "legacy")
    results = _run_units(wrf, tmp_path / "units", workers=1)

    assert not [r for r in results if r.error]
    legacy = _tree_bytes(tmp_path / "legacy")
    units = _tree_bytes(tmp_path / "units")
    assert set(units) == set(legacy)
    mismatched = [name for name in legacy if units[name] != legacy[name]]
    assert mismatched == []
    # Manifest covers everything written.
    reported = {name.rsplit("/", 1)[-1] for r in results for name in r.files}
    on_disk = {name.rsplit("/", 1)[-1] for name in units}
    assert reported == on_disk


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
    assert len(ok.files) == NT


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
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=os.environ | {"LABMIM_TEST_CRASH_UNIT": "pressure"},
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    rows = {label: (failed, n) for label, failed, n in __import__("json").loads(proc.stdout)}

    # The crashed unit is reported failed; the survivors completed fully.
    assert rows[f"{wrf.name}:pressure"][0] is True
    assert rows[f"{wrf.name}:temperature"] == (False, NT)
    assert rows[f"{wrf.name}:wind"] == (False, NT)
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
