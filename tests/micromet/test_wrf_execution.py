"""Tests for adaptive WRF execution planning."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import netCDF4
import pytest
from typer.testing import CliRunner

from micrometeorology.cli.export_wrf_geojson import app as wrf_geojson_app
from micrometeorology.wrf import execution
from micrometeorology.wrf.execution import (
    estimate_4d_working_set_bytes,
    resolve_wrf_execution_plan,
)
from tests.micromet.test_wrf_reader import _write_tiny_wrf_file


def _scratch_file(name: str, size: int = 16) -> Path:
    root = Path("scratch")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-{uuid.uuid4().hex}.nc"
    with open(path, "wb") as f:
        if size <= 4096:
            f.write(b"0" * size)
        else:
            f.truncate(size)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def test_auto_resolves_tiny_workload_to_eager_serial_for_single_worker():
    path = _scratch_file("tiny-auto")
    try:
        plan = resolve_wrf_execution_plan(paths=[path], workflow="json", workers=1)

        assert plan.reader == "eager"
        assert plan.chunks is None
        assert plan.json_worker_backend == "serial"
        assert "defaults to eager" in plan.reason
        assert "single worker" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_resolves_large_3d_input_to_eager_reader():
    # Files above the historical 512 MB / 1 GiB thresholds no longer flip to lazy.
    path = _scratch_file("large-auto", size=600 * 1024 * 1024)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=2,
            chunking_available=True,
        )

        assert plan.reader == "eager"
        assert plan.chunks is None
        assert "defaults to eager" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_figures_auto_resolves_lazy_when_4d_working_set_exceeds_budget(monkeypatch):
    path = _scratch_file("poteolico-auto")
    monkeypatch.delenv("LABMIM_EAGER_4D_BUDGET_GB", raising=False)
    monkeypatch.setattr(execution, "estimate_4d_working_set_bytes", lambda _path: 5 * 1024**3)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="figures",
            workers=1,
            requested_variables=["temperature", "poteolico"],
            chunking_available=True,
        )

        assert plan.reader == "lazy"
        assert plan.chunks == "auto"
        assert "4D working set exceeds eager budget" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_json_workflow_is_never_gated_by_4d_budget(monkeypatch):
    """The JSON workflow block-streams 4D extraction, so it always stays eager."""
    path = _scratch_file("poteolico-json-auto")
    monkeypatch.delenv("LABMIM_EAGER_4D_BUDGET_GB", raising=False)
    monkeypatch.setattr(execution, "estimate_4d_working_set_bytes", lambda _path: 512 * 1024**3)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=1,
            requested_variables=["temperature", "poteolico"],
        )

        assert plan.reader == "eager"
        assert "defaults to eager" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_figures_auto_stays_eager_when_4d_working_set_within_budget(monkeypatch):
    path = _scratch_file("poteolico-small-auto")
    monkeypatch.delenv("LABMIM_EAGER_4D_BUDGET_GB", raising=False)
    monkeypatch.setattr(execution, "estimate_4d_working_set_bytes", lambda _path: 1024**3)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="figures",
            workers=1,
            requested_variables=["poteolico100"],
        )

        assert plan.reader == "eager"
        assert "defaults to eager" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_4d_budget_gate_ignores_non_poteolico_variables(monkeypatch):
    path = _scratch_file("no-poteolico-auto")
    monkeypatch.setattr(execution, "estimate_4d_working_set_bytes", lambda _path: 512 * 1024**3)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="figures",
            workers=1,
            requested_variables=["temperature", "wind"],
        )

        assert plan.reader == "eager"
    finally:
        path.unlink(missing_ok=True)


def test_4d_budget_env_var_overrides_default(monkeypatch):
    path = _scratch_file("budget-env-auto")
    monkeypatch.setenv("LABMIM_EAGER_4D_BUDGET_GB", "0.001")
    monkeypatch.setattr(execution, "estimate_4d_working_set_bytes", lambda _path: 2 * 1024**2)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="figures",
            workers=1,
            requested_variables=["poteolico"],
            chunking_available=True,
        )

        assert plan.reader == "lazy"
        assert "4D working set exceeds eager budget" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_reader_with_explicit_chunks_resolves_lazy():
    path = _scratch_file("explicit-chunks-auto")
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            chunks_request="Time=1",
            workers=1,
            chunking_available=True,
        )

        assert plan.reader == "lazy"
        assert plan.chunks == {"Time": 1}
        assert "explicit chunk dimensions" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_json_backend_resolves_serial_for_small_payload_with_multiple_workers():
    path = _scratch_file("json-serial-auto")
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=4,
            estimated_json_payload_bytes=1024,
            json_task_count=128,
        )

        assert plan.json_worker_backend == "serial"
        assert "small JSON payload" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_json_backend_resolves_memmap_for_large_payload():
    path = _scratch_file("json-memmap-auto")
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=4,
            estimated_json_payload_bytes=64 * 1024 * 1024,
            json_task_count=4,
        )

        assert plan.json_worker_backend == "memmap"
        assert "large estimated JSON payload" in plan.reason
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("workflow", ["figures", "pipeline"])
def test_auto_figure_backends_keep_memmap_for_multi_worker_small_payload(workflow):
    path = _scratch_file("figures-memmap-auto")
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow=workflow,
            workers=4,
            estimated_json_payload_bytes=1024,
            json_task_count=4,
        )

        assert plan.json_worker_backend == "memmap"
        assert "multi-worker workload" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_explicit_reader_and_worker_overrides_auto_heuristics():
    path = _scratch_file("explicit-auto", size=128)
    try:
        eager_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            reader_request="eager",
            workers=4,
        )
        lazy_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            reader_request="lazy",
            chunks_request="none",
            workers=1,
        )
        memmap_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            json_worker_request="memmap",
            workers=1,
        )
        serial_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            json_worker_request="serial",
            workers=8,
        )

        assert eager_plan.reader == "eager"
        assert lazy_plan.reader == "lazy"
        assert lazy_plan.chunks is None
        assert memmap_plan.json_worker_backend == "memmap"
        assert serial_plan.json_worker_backend == "serial"
    finally:
        path.unlink(missing_ok=True)


def test_explicit_chunks_with_eager_reader_raise_clear_error():
    path = _scratch_file("bad-chunks")
    try:
        with pytest.raises(ValueError, match=r"--chunks.*--reader lazy") as exc_info:
            resolve_wrf_execution_plan(
                paths=[path],
                workflow="json",
                reader_request="eager",
                chunks_request="Time=1",
            )
        assert "--chunks" in str(exc_info.value)
        assert "--reader lazy" in str(exc_info.value)
    finally:
        path.unlink(missing_ok=True)


def test_explicit_chunks_without_dask_raise_clear_error_for_lazy_reader():
    path = _scratch_file("bad-dask-chunks")
    try:
        with pytest.raises(ValueError, match="dask-backed xarray chunking"):
            resolve_wrf_execution_plan(
                paths=[path],
                workflow="json",
                reader_request="lazy",
                chunks_request="Time=1",
                chunking_available=False,
            )
    finally:
        path.unlink(missing_ok=True)


def test_serial_worker_backend_with_tmp_dir_raises_clear_error(tmp_path):
    path = _scratch_file("bad-serial-tmp")
    try:
        with pytest.raises(ValueError, match="--tmp-dir"):
            resolve_wrf_execution_plan(
                paths=[path],
                workflow="json",
                json_worker_request="serial",
                tmp_dir=tmp_path,
            )
    finally:
        path.unlink(missing_ok=True)


def test_auto_chunks_without_dask_falls_back_to_unchunked_lazy():
    path = _scratch_file("auto-no-dask", size=128)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            reader_request="lazy",
            chunks_request="auto",
            chunking_available=False,
        )

        assert plan.reader == "lazy"
        assert plan.chunks is None
        assert "dask unavailable" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_resolved_plan_is_deterministic():
    path = _scratch_file("deterministic")
    try:
        kwargs: dict[str, Any] = {
            "paths": [path],
            "workflow": "json",
            "workers": 4,
            "estimated_json_payload_bytes": 2048,
            "large_json_payload_threshold_bytes": 1024,
        }
        assert resolve_wrf_execution_plan(**kwargs) == resolve_wrf_execution_plan(**kwargs)
    finally:
        path.unlink(missing_ok=True)


def test_estimate_4d_working_set_bytes_matches_dimension_formula(tmp_path):
    path = tmp_path / "wrfout_d01_4d_dims.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 5)
        ds.createDimension("bottom_top", 3)
        ds.createDimension("bottom_top_stag", 4)
        ds.createDimension("south_north", 6)
        ds.createDimension("west_east", 7)

    assert estimate_4d_working_set_bytes(path) == 11 * 5 * 4 * 6 * 7 * 4


def test_estimate_4d_working_set_bytes_returns_zero_when_dims_missing(tmp_path):
    path = tmp_path / "wrfout_d01_missing_dims.nc"
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("Time", 5)
        ds.createDimension("south_north", 6)

    assert estimate_4d_working_set_bytes(path) == 0


def test_estimate_4d_working_set_bytes_returns_zero_for_unreadable_path(tmp_path):
    assert estimate_4d_working_set_bytes(tmp_path / "does-not-exist.nc") == 0


def test_wrf_geojson_auto_matches_explicit_eager_serial_on_tiny_file():
    root = Path("scratch") / f"wrf-auto-equivalence-{uuid.uuid4().hex}"
    wrf_path = root / "wrfout_d01_synthetic_cli.nc"
    explicit_json = root / "explicit-json"
    explicit_geojson = root / "explicit-geojson"
    auto_json = root / "auto-json"
    auto_geojson = root / "auto-geojson"
    root.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(wrf_path)
        runner = CliRunner()

        explicit_result = runner.invoke(
            wrf_geojson_app,
            [
                "--dataset",
                str(wrf_path),
                "-o",
                str(explicit_json),
                "-g",
                str(explicit_geojson),
                "-v",
                "T2",
                "--reader",
                "eager",
                "--chunks",
                "none",
                "--worker-backend",
                "serial",
                "--workers",
                "1",
            ],
        )
        auto_result = runner.invoke(
            wrf_geojson_app,
            [
                "--dataset",
                str(wrf_path),
                "-o",
                str(auto_json),
                "-g",
                str(auto_geojson),
                "-v",
                "T2",
                "--workers",
                "1",
            ],
        )

        assert explicit_result.exit_code == 0, explicit_result.output
        assert auto_result.exit_code == 0, auto_result.output
        assert "reader: eager" in auto_result.output
        assert "worker backend: serial" in auto_result.output
        assert _read_json(auto_json / "D01_T2_000.json") == _read_json(
            explicit_json / "D01_T2_000.json"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
