"""Tests for adaptive WRF execution planning."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any, cast

import pytest
from click.testing import CliRunner

from micrometeorology.cli.export_wrf_geojson import main as wrf_geojson_main
from micrometeorology.wrf.execution import resolve_wrf_execution_plan
from tests.micromet.test_wrf_reader import _write_tiny_wrf_file


def _scratch_file(name: str, size: int = 16) -> Path:
    root = Path("scratch")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}-{uuid.uuid4().hex}.nc"
    with open(path, "wb") as f:
        f.write(b"0" * size)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def test_auto_resolves_tiny_workload_to_eager_pickle():
    path = _scratch_file("tiny-auto")
    try:
        plan = resolve_wrf_execution_plan(paths=[path], workflow="json", workers=1)

        assert plan.reader == "eager"
        assert plan.chunks is None
        assert plan.json_worker_backend == "serial"
        assert "small input" in plan.reason
        assert "single worker" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_resolves_large_input_to_lazy_auto_chunks():
    path = _scratch_file("large-auto", size=128)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=2,
            large_file_threshold_bytes=64,
            chunking_available=True,
        )

        assert plan.reader == "lazy"
        assert plan.chunks == "auto"
        assert "large input" in plan.reason
    finally:
        path.unlink(missing_ok=True)


def test_auto_resolves_large_payload_to_memmap_with_multiple_workers():
    path = _scratch_file("payload-auto")
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            workers=4,
            estimated_json_payload_bytes=2048,
            json_task_count=4,
            large_json_payload_threshold_bytes=1024,
        )

        assert plan.json_worker_backend == "memmap"
        assert "large estimated JSON payload" in plan.reason
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
            large_file_threshold_bytes=64,
        )
        lazy_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            reader_request="lazy",
            chunks_request="none",
            workers=1,
        )
        pickle_plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            json_worker_request="pickle",
            workers=4,
            estimated_json_payload_bytes=4096,
            large_json_payload_threshold_bytes=1024,
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
        assert pickle_plan.json_worker_backend == "pickle"
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


def test_auto_chunks_without_dask_falls_back_to_unchunked_lazy():
    path = _scratch_file("auto-no-dask", size=128)
    try:
        plan = resolve_wrf_execution_plan(
            paths=[path],
            workflow="json",
            reader_request="auto",
            chunks_request="auto",
            large_file_threshold_bytes=64,
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


def test_wrf_geojson_auto_matches_old_explicit_eager_pickle_on_tiny_file():
    root = Path("scratch") / f"wrf-auto-equivalence-{uuid.uuid4().hex}"
    wrf_path = root / "wrfout_d01_synthetic_cli.nc"
    old_json = root / "old-json"
    old_geojson = root / "old-geojson"
    auto_json = root / "auto-json"
    auto_geojson = root / "auto-geojson"
    root.mkdir(parents=True, exist_ok=True)

    try:
        _write_tiny_wrf_file(wrf_path)
        runner = CliRunner()

        old_result = runner.invoke(
            wrf_geojson_main,
            [
                "--dataset",
                str(wrf_path),
                "-o",
                str(old_json),
                "-g",
                str(old_geojson),
                "-v",
                "T2",
                "--reader",
                "eager",
                "--chunks",
                "none",
                "--worker-backend",
                "pickle",
                "--workers",
                "1",
            ],
        )
        auto_result = runner.invoke(
            wrf_geojson_main,
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

        assert old_result.exit_code == 0, old_result.output
        assert auto_result.exit_code == 0, auto_result.output
        assert "reader: eager" in auto_result.output
        assert "worker backend: serial" in auto_result.output
        assert _read_json(auto_json / "D01_T2_000.json") == _read_json(old_json / "D01_T2_000.json")
    finally:
        shutil.rmtree(root, ignore_errors=True)
