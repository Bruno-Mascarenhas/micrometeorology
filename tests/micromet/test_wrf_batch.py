"""Synthetic tests for WRF batch worker backends."""

from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from micrometeorology.wrf.batch import (
    FigureMemmapTask,
    FigureTask,
    JsonTask,
    MapConfig,
    run_figure_tasks,
    run_json_tasks,
)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return cast("dict[str, Any]", json.load(f))


def _fake_render_figure_memmap_payload(task: FigureMemmapTask) -> str:
    """Picklable stand-in for ``_render_figure_memmap`` (runs in pool workers)."""
    out = Path(task.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(np.load(task.data_path, mmap_mode="r"))
    out.write_text(
        json.dumps({"title": task.title, "values": data.tolist()}),
        encoding="utf-8",
    )
    return task.output_path


def test_json_memmap_backend_matches_serial_backend():
    root = Path("scratch") / f"json-backend-equivalence-{uuid.uuid4().hex}"
    serial_out = root / "serial.json"
    memmap_out = root / "memmap.json"
    tmp_dir = root / "tmp"
    root.mkdir(parents=True, exist_ok=True)

    try:
        data = np.ma.array(
            [[1.234, 2.345], [3.456, 4.567]],
            mask=[[False, True], [False, False]],
        )
        base_task = JsonTask(
            data=data,
            scale_min=0.0,
            scale_max=5.0,
            date_str="01/01/2024 00:00:00",
            output_path=str(serial_out),
            wind_data={"downsampled_angles": [180.0]},
        )
        memmap_task = base_task._replace(output_path=str(memmap_out))

        run_json_tasks([base_task], workers=1, backend="serial")
        run_json_tasks([memmap_task], workers=1, backend="memmap", tmp_dir=tmp_dir)

        assert _read_json(memmap_out) == _read_json(serial_out)
        assert _read_json(memmap_out)["values"][1] is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_json_memmap_backend_cleans_temporary_payload_directory():
    root = Path("scratch") / f"json-memmap-cleanup-{uuid.uuid4().hex}"
    tmp_dir = root / "tmp"
    out_path = root / "values.json"
    root.mkdir(parents=True, exist_ok=True)

    try:
        task = JsonTask(
            data=np.arange(4, dtype=np.float32).reshape(2, 2),
            scale_min=0.0,
            scale_max=3.0,
            date_str="01/01/2024 00:00:00",
            output_path=str(out_path),
            wind_data=None,
        )

        run_json_tasks([task], workers=1, backend="memmap", tmp_dir=tmp_dir)

        assert out_path.exists()
        assert tmp_dir.exists()
        assert not list(tmp_dir.iterdir())
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_json_auto_backend_matches_serial_backend_for_single_worker():
    root = Path("scratch") / f"json-serial-equivalence-{uuid.uuid4().hex}"
    auto_out = root / "auto.json"
    serial_out = root / "serial.json"
    root.mkdir(parents=True, exist_ok=True)

    try:
        data = np.array([[1.234, np.nan], [3.456, 4.567]], dtype=np.float32)
        auto_task = JsonTask(
            data=data,
            scale_min=0.0,
            scale_max=5.0,
            date_str="01/01/2024 00:00:00",
            output_path=str(auto_out),
            wind_data=None,
        )
        serial_task = auto_task._replace(output_path=str(serial_out))

        run_json_tasks([auto_task], workers=1, backend="auto")
        run_json_tasks([serial_task], workers=8, backend="serial")

        assert _read_json(serial_out) == _read_json(auto_out)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_json_memmap_backend_with_provided_executor_matches_owned_pool():
    root = Path("scratch") / f"json-executor-equivalence-{uuid.uuid4().hex}"
    tmp_dir = root / "tmp"
    root.mkdir(parents=True, exist_ok=True)

    datasets = [
        np.ma.array(
            [[1.234, 2.345], [3.456, 4.567]],
            mask=[[False, True], [False, False]],
        ),
        np.array([[0.5, np.nan], [7.89, -1.25]], dtype=np.float32),
    ]
    wind_payloads: list[dict[str, Any] | None] = [{"downsampled_angles": [180.0]}, None]

    def build_tasks(prefix: str) -> list[JsonTask]:
        return [
            JsonTask(
                data=data,
                scale_min=0.0,
                scale_max=8.0,
                date_str="01/01/2024 00:00:00",
                output_path=str(root / f"{prefix}_{i}.json"),
                wind_data=wind,
            )
            for i, (data, wind) in enumerate(zip(datasets, wind_payloads, strict=True))
        ]

    try:
        run_json_tasks(build_tasks("owned"), workers=2, backend="memmap", tmp_dir=tmp_dir)
        with ProcessPoolExecutor(max_workers=2) as pool:
            run_json_tasks(
                build_tasks("provided"),
                workers=2,
                backend="memmap",
                tmp_dir=tmp_dir,
                executor=pool,
            )
            # The provided executor must remain usable (not shut down).
            assert pool.submit(len, [1, 2, 3]).result() == 3

        for i in range(len(datasets)):
            provided = (root / f"provided_{i}.json").read_bytes()
            owned = (root / f"owned_{i}.json").read_bytes()
            assert provided == owned
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_broken_process_pool_propagates_instead_of_being_swallowed(tmp_path):
    """A broken pool dooms every remaining task: it must raise, not log-and-drop."""

    class _BrokenPoolExecutor:
        def submit(self, *_args, **_kwargs):
            future: Future[str] = Future()
            future.set_exception(BrokenProcessPool("worker died"))
            return future

    task = JsonTask(
        data=np.ones((2, 2), dtype=np.float32),
        scale_min=0.0,
        scale_max=1.0,
        date_str="01/01/2024 00:00:00",
        output_path=str(tmp_path / "values.json"),
        wind_data=None,
    )

    with pytest.raises(BrokenProcessPool):
        run_json_tasks(
            [task],
            workers=2,
            backend="memmap",
            tmp_dir=tmp_path / "memmap-tmp",
            executor=_BrokenPoolExecutor(),  # type: ignore[arg-type]
        )


def test_figure_memmap_backend_with_provided_executor_matches_owned_pool(monkeypatch):
    root = Path("scratch") / f"figure-executor-equivalence-{uuid.uuid4().hex}"
    tmp_dir = root / "tmp"
    root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "micrometeorology.wrf.batch._render_figure_memmap",
        _fake_render_figure_memmap_payload,
    )

    lon = np.zeros((2, 2), dtype=np.float32)
    lat = np.zeros((2, 2), dtype=np.float32)
    mc = MapConfig("D01", 0.0, 1.0, 0.0, 1.0, 1, 1, False, None)

    def build_tasks(prefix: str) -> list[FigureTask]:
        return [
            FigureTask(
                lon=lon,
                lat=lat,
                data=np.full((2, 2), float(i), dtype=np.float32),
                vmin=0.0,
                vmax=1.0,
                cmap_name="viridis",
                overlay_data=None,
                overlay_levels=None,
                u=None,
                v=None,
                title=f"frame {i}",
                output_path=str(root / f"{prefix}_{i}.png"),
                map_config=mc,
                dpi=50,
                saturation=1.0,
            )
            for i in range(2)
        ]

    try:
        run_figure_tasks(build_tasks("owned"), workers=2, backend="memmap", tmp_dir=tmp_dir)
        with ProcessPoolExecutor(max_workers=2) as pool:
            paths = run_figure_tasks(
                build_tasks("provided"),
                workers=2,
                backend="memmap",
                tmp_dir=tmp_dir,
                executor=pool,
            )
            # The provided executor must remain usable (not shut down).
            assert pool.submit(len, [1, 2, 3]).result() == 3

        assert sorted(paths) == [str(root / "provided_0.png"), str(root / "provided_1.png")]
        for i in range(2):
            provided = (root / f"provided_{i}.png").read_bytes()
            owned = (root / f"owned_{i}.png").read_bytes()
            assert provided == owned
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_figure_memmap_backend_cleans_payload_directory(monkeypatch):
    root = Path("scratch") / f"figure-memmap-cleanup-{uuid.uuid4().hex}"
    tmp_dir = root / "tmp"
    out_path = root / "figure.png"
    root.mkdir(parents=True, exist_ok=True)

    def fake_render(task):
        Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(task.output_path).write_text("ok", encoding="utf-8")
        return task.output_path

    monkeypatch.setattr("micrometeorology.wrf.batch._render_figure_memmap", fake_render)

    try:
        lon = np.zeros((2, 2), dtype=np.float32)
        lat = np.zeros((2, 2), dtype=np.float32)
        task = FigureTask(
            lon=lon,
            lat=lat,
            data=np.ones((2, 2), dtype=np.float32),
            vmin=0.0,
            vmax=1.0,
            cmap_name="viridis",
            overlay_data=None,
            overlay_levels=None,
            u=None,
            v=None,
            title="test",
            output_path=str(out_path),
            map_config=MapConfig("D01", 0.0, 1.0, 0.0, 1.0, 1, 1, False, None),
            dpi=50,
            saturation=1.0,
        )

        paths = run_figure_tasks([task], workers=1, backend="memmap", tmp_dir=tmp_dir)

        assert paths == [str(out_path)]
        assert out_path.exists()
        assert tmp_dir.exists()
        assert not list(tmp_dir.iterdir())
    finally:
        shutil.rmtree(root, ignore_errors=True)
