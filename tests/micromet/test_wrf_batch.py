"""Synthetic tests for the WRF figure-rendering worker backend."""

from __future__ import annotations

import json
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from micrometeorology.wrf.batch import (
    FigureMemmapTask,
    FigureTask,
    MapConfig,
    run_figure_tasks,
)


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


def _figure_task(output_path: Path, value: float = 1.0) -> FigureTask:
    return FigureTask(
        lon=np.zeros((2, 2), dtype=np.float32),
        lat=np.zeros((2, 2), dtype=np.float32),
        data=np.full((2, 2), value, dtype=np.float32),
        vmin=0.0,
        vmax=1.0,
        cmap_name="viridis",
        overlay_data=None,
        overlay_levels=None,
        u=None,
        v=None,
        title=f"frame {value}",
        output_path=str(output_path),
        map_config=MapConfig("D01", 0.0, 1.0, 0.0, 1.0, 1, 1, False, None),
        dpi=50,
        saturation=1.0,
    )


def test_figure_memmap_backend_with_provided_executor_matches_owned_pool(monkeypatch, tmp_path):
    tmp_dir = tmp_path / "tmp"

    monkeypatch.setattr(
        "micrometeorology.wrf.batch._render_figure_memmap",
        _fake_render_figure_memmap_payload,
    )

    def build_tasks(prefix: str) -> list[FigureTask]:
        return [_figure_task(tmp_path / f"{prefix}_{i}.png", value=float(i)) for i in range(2)]

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

    assert sorted(paths) == [str(tmp_path / "provided_0.png"), str(tmp_path / "provided_1.png")]
    for i in range(2):
        provided = (tmp_path / f"provided_{i}.png").read_bytes()
        owned = (tmp_path / f"owned_{i}.png").read_bytes()
        assert provided == owned


def test_figure_memmap_backend_cleans_payload_directory(monkeypatch, tmp_path):
    tmp_dir = tmp_path / "tmp"
    out_path = tmp_path / "figure.png"

    def fake_render(task):
        Path(task.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(task.output_path).write_text("ok", encoding="utf-8")
        return task.output_path

    monkeypatch.setattr("micrometeorology.wrf.batch._render_figure_memmap", fake_render)

    paths = run_figure_tasks([_figure_task(out_path)], workers=1, backend="memmap", tmp_dir=tmp_dir)

    assert paths == [str(out_path)]
    assert out_path.exists()
    assert tmp_dir.exists()
    assert not list(tmp_dir.iterdir())


def test_broken_process_pool_propagates_instead_of_being_swallowed(tmp_path):
    """A broken pool dooms every remaining task: it must raise, not log-and-drop."""

    class _BrokenPoolExecutor:
        def submit(self, *_args, **_kwargs):
            future: Future[str] = Future()
            future.set_exception(BrokenProcessPool("worker died"))
            return future

    task = _figure_task(tmp_path / "figure.png")

    with pytest.raises(BrokenProcessPool):
        run_figure_tasks(
            [task],
            workers=2,
            backend="memmap",
            tmp_dir=tmp_path / "memmap-tmp",
            executor=_BrokenPoolExecutor(),  # type: ignore[arg-type]
        )
