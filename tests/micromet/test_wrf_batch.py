"""Synthetic tests for WRF batch worker backends."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from micrometeorology.wrf.batch import JsonTask, run_json_tasks


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_json_memmap_backend_matches_pickle_backend():
    root = Path("scratch") / f"json-backend-equivalence-{uuid.uuid4().hex}"
    pickle_out = root / "pickle.json"
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
            output_path=str(pickle_out),
            wind_data={"downsampled_angles": [180.0]},
        )
        memmap_task = base_task._replace(output_path=str(memmap_out))

        run_json_tasks([base_task], workers=1, backend="pickle")
        run_json_tasks([memmap_task], workers=1, backend="memmap", tmp_dir=tmp_dir)

        assert _read_json(memmap_out) == _read_json(pickle_out)
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
