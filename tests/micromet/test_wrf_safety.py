"""Regression tests for WRF memory and broadcasting guardrails."""

from __future__ import annotations

import inspect
import operator
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
import xarray as xr

from micrometeorology.wrf.batch import JsonTask, run_json_tasks
from micrometeorology.wrf.interpolation import _xarray_vertical_interpolate
from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
    destagger_dataarray,
    safe_dataarray_binary_op,
)
from micrometeorology.wrf.variables import compute_adjusted_heights

if TYPE_CHECKING:
    from micrometeorology.wrf.reader import WRFReader


def test_safe_dataarray_binary_op_refuses_staggered_outer_product_dims():
    u = xr.DataArray(
        np.ones((1, 2, 3), dtype=np.float32),
        dims=("Time", "south_north", "west_east_stag"),
    )
    v = xr.DataArray(
        np.ones((1, 3, 2), dtype=np.float32),
        dims=("Time", "south_north_stag", "west_east"),
    )

    with pytest.raises(ValueError, match="Refusing automatic xarray alignment"):
        safe_dataarray_binary_op(u, v, operator.add, context="unsafe staggered add")


def test_destagger_dataarray_uses_positional_average_and_mass_grid_dim():
    raw = xr.DataArray(
        np.array([[[[0.0, 2.0, 4.0, 6.0]]]], dtype=np.float32),
        dims=("Time", "bottom_top", "south_north", "west_east_stag"),
        coords={"west_east_stag": [10, 20, 30, 40]},
    )

    centered = destagger_dataarray(
        raw,
        staggered_dim="west_east_stag",
        target_dim="west_east",
        context="test U destagger",
    )

    assert centered.dims == ("Time", "bottom_top", "south_north", "west_east")
    np.testing.assert_allclose(centered.to_numpy(), [[[[1.0, 3.0, 5.0]]]])


class _TinyWRFReader:
    path = Path("wrfout_d01_tiny")

    def __init__(self) -> None:
        self._vars = {
            "U": xr.DataArray(
                np.arange(1 * 2 * 2 * 4, dtype=np.float32).reshape(1, 2, 2, 4),
                dims=("Time", "bottom_top", "south_north", "west_east_stag"),
            ),
            "V": xr.DataArray(
                np.ones((1, 2, 3, 3), dtype=np.float32),
                dims=("Time", "bottom_top", "south_north_stag", "west_east"),
            ),
            "PH": xr.DataArray(
                np.full((1, 3, 2, 3), 9.81, dtype=np.float32),
                dims=("Time", "bottom_top_stag", "south_north", "west_east"),
            ),
            "PHB": xr.DataArray(
                np.full((1, 3, 2, 3), 9.81, dtype=np.float32),
                dims=("Time", "bottom_top_stag", "south_north", "west_east"),
            ),
            "HGT": xr.DataArray(
                np.zeros((1, 2, 3), dtype=np.float32),
                dims=("Time", "south_north", "west_east"),
            ),
        }

    def get_variable(self, name: str):
        return self._vars[name]


def test_compute_adjusted_heights_returns_explicit_mass_grid_dims():
    u, v, height, speed = compute_adjusted_heights(cast("WRFReader", _TinyWRFReader()))

    expected_dims = ("Time", "bottom_top", "south_north", "west_east")
    assert u.dims == expected_dims
    assert v.dims == expected_dims
    assert height.dims == expected_dims
    assert speed.dims == expected_dims
    assert speed.shape == (1, 2, 2, 3)


def test_memory_guard_fails_before_large_allocation():
    with pytest.raises(MemoryError, match="test allocation"):
        assert_reasonable_array_size(
            (1024, 1024),
            np.float64,
            max_gb=0.001,
            context="test allocation",
        )


def test_xarray_vertical_interpolate_source_keeps_vectorize_false():
    source = inspect.getsource(_xarray_vertical_interpolate)

    assert "vectorize=False" in source
    assert "vectorize=True" not in source


def test_pickle_backend_is_removed():
    root = Path("scratch") / f"pickle-removed-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    task1 = JsonTask(
        data=np.ones((2, 2), dtype=np.float32),
        scale_min=0.0,
        scale_max=1.0,
        date_str="01/01/2024 00:00:00",
        output_path=str(root / "out1.json"),
        wind_data=None,
    )
    task2 = task1._replace(output_path=str(root / "out2.json"))

    try:
        with pytest.raises(ValueError, match="Unknown JSON worker backend"):
            run_json_tasks([task1, task2], workers=2, backend=cast("Any", "pickle"))
    finally:
        shutil.rmtree(root, ignore_errors=True)
