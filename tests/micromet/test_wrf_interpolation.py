"""Synthetic tests for WRF interpolation utilities."""

from __future__ import annotations

import numpy as np
import xarray as xr

from micrometeorology.wrf.interpolation import interpolate_speed_to_height


def test_interpolate_speed_to_height_preserves_xarray_until_result_materialization():
    speed = xr.DataArray(
        np.array([[[[1.0]], [[3.0]]], [[[2.0]], [[4.0]]]], dtype=np.float32),
        dims=("Time", "bottom_top", "south_north", "west_east"),
    )
    heights = xr.DataArray(
        np.array([[[[0.0]], [[100.0]]], [[[0.0]], [[100.0]]]], dtype=np.float32),
        dims=speed.dims,
    )

    result = interpolate_speed_to_height(speed, heights, 50.0)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("Time", "south_north", "west_east")
    np.testing.assert_allclose(result.to_numpy().ravel(), [2.0, 3.0])
