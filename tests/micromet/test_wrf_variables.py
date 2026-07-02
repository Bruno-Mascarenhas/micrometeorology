"""Regression tests for WRF variable extraction fixes.

Covers the Phase 6 behavior bug fixes:
- rain step 0 publishes zeros instead of the cumulative total (Fix 2);
- min/max helpers must not crash on single-timestep inputs (Fix 3).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from micrometeorology.wrf.variables import (
    extract_rain_step,
    get_low_high,
    get_low_high_rain,
    get_low_high_wind,
)

# Cumulative totals with a deliberately NONZERO first frame, as after a
# restart: the accumulated-since-simulation-start value must never be
# published as a per-step increment.
_CUMULATIVE = np.array(
    [
        [[5.0, 7.0], [9.0, 11.0]],
        [[6.0, 7.5], [9.0, 14.0]],
        [[8.0, 10.0], [9.5, 20.0]],
    ],
    dtype=np.float32,
)


def _cumulative_dataarray() -> xr.DataArray:
    return xr.DataArray(_CUMULATIVE.copy(), dims=("Time", "south_north", "west_east"))


# ---------------------------------------------------------------------------
# Fix 2 — rain increments for the first frames
# ---------------------------------------------------------------------------


def test_extract_rain_step_zero_is_zeros_not_cumulative_numpy():
    step0 = extract_rain_step(_CUMULATIVE.copy(), 0)

    assert step0.shape == (2, 2)
    assert step0.dtype == np.float32
    np.testing.assert_array_equal(step0, np.zeros((2, 2), dtype=np.float32))


def test_extract_rain_step_later_steps_are_increments_numpy():
    total = _CUMULATIVE.copy()

    np.testing.assert_array_equal(extract_rain_step(total, 1), total[1] - total[0])
    np.testing.assert_array_equal(extract_rain_step(total, 2), total[2] - total[1])


def test_extract_rain_step_zero_is_zeros_not_cumulative_xarray():
    total = _cumulative_dataarray()

    step0 = extract_rain_step(total, 0)

    assert isinstance(step0, xr.DataArray)
    assert step0.shape == (2, 2)
    assert step0.dtype == np.float32
    np.testing.assert_array_equal(step0.to_numpy(), np.zeros((2, 2), dtype=np.float32))


def test_extract_rain_step_later_steps_are_increments_xarray():
    total = _cumulative_dataarray()

    step1 = extract_rain_step(total, 1)
    step2 = extract_rain_step(total, 2)

    np.testing.assert_array_equal(step1.to_numpy(), _CUMULATIVE[1] - _CUMULATIVE[0])
    np.testing.assert_array_equal(step2.to_numpy(), _CUMULATIVE[2] - _CUMULATIVE[1])


# ---------------------------------------------------------------------------
# Fix 3 — single-timestep inputs must not crash the bounds helpers
# ---------------------------------------------------------------------------


def test_get_low_high_single_timestep_uses_full_array():
    single = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    low, high = get_low_high(single)

    assert low == 1.0
    assert high == float(np.nanpercentile(single.ravel(), 98))


def test_get_low_high_single_timestep_xarray_uses_full_array():
    single = xr.DataArray(
        np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32),
        dims=("Time", "south_north", "west_east"),
    )

    low, high = get_low_high(single)

    assert low == 1.0
    assert high == pytest.approx(3.94)  # 98th percentile of the FULL array


def test_get_low_high_multi_step_still_skips_first_step():
    arr = np.array(
        [
            [[100.0, 100.0], [100.0, 100.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        dtype=np.float32,
    )

    low, high = get_low_high(arr)

    assert low == 1.0
    assert high == float(np.nanpercentile(arr[1:].ravel(), 98))


def test_get_low_high_wind_single_timestep_uses_full_arrays():
    u = np.array([[[3.0, 0.0], [0.0, 3.0]]], dtype=np.float32)
    v = np.array([[[4.0, 0.0], [0.0, 4.0]]], dtype=np.float32)

    low, high = get_low_high_wind(u, v)

    assert low == 0.0
    assert high == 5.0


def test_get_low_high_wind_single_timestep_xarray_uses_full_arrays():
    dims = ("Time", "south_north", "west_east")
    u = xr.DataArray(np.array([[[3.0, 0.0], [0.0, 3.0]]], dtype=np.float32), dims=dims)
    v = xr.DataArray(np.array([[[4.0, 0.0], [0.0, 4.0]]], dtype=np.float32), dims=dims)

    low, high = get_low_high_wind(u, v)

    assert low == 0.0
    assert high == 5.0


def test_get_low_high_rain_single_timestep_returns_zero_bounds():
    single = np.array([[[5.0, 7.0], [9.0, 11.0]]], dtype=np.float32)

    assert get_low_high_rain(single) == (0.0, 0.0)

    single_da = xr.DataArray(single, dims=("Time", "south_north", "west_east"))
    assert get_low_high_rain(single_da) == (0.0, 0.0)
