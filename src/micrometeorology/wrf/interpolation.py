"""Vertical interpolation utilities for WRF data.

Replaces ``wrf-python``'s ``interplevel`` with a fully vectorized
implementation that has no external dependency beyond NumPy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
    assert_same_dims_and_shape,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def vertical_interpolate(
    values: NDArray,
    heights: NDArray,
    target_height: float,
    axis: int = 0,
) -> NDArray:
    """Interpolate *values* from model levels to *target_height* (meters AGL).

    Parameters
    ----------
    values:
        N-D array of the field to interpolate.
    heights:
        N-D array of heights at each level (meters AGL), matching *values* shape.
    target_height:
        Desired height in meters above ground level.
    axis:
        The axis corresponding to the vertical levels (default 0).

    Returns
    -------
    NDArray
        (N-1)-D array with interpolated values.
    """
    values_arr = np.asarray(values)
    heights_arr = np.asarray(heights)
    if values_arr.shape != heights_arr.shape:
        raise ValueError("values and heights must have the same shape")

    dtype = np.result_type(values_arr.dtype, heights_arr.dtype, np.float32)
    assert_reasonable_array_size(
        values_arr.shape,
        dtype,
        context="vertical interpolation block",
        multiplier=6.0,
    )

    values = values_arr.astype(dtype, copy=False)
    heights = heights_arr.astype(dtype, copy=False)
    if values.shape != heights.shape:
        raise ValueError("values and heights must have the same shape")

    levels = values.shape[axis]

    # Move the interpolation axis to the front
    v_moved = np.moveaxis(values, axis, 0)
    h_moved = np.moveaxis(heights, axis, 0)

    # Flatten the rest of the dimensions
    n_cols = int(np.prod(v_moved.shape[1:]))
    h = h_moved.reshape(levels, n_cols)
    s = v_moved.reshape(levels, n_cols)

    # Sort by height (NaNs pushed to end)
    order = np.argsort(h, axis=0)
    h_sorted = np.take_along_axis(h, order, axis=0)
    s_sorted = np.take_along_axis(s, order, axis=0)

    valid = ~np.isnan(h_sorted) & ~np.isnan(s_sorted)
    valid_count = np.sum(valid, axis=0)

    result = np.full(n_cols, np.nan, dtype=dtype)

    # Single valid level → use that value
    single_mask = valid_count == 1
    if np.any(single_mask):
        idx_single = np.argmax(valid, axis=0)
        cols = np.where(single_mask)[0]
        result[cols] = s_sorted[idx_single[cols], cols]

    # Two or more → linear interpolation
    multi_mask = valid_count >= 2
    if np.any(multi_mask):
        cols = np.where(multi_mask)[0]
        h_m = h_sorted[:, cols]
        s_m = s_sorted[:, cols]

        greater = h_m > target_height
        any_greater = np.any(greater, axis=0)
        first_gt = np.argmax(greater, axis=0)

        lower_idx = np.where(any_greater, first_gt - 1, valid_count[cols] - 2)
        lower_idx = np.clip(lower_idx, 0, levels - 2)

        col_idx = np.arange(cols.size)
        h1 = h_m[lower_idx, col_idx]
        h2 = h_m[lower_idx + 1, col_idx]
        s1 = s_m[lower_idx, col_idx]
        s2 = s_m[lower_idx + 1, col_idx]

        denom = h2 - h1
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (target_height - h1) / denom
        frac = np.where(np.isfinite(frac), frac, 0.0)

        result[cols] = s1 + frac * (s2 - s1)

    result_shape = list(values.shape)
    result_shape.pop(axis)
    return result.reshape(result_shape)


class VerticalInterpolator:
    """Reusable vertical interpolator that prepares the height stack once.

    The pipeline calls :func:`vertical_interpolate` once per (target height x
    field) against the *same* height stack, re-sorting the heights every time.
    This class validates the heights once and, when every column is NaN-free
    and strictly increasing along the vertical axis, interpolates via a
    monotonic bracket search that is bitwise-identical to
    :func:`vertical_interpolate` while skipping the per-call ``argsort``.
    Per-target brackets are cached so interpolating several fields to the same
    height reuses them.  Whenever the fast-path preconditions do not hold
    (NaN heights, non-monotonic columns, NaN values, fewer than two levels),
    the call falls back to :func:`vertical_interpolate`, so results are always
    identical to the eager reference.

    Parameters
    ----------
    heights:
        N-D array of heights at each level (meters AGL), shared by every
        field passed to :meth:`interpolate`.
    axis:
        The axis corresponding to the vertical levels (default 1, matching
        WRF ``(time, levels, ny, nx)`` blocks).
    """

    def __init__(self, heights: NDArray, axis: int = 1) -> None:
        heights_arr = np.asarray(heights)
        self.axis = axis
        self._heights = heights_arr
        self._shape = heights_arr.shape

        h_moved = np.moveaxis(heights_arr, axis, 0)
        self._levels = h_moved.shape[0]
        self._n_cols = int(np.prod(h_moved.shape[1:]))
        self._h2d = h_moved.reshape(self._levels, self._n_cols)
        self._cols = np.arange(self._n_cols)

        self._fast_ok = (
            self._levels >= 2
            and not np.isnan(self._h2d).any()
            and bool((np.diff(self._h2d, axis=0) > 0).all())
        )
        # target height -> (lower_idx, frac, dtype) for the bracket fast path.
        self._bracket_cache: dict[float, tuple[NDArray, NDArray, np.dtype]] = {}

    def _bracket(self, target_height: float, dtype: np.dtype) -> tuple[NDArray, NDArray]:
        """Return cached ``(lower_idx, frac)`` for *target_height* in *dtype*."""
        cached = self._bracket_cache.get(target_height)
        if cached is not None and cached[2] == dtype:
            return cached[0], cached[1]

        h = self._h2d.astype(dtype, copy=False)
        greater = h > target_height
        any_greater = np.any(greater, axis=0)
        first_gt = np.argmax(greater, axis=0)

        lower_idx = np.where(any_greater, first_gt - 1, self._levels - 2)
        lower_idx = np.clip(lower_idx, 0, self._levels - 2)

        h1 = h[lower_idx, self._cols]
        h2 = h[lower_idx + 1, self._cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (target_height - h1) / (h2 - h1)
        frac = np.where(np.isfinite(frac), frac, 0.0)

        self._bracket_cache[target_height] = (lower_idx, frac, dtype)
        return lower_idx, frac

    def interpolate(self, values: NDArray, target: float) -> NDArray:
        """Interpolate *values* to *target* height (meters AGL).

        Parameters
        ----------
        values:
            N-D array of the field to interpolate, same shape as the heights
            passed to the constructor.
        target:
            Desired height in meters above ground level.

        Returns
        -------
        NDArray
            (N-1)-D array with interpolated values, bitwise-identical to
            ``vertical_interpolate(values, heights, target, axis=self.axis)``.
        """
        target = float(target)
        values_arr = np.asarray(values)
        if values_arr.shape != self._shape:
            raise ValueError("values and heights must have the same shape")

        if not self._fast_ok or np.isnan(values_arr).any():
            return vertical_interpolate(values_arr, self._heights, target, axis=self.axis)

        dtype = np.result_type(values_arr.dtype, self._heights.dtype, np.float32)
        assert_reasonable_array_size(
            values_arr.shape,
            dtype,
            context="vertical interpolation block",
            multiplier=6.0,
        )
        lower_idx, frac = self._bracket(target, dtype)

        v = values_arr.astype(dtype, copy=False)
        s = np.moveaxis(v, self.axis, 0).reshape(self._levels, self._n_cols)
        s1 = s[lower_idx, self._cols]
        s2 = s[lower_idx + 1, self._cols]
        result: NDArray = s1 + frac * (s2 - s1)

        result_shape = list(values_arr.shape)
        result_shape.pop(self.axis)
        return result.reshape(result_shape)


def _is_xarray(value: object) -> bool:
    return isinstance(value, xr.DataArray)


def _vertical_dim(value: xr.DataArray) -> str:
    return str(value.dims[1])


def _xarray_vertical_interpolate(
    values: xr.DataArray,
    heights: xr.DataArray,
    target_height: float,
) -> xr.DataArray:
    """Interpolate an xarray field along the vertical dimension.

    ``apply_ufunc`` with ``input_core_dims`` moves the level axis to the
    **last** position in the underlying NumPy array, so we pass ``axis=-1``
    to ``vertical_interpolate``.  ``vectorize=False`` keeps the call
    vectorized over the full (time, ny, nx) block; setting it to ``True``
    would dispatch one Python call per grid cell, which is catastrophically
    slow for WRF grids.
    """
    assert_same_dims_and_shape(values, heights, context="xarray vertical interpolation")
    level_dim = _vertical_dim(values)
    return cast(
        "xr.DataArray",
        xr.apply_ufunc(
            lambda value_profile, height_profile: vertical_interpolate(
                value_profile,
                height_profile,
                target_height,
                axis=-1,
            ),
            values,
            heights,
            input_core_dims=[[level_dim], [level_dim]],
            output_core_dims=[[]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[float],
        ),
    )


def interpolate_speed_to_height(
    speed_4d: Any,
    heights: Any,
    target_height: float,
) -> Any:
    """Interpolate wind speed to a target height for all time steps.

    Parameters
    ----------
    speed_4d:
        4-D array ``(time, levels, ny, nx)`` of wind speed.
    heights:
        4-D array ``(time, levels, ny, nx)`` of adjusted heights.
    target_height:
        Target height in meters AGL.

    Returns
    -------
    speed_3d:
        3-D array ``(time, ny, nx)`` with interpolated speeds.
    """
    if _is_xarray(speed_4d) or _is_xarray(heights):
        return _xarray_vertical_interpolate(speed_4d, heights, target_height)
    return vertical_interpolate(speed_4d, heights, target_height, axis=1)


def compute_weibull_k(speed_3d: NDArray) -> NDArray:
    """Compute the Weibull shape factor *k* from a time series of wind speed fields.

    Parameters
    ----------
    speed_3d:
        3-D array ``(time, ny, nx)``.  The first time step is excluded.

    Returns
    -------
    fator_k:
        2-D array ``(ny, nx)`` of Weibull k values.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.nanstd(speed_3d[1:, ...], axis=0)
        mean = np.nanmean(speed_3d[1:, ...], axis=0)
        ratio = np.where(mean > 0, std / mean, np.nan)
        fator_k = np.power(ratio, -1.086)
    return fator_k


def compute_wind_vectors_at_height(
    u_central: Any,
    v_central: Any,
    height_adjusted: Any,
    target_height: float,
    downsampling: int = 4,
) -> dict:
    """Compute wind vectors interpolated to *target_height* with down-sampling.

    Returns a dict with keys:
    - ``downsampled_angles``: wind direction angles (degrees, meteorological convention)
    - ``downsampled_magnitudes``: wind speed (m/s)
    - ``downsampled_linear_indices``: row-major linear indices for the sampled points
    """
    ny, nx = u_central.shape[2], u_central.shape[3]

    if _is_xarray(u_central) or _is_xarray(v_central) or _is_xarray(height_adjusted):
        u_all = interpolate_speed_to_height(u_central, height_adjusted, target_height)
        v_all = interpolate_speed_to_height(v_central, height_adjusted, target_height)
        time_dim = "Time" if "Time" in u_all.dims else str(u_all.dims[0])
        u_target = u_all.mean(dim=time_dim, skipna=True)
        v_target = v_all.mean(dim=time_dim, skipna=True)
        magnitude = np.hypot(u_target, v_target)
        angle = np.arctan2(u_target, v_target) * 180.0 / np.pi
        angle = angle.where(angle >= 0, angle + 360.0)

        angle_values = angle.isel(
            {
                angle.dims[-2]: slice(0, None, downsampling),
                angle.dims[-1]: slice(0, None, downsampling),
            }
        ).to_numpy()
        magnitude_values = magnitude.isel(
            {
                magnitude.dims[-2]: slice(0, None, downsampling),
                magnitude.dims[-1]: slice(0, None, downsampling),
            }
        ).to_numpy()
        i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
        angles_flat = angle_values.ravel()
        mags_flat = magnitude_values.ravel()
        i_flat = i_idx.ravel()
        j_flat = j_idx.ravel()
        valid = ~np.isnan(angles_flat)
        linear_indices = (i_flat * nx + j_flat)[valid]
        return {
            "downsampled_angles": angles_flat[valid].tolist(),
            "downsampled_magnitudes": mags_flat[valid].tolist(),
            "downsampled_linear_indices": linear_indices.tolist(),
        }

    # Vectorized interpolation for all time steps at once
    u_all = vertical_interpolate(u_central, height_adjusted, target_height, axis=1)
    v_all = vertical_interpolate(v_central, height_adjusted, target_height, axis=1)

    # Average over time ignoring NaNs
    with np.errstate(invalid="ignore"):
        u_target = np.nanmean(u_all, axis=0)
        v_target = np.nanmean(v_all, axis=0)

    magnitude = np.hypot(u_target, v_target)
    angle = np.arctan2(u_target, v_target) * 180.0 / np.pi
    angle = np.where(angle < 0, angle + 360.0, angle)

    # Fast downsampling with advanced slicing
    i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    angles_flat = angle[i_flat, j_flat]
    mags_flat = magnitude[i_flat, j_flat]

    valid = ~np.isnan(angles_flat)

    linear_indices = (i_flat * nx + j_flat)[valid]

    return {
        "downsampled_angles": angles_flat[valid].tolist(),
        "downsampled_magnitudes": mags_flat[valid].tolist(),
        "downsampled_linear_indices": linear_indices.tolist(),
    }
