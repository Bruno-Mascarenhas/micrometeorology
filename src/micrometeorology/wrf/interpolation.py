"""Vertical interpolation utilities for WRF data.

``vertical_interpolate`` is the fully vectorized reference implementation
(argsort-based, NaN-robust); ``VerticalInterpolator`` prepares a height stack
once and serves the monotonic bracket fast path with automatic fallback.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from micrometeorology.wrf.safety import assert_reasonable_array_size

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
