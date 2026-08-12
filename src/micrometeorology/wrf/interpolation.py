"""Vertical interpolation utilities for WRF data.

``vertical_interpolate`` is the fully vectorized reference implementation
(argsort-based, NaN-robust); ``VerticalInterpolator`` prepares a height stack
once and serves the monotonic bracket fast path with automatic fallback.
"""

import logging
from collections.abc import Sequence

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
        N-D array of the field to interpolate, in the field's own unit.
    heights:
        N-D array of heights at each level (meters AGL), matching *values* shape.
    target_height:
        Desired height in meters above ground level.
    axis:
        The axis corresponding to the vertical levels (default 0).

    Returns
    -------
    NDArray
        (N-1)-D array with interpolated values, in the unit of *values* and at
        least float32. A column with no valid level comes back ``NaN``; one with
        exactly one valid level comes back that level's value, unextrapolated;
        one whose bracket collapses (``h2 == h1``) forces the fraction to zero
        and so comes back the lower level's value.

    Raises
    ------
    ValueError
        When *values* and *heights* have different shapes.
    MemoryError
        When the block would exceed the array-size ceiling.
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

    v_moved = np.moveaxis(values, axis, 0)
    h_moved = np.moveaxis(heights, axis, 0)

    n_cols = int(np.prod(v_moved.shape[1:]))
    h = h_moved.reshape(levels, n_cols)
    s = v_moved.reshape(levels, n_cols)

    # ``argsort`` places NaNs last, which is what lets the valid-count logic
    # below treat the leading rows of each column as its usable levels.
    order = np.argsort(h, axis=0)
    h_sorted = np.take_along_axis(h, order, axis=0)
    s_sorted = np.take_along_axis(s, order, axis=0)

    valid = ~np.isnan(h_sorted) & ~np.isnan(s_sorted)
    valid_count = np.sum(valid, axis=0)

    result = np.full(n_cols, np.nan, dtype=dtype)

    single_mask = valid_count == 1
    if np.any(single_mask):
        idx_single = np.argmax(valid, axis=0)
        cols = np.where(single_mask)[0]
        result[cols] = s_sorted[idx_single[cols], cols]

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

    The heights are validated once.  When every column is NaN-free and strictly
    increasing along the vertical axis, interpolation uses a monotonic bracket
    search that is bitwise-identical to :func:`vertical_interpolate` while
    skipping the per-call ``argsort``, and per-target brackets are cached so
    several fields interpolated to the same height reuse them.  Whenever the
    fast-path preconditions do not hold (NaN heights, non-monotonic columns,
    NaN values, <2 levels) the call falls back to :func:`vertical_interpolate`,
    so results are always identical to the eager reference.

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
        self._levels = heights_arr.shape[axis]

        # Strictly-increasing test in the array's own layout: comparing the
        # level-shifted slices avoids materializing a full-size np.diff array.
        above = [slice(None)] * heights_arr.ndim
        below = list(above)
        above[axis] = slice(1, None)
        below[axis] = slice(None, -1)
        self._fast_ok = (
            self._levels >= 2
            and not np.isnan(heights_arr).any()
            and bool((heights_arr[tuple(above)] > heights_arr[tuple(below)]).all())
        )
        # target height -> (lower_idx, upper_idx, frac, dtype). Both index
        # arrays are cached: recomputing ``lower_idx + 1`` per gather
        # reallocates a full-size int64 array for nothing.
        self._bracket_cache: dict[float, tuple[NDArray, NDArray, NDArray, np.dtype]] = {}

    def _bracket(self, target_height: float, dtype: np.dtype) -> tuple[NDArray, NDArray, NDArray]:
        """Return cached ``(lower_idx, upper_idx, frac)`` for *target_height* in *dtype*.

        The index arrays keep a singleton vertical axis so they gather with
        :func:`numpy.take_along_axis` straight from the block's native layout.
        """
        cached = self._bracket_cache.get(target_height)
        if cached is not None and cached[3] == dtype:
            return cached[0], cached[1], cached[2]

        h = self._heights.astype(dtype, copy=False)
        greater = h > target_height
        any_greater = np.any(greater, axis=self.axis)
        first_gt = np.argmax(greater, axis=self.axis)

        lower = np.where(any_greater, first_gt - 1, self._levels - 2)
        lower_idx = np.expand_dims(np.clip(lower, 0, self._levels - 2), self.axis)
        upper_idx = lower_idx + 1

        h1 = np.take_along_axis(h, lower_idx, self.axis)
        h2 = np.take_along_axis(h, upper_idx, self.axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = (target_height - h1) / (h2 - h1)
        frac = np.where(np.isfinite(frac), frac, 0.0)

        self._bracket_cache[target_height] = (lower_idx, upper_idx, frac, dtype)
        return lower_idx, upper_idx, frac

    def interpolate_many(self, values: NDArray, targets: Sequence[float]) -> list[NDArray]:
        """Interpolate *values* to every height in *targets* (meters AGL).

        The shape check, the NaN scan and the dtype cast run once for the whole
        field instead of once per target — the scan alone is a full pass over
        the block, and the pipeline asks for the same field at three heights.

        Parameters
        ----------
        values:
            N-D array of the field to interpolate, same shape as the heights
            passed to the constructor.
        targets:
            Heights in meters above ground level.

        Returns
        -------
        list[NDArray]
            One (N-1)-D array per target, in *targets* order, each
            bitwise-identical to :meth:`interpolate` for that target.

        Raises
        ------
        ValueError
            When *values* does not match the constructor's height shape.
        """
        values_arr = np.asarray(values)
        if values_arr.shape != self._shape:
            raise ValueError("values and heights must have the same shape")

        if not self._fast_ok or np.isnan(values_arr).any():
            return [
                vertical_interpolate(values_arr, self._heights, float(target), axis=self.axis)
                for target in targets
            ]

        dtype = np.result_type(values_arr.dtype, self._heights.dtype, np.float32)
        assert_reasonable_array_size(
            values_arr.shape,
            dtype,
            context="vertical interpolation block",
            multiplier=6.0,
        )
        v = values_arr.astype(dtype, copy=False)

        results: list[NDArray] = []
        for target in targets:
            lower_idx, upper_idx, frac = self._bracket(float(target), dtype)
            s1 = np.take_along_axis(v, lower_idx, self.axis)
            s2 = np.take_along_axis(v, upper_idx, self.axis)
            results.append(np.squeeze(s1 + frac * (s2 - s1), axis=self.axis))
        return results

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
        return self.interpolate_many(values, (target,))[0]
