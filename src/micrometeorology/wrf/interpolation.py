"""Vertical interpolation utilities for WRF data.

Replaces ``wrf-python``'s ``interplevel`` with a fully vectorized
implementation that has no external dependency beyond NumPy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

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
    values = np.asarray(values, dtype=float)
    heights = np.asarray(heights, dtype=float)
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

    result = np.full(n_cols, np.nan, dtype=float)

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


def interpolate_speed_to_height(
    speed_4d: NDArray,
    heights: NDArray,
    target_height: float,
) -> NDArray:
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
    u_central: NDArray,
    v_central: NDArray,
    height_adjusted: NDArray,
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
