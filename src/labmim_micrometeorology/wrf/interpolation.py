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
) -> NDArray:
    """Interpolate *values* from model levels to *target_height* (meters AGL).

    Parameters
    ----------
    values:
        3-D array ``(levels, ny, nx)`` of the field to interpolate.
    heights:
        3-D array ``(levels, ny, nx)`` of heights at each level (meters AGL).
    target_height:
        Desired height in meters above ground level.

    Returns
    -------
    NDArray
        2-D array ``(ny, nx)`` with interpolated values.

    Notes
    -----
    The algorithm is fully vectorized: it sorts levels by height at each
    column, locates the two levels bracketing ``target_height``, and
    performs linear interpolation.  Columns with fewer than 2 valid levels
    use the single available value (nearest-neighbour fallback).
    """
    values = np.asarray(values, dtype=float)
    heights = np.asarray(heights, dtype=float)
    if values.ndim != 3 or heights.ndim != 3:
        raise ValueError("values and heights must be 3-D arrays (levels, ny, nx)")

    levels, ny, nx = values.shape
    n_cols = ny * nx

    h = heights.reshape(levels, n_cols)
    s = values.reshape(levels, n_cols)

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

    return result.reshape(ny, nx)


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
    nt = speed_4d.shape[0]
    ny, nx = speed_4d.shape[2], speed_4d.shape[3]
    speed_3d = np.empty((nt, ny, nx), dtype=float)

    for t in range(nt):
        speed_3d[t, :, :] = vertical_interpolate(
            speed_4d[t, :, :, :], heights[t, :, :, :], target_height
        )

    return speed_3d


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
    nt, _, ny, nx = u_central.shape

    # Running accumulator — avoids allocating stacked arrays each iteration
    u_sum = np.zeros((ny, nx), dtype=np.float64)
    v_sum = np.zeros((ny, nx), dtype=np.float64)
    count = np.zeros((ny, nx), dtype=np.int32)

    for t in range(nt):
        u_t = vertical_interpolate(u_central[t], height_adjusted[t], target_height)
        v_t = vertical_interpolate(v_central[t], height_adjusted[t], target_height)
        valid = ~np.isnan(u_t)
        u_sum[valid] += u_t[valid]
        v_sum[valid] += v_t[valid]
        count[valid] += 1

    # Avoid division by zero
    with np.errstate(invalid="ignore"):
        u_target = np.where(count > 0, u_sum / count, np.nan)
        v_target = np.where(count > 0, v_sum / count, np.nan)

    magnitude = np.hypot(u_target, v_target)
    angle = np.arctan2(u_target, v_target) * 180.0 / np.pi
    angle = np.where(angle < 0, angle + 360.0, angle)

    ds_angles: list[float] = []
    ds_magnitudes: list[float] = []
    ds_indices: list[int] = []

    for i in range(0, ny, downsampling):
        for j in range(0, nx, downsampling):
            if not np.isnan(angle[i, j]):
                ds_indices.append(int(i * nx + j))
                ds_angles.append(float(angle[i, j]))
                ds_magnitudes.append(float(magnitude[i, j]))

    return {
        "downsampled_angles": ds_angles,
        "downsampled_magnitudes": ds_magnitudes,
        "downsampled_linear_indices": ds_indices,
    }
