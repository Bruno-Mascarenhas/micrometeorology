"""Safety guards for large WRF/xarray array operations."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike

DEFAULT_MAX_ARRAY_GB = float(os.environ.get("LABMIM_MAX_ARRAY_GB", "16"))


def estimate_array_nbytes(shape: tuple[int, ...] | list[int], dtype: Any = np.float64) -> int:
    """Estimate ndarray bytes from shape and dtype without allocating."""
    itemsize = np.dtype(dtype).itemsize
    elements = 1
    for size in shape:
        size_int = int(size)
        if size_int < 0:
            raise ValueError(f"Invalid negative dimension size: {shape!r}")
        elements *= size_int
    return int(elements * itemsize)


def assert_reasonable_array_size(
    shape: tuple[int, ...] | list[int],
    dtype: Any = np.float64,
    *,
    max_gb: float = DEFAULT_MAX_ARRAY_GB,
    context: str = "array operation",
    multiplier: float = 1.0,
) -> None:
    """Fail before an operation would allocate an unreasonable array."""
    base_bytes = estimate_array_nbytes(shape, dtype)
    estimated_bytes = int(base_bytes * multiplier)
    limit_bytes = int(max_gb * 1024**3)
    if estimated_bytes > limit_bytes:
        estimated_gb = estimated_bytes / 1024**3
        raise MemoryError(
            f"{context} would require about {estimated_gb:.2f} GiB "
            f"(shape={tuple(shape)!r}, dtype={np.dtype(dtype)}, multiplier={multiplier:g}); "
            f"limit is {max_gb:.2f} GiB. Use lazy/chunked WRF reading or reduce the workload."
        )


def dataarray_nbytes(value: xr.DataArray, *, multiplier: float = 1.0) -> int:
    """Estimate bytes for an xarray DataArray from metadata only."""
    return int(estimate_array_nbytes(value.shape, value.dtype) * multiplier)


def assert_same_dims_and_shape(
    left: xr.DataArray,
    right: xr.DataArray,
    *,
    context: str,
) -> None:
    """Require exact xarray dimension names and sizes before binary operations."""
    if left.dims != right.dims:
        raise ValueError(
            f"{context}: incompatible dimensions {left.dims!r} vs {right.dims!r}. "
            "Refusing automatic xarray alignment because it can create an outer-product grid."
        )
    left_sizes = tuple(left.sizes[dim] for dim in left.dims)
    right_sizes = tuple(right.sizes[dim] for dim in right.dims)
    if left_sizes != right_sizes:
        raise ValueError(
            f"{context}: incompatible shapes {left_sizes!r} vs {right_sizes!r} "
            f"for dims {left.dims!r}."
        )


def _safe_coords_without_dim(value: xr.DataArray, dim: str) -> dict[str, xr.DataArray]:
    return {
        cast("str", name): coord
        for name, coord in value.coords.items()
        if dim not in coord.dims and name != dim
    }


def destagger_dataarray(
    value: xr.DataArray,
    *,
    staggered_dim: str,
    target_dim: str,
    context: str,
    max_gb: float = DEFAULT_MAX_ARRAY_GB,
) -> xr.DataArray:
    """Average adjacent staggered-grid points positionally and rename the dimension."""
    if staggered_dim not in value.dims:
        raise ValueError(f"{context}: missing staggered dimension {staggered_dim!r}")
    axis = value.dims.index(staggered_dim)
    if value.shape[axis] < 2:
        raise ValueError(f"{context}: staggered dimension {staggered_dim!r} has fewer than 2 cells")

    result_shape = list(value.shape)
    result_shape[axis] -= 1
    assert_reasonable_array_size(
        result_shape,
        value.dtype,
        max_gb=max_gb,
        context=f"{context} destagger result",
    )

    left = value.isel({staggered_dim: slice(0, -1)})
    right = value.isel({staggered_dim: slice(1, None)})
    data = (left.data + right.data) * 0.5
    dims = tuple(target_dim if dim == staggered_dim else dim for dim in value.dims)
    return xr.DataArray(
        data,
        dims=dims,
        coords=_safe_coords_without_dim(left, staggered_dim),
        attrs=value.attrs,
        name=value.name,
    )


def safe_dataarray_binary_op(
    left: xr.DataArray,
    right: xr.DataArray,
    op: Callable[[Any, Any], Any],
    *,
    context: str,
    max_gb: float = DEFAULT_MAX_ARRAY_GB,
    result_dtype: Any | None = None,
) -> xr.DataArray:
    """Apply a binary operation without allowing xarray auto-alignment."""
    assert_same_dims_and_shape(left, right, context=context)
    dtype = result_dtype or np.result_type(left.dtype, right.dtype)
    assert_reasonable_array_size(left.shape, dtype, max_gb=max_gb, context=context)
    data = op(left.data, right.data)
    return xr.DataArray(
        data,
        dims=left.dims,
        coords=left.coords,
        attrs=left.attrs,
        name=left.name,
    )


def safe_binary_op(
    left: Any,
    right: Any,
    op: Callable[[Any, Any], Any],
    *,
    context: str,
    max_gb: float = DEFAULT_MAX_ARRAY_GB,
    result_dtype: Any | None = None,
    allow_numpy_broadcast: bool = False,
) -> Any:
    """Safe binary op for DataArrays and ndarray-like values."""
    if isinstance(left, xr.DataArray) or isinstance(right, xr.DataArray):
        if not isinstance(left, xr.DataArray) or not isinstance(right, xr.DataArray):
            raise TypeError(f"{context}: mixed xarray/non-xarray binary operation is not allowed")
        return safe_dataarray_binary_op(
            left,
            right,
            op,
            context=context,
            max_gb=max_gb,
            result_dtype=result_dtype,
        )

    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    shape = np.broadcast_shapes(left_arr.shape, right_arr.shape)
    if not allow_numpy_broadcast and left_arr.shape != right_arr.shape:
        raise ValueError(
            f"{context}: incompatible ndarray shapes {left_arr.shape!r} vs {right_arr.shape!r}"
        )
    dtype = result_dtype or np.result_type(left_arr.dtype, right_arr.dtype)
    assert_reasonable_array_size(shape, dtype, max_gb=max_gb, context=context)
    return op(left_arr, right_arr)


def as_array_metadata(value: ArrayLike | xr.DataArray) -> tuple[tuple[int, ...], np.dtype]:
    """Return shape/dtype metadata without computing xarray data."""
    if isinstance(value, xr.DataArray):
        return tuple(int(size) for size in value.shape), np.dtype(value.dtype)
    arr = np.asarray(value)
    return arr.shape, arr.dtype
