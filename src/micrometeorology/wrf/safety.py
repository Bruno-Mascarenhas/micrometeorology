"""Memory-safety guards for large array operations."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

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
