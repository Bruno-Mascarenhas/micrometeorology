"""Memory-safety guards for large array operations.

The ceiling is per-array, not per-process, and comes from the
``LABMIM_MAX_ARRAY_GB`` environment variable (16 GiB when unset) so a workstation
run and a container run can differ without a code change.
"""

import os
from typing import Any

import numpy as np

DEFAULT_MAX_ARRAY_GB = float(os.environ.get("LABMIM_MAX_ARRAY_GB", "16"))


def estimate_array_nbytes(shape: tuple[int, ...] | list[int], dtype: Any = np.float64) -> int:
    """Estimate ndarray bytes from shape and dtype without allocating.

    Parameters
    ----------
    shape:
        Symbolic shape of the array that would be allocated, e.g. ``(T, ny, nx)``.
    dtype:
        Anything ``numpy.dtype`` accepts; only its item size is used.

    Returns
    -------
    int
        Byte count of a dense array with that shape and dtype.

    Raises
    ------
    ValueError
        When any dimension is negative.
    """
    itemsize = np.dtype(dtype).itemsize
    elements = 1
    for dimension in shape:
        length = int(dimension)
        if length < 0:
            raise ValueError(f"Invalid negative dimension size: {shape!r}")
        elements *= length
    return int(elements * itemsize)


def assert_reasonable_array_size(
    shape: tuple[int, ...] | list[int],
    dtype: Any = np.float64,
    *,
    max_gb: float = DEFAULT_MAX_ARRAY_GB,
    context: str = "array operation",
    multiplier: float = 1.0,
) -> None:
    """Fail before an operation would allocate an unreasonable array.

    Parameters
    ----------
    shape:
        Symbolic shape of the array the operation would allocate.
    dtype:
        Element type of that array.
    max_gb:
        Ceiling in GiB, defaulting to :data:`DEFAULT_MAX_ARRAY_GB`.
    context:
        Phrase naming the operation, quoted verbatim in the error message.
    multiplier:
        How many copies of the base array the operation holds at peak, so a
        caller that materializes intermediates passes more than 1.0.

    Raises
    ------
    MemoryError
        When ``shape x dtype x multiplier`` exceeds *max_gb*.
    ValueError
        When any dimension is negative.
    """
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
