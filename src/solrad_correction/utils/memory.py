"""Memory guardrails for tabular and sequence ML pipelines.

Sliding-window expansion multiplies a table by ``sequence_length``, so a
configuration that looks harmless can ask for an array larger than the machine
has. Every allocation of that kind is preceded by :func:`assert_array_size`,
which fails with the stage's name instead of letting the process be killed by
the OOM killer with no traceback.

``SOLRAD_MAX_ARRAY_GB`` sets the ceiling in GiB (default 8) and is read from the
environment once at import, so a workstation with more memory raises it for a
whole run rather than per call site.
"""

import os
from typing import Any

import numpy as np
import pandas as pd

SOLRAD_MAX_ARRAY_GB = float(os.environ.get("SOLRAD_MAX_ARRAY_GB", "8"))


def estimate_array_nbytes(shape: tuple[int, ...] | list[int], dtype: Any = np.float32) -> int:
    """Estimate ndarray bytes from shape/dtype without allocating.

    Parameters
    ----------
    shape:
        Symbolic array shape; an empty shape is a scalar and estimates one
        itemsize.
    dtype:
        Anything ``np.dtype`` accepts.

    Returns
    -------
    int
        Exact byte count of the array's data buffer, excluding object overhead.

    Raises
    ------
    ValueError
        If any axis is negative — the product would silently understate the
        allocation the caller is about to guard.
    """
    elements = 1
    for size in shape:
        size_int = int(size)
        if size_int < 0:
            raise ValueError(f"Invalid negative shape: {shape!r}")
        elements *= size_int
    return int(elements * np.dtype(dtype).itemsize)


def assert_array_size(
    shape: tuple[int, ...] | list[int],
    dtype: Any = np.float32,
    *,
    context: str,
    max_gb: float | None = None,
    multiplier: float = 1.0,
) -> None:
    """Fail before materializing a large ML array.

    Parameters
    ----------
    shape, dtype:
        Shape and dtype of the array about to be built.
    context:
        Phrase naming the allocation, quoted verbatim in the error so the
        message says which stage of the pipeline refused.
    max_gb:
        Ceiling in GiB; defaults to ``SOLRAD_MAX_ARRAY_GB``, read from the
        environment at import time (8 GiB when unset).
    multiplier:
        Peak-to-final ratio of the allocation. Pass above 1.0 where a
        transformation holds a temporary copy alongside the result, so the check
        guards the peak rather than the final array.

    Raises
    ------
    MemoryError
        If the estimated peak exceeds the ceiling.
    """
    limit_gb = SOLRAD_MAX_ARRAY_GB if max_gb is None else max_gb
    nbytes = int(estimate_array_nbytes(shape, dtype) * multiplier)
    limit = int(limit_gb * 1024**3)
    if nbytes > limit:
        raise MemoryError(
            f"{context} would materialize about {nbytes / 1024**3:.2f} GiB "
            f"(shape={tuple(shape)!r}, dtype={np.dtype(dtype)}, multiplier={multiplier:g}); "
            f"limit is {limit_gb:.2f} GiB. Reduce rows/features or raise SOLRAD_MAX_ARRAY_GB."
        )


def dataframe_to_float32_numpy(
    df: pd.DataFrame,
    columns: list[str],
    *,
    context: str,
) -> np.ndarray:
    """Convert selected DataFrame columns to float32 with a preflight size check.

    Returns
    -------
    np.ndarray
        ``float32``, shape ``(len(df), len(columns))``, columns in the order
        requested and in their source units.

    Raises
    ------
    MemoryError
        If the conversion would exceed the guardrail; the ``2.0`` multiplier
        covers the copy pandas makes when the frame is not already float32.
    """
    shape = (len(df), len(columns))
    assert_array_size(shape, np.float32, context=context, multiplier=2.0)
    return df.loc[:, columns].to_numpy(dtype=np.float32, copy=False)


def series_to_float32_numpy(series: pd.Series, *, context: str) -> np.ndarray:
    """Convert a Series to float32 with a preflight size check.

    Returns
    -------
    np.ndarray
        ``float32``, shape ``(len(series),)``, in the series' source unit.

    Raises
    ------
    MemoryError
        If the conversion would exceed the guardrail; the ``2.0`` multiplier
        covers the copy pandas makes when the series is not already float32.
    """
    assert_array_size((len(series),), np.float32, context=context, multiplier=2.0)
    return series.to_numpy(dtype=np.float32, copy=False)
