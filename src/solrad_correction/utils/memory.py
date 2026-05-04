"""Memory guardrails for tabular and sequence ML pipelines."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

SOLRAD_MAX_ARRAY_GB = float(os.environ.get("SOLRAD_MAX_ARRAY_GB", "8"))


def estimate_array_nbytes(shape: tuple[int, ...] | list[int], dtype: Any = np.float32) -> int:
    """Estimate ndarray bytes from shape/dtype without allocating."""
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
    """Fail before materializing a large ML array."""
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
    """Convert selected DataFrame columns to float32 with a preflight size check."""
    shape = (len(df), len(columns))
    assert_array_size(shape, np.float32, context=context, multiplier=2.0)
    return df.loc[:, columns].to_numpy(dtype=np.float32, copy=False)


def series_to_float32_numpy(series: pd.Series, *, context: str) -> np.ndarray:
    """Convert a Series to float32 with a preflight size check."""
    assert_array_size((len(series),), np.float32, context=context, multiplier=2.0)
    return series.to_numpy(dtype=np.float32, copy=False)
