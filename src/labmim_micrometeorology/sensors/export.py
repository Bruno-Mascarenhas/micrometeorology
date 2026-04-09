"""Formatted export of processed sensor data."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd  # noqa: TC002 — used at runtime

from labmim_micrometeorology.common.paths import ensure_dir

logger = logging.getLogger(__name__)


def export_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    separator: str = ",",
    na_rep: str = "nan",
    float_format: str = "%.3f",
    include_datetime_columns: bool = False,
) -> Path:
    """Export a DataFrame to CSV with standard formatting.

    Parameters
    ----------
    df:
        DataFrame to export (must have a DatetimeIndex).
    output_path:
        Output file path.
    separator:
        Column separator.
    na_rep:
        String representation for NaN values.
    float_format:
        Format string for floating point values.
    include_datetime_columns:
        If True, add ``year``, ``month``, ``day``, ``hour`` columns.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    out = Path(output_path)
    ensure_dir(out.parent)

    export_df = df.copy()

    if include_datetime_columns:
        export_df.insert(0, "year", export_df.index.year)
        export_df.insert(1, "month", export_df.index.month)
        export_df.insert(2, "day", export_df.index.day)
        export_df.insert(3, "hour", export_df.index.hour)

    export_df.to_csv(
        out,
        sep=separator,
        na_rep=na_rep,
        float_format=float_format,
        index=not include_datetime_columns,
    )

    logger.info("Exported %d rows to %s", len(export_df), out)
    return out
