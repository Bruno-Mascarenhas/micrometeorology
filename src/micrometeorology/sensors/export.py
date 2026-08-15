"""Formatted export of processed sensor data."""

import logging
from pathlib import Path

import pandas as pd

from micrometeorology.common.paths import ensure_dir

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
        DataFrame to export (must have a naive station-local ``DatetimeIndex``).
        Values are written in whatever units the frame carries; this step
        converts nothing.
    output_path:
        Output file path. Parent directories are created.
    separator:
        Column separator.
    na_rep:
        String representation for NaN values. It is written for every missing
        sample rather than the row being dropped, so a gap in the record stays
        visible as a gap.
    float_format:
        Format string for floating point values. It rounds what is written, so
        a CSV is a presentation of the frame, not a round-trip of it.
    include_datetime_columns:
        If True, prepend ``year``, ``month``, ``day``, ``hour`` columns. They
        replace the index rather than joining it: the timestamp is then written
        only as those four columns, never also as a leading index column. The
        frame must therefore be hourly or coarser.

    Returns
    -------
    Path
        Path to the written CSV file.

    Raises
    ------
    TypeError
        If ``include_datetime_columns`` is requested but the index is not a
        ``DatetimeIndex`` (there would be no year/month/day/hour to split out).
    ValueError
        If ``include_datetime_columns`` is requested on a sub-hourly index,
        whose minute the four columns cannot carry, or on an index carrying
        ``NaT``, which four integer columns cannot represent at all.
    """
    out = Path(output_path)
    ensure_dir(out.parent)

    export_df = df.copy()

    if include_datetime_columns:
        index = export_df.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(
                "include_datetime_columns requires a DatetimeIndex; "
                f"got {type(index).__name__}. Parse the timestamp column first."
            )
        labelled = index[index.notna()]
        if len(labelled) < len(index):
            raise ValueError(
                "include_datetime_columns writes four integer date columns, which "
                "cannot represent a missing timestamp, and the index carries NaT. "
                "Drop those rows or export the index instead."
            )
        if bool((labelled != labelled.floor("h")).any()):
            raise ValueError(
                "include_datetime_columns writes only year/month/day/hour, so a "
                "sub-hourly index would lose the minute and collapse every row of "
                "the same clock hour onto one key. Aggregate to 1h, or export the "
                "index instead."
            )
        export_df.insert(0, "year", index.year)
        export_df.insert(1, "month", index.month)
        export_df.insert(2, "day", index.day)
        export_df.insert(3, "hour", index.hour)

    export_df.to_csv(
        out,
        sep=separator,
        na_rep=na_rep,
        float_format=float_format,
        index=not include_datetime_columns,
    )

    logger.info("Exported %d rows to %s", len(export_df), out)
    return out
