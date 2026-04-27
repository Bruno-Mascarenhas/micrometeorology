"""Data loaders for solrad_correction experiments.

The experiment runner uses ``load_table`` for preprocessed tabular inputs.
It supports CSV and Parquet with format detection, column projection, date
index parsing, dtype hints, and development row limits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

logger = logging.getLogger(__name__)

DataFormat = Literal["auto", "csv", "parquet"]


def detect_table_format(path: str | Path, requested: DataFormat = "auto") -> Literal["csv", "parquet"]:
    """Resolve a table format from an explicit value or file suffix."""
    if requested != "auto":
        if requested not in {"csv", "parquet"}:
            raise ValueError("data format must be one of: auto, csv, parquet")
        return requested

    suffix = Path(path).suffix.lower()
    if suffix in {".csv", ".txt"}:
        return "csv"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(
        f"Could not detect tabular data format from suffix '{suffix}'. "
        "Set data.source_format to 'csv' or 'parquet'."
    )


def load_table(
    path: str | Path,
    *,
    source_format: DataFormat = "auto",
    columns: list[str] | None = None,
    datetime_column: str | int | None = 0,
    datetime_index: bool = True,
    dtype_map: dict[str, str] | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Load a CSV or Parquet table with projection and lightweight typing.

    Parameters
    ----------
    path:
        Input table path. No paths under ``data/`` are touched by tests.
    source_format:
        ``auto``, ``csv``, or ``parquet``.
    columns:
        Data columns to project. The datetime column is included automatically
        when it is a named column.
    datetime_column:
        Name or zero-based CSV column position used as the datetime index.
        ``None`` leaves the source index/columns unchanged.
    datetime_index:
        Whether to set/parse ``datetime_column`` as the DataFrame index.
    dtype_map:
        Optional pandas dtype mapping for projected data columns.
    limit_rows:
        Optional positive row cap applied during read when supported.
    """
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be positive when set")

    fmt = detect_table_format(path, source_format)
    p = Path(path)
    dtype_map = dtype_map or {}
    projected = list(dict.fromkeys(columns or []))

    if fmt == "csv":
        df = _read_csv_table(
            p,
            columns=projected or None,
            datetime_column=datetime_column,
            datetime_index=datetime_index,
            dtype_map=dtype_map,
            limit_rows=limit_rows,
        )
    else:
        df = _read_parquet_table(
            p,
            columns=projected or None,
            datetime_column=datetime_column,
            datetime_index=datetime_index,
            limit_rows=limit_rows,
        )
        if dtype_map:
            applicable = {k: v for k, v in dtype_map.items() if k in df.columns}
            if applicable:
                df = df.astype(applicable)

    logger.info("Loaded %s data: %d rows, %d cols", fmt, len(df), len(df.columns))
    return df


def _read_csv_table(
    path: Path,
    *,
    columns: list[str] | None,
    datetime_column: str | int | None,
    datetime_index: bool,
    dtype_map: dict[str, str],
    limit_rows: int | None,
) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    index_col: str | int | None = None
    parse_dates: list[str] | None = None
    usecols = columns

    resolved_datetime_column = _resolve_datetime_column(header.columns.tolist(), datetime_column)
    if datetime_index and resolved_datetime_column is not None:
        index_col = resolved_datetime_column
        parse_dates = [resolved_datetime_column]
        if columns is not None:
            usecols = [resolved_datetime_column, *columns]

    csv_dtype = {k: v for k, v in dtype_map.items() if columns is None or k in columns} or None
    kwargs: dict[str, Any] = {
        "usecols": usecols,
        "dtype": csv_dtype,
        "parse_dates": parse_dates,
        "index_col": index_col,
        "nrows": limit_rows,
    }
    return cast("pd.DataFrame", pd.read_csv(path, **kwargs))


def _read_parquet_table(
    path: Path,
    *,
    columns: list[str] | None,
    datetime_column: str | int | None,
    datetime_index: bool,
    limit_rows: int | None,
) -> pd.DataFrame:
    parquet_columns = columns
    if datetime_index and isinstance(datetime_column, str) and columns is not None:
        parquet_columns = [datetime_column, *columns]

    df = pd.read_parquet(path, columns=parquet_columns)
    if limit_rows is not None:
        df = df.iloc[:limit_rows].copy()

    if datetime_index and datetime_column is not None:
        if isinstance(datetime_column, int):
            if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            else:
                name = df.columns[datetime_column]
                df[name] = pd.to_datetime(df[name])
                df = df.set_index(name)
        elif datetime_column in df.columns:
            df[datetime_column] = pd.to_datetime(df[datetime_column])
            df = df.set_index(datetime_column)
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"datetime_column '{datetime_column}' not found in parquet table")

    return df


def _resolve_datetime_column(columns: list[str], datetime_column: str | int | None) -> str | None:
    if datetime_column is None:
        return None
    if isinstance(datetime_column, int):
        try:
            return columns[datetime_column]
        except IndexError as exc:
            raise ValueError(f"datetime_column index {datetime_column} is out of range") from exc
    if datetime_column not in columns:
        raise ValueError(f"datetime_column '{datetime_column}' not found in CSV header")
    return datetime_column


def load_sensor_raw(
    data_dir: str | Path,
    *,
    pattern: str = "*.dat",
    calibrations_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load raw sensor data using the micrometeorology ingestion pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``.dat`` files.
    pattern:
        Glob pattern for file selection.
    calibrations_path:
        Path to calibrations YAML.  If provided, calibrations are applied.
    """
    from micrometeorology.common.paths import find_files
    from micrometeorology.sensors.ingestion import merge_dat_files

    files = find_files(data_dir, pattern)
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")

    df = merge_dat_files(files)  # type: ignore

    if calibrations_path and Path(calibrations_path).exists():
        from micrometeorology.sensors.calibration import (
            apply_calibrations,
            load_calibrations,
        )

        cals = load_calibrations(calibrations_path)
        df = apply_calibrations(df, cals)

    logger.info("Loaded raw sensor data: %d rows, %d cols", len(df), len(df.columns))
    return df


def load_sensor_hourly(
    path: str | Path,
    *,
    source_format: DataFormat = "auto",
    columns: list[str] | None = None,
    datetime_column: str | int | None = 0,
    datetime_index: bool = True,
    dtype_map: dict[str, str] | None = None,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    """Load preprocessed hourly sensor data from CSV or Parquet.

    CSV inputs preserve the historical default of using the first column as
    the datetime index.
    """
    return load_table(
        path,
        source_format=source_format,
        columns=columns,
        datetime_column=datetime_column,
        datetime_index=datetime_index,
        dtype_map=dtype_map,
        limit_rows=limit_rows,
    )


def load_wrf_series(
    wrf_files: list[str | Path],
    lat: float,
    lon: float,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Extract WRF point time-series at (lat, lon).

    Delegates to ``micrometeorology.wrf.series.extract_point_series``.
    """
    from micrometeorology.wrf.series import extract_point_series

    paths = [Path(f) for f in wrf_files]
    df = extract_point_series(paths, lat, lon, variables)
    logger.info("Loaded WRF series: %d rows at (%.4f, %.4f)", len(df), lat, lon)
    return df
