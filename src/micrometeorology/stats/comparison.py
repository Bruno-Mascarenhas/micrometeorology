"""Model vs. observation comparison.

Reads, aligns and pairs observation/model datasets and reduces them to metric
tables. Pure pandas on purpose: ``labmim-metrics`` imports from here and would
otherwise pay ~0.5 s of matplotlib import on every invocation. The figure that
consumes these frames lives in :mod:`micrometeorology.stats.comparison_plots`.
"""

import logging
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

from micrometeorology.stats.metrics import compute_all, is_circular_column, valid_pairs

logger = logging.getLogger(__name__)


def read_dataset(
    path: str | Path,
    *,
    separator: str = ",",
    timestamp_columns: list[str] | None = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Read a CSV/DAT file into a DatetimeIndex DataFrame.

    Parameters
    ----------
    path:
        Path to the data file.
    separator:
        Column separator.
    timestamp_columns:
        Columns to combine into a datetime index (e.g. ``['year','month','day','hour']``).
        If ``None``, tries ``TIMESTAMP`` column or the unnamed leading index column.
    """
    df = pd.read_csv(path, sep=separator, low_memory=False)

    if timestamp_columns and all(c in df.columns for c in timestamp_columns):
        df.index = pd.to_datetime(df[timestamp_columns])
        df = df.drop(columns=timestamp_columns, errors="ignore")
    elif "TIMESTAMP" in df.columns:
        df.index = pd.to_datetime(df["TIMESTAMP"])
        df = df.drop(columns=["TIMESTAMP"], errors="ignore")
    elif parse_dates:
        date_parts = [c for c in ("year", "month", "day", "hour") if c in df.columns]
        if len(date_parts) >= 3:
            df.index = pd.to_datetime(df[date_parts])
            df = df.drop(columns=date_parts, errors="ignore")
        elif len(df.columns) > 0:
            # The leading index column ``sensors.export.export_csv`` writes, plus
            # the named one ``DataFrame.to_csv`` writes for a ``timestamp`` index.
            # Numeric leading columns are excluded on purpose: ``pd.to_datetime``
            # reads an integer RECORD/year column as nanoseconds since the epoch
            # and would fabricate a 1970 index.
            leading = df.columns[0]
            label = str(leading).strip().casefold()
            is_unnamed = str(leading).startswith("Unnamed:") or not str(leading).strip()
            named_like_a_stamp = label in {"timestamp", "datetime", "date", "time"}
            if (is_unnamed or named_like_a_stamp) and not is_numeric_dtype(df[leading]):
                parsed = pd.to_datetime(df[leading], errors="coerce")
                if parsed.notna().all():
                    df.index = parsed
                    df = df.drop(columns=[leading])

    # Deliberately NOT an error when no time index could be built: a RangeIndex
    # is a supported state here, because ``labmim-metrics`` aligns such a pair by
    # ROW ORDER and says so. Refusing it belongs to ``compare_wrf_observations``,
    # which can name the offending file.
    df.index.name = None

    # Both dtype names are needed: pandas 3 reads text as ``str`` dtype, and
    # ``include=["object"]`` matches it only through a shim due for removal.
    text_columns = df.select_dtypes(include=["object", "str"]).columns
    if len(text_columns) > 0:
        df[text_columns] = df[text_columns].apply(pd.to_numeric, errors="coerce")

    return df


def pair_dataframes(
    obs: pd.DataFrame,
    model: pd.DataFrame,
    *,
    tolerance: str = "30min",
) -> pd.DataFrame:
    """Align observation and model DataFrames by nearest timestamp.

    Returns a DataFrame with multi-level columns: ``('obs', var)`` and ``('model', var)``.
    Only common columns are included.
    """
    common_cols = sorted(set(obs.columns) & set(model.columns))
    if not common_cols:
        logger.warning("No common columns between observation and model datasets")
        return pd.DataFrame()

    if not isinstance(obs.index, pd.DatetimeIndex) or not isinstance(model.index, pd.DatetimeIndex):
        raise TypeError(
            "pair_dataframes needs a DatetimeIndex on both frames, got "
            f"{type(obs.index).__name__} (observation) and {type(model.index).__name__} (model)"
        )

    obs_subset = obs[common_cols].sort_index()
    model_subset = model[common_cols].sort_index()

    merged = pd.merge_asof(
        obs_subset.rename_axis("time").reset_index(),
        model_subset.rename_axis("time").reset_index(),
        on="time",
        tolerance=pd.Timedelta(tolerance),
        suffixes=("_obs", "_model"),
        direction="nearest",
    )
    merged = merged.set_index("time")
    return merged


def compare_variables(
    paired_df: pd.DataFrame,
    variable: str,
) -> dict[str, float]:
    """Compute all metrics for a single variable from a paired DataFrame."""
    obs_col = f"{variable}_obs"
    model_col = f"{variable}_model"

    if obs_col not in paired_df.columns or model_col not in paired_df.columns:
        logger.warning("Variable %s not found in paired DataFrame", variable)
        return {}

    obs = paired_df[obs_col].to_numpy()
    model = paired_df[model_col].to_numpy()
    circular = is_circular_column(variable)
    if circular:
        # Name-based detection changes the published numbers, so it says so.
        logger.info(
            "%s: circular quantity — residuals wrapped to [-180, 180); "
            "R2/r/d/IOA/NRMSE suppressed (an arithmetic mean of bearings is meaningless)",
            variable,
        )
    # ``n`` leads the record on purpose: it is the denominator of every number
    # after it, and it is not the row count of *paired_df*.
    return {"n": float(valid_pairs(obs, model)), **compute_all(obs, model, circular=circular)}


def compare_all_variables(paired_df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for all paired variables.

    Returns a DataFrame with metrics as rows and variables as columns.
    """
    variables = sorted(
        {
            c.replace("_obs", "")
            for c in paired_df.columns
            if c.endswith("_obs") and c.replace("_obs", "_model") in paired_df.columns
        }
    )

    results: dict[str, dict[str, float]] = {}
    for variable in variables:
        results[variable] = compare_variables(paired_df, variable)

    return pd.DataFrame(results)
