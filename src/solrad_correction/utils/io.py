"""I/O utilities for experiment artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict:
    """Load a JSON file whose top-level value is an object.

    Every artifact this reads back (``metrics.json``, ``manifest.json``,
    ``config.json``) is written by :func:`save_json` from a dict, so a top-level
    array or scalar means the file is not the artifact it was taken for. Saying
    so here beats returning it as a ``dict`` and failing on the first key access
    somewhere downstream.

    Raises
    ------
    TypeError
        If the file parses to something other than a JSON object.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object, got a top-level {type(data).__name__}")
    return data


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    index: pd.DatetimeIndex | None = None,
) -> None:
    """Save ground truth and predictions as CSV.

    Raises
    ------
    ValueError
        If ``index`` is provided but its length does not match the predictions;
        silently dropping the index would produce untrackable artifacts.
    """
    df = pd.DataFrame(
        {"y_true": np.asarray(y_true).flatten(), "y_pred": np.asarray(y_pred).flatten()}
    )
    if index is not None:
        if len(index) != len(df):
            raise ValueError(
                f"predictions index length ({len(index)}) does not match "
                f"number of prediction rows ({len(df)})"
            )
        df.index = index
        if df.index.name is None:
            df.index.name = "timestamp"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, float_format="%.4f")
