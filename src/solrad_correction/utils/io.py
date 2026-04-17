"""I/O utilities for experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


def save_json(data: dict, path: str | Path) -> None:
    """Save a dict as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: str | Path,
    index: pd.DatetimeIndex | None = None,
) -> None:
    """Save ground truth and predictions as CSV."""
    import numpy as np  # noqa: TC002
    import pandas as pd  # noqa: TC002

    df = pd.DataFrame(
        {"y_true": np.asarray(y_true).flatten(), "y_pred": np.asarray(y_pred).flatten()}
    )
    if index is not None and len(index) == len(df):
        df.index = index
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, float_format="%.4f")
