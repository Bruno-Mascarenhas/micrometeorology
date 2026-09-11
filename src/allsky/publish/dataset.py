"""What the publisher needs from the prepared dataset the served checkpoints trained on."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["train_max_solar_elevation_deg"]

MANIFEST_FILENAME = "manifest.parquet"
SPLITS_FILENAME = "splits.json"


def train_max_solar_elevation_deg(dataset_dir: str | Path) -> float | None:
    """Highest solar elevation (degrees) in the training split's rows, or ``None`` without one.

    Live frames and test days above it are extrapolation: the network never
    saw a sun that high.
    """
    root = Path(dataset_dir)
    splits_path, manifest_path = root / SPLITS_FILENAME, root / MANIFEST_FILENAME
    if not splits_path.is_file() or not manifest_path.is_file():
        return None
    splits: Any = json.loads(splits_path.read_text(encoding="utf-8"))
    assignment = splits.get("assignment") if isinstance(splits, dict) else None
    if not isinstance(assignment, dict):
        return None
    train_days = {day for day, split in assignment.items() if split == "train"}
    frame = pd.read_parquet(manifest_path, columns=["day_id", "solar_elevation"])
    rows = frame[frame["day_id"].astype(str).isin(train_days)]
    elevation = rows["solar_elevation"].to_numpy(dtype=np.float64)
    finite = elevation[np.isfinite(elevation)]
    return float(finite.max()) if finite.size else None
