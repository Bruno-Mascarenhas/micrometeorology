"""What the publisher needs from the prepared dataset the served checkpoints trained on."""

from pathlib import Path

import numpy as np
import pandas as pd

from allsky.config import DATASET_MANIFEST_FILENAME, DATASET_SPLIT_FILENAME
from allsky.data.splits import load_split_artifact

__all__ = ["train_max_solar_elevation_deg"]


def train_max_solar_elevation_deg(dataset_dir: str | Path) -> float | None:
    """Highest solar elevation (degrees) in the training split's rows, or ``None`` without one.

    Live frames and test days above it are extrapolation: the network never
    saw a sun that high. ``None`` when the dataset directory holds no split
    or no manifest; a split artifact that is present but corrupt raises, as
    :func:`allsky.data.splits.load_split_artifact` documents.
    """
    root = Path(dataset_dir)
    splits_path = root / DATASET_SPLIT_FILENAME
    manifest_path = root / DATASET_MANIFEST_FILENAME
    if not splits_path.is_file() or not manifest_path.is_file():
        return None
    assignment = load_split_artifact(splits_path).assignment
    train_days = {day for day, split in assignment.items() if split == "train"}
    frame = pd.read_parquet(manifest_path, columns=["day_id", "solar_elevation"])
    rows = frame[frame["day_id"].astype(str).isin(train_days)]
    elevation = rows["solar_elevation"].to_numpy(dtype=np.float64)
    finite = elevation[np.isfinite(elevation)]
    return float(finite.max()) if finite.size else None
