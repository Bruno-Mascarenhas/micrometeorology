"""Dataset artifact serialization for the v2 experiment layout."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from solrad_correction.datasets.sequence import WindowedSequenceDataset, WindowedSequenceDatasetMeta
from solrad_correction.datasets.tabular import TabularDataset


def save_dataset(
    dataset: object, path: str | Path, *, feature_names: list[str] | None = None
) -> None:
    """Persist a supported dataset without leaking artifact logic to callers."""
    if isinstance(dataset, TabularDataset):
        save_tabular_dataset(dataset, path)
        return
    if isinstance(dataset, WindowedSequenceDataset):
        save_windowed_sequence_dataset(dataset, path, feature_names=feature_names)
        return
    raise TypeError(f"Unsupported dataset type for serialization: {type(dataset).__name__}")


def save_tabular_dataset(dataset: TabularDataset, path: str | Path) -> None:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "data.npz", X=dataset.X, y=dataset.y)
    pd.DataFrame({"feature_names": dataset.feature_names}).to_csv(
        p / "feature_names.csv",
        index=False,
    )
    if dataset.index is not None:
        pd.Series(dataset.index).to_csv(p / "index.csv", index=False)


def load_tabular_dataset(path: str | Path) -> TabularDataset:
    p = Path(path)
    data = np.load(p / "data.npz")
    meta = pd.read_csv(p / "feature_names.csv")
    index = _load_optional_index(p / "index.csv")
    return TabularDataset(
        X=data["X"],
        y=data["y"],
        feature_names=meta["feature_names"].tolist(),
        index=index,
    )


def save_windowed_sequence_dataset(
    dataset: WindowedSequenceDataset,
    path: str | Path,
    *,
    feature_names: list[str] | None = None,
    index: pd.DatetimeIndex | None = None,
) -> None:
    WindowedSequenceDatasetMeta.from_dataset(
        dataset,
        feature_names=feature_names or [],
        index=index,
    ).save(path)


def load_windowed_sequence_dataset(path: str | Path) -> WindowedSequenceDataset:
    return WindowedSequenceDatasetMeta.load(path).to_torch_dataset()


def _load_optional_index(path: Path) -> pd.DatetimeIndex | None:
    if not path.exists():
        return None
    idx_df = pd.read_csv(path)
    return pd.DatetimeIndex(pd.to_datetime(idx_df.iloc[:, 0]))
