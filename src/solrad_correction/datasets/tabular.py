"""Tabular dataset for scikit-learn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class TabularDataset:
    """Holds feature matrix X, target vector y, and metadata.

    Designed for sklearn-style models where each row is independent.
    """

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    index: pd.DatetimeIndex | None = None

    def __len__(self) -> int:
        return len(self.X)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        drop_na: bool = True,
    ) -> TabularDataset:
        """Create a dataset from a DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame with DatetimeIndex.
        feature_columns:
            Names of feature columns.
        target_column:
            Name of the target column.
        drop_na:
            If True, drop rows with any NaN in features or target.
        """
        subset = df[feature_columns + [target_column]].copy()
        if drop_na:
            subset = subset.dropna()

        features = subset[feature_columns].values.astype(np.float32)
        targets = subset[target_column].values.astype(np.float32)
        index = subset.index if isinstance(subset.index, pd.DatetimeIndex) else None

        return cls(X=features, y=targets, feature_names=list(feature_columns), index=index)

    def save(self, path: str | Path) -> None:
        """Save dataset to disk for reproducibility.

        Saves features, target, feature names, and index as NPZ + CSV.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(p / "data.npz", X=self.X, y=self.y)

        # Save metadata
        meta = pd.DataFrame({"feature_names": self.feature_names})
        meta.to_csv(p / "feature_names.csv", index=False)

        if self.index is not None:
            pd.Series(self.index).to_csv(p / "index.csv", index=False)

    @classmethod
    def load(cls, path: str | Path) -> TabularDataset:
        """Load a previously saved dataset."""
        p = Path(path)
        data = np.load(p / "data.npz")
        features, targets = data["X"], data["y"]

        meta = pd.read_csv(p / "feature_names.csv")
        feature_names = meta["feature_names"].tolist()

        index = None
        idx_path = p / "index.csv"
        if idx_path.exists():
            idx_series = pd.read_csv(idx_path, squeeze=False)
            index = pd.to_datetime(idx_series.iloc[:, 0])

        return cls(X=features, y=targets, feature_names=feature_names, index=index)
