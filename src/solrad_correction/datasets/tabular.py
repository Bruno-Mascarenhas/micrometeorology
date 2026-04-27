"""Tabular dataset for scikit-learn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path


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
        subset = df[[*feature_columns, target_column]].copy()
        if drop_na:
            subset = subset.dropna()

        features = subset[feature_columns].to_numpy().astype(np.float32)
        targets = subset[target_column].to_numpy().astype(np.float32)
        index = subset.index if isinstance(subset.index, pd.DatetimeIndex) else None

        return cls(X=features, y=targets, feature_names=list(feature_columns), index=index)

    def save(self, path: str | Path) -> None:
        """Save dataset to disk for reproducibility.

        Saves features, target, feature names, and index as NPZ + CSV.
        """
        from solrad_correction.datasets.serialization import save_tabular_dataset

        save_tabular_dataset(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TabularDataset:
        """Load a previously saved dataset."""
        from solrad_correction.datasets.serialization import load_tabular_dataset

        return load_tabular_dataset(path)
