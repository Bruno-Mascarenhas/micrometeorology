"""Tabular dataset for scikit-learn models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from solrad_correction.utils.memory import dataframe_to_float32_numpy, series_to_float32_numpy

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


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
        subset = df.loc[:, [*feature_columns, target_column]]
        if drop_na:
            subset = subset.dropna()

        features = dataframe_to_float32_numpy(
            subset,
            feature_columns,
            context="TabularDataset feature matrix",
        )
        targets = series_to_float32_numpy(
            subset[target_column],
            context="TabularDataset target vector",
        )
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
