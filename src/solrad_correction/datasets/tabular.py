"""Tabular dataset for scikit-learn models."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from solrad_correction.utils.memory import dataframe_to_float32_numpy, series_to_float32_numpy


@dataclass(frozen=True, slots=True)
class TabularDataset:
    """Holds feature matrix X, target vector y, and metadata.

    Designed for sklearn-style models where each row is independent.

    Attributes
    ----------
    X:
        ``float32``, shape ``(n_samples, n_features)``, columns ordered as
        ``feature_names``. Preprocessed (scaled) values, so dimensionless unless
        the experiment ran with ``scaler_type='none'``.
    y:
        ``float32``, shape ``(n_samples,)``, one target per row of ``X`` and in
        the same scaling as ``X``.
    feature_names:
        Column names of ``X``, length ``n_features``.
    index:
        Timestamps of the rows, shape ``(n_samples,)``, or ``None`` when the
        source frame was not datetime-indexed. Carried so predictions can be
        written back against the time they belong to.
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

        Returns
        -------
        TabularDataset
            ``X`` of shape ``(n_kept_rows, len(feature_columns))`` and ``y`` of
            shape ``(n_kept_rows,)``, both ``float32``, carrying ``df``'s values
            unchanged. ``index`` is the surviving timestamps, or ``None`` when
            ``df`` is not datetime-indexed.

        Raises
        ------
        KeyError
            If a requested feature or target column is absent from ``df``.
        MemoryError
            If the extracted arrays would exceed the ``SOLRAD_MAX_ARRAY_GB``
            guardrail.
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

        Writes ``path`` as a directory holding ``data.npz`` (the ``X`` and ``y``
        arrays), ``feature_names.csv`` and, when an index is set, ``index.csv``.
        Read back by :meth:`load`.
        """
        from solrad_correction.datasets.serialization import save_tabular_dataset

        save_tabular_dataset(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TabularDataset:
        """Load a dataset directory written by :meth:`save`.

        The index sidecar is optional, so a dataset saved without timestamps
        loads back with ``index=None`` rather than failing.
        """
        from solrad_correction.datasets.serialization import load_tabular_dataset

        return load_tabular_dataset(path)
