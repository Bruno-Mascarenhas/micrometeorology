"""Preprocessing pipeline with train-only fitting to prevent data leakage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """Stateful preprocessing pipeline that fits only on training data.

    Ensures no leakage: scalers and imputers are fit on ``fit()`` and
    applied identically on ``transform()`` for val/test sets.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        impute_strategy: str = "drop",
        drop_na_threshold: float = 0.5,
    ) -> None:
        self.scaler_type = scaler_type
        self.impute_strategy = impute_strategy
        self.drop_na_threshold = drop_na_threshold

        # State (fitted on train data only)
        self._fitted = False
        self._columns: list[str] = []
        self._fill_values: dict[str, float] = {}
        self._mean: pd.Series | None = None
        self._std: pd.Series | None = None
        self._min: pd.Series | None = None
        self._max: pd.Series | None = None
        self._drop_cols: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, df: pd.DataFrame) -> PreprocessingPipeline:
        """Fit the pipeline on training data only.

        Learns: columns to keep, fill values, scaler parameters.
        """
        # Drop columns with too many NaNs
        na_ratio = df.isna().mean()
        self._drop_cols = list(na_ratio[na_ratio > self.drop_na_threshold].index)
        df_clean = df.drop(columns=self._drop_cols, errors="ignore")
        self._columns = list(df_clean.columns)

        # Imputation fill values (learned from train)
        if self.impute_strategy == "mean":
            self._fill_values = df_clean.mean().to_dict()
        elif self.impute_strategy == "ffill":
            self._fill_values = {}  # ffill doesn't need pre-computed values

        # Scaler parameters
        if self.scaler_type == "standard":
            self._mean = df_clean.mean()
            self._std = df_clean.std().replace(0, 1)  # avoid div by zero
        elif self.scaler_type == "minmax":
            self._min = df_clean.min()
            self._max = df_clean.max()
            diff = self._max - self._min
            diff[diff == 0] = 1
            self._max = self._min + diff  # avoid div by zero

        self._fitted = True
        logger.info(
            "Pipeline fitted: %d cols, dropped %d (>%.0f%% NaN), scaler=%s",
            len(self._columns),
            len(self._drop_cols),
            self.drop_na_threshold * 100,
            self.scaler_type,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted parameters.

        Must call ``fit()`` first.
        """
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")

        # Keep only trained columns (handle missing cols gracefully)
        available = [c for c in self._columns if c in df.columns]
        out = df[available].copy()

        # Add missing columns as NaN
        for c in self._columns:
            if c not in out.columns:
                out[c] = np.nan
        out = out[self._columns]

        # Imputation
        if self.impute_strategy == "drop":
            out = out.dropna()
        elif self.impute_strategy == "ffill":
            out = out.ffill().bfill()
        elif self.impute_strategy == "mean":
            out = out.fillna(self._fill_values)
        elif self.impute_strategy == "interpolate":
            out = out.interpolate(method="time").bfill().ffill()

        # Scaling
        if self.scaler_type == "standard" and self._mean is not None and self._std is not None:
            out = (out - self._mean) / self._std
        elif self.scaler_type == "minmax" and self._min is not None and self._max is not None:
            out = (out - self._min) / (self._max - self._min)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on df and return transformed result."""
        return self.fit(df).transform(df)

    def inverse_transform_column(self, values: np.ndarray, column: str) -> np.ndarray:
        """Inverse-transform a single column (e.g. to get predictions in original scale)."""
        if self.scaler_type == "standard" and self._mean is not None and self._std is not None:
            return values * self._std[column] + self._mean[column]
        elif self.scaler_type == "minmax" and self._min is not None and self._max is not None:
            return values * (self._max[column] - self._min[column]) + self._min[column]
        return values

    def save(self, path: str | Path) -> None:
        """Save pipeline state."""
        import joblib

        state: dict[str, Any] = {
            "scaler_type": self.scaler_type,
            "impute_strategy": self.impute_strategy,
            "drop_na_threshold": self.drop_na_threshold,
            "columns": self._columns,
            "fill_values": self._fill_values,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max,
            "drop_cols": self._drop_cols,
            "fitted": self._fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)

    @classmethod
    def load(cls, path: str | Path) -> PreprocessingPipeline:
        """Load a previously saved pipeline."""
        import joblib

        state = joblib.load(path)
        pipeline = cls(
            scaler_type=state["scaler_type"],
            impute_strategy=state["impute_strategy"],
            drop_na_threshold=state["drop_na_threshold"],
        )
        pipeline._columns = state["columns"]
        pipeline._fill_values = state["fill_values"]
        pipeline._mean = state["mean"]
        pipeline._std = state["std"]
        pipeline._min = state["min"]
        pipeline._max = state["max"]
        pipeline._drop_cols = state["drop_cols"]
        pipeline._fitted = state["fitted"]
        return pipeline
