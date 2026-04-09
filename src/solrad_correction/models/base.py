"""Abstract base class for all regression models.

Every model in the project (SVM, LSTM, Transformer, future additions)
inherits from ``BaseRegressorModel`` to guarantee a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig


class BaseRegressorModel(ABC):
    """Unified interface for all regressors — sklearn and PyTorch alike."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""

    @abstractmethod
    def fit(
        self,
        train_data: Any,
        val_data: Any | None = None,
        config: ModelConfig | None = None,
    ) -> BaseRegressorModel:
        """Train the model.

        Parameters
        ----------
        train_data:
            Training data (TabularDataset or SequenceDataset).
        val_data:
            Optional validation data for early stopping / monitoring.
        config:
            Model configuration overrides.
        """

    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Generate predictions.

        Returns a 1-D array of predicted values.
        """

    def evaluate(
        self,
        data: Any,
        metrics: dict[str, callable] | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Default implementation: predict then compute metrics.
        """
        from labmim_micrometeorology.stats.metrics import ALL_METRICS

        if metrics is None:
            metrics = ALL_METRICS

        y_pred = self.predict(data)

        # Extract ground truth
        if hasattr(data, "y"):
            y_true = np.asarray(data.y).flatten()
        else:
            raise ValueError("Data must have a 'y' attribute for evaluation")

        return {name: fn(y_true, y_pred) for name, fn in metrics.items()}

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseRegressorModel:
        """Load model from disk."""
