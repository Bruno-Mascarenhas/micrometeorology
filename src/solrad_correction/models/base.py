"""Abstract base class for all regression models.

Every model in the project (SVM, LSTM, Transformer, future additions)
inherits from ``BaseRegressorModel`` to guarantee a consistent interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.sequence import SequenceDataset, WindowedSequenceDataset
    from solrad_correction.datasets.tabular import TabularDataset
    from solrad_correction.evaluation.metrics import MetricFn


@dataclass
class TrainingResult:
    """Result of a model training session."""

    model: BaseRegressorModel
    history: dict[str, list[float]] = field(default_factory=dict)


class BaseRegressorModel(ABC):
    """Unified interface for all regressors: sklearn and PyTorch alike."""

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
        **kwargs: Any,
    ) -> TrainingResult:
        """Train the model.

        Parameters
        ----------
        train_data:
            Training data (TabularDataset or SequenceDataset).
        val_data:
            Optional validation data for early stopping / monitoring.
        config:
            Model configuration overrides.
        **kwargs:
            Additional training arguments (like runtime config).
        """

    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Generate predictions.

        Returns a 1-D array of predicted values.
        """

    def evaluate(
        self,
        data: Any,
        metrics: dict[str, MetricFn] | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a dataset.

        Default implementation: predict then compute metrics.

        ``target_values()`` takes precedence over ``y`` so that windowed
        datasets compare predictions against the window-aligned targets
        instead of the full-length base target vector.
        """
        from solrad_correction.evaluation.metrics import REGRESSION_METRICS

        if metrics is None:
            metrics = REGRESSION_METRICS

        y_pred = self.predict(data)

        target_values = getattr(data, "target_values", None)
        if callable(target_values):
            y_true = np.asarray(target_values()).flatten()
        elif hasattr(data, "y"):
            y_true = np.asarray(data.y).flatten()
        else:
            raise TypeError(
                f"Data of type {type(data).__name__} does not expose y or target_values()"
            )

        if y_true.shape[0] != y_pred.shape[0]:
            raise ValueError(
                f"Targets ({y_true.shape[0]}) and predictions ({y_pred.shape[0]}) have "
                "different lengths; dataset targets are misaligned with model outputs"
            )

        return {name: fn(y_true, y_pred) for name, fn in metrics.items()}

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseRegressorModel:
        """Load model from disk."""


class TabularRegressorModel(BaseRegressorModel):
    """Strictly typed interface for tabular models."""

    @abstractmethod
    def fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train the model on tabular data."""

    @abstractmethod
    def predict(self, data: TabularDataset | np.ndarray) -> np.ndarray:
        """Predict on tabular data."""


class SequenceRegressorModel(BaseRegressorModel):
    """Strictly typed interface for sequence models."""

    @abstractmethod
    def fit(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train the model on sequential data."""

    @abstractmethod
    def predict(self, data: SequenceDataset | WindowedSequenceDataset | np.ndarray) -> np.ndarray:
        """Predict on sequential data."""
