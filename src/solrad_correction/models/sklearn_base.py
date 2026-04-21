"""Base class for scikit-learn-based regressors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.utils.serialization import load_sklearn_model, save_sklearn_model

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.base import RegressorMixin

    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.tabular import TabularDataset

logger = logging.getLogger(__name__)


class SklearnRegressorModel(BaseRegressorModel):
    """Wrapper for any scikit-learn regressor.

    Subclasses must set ``self._estimator`` in ``__init__``.
    """

    _estimator: RegressorMixin

    def fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset | None = None,
        _config: ModelConfig | None = None,
    ) -> SklearnRegressorModel:
        """Fit the sklearn estimator on tabular data."""
        logger.info("Training %s on %d samples", self.name, len(train_data))
        self._estimator.fit(train_data.X, train_data.y)

        if val_data is not None:
            val_metrics = self.evaluate(val_data)
            logger.info("Validation: %s", val_metrics)

        return self

    def predict(self, data: TabularDataset | np.ndarray) -> np.ndarray:
        """Predict using the fitted estimator."""
        x_input = data.X if hasattr(data, "X") else np.asarray(data)
        return self._estimator.predict(x_input).astype(np.float32)  # type: ignore

    def save(self, path: str | Path) -> None:
        """Save model via joblib."""
        save_sklearn_model(self._estimator, path)

    @classmethod
    def load(cls, path: str | Path) -> SklearnRegressorModel:
        """Load model via joblib."""
        estimator = load_sklearn_model(path)
        instance = cls.__new__(cls)
        instance._estimator = estimator
        return instance
