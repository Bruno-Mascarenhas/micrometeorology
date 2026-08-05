"""Base class for scikit-learn-based regressors."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import RegressorMixin

from solrad_correction.config import ModelConfig
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.models.base import TabularRegressorModel, TrainingResult
from solrad_correction.utils.memory import assert_array_size
from solrad_correction.utils.serialization import load_sklearn_model, save_sklearn_model

logger = logging.getLogger(__name__)


class SklearnRegressorModel(TabularRegressorModel):
    """Wrapper for any scikit-learn regressor.

    Subclasses must set ``self._estimator`` in ``__init__``.
    """

    _estimator: RegressorMixin

    def fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Fit the sklearn estimator on tabular data."""
        _ = config, kwargs
        logger.info("Training %s on %d samples", self.name, len(train_data))
        self._estimator.fit(train_data.X, train_data.y)

        if val_data is not None:
            val_metrics = self.evaluate(val_data)
            logger.info("Validation: %s", val_metrics)

        return TrainingResult(model=self)

    def predict(self, data: TabularDataset | np.ndarray) -> np.ndarray:
        """Predict using the fitted estimator."""
        x_input = data.X if hasattr(data, "X") else np.asarray(data)
        assert_array_size(x_input.shape, np.float32, context="sklearn prediction input array")
        # scikit-learn ships no type information, so `predict` is untyped and its
        # result has to be re-established as an array before `.astype` can be
        # trusted to return one. `np.asarray` on the ndarray sklearn returns is
        # the identity, so the copy `.astype` makes is still the only one.
        predictions = np.asarray(self._estimator.predict(x_input))
        return predictions.astype(np.float32)

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
