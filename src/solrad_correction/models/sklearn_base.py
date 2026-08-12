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
        """Fit the sklearn estimator on tabular data.

        Parameters
        ----------
        train_data:
            Preprocessed training rows: features ``(N, F)`` and target ``(N,)``,
            both scaled by the fitted preprocessing pipeline.
        val_data:
            Optional validation rows. sklearn estimators fit in one closed-form
            pass, so this only produces a logged score — there is no early
            stopping to feed and no epoch at which to intervene.
        config:
            Unused: the hyperparameters were already resolved when the
            estimator was constructed, in ``from_config``.
        **kwargs:
            Unused; accepted so every model shares one ``fit`` signature.

        Returns
        -------
        TrainingResult
            The fitted model, with an empty history (there are no epochs).
        """
        _ = config, kwargs
        logger.info("Training %s on %d samples", self.name, len(train_data))
        self._estimator.fit(train_data.X, train_data.y)

        if val_data is not None:
            val_metrics = self.evaluate(val_data)
            logger.info("Validation: %s", val_metrics)

        return TrainingResult(model=self)

    def predict(self, data: TabularDataset | np.ndarray) -> np.ndarray:
        """Predict using the fitted estimator.

        Parameters
        ----------
        data:
            Tabular dataset, or a bare feature matrix of shape ``(N, F)``, in
            the same scaled feature space the estimator was fitted on.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(N,)``, ``float32``, in preprocessed target
            units.

        Notes
        -----
        scikit-learn ships no type information, so ``predict`` is untyped and
        its result has to be re-established as an array before ``.astype`` can
        be trusted to return one. ``np.asarray`` on the ndarray sklearn returns
        is the identity, so the copy ``.astype`` makes is still the only one.
        """
        x_input = data.X if hasattr(data, "X") else np.asarray(data)
        assert_array_size(x_input.shape, np.float32, context="sklearn prediction input array")
        predictions = np.asarray(self._estimator.predict(x_input))
        return predictions.astype(np.float32)

    def save(self, path: str | Path) -> None:
        """Pickle the fitted estimator to ``path`` with joblib."""
        save_sklearn_model(self._estimator, path)

    @classmethod
    def load(cls, path: str | Path) -> SklearnRegressorModel:
        """Rebuild the wrapper around an estimator unpickled from ``path``.

        The instance is created without running ``__init__``, so it holds the
        estimator exactly as it was fitted rather than a freshly constructed
        one carrying this class's default hyperparameters.
        """
        estimator = load_sklearn_model(path)
        instance = cls.__new__(cls)
        instance._estimator = estimator
        return instance
