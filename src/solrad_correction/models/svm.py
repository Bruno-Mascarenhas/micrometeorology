"""Support Vector Regression (SVR) model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.svm import SVR

from solrad_correction.models.sklearn_base import SklearnRegressorModel

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig

logger = logging.getLogger(__name__)


class SVMRegressor(SklearnRegressorModel):
    """SVR wrapper following the project's regressor interface.

    Example::

        model = SVMRegressor(kernel="rbf", C=10.0, epsilon=0.1)
        model.fit(train_dataset)
        preds = model.predict(test_dataset)
    """

    @property
    def name(self) -> str:
        return f"SVM({self._estimator.kernel})"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,  # noqa: N803
        epsilon: float = 0.1,
        gamma: str = "scale",
    ) -> None:
        self._estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    @classmethod
    def from_config(cls, config: ModelConfig) -> SVMRegressor:
        """Create from experiment config."""
        return cls(
            kernel=config.svm_kernel,
            C=config.svm_c,
            epsilon=config.svm_epsilon,
            gamma=config.svm_gamma,
        )

    @classmethod
    def load(cls, path: str | Path) -> SVMRegressor:
        """Load a saved SVM model."""
        from solrad_correction.utils.serialization import load_sklearn_model

        estimator = load_sklearn_model(path)
        instance = cls.__new__(cls)
        instance._estimator = estimator
        return instance
