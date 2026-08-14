"""Support Vector Regression (SVR) model."""

import logging
from pathlib import Path

from sklearn.svm import SVR

from solrad_correction.config import ModelConfig
from solrad_correction.models.sklearn_base import SklearnRegressorModel

logger = logging.getLogger(__name__)


class SVMRegressor(SklearnRegressorModel):
    """SVR wrapper following the project's regressor interface.

    Parameters
    ----------
    kernel:
        Kernel passed to :class:`sklearn.svm.SVR` (``"rbf"``, ``"linear"``,
        ``"poly"``, ``"sigmoid"``).
    C:
        Regularization strength: larger values buy a tighter fit at the cost of
        a wider margin. Spelled uppercase because it is sklearn's own
        parameter name.
    epsilon:
        Half-width of the tube inside which errors cost nothing, in the units
        the target arrives in — the preprocessed target space, not the
        original physical one, so its meaning depends on the configured
        scaler.
    gamma:
        Kernel coefficient for the non-linear kernels.

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
        C: float = 1.0,  # noqa: N803 - scikit-learn's own SVR parameter name, part of this public API
        epsilon: float = 0.1,
        gamma: str = "scale",
    ) -> None:
        self._estimator = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    @classmethod
    def from_config(cls, config: ModelConfig) -> SVMRegressor:
        """Create an unfitted regressor from the ``svm_*`` fields of a config."""
        return cls(
            kernel=config.svm_kernel,
            C=config.svm_c,
            epsilon=config.svm_epsilon,
            gamma=config.svm_gamma,
        )

    @classmethod
    def load(cls, path: str | Path) -> SVMRegressor:
        """Load a saved SVM model, keeping the estimator exactly as fitted.

        The instance is built without ``__init__`` so the unpickled estimator
        is not replaced by a fresh one carrying the default hyperparameters.
        """
        from solrad_correction.utils.serialization import load_sklearn_model

        estimator = load_sklearn_model(path)
        instance = cls.__new__(cls)
        instance._estimator = estimator
        return instance
