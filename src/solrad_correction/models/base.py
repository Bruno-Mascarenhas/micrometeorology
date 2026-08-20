"""Abstract base class for all regression models.

Every model in the project (SVM, LSTM, Transformer, future additions)
inherits from ``BaseRegressorModel`` to guarantee a consistent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from solrad_correction.config import ModelConfig
from solrad_correction.datasets.sequence import SequenceDataset, WindowedSequenceDataset
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.metrics import REGRESSION_METRICS, MetricFn


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Result of a model training session.

    Attributes
    ----------
    model:
        The trained model. Torch models return themselves carrying the best
        weights seen during the run, not the last epoch's.
    history:
        Per-epoch curves keyed by ``"epoch"``, ``"train_loss"`` and
        ``"val_loss"``. ``"val_loss"`` is empty without a validation set; the
        others hold one entry per epoch trained. Losses are in the preprocessed
        target units the criterion saw. Models that train in a single
        closed-form fit, such as the sklearn ones, leave the whole dict empty.
    """

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
            Training data (``TabularDataset`` or a sequence dataset), already
            preprocessed: features and target arrive in the scaled space
            the fitted pipeline produced.
        val_data:
            Optional validation data for early stopping and monitoring.
        config:
            Model configuration supplying the hyperparameters of this run.
        **kwargs:
            Additional training arguments, such as ``runtime`` (a
            ``RuntimeConfig``) and ``preprocessing_fingerprint``.

        Returns
        -------
        TrainingResult
            The trained model and its per-epoch history.
        """

    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Generate predictions for the rows of ``data`` the model evaluates.

        Parameters
        ----------
        data:
            Dataset, or a feature array, of the kind the concrete model
            accepts.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(N,)``, ``float32``, in the preprocessed
            target units the model was fitted on. ``N`` counts the rows the
            model actually predicts, which for a windowed sequence dataset is
            fewer than the rows of the frame it was built from.
        """

    def evaluate(
        self,
        data: Any,
        metrics: dict[str, MetricFn] | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a dataset: predict, then score.

        ``target_values()`` takes precedence over ``y`` so that windowed
        datasets compare predictions against the window-aligned targets
        instead of the full-length base target vector.

        Parameters
        ----------
        data:
            Dataset exposing its targets as ``target_values()`` or as ``y``,
            of shape ``(N,)``.
        metrics:
            Metric callables keyed by display name; defaults to
            :data:`~solrad_correction.evaluation.metrics.REGRESSION_METRICS`.

        Returns
        -------
        dict of str to float
            One score per metric, expressed in whatever units the dataset's
            targets carry — the preprocessed target space, unless the caller
            inverse-transformed them first.

        Raises
        ------
        TypeError
            If ``data`` exposes neither ``target_values()`` nor ``y``.
        ValueError
            If targets and predictions differ in length, which means the
            dataset rows are misaligned with the model outputs.
        """
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
        """Write the fitted model to ``path`` in the backend's own format.

        The file carries the weights and the architecture arguments needed to
        rebuild the model, but never the preprocessing: the pipeline persists
        its fitted scaler separately, under the experiment's
        ``preprocessing/`` directory.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> BaseRegressorModel:
        """Rebuild a model previously written by :meth:`save`.

        The returned instance is ready to :meth:`predict`, but it carries no
        preprocessing: callers must feed it features scaled with the same
        fitted pipeline the weights were trained under.
        """


class TabularRegressorModel(BaseRegressorModel):
    """Interface for models whose samples are independent rows."""

    @abstractmethod
    def fit(
        self,
        train_data: TabularDataset,
        val_data: TabularDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train on a feature matrix of shape ``(N, F)`` and targets ``(N,)``."""

    @abstractmethod
    def predict(self, data: TabularDataset | np.ndarray) -> np.ndarray:
        """Predict one value per row of an ``(N, F)`` feature matrix.

        Returns an array of shape ``(N,)``, ``float32``, in preprocessed
        target units.
        """


class SequenceRegressorModel(BaseRegressorModel):
    """Interface for models whose samples are windows over the time axis."""

    @abstractmethod
    def fit(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train on batches of windows ``(B, T, F)`` with targets ``(B,)``."""

    @abstractmethod
    def predict(self, data: SequenceDataset | WindowedSequenceDataset | np.ndarray) -> np.ndarray:
        """Predict one value per window.

        Returns an array of shape ``(N,)``, ``float32``, in preprocessed target
        units, where ``N`` is the number of windows the dataset yields — fewer
        than its rows, since no window ends before position ``T - 1`` and
        windows spanning a temporal gap are dropped.
        """
