"""Typed stage results for experiment orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.evaluation.reports import ExperimentReport
from solrad_correction.models.base import BaseRegressorModel, TrainingResult


@dataclass(slots=True)
class PipelineProfile:
    """Stage timing accumulator."""

    stage_seconds: dict[str, float]

    def time_stage(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call ``fn`` and record its wall-clock duration under ``name``.

        The elapsed time is stored even if ``fn`` raises, and the return value
        (or exception) is propagated unchanged.
        """
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stage_seconds[name] = time.monotonic() - started


@dataclass(slots=True)
class LoadedData:
    """Input data loaded from the configured source."""

    frame: pd.DataFrame


@dataclass(slots=True)
class FeatureFrame:
    """Feature-engineered data and resolved model input columns."""

    frame: pd.DataFrame
    feature_cols: list[str]


@dataclass(slots=True)
class SplitFrames:
    """Chronological train/validation/test dataframes."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(slots=True)
class PreprocessedSplits:
    """Preprocessed train/validation/test frames plus fitted state."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    pipeline: PreprocessingPipeline
    feature_cols: list[str]


@dataclass(slots=True)
class DatasetBundle:
    """Datasets and evaluation payload for a model family."""

    train: Any
    val: Any | None
    test: Any
    input_size: int | None
    y_true: np.ndarray
    prediction_index: pd.DatetimeIndex | None


@dataclass(slots=True)
class TrainingOutput:
    """Trained model and training metadata."""

    duration_seconds: float
    result: TrainingResult


@dataclass(slots=True)
class PredictionOutput:
    """Predictions in preprocessed target space."""

    y_true: np.ndarray
    y_pred: np.ndarray
    index: pd.DatetimeIndex | None


@dataclass(slots=True)
class EvaluationResult:
    """Predictions in original target space plus computed metrics."""

    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]


@dataclass(slots=True)
class ExperimentResult:
    """Complete experiment result ready for artifact writing."""

    report: ExperimentReport
    processed: PreprocessedSplits
    datasets: DatasetBundle
    model: BaseRegressorModel
    predictions: PredictionOutput
    evaluation: EvaluationResult
