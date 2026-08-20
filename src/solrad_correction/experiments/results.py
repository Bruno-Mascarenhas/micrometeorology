"""Typed stage results for experiment orchestration."""

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.datasets.sequence import SequenceDataset, WindowedSequenceDataset
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.reports import ExperimentReport
from solrad_correction.models.base import BaseRegressorModel, TrainingResult

ExperimentDataset = TabularDataset | SequenceDataset | WindowedSequenceDataset


@dataclass(slots=True)
class PipelineProfile:
    """Stage timing accumulator.

    Attributes
    ----------
    stage_seconds:
        Wall-clock seconds per pipeline stage, keyed by stage name and written
        to ``profile.json`` when profiling is enabled.
    """

    stage_seconds: dict[str, float]

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Record the wall-clock duration of the wrapped block under ``name``.

        Wrapping the call site instead of taking the stage function as an
        argument keeps every stage's real argument and return types visible to
        the type checker. The elapsed time is stored even if the block raises,
        and the exception propagates unchanged.
        """
        started = time.monotonic()
        try:
            yield
        finally:
            self.stage_seconds[name] = time.monotonic() - started


@dataclass(frozen=True, slots=True)
class LoadedData:
    """Input data loaded from the configured source, before any engineering.

    Attributes
    ----------
    frame:
        Raw table as the loader returned it, indexed by timestamp when the
        source carries one.
    """

    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class FeatureFrame:
    """Feature-engineered data and resolved model input columns.

    Attributes
    ----------
    frame:
        The loaded table plus every engineered column, chronologically sorted.
    feature_cols:
        Names of the columns the model reads, target excluded.
    """

    frame: pd.DataFrame
    feature_cols: list[str]


@dataclass(frozen=True, slots=True)
class SplitFrames:
    """Chronological train/validation/test dataframes.

    Split by position in time, never shuffled by default: consecutive rows of
    an irradiance series are near-duplicates, so a random split would leak the
    test period into training.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class PreprocessedSplits:
    """Preprocessed train/validation/test frames plus fitted state.

    Attributes
    ----------
    train, val, test:
        Frames in the scaled feature/target space.
    pipeline:
        The pipeline, fitted on the training split alone and merely applied to
        the other two, whose inverse transform returns predictions to the
        original target units.
    feature_cols:
        Feature columns that survived preprocessing, which can be fewer than
        those requested if a column was dropped for missingness.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    pipeline: PreprocessingPipeline
    feature_cols: list[str]


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Datasets and evaluation payload for a model family.

    Attributes
    ----------
    train, val, test:
        Datasets of the shape the model family consumes: independent rows for a
        tabular model, sliding windows for a sequence model.
    input_size:
        Number of features ``F`` per time step, for the sequence models;
        ``None`` for tabular ones, which need no such declaration.
    y_true:
        Test targets of shape ``(N,)``, ``float32``, in scaled target units,
        taken from the built test dataset so they are row-aligned with what the
        model will predict.
    prediction_index:
        Timestamps for those same ``N`` rows, or ``None`` when the source index
        is not temporal.
    """

    train: ExperimentDataset
    val: ExperimentDataset | None
    test: ExperimentDataset
    input_size: int | None
    y_true: np.ndarray
    prediction_index: pd.DatetimeIndex | None


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    """Trained model and training metadata.

    Attributes
    ----------
    duration_seconds:
        Wall-clock time of the fit, recorded in the run metadata.
    result:
        Trained model and per-epoch history.
    """

    duration_seconds: float
    result: TrainingResult


@dataclass(frozen=True, slots=True)
class PredictionOutput:
    """Predictions in preprocessed target space.

    Attributes
    ----------
    y_true, y_pred:
        Arrays of shape ``(N,)``, ``float32``, in scaled target units and
        row-aligned with each other.
    index:
        Timestamps of those rows, or ``None`` for a non-temporal index.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    index: pd.DatetimeIndex | None


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Predictions in original target space plus computed metrics.

    Attributes
    ----------
    y_true, y_pred:
        Arrays of shape ``(N,)``, inverse-transformed back into the physical
        units of the configured target column.
    metrics:
        Regression scores computed in those same units, so they are readable as
        physical errors rather than in the scaled space the model saw.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete experiment result ready for artifact writing."""

    report: ExperimentReport
    processed: PreprocessedSplits
    datasets: DatasetBundle
    model: BaseRegressorModel
    predictions: PredictionOutput
    evaluation: EvaluationResult
