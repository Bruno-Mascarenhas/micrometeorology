"""Composable experiment pipeline stages."""

import logging
import time
from typing import Literal

import pandas as pd

from solrad_correction.config import DataConfig, ExperimentConfig
from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.data.splits import temporal_train_val_test_split
from solrad_correction.datasets.sequence import WindowedSequenceDataset
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.metrics import compute_regression_metrics
from solrad_correction.evaluation.policy import align_test_frame
from solrad_correction.evaluation.reports import ExperimentReport
from solrad_correction.experiments.artifacts import ArtifactLayout
from solrad_correction.experiments.results import (
    DatasetBundle,
    EvaluationResult,
    ExperimentResult,
    FeatureFrame,
    LoadedData,
    PipelineProfile,
    PredictionOutput,
    PreprocessedSplits,
    SplitFrames,
    TrainingOutput,
)
from solrad_correction.experiments.writer import ExperimentWriter
from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.models.registry import build_model, get_model_spec
from solrad_correction.training.dataloaders import resolve_device
from solrad_correction.utils.memory import dataframe_to_float32_numpy, series_to_float32_numpy
from solrad_correction.utils.metadata import collect_run_metadata
from solrad_correction.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)


def prepare_runtime(config: ExperimentConfig) -> None:
    """Resolve output-coupled runtime defaults in-place.

    A torch run with no configured checkpoint directory gets the one inside its
    own experiment directory, so checkpoints land beside the run's other
    artifacts instead of nowhere.
    """
    model_type = config.model.model_type.lower()
    if model_type in {"lstm", "transformer"} and config.runtime.checkpoint_dir is None:
        artifact_layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
        config.runtime.checkpoint_dir = str(artifact_layout.checkpoints_dir)


def load_data(config: ExperimentConfig) -> LoadedData:
    """Load configured input data.

    ``data.use_raw`` decides only when a config populates both input paths;
    otherwise the populated path selects the loader, so every config that
    already works keeps its branch.

    Returns
    -------
    LoadedData
        The loaded table, truncated to ``runtime.limit_rows`` when set.

    Raises
    ------
    ValueError
        If the config declares no input path at all.
    """
    _warn_about_ignored_site_coordinates(config)
    if config.data.use_raw and config.data.sensor_data_path:
        loaded_frame = _load_raw_sensor_frame(config, config.data.sensor_data_path)
    elif config.data.hourly_data_path:
        from solrad_correction.data.loaders import load_sensor_hourly

        projected_columns = config.data.load_columns or None
        if projected_columns is None and config.data.feature_columns:
            projected_columns = [*config.data.feature_columns, config.data.target_column]
        loaded_frame = load_sensor_hourly(
            config.data.hourly_data_path,
            source_format=_resolve_source_format(config.data.source_format),
            columns=projected_columns,
            datetime_column=config.data.datetime_column,
            datetime_index=config.data.datetime_index,
            dtype_map=config.data.dtype_map,
            limit_rows=config.runtime.limit_rows,
            cache_dir=config.data.cache_dir,
        )
    elif config.data.sensor_data_path:
        loaded_frame = _load_raw_sensor_frame(config, config.data.sensor_data_path)
    else:
        raise ValueError("No data path provided in config")

    return LoadedData(frame=loaded_frame)


def _resolve_source_format(source_format: str) -> Literal["auto", "csv", "parquet"]:
    """Re-establish the table format the loader declares from the config string.

    ``DataConfig.source_format`` is a plain ``str`` because YAML hands over
    nothing narrower, while ``load_sensor_hourly`` accepts only the three values
    ``detect_table_format`` knows. ``ExperimentConfig.validate`` rejects anything
    else long before a run reaches this point; the message below is the one
    ``detect_table_format`` would have raised, so a stage driven without
    ``validate()`` still fails identically.

    Raises
    ------
    ValueError
        If ``source_format`` is not one of ``auto``, ``csv`` or ``parquet``.
    """
    match source_format:
        case "auto" | "csv" | "parquet":
            return source_format
    raise ValueError("data format must be one of: auto, csv, parquet")


def _load_raw_sensor_frame(config: ExperimentConfig, sensor_data_path: str) -> pd.DataFrame:
    """Ingest the raw ``.dat`` sensor tree configured by ``data.sensor_data_path``."""
    from solrad_correction.data.loaders import load_sensor_raw

    loaded_frame = load_sensor_raw(
        sensor_data_path,
        pattern=config.data.sensor_pattern,
        calibrations_path=config.data.calibrations_path,
        resample_freq=config.data.resample_freq,
        min_samples=config.data.sensor_min_samples,
    )
    if config.runtime.limit_rows is not None:
        loaded_frame = loaded_frame.iloc[: config.runtime.limit_rows].copy()
    return loaded_frame


def _warn_about_ignored_site_coordinates(config: ExperimentConfig) -> None:
    """Say out loud that station coordinates cannot retarget a run to another site."""
    coordinate_defaults = DataConfig()
    if (config.data.station_lat, config.data.station_lon) != (
        coordinate_defaults.station_lat,
        coordinate_defaults.station_lon,
    ):
        logger.warning(
            "data.station_lat/data.station_lon are accepted for backward compatibility "
            "and no feature stage reads them; the site comes from the configured data path"
        )


def build_features(loaded: LoadedData, config: ExperimentConfig) -> FeatureFrame:
    """Apply feature engineering according to config.

    When ``data.feature_columns`` is set, the model inputs are the requested
    base columns plus every column added by the enabled feature stages
    (temporal/cyclic/lag/rolling/diff) — engineered columns are included
    explicitly because they were requested via the feature config, never by
    name-prefix accident.

    Rolling and diff features read the current row, so they are built from the
    declared base columns *other than the target*: ``{target}_roll_*`` and
    ``{target}_diff_*`` are same-row functions of ``y[t]`` and would leak the
    target into the features. Lags are built from every declared base column,
    because ``{target}_lag_{n}`` with ``n >= 1`` (enforced by
    ``ExperimentConfig.validate``) is legitimate autoregression.

    The frame is sorted chronologically FIRST, because every stage below is
    POSITIONAL (``shift``, ``rolling``, ``diff``) while the chronological sort
    otherwise happens later, inside ``temporal_train_val_test_split``. On an
    out-of-order source — two concatenated per-period exports, say —
    ``{col}_lag_1`` therefore meant "one row earlier in the file", and after
    the sort those values were attached to rows that PRECEDE them in time: a
    future value inside a training row, with the disorder invisible downstream
    because the split had already hidden it.

    Returns
    -------
    FeatureFrame
        The engineered frame and the resolved model input columns.
    """
    engineered_frame = loaded.frame
    index = engineered_frame.index
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        logger.warning(
            "source rows are not in chronological order; sorting before building "
            "lag/rolling/diff features, which are positional"
        )
        engineered_frame = engineered_frame.sort_index()
    source_columns = set(engineered_frame.columns)
    non_target_feature_columns = [
        column for column in config.data.feature_columns if column != config.data.target_column
    ]
    if config.features.add_temporal:
        from solrad_correction.features.temporal import (
            add_all_cyclic_encodings,
            add_temporal_features,
        )

        engineered_frame = add_temporal_features(engineered_frame)
        if config.features.cyclic_encoding:
            engineered_frame = add_all_cyclic_encodings(engineered_frame)

    if config.features.lag_steps:
        from solrad_correction.features.engineering import add_lag_features

        engineered_frame = add_lag_features(
            engineered_frame,
            config.data.feature_columns,
            config.features.lag_steps,
        )

    if config.features.rolling_windows:
        from solrad_correction.features.engineering import add_rolling_features

        engineered_frame = add_rolling_features(
            engineered_frame,
            non_target_feature_columns,
            config.features.rolling_windows,
            config.features.rolling_aggs,
        )

    if config.features.add_diffs:
        from solrad_correction.features.engineering import add_diff_features

        engineered_frame = add_diff_features(engineered_frame, non_target_feature_columns)

    feature_columns = [
        column for column in engineered_frame.columns if column != config.data.target_column
    ]
    if config.data.feature_columns:
        engineered_columns = [
            column for column in engineered_frame.columns if column not in source_columns
        ]
        requested_columns = set(config.data.feature_columns) | set(engineered_columns)
        feature_columns = [
            column
            for column in engineered_frame.columns
            if column in requested_columns and column != config.data.target_column
        ]
    return FeatureFrame(frame=engineered_frame, feature_cols=feature_columns)


def split_data(config: ExperimentConfig, features: FeatureFrame) -> SplitFrames:
    """Split data chronologically according to config.

    The three splits are contiguous blocks in time, so the validation and test
    periods sit strictly after the training one.
    """
    training_frame, validation_frame, test_frame = temporal_train_val_test_split(
        features.frame,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        shuffle=config.split.shuffle,
    )
    return SplitFrames(train=training_frame, val=validation_frame, test=test_frame)


def preprocess_splits(
    config: ExperimentConfig,
    features: FeatureFrame,
    splits: SplitFrames,
) -> PreprocessedSplits:
    """Fit preprocessing on train and transform all splits.

    The scaler and the imputation statistics are estimated on the training
    split alone and merely applied to validation and test: fitting them over
    the whole frame would let the test period's mean and scale reach the model.

    Returns
    -------
    PreprocessedSplits
        The three transformed frames, the fitted pipeline, and the feature
        columns that survived it.

    Raises
    ------
    ValueError
        If preprocessing dropped the target column, which would leave nothing
        to regress against.
    """
    preprocessing_pipeline = PreprocessingPipeline(
        scaler_type=config.preprocess.scaler_type,
        impute_strategy=config.preprocess.impute_strategy,
        drop_na_threshold=config.preprocess.drop_na_threshold,
        feature_columns=features.feature_cols,
        target_column=config.data.target_column,
    )
    model_columns = [*features.feature_cols, config.data.target_column]
    preprocessed_training_frame = preprocessing_pipeline.fit_transform(splits.train[model_columns])
    if config.data.target_column not in preprocessed_training_frame.columns:
        raise ValueError(
            f"Target column '{config.data.target_column}' was dropped during preprocessing"
        )
    retained_feature_columns = [
        column for column in features.feature_cols if column in preprocessing_pipeline.columns
    ]
    return PreprocessedSplits(
        train=preprocessed_training_frame,
        val=preprocessing_pipeline.transform(splits.val[model_columns]),
        test=preprocessing_pipeline.transform(splits.test[model_columns]),
        pipeline=preprocessing_pipeline,
        feature_cols=retained_feature_columns,
    )


def build_datasets(config: ExperimentConfig, processed: PreprocessedSplits) -> DatasetBundle:
    """Build train/validation/test datasets and preserve artifact schemas.

    The prediction index is taken from the built test dataset itself so it is
    always row-aligned with the model's predictions (including rows dropped
    for NaNs or temporal gaps) under every evaluation policy.

    Returns
    -------
    DatasetBundle
        Train/validation/test datasets of the kind the configured model
        consumes, plus the test targets ``(N,)`` and their timestamps.
    """
    preprocessed_splits = processed
    model_type = config.model.model_type.lower()
    model_spec = get_model_spec(model_type)
    if model_spec.kind == "tabular":
        return _build_tabular_dataset_bundle(
            config,
            preprocessed_splits,
            model_type=model_type,
        )
    return _build_sequence_dataset_bundle(config, preprocessed_splits)


def _build_tabular_dataset_bundle(
    config: ExperimentConfig,
    preprocessed_splits: PreprocessedSplits,
    *,
    model_type: str,
) -> DatasetBundle:
    """Build independent-row datasets with an evaluation-aligned test frame.

    Only the test frame is trimmed by the evaluation policy: under
    ``common_sequence_horizon`` the tabular model is scored on exactly the rows
    a sequence model over the same frame would predict, which is what makes the
    two families' metrics comparable. Training and validation keep every row.
    """
    feature_columns = preprocessed_splits.feature_cols
    target_column = config.data.target_column
    training_dataset = TabularDataset.from_dataframe(
        preprocessed_splits.train,
        feature_columns,
        target_column,
    )
    validation_dataset = TabularDataset.from_dataframe(
        preprocessed_splits.val,
        feature_columns,
        target_column,
    )
    aligned_test_frame = align_test_frame(
        preprocessed_splits.test,
        model_type=model_type,
        sequence_length=config.model.sequence_length,
        evaluation_policy=config.model.evaluation_policy,
    )
    test_dataset = TabularDataset.from_dataframe(
        aligned_test_frame,
        feature_columns,
        target_column,
    )
    return DatasetBundle(
        train=training_dataset,
        val=validation_dataset,
        test=test_dataset,
        input_size=None,
        y_true=test_dataset.y,
        prediction_index=test_dataset.index,
    )


def _build_sequence_dataset_bundle(
    config: ExperimentConfig,
    preprocessed_splits: PreprocessedSplits,
) -> DatasetBundle:
    """Build lazy sliding-window datasets and their aligned evaluation payload.

    The windows are materialized on demand rather than stacked up front, so
    memory stays proportional to the frame and not to ``rows * T``. No
    evaluation-policy trimming happens here: a sequence model already predicts
    exactly the window-target rows the policy defines.
    """
    feature_columns = preprocessed_splits.feature_cols
    target_column = config.data.target_column
    sequence_length = config.model.sequence_length
    training_dataset = _build_windowed_dataset(
        preprocessed_splits.train,
        "train",
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )
    validation_dataset = _build_windowed_dataset(
        preprocessed_splits.val,
        "validation",
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )
    test_dataset = _build_windowed_dataset(
        preprocessed_splits.test,
        "test",
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )
    return DatasetBundle(
        train=training_dataset,
        val=validation_dataset,
        test=test_dataset,
        input_size=training_dataset.n_features,
        y_true=test_dataset.target_values(),
        prediction_index=test_dataset.prediction_index(),
    )


def _build_windowed_dataset(
    frame: pd.DataFrame,
    split_name: str,
    *,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
) -> WindowedSequenceDataset:
    """Convert one preprocessed split into a lazy sliding-window dataset.

    Features become a ``(rows, F)`` ``float32`` matrix and targets a
    ``(rows,)`` ``float32`` vector, both in the scaled space. The frame's index
    travels with them when it is temporal, which is what lets the dataset drop
    windows that straddle an acquisition gap.
    """
    feature_matrix = dataframe_to_float32_numpy(
        frame,
        feature_columns,
        context=f"{split_name} sequence feature matrix",
    )
    target_values = series_to_float32_numpy(
        frame[target_column],
        context=f"{split_name} sequence target vector",
    )
    return WindowedSequenceDataset(
        feature_matrix,
        target_values,
        sequence_length,
        index=_datetime_index_or_none(frame),
    )


def _datetime_index_or_none(frame: pd.DataFrame) -> pd.DatetimeIndex | None:
    """Return the frame's DatetimeIndex, or None when the index is not temporal."""
    return frame.index if isinstance(frame.index, pd.DatetimeIndex) else None


def build_configured_model(config: ExperimentConfig, bundle: DatasetBundle) -> BaseRegressorModel:
    """Build the configured model through the registry.

    The input width comes from the bundle, not the config: it is the number of
    feature columns that survived preprocessing. The device is resolved here so
    the model is constructed straight onto it.
    """
    device = resolve_device(config.runtime.device)
    return build_model(config.model, input_size=bundle.input_size, device=device)


def train_model(
    config: ExperimentConfig,
    model: BaseRegressorModel,
    bundle: DatasetBundle,
    *,
    preprocessing_fingerprint: str | None = None,
) -> TrainingOutput:
    """Train the configured model.

    ``preprocessing_fingerprint`` identifies the transform this run's features
    were scaled with. It is baked into every checkpoint and re-checked on a
    resume: a resume refits the scaler from the data on disk now and never loads
    the checkpoint's own transform, so without the check a changed feature set
    or re-fitted mean/scale feeds the restored weights differently-scaled inputs.

    Only the sequence models are handed the runtime config and the fingerprint;
    the tabular ones fit in a single closed-form pass with nothing to resume.

    Returns
    -------
    TrainingOutput
        The training result and the wall-clock duration of the fit.
    """
    started = time.monotonic()
    if get_model_spec(config.model.model_type).kind == "sequence":
        training_result = model.fit(
            bundle.train,
            bundle.val,
            config.model,
            runtime=config.runtime,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )
    else:
        training_result = model.fit(bundle.train, bundle.val, config.model)
    return TrainingOutput(duration_seconds=time.monotonic() - started, result=training_result)


def predict_model(model: BaseRegressorModel, bundle: DatasetBundle) -> PredictionOutput:
    """Generate model predictions for the test dataset.

    Targets and timestamps are carried over from the bundle rather than
    recomputed, so predictions, observations and index stay one aligned set of
    ``N`` rows. Everything is still in scaled target units at this point.
    """
    return PredictionOutput(
        y_true=bundle.y_true,
        y_pred=model.predict(bundle.test),
        index=bundle.prediction_index,
    )


def evaluate_predictions(
    processed: PreprocessedSplits,
    config: ExperimentConfig,
    predictions: PredictionOutput,
) -> EvaluationResult:
    """Inverse-transform and compute regression metrics.

    Both vectors are returned to the physical units of the configured target
    column before any metric is computed, so RMSE, MAE and MBE are reported as
    physical errors rather than as multiples of the training standard
    deviation.

    Returns
    -------
    EvaluationResult
        Observed and predicted arrays of shape ``(N,)`` in original target
        units, and the metrics computed from them.
    """
    y_true_orig = processed.pipeline.inverse_transform_column(
        predictions.y_true,
        config.data.target_column,
    )
    y_pred_orig = processed.pipeline.inverse_transform_column(
        predictions.y_pred,
        config.data.target_column,
    )
    metrics = compute_regression_metrics(y_true_orig, y_pred_orig)
    return EvaluationResult(y_true=y_true_orig, y_pred=y_pred_orig, metrics=metrics)


def run_pipeline(config: ExperimentConfig) -> ExperimentReport:
    """Run an experiment through composable stages.

    The config is validated before anything is loaded and the global seed is
    set before anything random happens, so the run is reproducible from the
    versioned config alone.

    Two artifacts are persisted mid-pipeline rather than at the end: the fitted
    preprocessing, as soon as it exists, so a crash in a later stage never
    discards reusable state; and the trained model, immediately after the fit,
    so a crash during prediction or evaluation still leaves the weights
    recoverable on disk.

    Parameters
    ----------
    config:
        Fully populated experiment config.

    Returns
    -------
    ExperimentReport
        Metrics, resolved config, training history and run metadata — the same
        object whose contents were written under the experiment directory.
    """
    config.validate()
    prepare_runtime(config)
    experiment_started_at = time.monotonic()
    pipeline_profile = PipelineProfile(stage_seconds={})
    set_global_seed(config.seed)
    experiment_writer = ExperimentWriter.from_config(config)
    experiment_writer.prepare()

    with pipeline_profile.stage("load_data"):
        loaded_data = load_data(config)
    with pipeline_profile.stage("build_features"):
        feature_frame = build_features(loaded_data, config)
    with pipeline_profile.stage("split_data"):
        split_frames = split_data(config, feature_frame)
    with pipeline_profile.stage("preprocess_splits"):
        preprocessed_splits = preprocess_splits(config, feature_frame, split_frames)
    with pipeline_profile.stage("persist_preprocessing"):
        experiment_writer.write_preprocessing(preprocessed_splits.pipeline)
    with pipeline_profile.stage("build_datasets"):
        dataset_bundle = build_datasets(config, preprocessed_splits)
    with pipeline_profile.stage("build_model"):
        configured_model = build_configured_model(config, dataset_bundle)
    with pipeline_profile.stage("train_model"):
        training_output = train_model(
            config,
            configured_model,
            dataset_bundle,
            preprocessing_fingerprint=preprocessed_splits.pipeline.state.fingerprint(),
        )
    trained_model = training_output.result.model
    with pipeline_profile.stage("persist_model"):
        experiment_writer.write_model(config, trained_model)
    with pipeline_profile.stage("predict_model"):
        prediction_output = predict_model(trained_model, dataset_bundle)
    with pipeline_profile.stage("evaluate_predictions"):
        evaluation_result = evaluate_predictions(preprocessed_splits, config, prediction_output)

    experiment_report = ExperimentReport(
        experiment_name=config.name,
        model_name=config.model.model_type.lower(),
        metrics=evaluation_result.metrics,
        config=config.to_dict(),
        train_history=training_output.result.history,
        metadata=collect_run_metadata(
            config=config,
            model=trained_model,
            started_at=experiment_started_at,
            training_duration_seconds=training_output.duration_seconds,
        ),
    )
    experiment_result = ExperimentResult(
        report=experiment_report,
        processed=preprocessed_splits,
        datasets=dataset_bundle,
        model=trained_model,
        predictions=prediction_output,
        evaluation=evaluation_result,
    )
    experiment_writer.write_result(
        config=config,
        result=experiment_result,
        profile=pipeline_profile,
    )
    experiment_report.print_summary()
    return experiment_report
