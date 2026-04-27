"""Composable experiment pipeline stages."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.data.splits import temporal_train_val_test_split
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.metrics import compute_regression_metrics
from solrad_correction.evaluation.policy import align_test_frame, prediction_index
from solrad_correction.evaluation.reports import ExperimentReport
from solrad_correction.experiments.artifacts import ArtifactLayout
from solrad_correction.experiments.results import (
    DatasetBundle,
    EvaluationResult,
    ExperimentResult,
    FeatureFrame,
    LoadedData,
    PredictionOutput,
    PreprocessedSplits,
    SplitFrames,
    TrainingOutput,
)
from solrad_correction.experiments.writer import ExperimentWriter
from solrad_correction.models.registry import build_model, get_model_spec
from solrad_correction.training.dataloaders import resolve_device
from solrad_correction.utils.metadata import collect_run_metadata
from solrad_correction.utils.seeds import set_global_seed

if TYPE_CHECKING:
    import pandas as pd

    from solrad_correction.config import ExperimentConfig
    from solrad_correction.models.base import BaseRegressorModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineProfile:
    """Stage timing accumulator."""

    stage_seconds: dict[str, float]

    def time_stage(self, name: str, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stage_seconds[name] = time.monotonic() - started


def prepare_runtime(config: ExperimentConfig) -> None:
    """Resolve output-coupled runtime defaults in-place."""
    model_type = config.model.model_type.lower()
    if model_type in {"lstm", "transformer"} and config.runtime.checkpoint_dir is None:
        layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
        config.runtime.checkpoint_dir = str(layout.checkpoints_dir)


def load_data(config: ExperimentConfig) -> LoadedData:
    """Load configured input data."""
    if config.data.hourly_data_path:
        from solrad_correction.data.loaders import load_sensor_hourly

        projected_columns = config.data.load_columns or None
        if projected_columns is None and config.data.feature_columns:
            projected_columns = [*config.data.feature_columns, config.data.target_column]
        df = load_sensor_hourly(
            config.data.hourly_data_path,
            source_format=config.data.source_format,  # type: ignore[arg-type]
            columns=projected_columns,
            datetime_column=config.data.datetime_column,
            datetime_index=config.data.datetime_index,
            dtype_map=config.data.dtype_map,
            limit_rows=config.runtime.limit_rows,
            cache_dir=config.data.cache_dir,
        )
    elif config.data.sensor_data_path:
        from solrad_correction.data.loaders import load_sensor_raw

        df = load_sensor_raw(
            config.data.sensor_data_path,
            pattern=config.data.sensor_pattern,
            calibrations_path=config.data.calibrations_path,
            resample_freq=config.data.resample_freq,
            min_samples=config.data.sensor_min_samples,
        )
        if config.runtime.limit_rows is not None:
            df = df.iloc[: config.runtime.limit_rows].copy()
    else:
        raise ValueError("No data path provided in config")

    return LoadedData(frame=df)


def build_features(loaded: LoadedData, config: ExperimentConfig) -> FeatureFrame:
    """Apply feature engineering according to config."""
    df = loaded.frame
    if config.features.add_temporal:
        from solrad_correction.features.temporal import (
            add_all_cyclic_encodings,
            add_temporal_features,
        )

        df = add_temporal_features(df)
        if config.features.cyclic_encoding:
            df = add_all_cyclic_encodings(df)

    if config.features.lag_steps:
        from solrad_correction.features.engineering import add_lag_features

        df = add_lag_features(df, config.data.feature_columns, config.features.lag_steps)

    if config.features.rolling_windows:
        from solrad_correction.features.engineering import add_rolling_features

        df = add_rolling_features(
            df,
            config.data.feature_columns,
            config.features.rolling_windows,
            config.features.rolling_aggs,
        )

    if config.features.add_diffs:
        from solrad_correction.features.engineering import add_diff_features

        df = add_diff_features(df, config.data.feature_columns)

    feature_cols = [c for c in df.columns if c != config.data.target_column]
    if config.data.feature_columns:
        base = set(config.data.feature_columns)
        feature_cols = [
            c for c in df.columns if c in base or any(c.startswith(f"{b}_") for b in base)
        ]
        feature_cols = [c for c in feature_cols if c != config.data.target_column]
    return FeatureFrame(frame=df, feature_cols=feature_cols)


def split_data(config: ExperimentConfig, features: FeatureFrame) -> SplitFrames:
    """Split data chronologically according to config."""
    train, val, test = temporal_train_val_test_split(
        features.frame,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        shuffle=config.split.shuffle,
    )
    return SplitFrames(train=train, val=val, test=test)


def preprocess_splits(
    config: ExperimentConfig,
    features: FeatureFrame,
    splits: SplitFrames,
) -> PreprocessedSplits:
    """Fit preprocessing on train and transform all splits."""
    pipeline = PreprocessingPipeline(
        scaler_type=config.preprocess.scaler_type,
        impute_strategy=config.preprocess.impute_strategy,
        drop_na_threshold=config.preprocess.drop_na_threshold,
        feature_columns=features.feature_cols,
        target_column=config.data.target_column,
    )
    all_cols = [*features.feature_cols, config.data.target_column]
    train = pipeline.fit_transform(splits.train[all_cols])
    if config.data.target_column not in train.columns:
        raise ValueError(
            f"Target column '{config.data.target_column}' was dropped during preprocessing"
        )
    retained_features = [c for c in features.feature_cols if c in pipeline.columns]
    return PreprocessedSplits(
        train=train,
        val=pipeline.transform(splits.val[all_cols]),
        test=pipeline.transform(splits.test[all_cols]),
        pipeline=pipeline,
        feature_cols=retained_features,
    )


def build_datasets(config: ExperimentConfig, processed: PreprocessedSplits) -> DatasetBundle:
    """Build train/validation/test datasets and preserve artifact schemas."""
    model_type = config.model.model_type.lower()
    eval_policy = config.model.evaluation_policy
    spec = get_model_spec(model_type)
    feature_cols = processed.feature_cols

    if spec.kind == "tabular":
        train_ds = TabularDataset.from_dataframe(
            processed.train, feature_cols, config.data.target_column
        )
        val_ds = TabularDataset.from_dataframe(
            processed.val, feature_cols, config.data.target_column
        )
        test_eval = align_test_frame(
            processed.test,
            model_type=model_type,
            sequence_length=config.model.sequence_length,
            evaluation_policy=eval_policy,
        )
        test_ds = TabularDataset.from_dataframe(test_eval, feature_cols, config.data.target_column)
        pred_index = prediction_index(
            cast("pd.DatetimeIndex", processed.test.index),
            model_type=model_type,
            sequence_length=config.model.sequence_length,
            evaluation_policy=eval_policy,
        )
        return DatasetBundle(
            train=train_ds,
            val=val_ds,
            test=test_ds,
            input_size=None,
            y_true=test_ds.y,
            prediction_index=pred_index,
        )

    from solrad_correction.datasets.sequence import WindowedSequenceDataset

    seq_len = config.model.sequence_length
    train_features = processed.train[feature_cols].to_numpy()
    val_features = processed.val[feature_cols].to_numpy()
    test_features = processed.test[feature_cols].to_numpy()
    train_target = processed.train[config.data.target_column].to_numpy()
    val_target = processed.val[config.data.target_column].to_numpy()
    test_target = processed.test[config.data.target_column].to_numpy()

    train_seq = WindowedSequenceDataset(train_features, train_target, seq_len)
    val_seq = WindowedSequenceDataset(val_features, val_target, seq_len)
    test_seq = WindowedSequenceDataset(test_features, test_target, seq_len)
    pred_index = prediction_index(
        cast("pd.DatetimeIndex", processed.test.index),
        model_type=model_type,
        sequence_length=seq_len,
        evaluation_policy=eval_policy,
    )
    return DatasetBundle(
        train=train_seq,
        val=val_seq,
        test=test_seq,
        input_size=train_features.shape[1],
        y_true=test_seq.target_values(),
        prediction_index=pred_index,
    )


def build_configured_model(config: ExperimentConfig, bundle: DatasetBundle) -> BaseRegressorModel:
    """Build the configured model through the registry."""
    device = resolve_device(config.runtime.device)
    return build_model(config.model, input_size=bundle.input_size, device=device)


def train_model(
    config: ExperimentConfig,
    model: BaseRegressorModel,
    bundle: DatasetBundle,
) -> TrainingOutput:
    """Train the configured model."""
    started = time.monotonic()
    if get_model_spec(config.model.model_type).kind == "sequence":
        res = model.fit(bundle.train, bundle.val, config.model, runtime=config.runtime)
    else:
        res = model.fit(bundle.train, bundle.val, config.model)
    return TrainingOutput(duration_seconds=time.monotonic() - started, result=res)


def predict_model(model: BaseRegressorModel, bundle: DatasetBundle) -> PredictionOutput:
    """Generate model predictions for the test dataset."""
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
    """Inverse-transform and compute regression metrics."""
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
    """Run an experiment through composable stages."""
    config.validate()
    prepare_runtime(config)
    experiment_started = time.monotonic()
    profile = PipelineProfile(stage_seconds={})
    set_global_seed(config.seed)
    writer = ExperimentWriter.from_config(config)
    writer.prepare()

    loaded = profile.time_stage("load_data", load_data, config)
    features = profile.time_stage("build_features", build_features, loaded, config)
    splits = profile.time_stage("split_data", split_data, config, features)
    processed = profile.time_stage(
        "preprocess_splits",
        preprocess_splits,
        config,
        features,
        splits,
    )
    bundle = profile.time_stage("build_datasets", build_datasets, config, processed)
    model = profile.time_stage("build_model", build_configured_model, config, bundle)
    training = profile.time_stage("train_model", train_model, config, model, bundle)
    model = training.result.model
    predictions = profile.time_stage("predict_model", predict_model, model, bundle)
    evaluation = profile.time_stage(
        "evaluate_predictions",
        evaluate_predictions,
        processed,
        config,
        predictions,
    )

    report = ExperimentReport(
        experiment_name=config.name,
        model_name=config.model.model_type.lower(),
        metrics=evaluation.metrics,
        config=config.to_dict(),
        train_history=training.result.history,
        metadata=collect_run_metadata(
            config=config,
            model=model,
            started_at=experiment_started,
            training_duration_seconds=training.duration_seconds,
        ),
    )
    result = ExperimentResult(
        report=report,
        processed=processed,
        datasets=bundle,
        model=model,
        predictions=predictions,
        evaluation=evaluation,
    )
    profile.time_stage(
        "write_experiment_results",
        writer.write_result,
        config=config,
        result=result,
        profile=profile,
    )
    report.print_summary()
    return report
