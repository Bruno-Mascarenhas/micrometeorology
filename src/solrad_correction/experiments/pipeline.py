"""Composable experiment pipeline stages."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.data.splits import temporal_train_val_test_split
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.metrics import compute_regression_metrics
from solrad_correction.evaluation.reports import ExperimentReport, save_experiment_results
from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
from solrad_correction.models.registry import build_model, get_model_spec
from solrad_correction.training.dataloaders import resolve_device
from solrad_correction.utils.io import save_json
from solrad_correction.utils.metadata import collect_run_metadata
from solrad_correction.utils.seeds import set_global_seed

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from solrad_correction.config import ExperimentConfig
    from solrad_correction.models.base import BaseRegressorModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessedSplits:
    """Preprocessed train/validation/test frames plus state."""

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
class PipelineProfile:
    """Stage timing accumulator."""

    stage_seconds: dict[str, float]

    def time_stage(self, name: str, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        started = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.stage_seconds[name] = time.monotonic() - started


def _prediction_index_for_policy(
    index: pd.DatetimeIndex,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DatetimeIndex | None:
    """Return prediction index for explicit non-default evaluation policies."""
    _ = model_type
    if evaluation_policy == "model_native":
        return None
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    return index[sequence_length:]


def _test_frame_for_policy(
    test_df: pd.DataFrame,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DataFrame:
    """Apply explicit evaluation row policy without changing default behavior."""
    if evaluation_policy == "model_native":
        return test_df
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")
    if model_type == "svm":
        return test_df.iloc[sequence_length:]
    return test_df


def prepare_runtime(config: ExperimentConfig) -> None:
    """Resolve output-coupled runtime defaults in-place."""
    model_type = config.model.model_type.lower()
    if model_type in {"lstm", "transformer"} and config.runtime.checkpoint_dir is None:
        layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
        config.runtime.checkpoint_dir = str(layout.checkpoints_dir)


def load_data(config: ExperimentConfig) -> pd.DataFrame:
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
        )
    elif config.data.sensor_data_path:
        from solrad_correction.data.loaders import load_sensor_raw

        df = load_sensor_raw(config.data.sensor_data_path)
        if config.runtime.limit_rows is not None:
            df = df.iloc[: config.runtime.limit_rows].copy()
    else:
        raise ValueError("No data path provided in config")

    return df


def build_features(df: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Apply feature engineering according to config."""
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

    return df


def select_feature_columns(df: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    """Resolve final feature columns after feature engineering."""
    feature_cols = [c for c in df.columns if c != config.data.target_column]
    if config.data.feature_columns:
        base = set(config.data.feature_columns)
        feature_cols = [
            c for c in df.columns if c in base or any(c.startswith(f"{b}_") for b in base)
        ]
        feature_cols = [c for c in feature_cols if c != config.data.target_column]
    return feature_cols


def split_data(
    config: ExperimentConfig, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically according to config."""
    return temporal_train_val_test_split(
        df,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        shuffle=config.split.shuffle,
    )


def preprocess_splits(
    config: ExperimentConfig,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> ProcessedSplits:
    """Fit preprocessing on train and transform all splits."""
    pipeline = PreprocessingPipeline(
        scaler_type=config.preprocess.scaler_type,
        impute_strategy=config.preprocess.impute_strategy,
        drop_na_threshold=config.preprocess.drop_na_threshold,
        feature_columns=feature_cols,
        target_column=config.data.target_column,
    )
    all_cols = [*feature_cols, config.data.target_column]
    train = pipeline.fit_transform(train_df[all_cols])
    if config.data.target_column not in train.columns:
        raise ValueError(
            f"Target column '{config.data.target_column}' was dropped during preprocessing"
        )
    retained_features = [c for c in feature_cols if c in pipeline.columns]
    return ProcessedSplits(
        train=train,
        val=pipeline.transform(val_df[all_cols]),
        test=pipeline.transform(test_df[all_cols]),
        pipeline=pipeline,
        feature_cols=retained_features,
    )


def save_preprocessing_state(config: ExperimentConfig, processed: ProcessedSplits) -> None:
    """Persist preprocessing state."""
    layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
    processed.pipeline.save(layout.preprocessing_joblib)
    processed.pipeline.save_state_json(layout.preprocessing_state)


def build_datasets(config: ExperimentConfig, processed: ProcessedSplits) -> DatasetBundle:
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
        test_eval = _test_frame_for_policy(
            processed.test,
            model_type=model_type,
            sequence_length=config.model.sequence_length,
            evaluation_policy=eval_policy,
        )
        test_ds = TabularDataset.from_dataframe(test_eval, feature_cols, config.data.target_column)
        layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
        train_ds.save(layout.datasets_dir / "train")
        val_ds.save(layout.datasets_dir / "val")
        test_ds.save(layout.datasets_dir / "test")
        prediction_index = _prediction_index_for_policy(
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
            prediction_index=prediction_index,
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
    layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
    train_seq.save(layout.datasets_dir / "train", feature_names=feature_cols)
    val_seq.save(layout.datasets_dir / "val", feature_names=feature_cols)
    test_seq.save(layout.datasets_dir / "test", feature_names=feature_cols)

    prediction_index = _prediction_index_for_policy(
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
        prediction_index=prediction_index,
    )


def build_configured_model(config: ExperimentConfig, bundle: DatasetBundle) -> BaseRegressorModel:
    """Build the configured model through the registry."""
    device = resolve_device(config.runtime.device)
    return build_model(config.model, input_size=bundle.input_size, device=device)


def train_model(
    config: ExperimentConfig,
    model: BaseRegressorModel,
    bundle: DatasetBundle,
) -> float:
    """Train and persist the configured model."""
    started = time.monotonic()
    if get_model_spec(config.model.model_type).kind == "sequence":
        model.fit(bundle.train, bundle.val, config.model, runtime=config.runtime)  # type: ignore[call-arg]
        model.save(ArtifactLayout.from_experiment_dir(config.experiment_dir).model_pt)
    else:
        model.fit(bundle.train, bundle.val, config.model)
        model.save(ArtifactLayout.from_experiment_dir(config.experiment_dir).model_joblib)
    return time.monotonic() - started


def predict_model(model: BaseRegressorModel, bundle: DatasetBundle) -> np.ndarray:
    """Generate model predictions for the test dataset."""
    return model.predict(bundle.test)


def evaluate_predictions(
    processed: ProcessedSplits,
    config: ExperimentConfig,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Inverse-transform and compute regression metrics."""
    y_true_orig = processed.pipeline.inverse_transform_column(y_true, config.data.target_column)
    y_pred_orig = processed.pipeline.inverse_transform_column(y_pred, config.data.target_column)
    metrics = compute_regression_metrics(y_true_orig, y_pred_orig)
    return y_true_orig, y_pred_orig, metrics


def save_profile(config: ExperimentConfig, profile: PipelineProfile) -> None:
    """Save optional stage timing profile."""
    if config.runtime.profile:
        save_json(
            {
                "schema_version": 1,
                "stage_seconds": profile.stage_seconds,
                "total_stage_seconds": sum(profile.stage_seconds.values()),
            },
            ArtifactLayout.from_experiment_dir(config.experiment_dir).profile,
        )


def run_pipeline(config: ExperimentConfig) -> ExperimentReport:
    """Run an experiment through composable stages."""
    config.validate()
    prepare_runtime(config)
    experiment_started = time.monotonic()
    profile = PipelineProfile(stage_seconds={})
    set_global_seed(config.seed)
    layout = ArtifactLayout.from_experiment_dir(config.experiment_dir)
    layout.ensure_directories()
    config.save(layout.config_yaml)

    df = profile.time_stage("load_data", load_data, config)
    df = profile.time_stage("build_features", build_features, df, config)
    feature_cols = profile.time_stage("select_feature_columns", select_feature_columns, df, config)
    train_df, val_df, test_df = profile.time_stage("split_data", split_data, config, df)
    processed = profile.time_stage(
        "preprocess_splits",
        preprocess_splits,
        config,
        feature_cols,
        train_df,
        val_df,
        test_df,
    )
    profile.time_stage("save_preprocessing_state", save_preprocessing_state, config, processed)
    bundle = profile.time_stage("build_datasets", build_datasets, config, processed)
    model = profile.time_stage("build_model", build_configured_model, config, bundle)
    training_duration = profile.time_stage("train_model", train_model, config, model, bundle)
    y_pred = profile.time_stage("predict_model", predict_model, model, bundle)
    y_true_orig, y_pred_orig, metrics = profile.time_stage(
        "evaluate_predictions",
        evaluate_predictions,
        processed,
        config,
        bundle.y_true,
        y_pred,
    )

    report = ExperimentReport(
        experiment_name=config.name,
        model_name=config.model.model_type.lower(),
        metrics=metrics,
        config=config.to_dict(),
        train_history=getattr(model, "training_history", {}),
        metadata=collect_run_metadata(
            config=config,
            model=model,
            started_at=experiment_started,
            training_duration_seconds=training_duration,
        ),
    )
    profile.time_stage(
        "save_experiment_results",
        save_experiment_results,
        report,
        y_true_orig,
        y_pred_orig,
        config.experiment_dir,
        bundle.prediction_index,
    )
    save_profile(config, profile)
    write_manifest(
        layout,
        extra={
            "experiment_name": config.name,
            "model_type": config.model.model_type.lower(),
            "profile_enabled": config.runtime.profile,
        },
    )
    report.print_summary()
    return report
