"""Experiment runner — full pipeline from config to results."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, cast

from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.data.splits import temporal_train_val_test_split
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.evaluation.metrics import compute_regression_metrics
from solrad_correction.evaluation.reports import ExperimentReport, save_experiment_results
from solrad_correction.utils.metadata import collect_run_metadata
from solrad_correction.utils.seeds import set_global_seed

if TYPE_CHECKING:
    import pandas as pd

    from solrad_correction.config import ExperimentConfig

logger = logging.getLogger(__name__)


def _prediction_index_for_policy(
    index: pd.DatetimeIndex,
    *,
    model_type: str,
    sequence_length: int,
    evaluation_policy: str,
) -> pd.DatetimeIndex | None:
    """Return prediction index for explicit non-default evaluation policies."""
    if evaluation_policy == "model_native":
        return None
    if evaluation_policy != "common_sequence_horizon":
        raise ValueError(f"Unknown evaluation_policy: {evaluation_policy}")

    if model_type == "svm":
        return index[sequence_length:]
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


def run_experiment(config: ExperimentConfig) -> ExperimentReport:
    """Execute a complete experiment from config.

    Pipeline:
    1. Load data
    2. Feature engineering
    3. Split (chronological)
    4. Preprocess (fit on train only)
    5. Build datasets
    6. Train model
    7. Evaluate
    8. Save results

    Returns an ExperimentReport.
    """
    config.validate()
    experiment_started = time.monotonic()
    training_duration_seconds: float | None = None
    set_global_seed(config.seed)
    exp_dir = config.experiment_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config.save(exp_dir / "config.yaml")

    # ── 1. Load data ──

    if config.data.hourly_data_path:
        from solrad_correction.data.loaders import load_sensor_hourly

        df = load_sensor_hourly(config.data.hourly_data_path)
    elif config.data.sensor_data_path:
        from solrad_correction.data.loaders import load_sensor_raw

        df = load_sensor_raw(config.data.sensor_data_path)
    else:
        raise ValueError("No data path provided in config")

    # ── 2. Feature engineering ──
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

    # Determine final feature columns
    feature_cols = [c for c in df.columns if c != config.data.target_column]
    if config.data.feature_columns:
        # Use specified + generated
        base = set(config.data.feature_columns)
        feature_cols = [
            c for c in df.columns if c in base or any(c.startswith(f"{b}_") for b in base)
        ]
        feature_cols = [c for c in feature_cols if c != config.data.target_column]

    # ── 3. Split ──
    train_df, val_df, test_df = temporal_train_val_test_split(
        df,
        config.split.train_ratio,
        config.split.val_ratio,
        config.split.test_ratio,
        shuffle=config.split.shuffle,
    )

    # ── 4. Preprocess ──
    pipeline = PreprocessingPipeline(
        scaler_type=config.preprocess.scaler_type,
        impute_strategy=config.preprocess.impute_strategy,
        drop_na_threshold=config.preprocess.drop_na_threshold,
    )
    all_cols = [*feature_cols, config.data.target_column]
    train_pp = pipeline.fit_transform(train_df[all_cols])
    val_pp = pipeline.transform(val_df[all_cols])
    test_pp = pipeline.transform(test_df[all_cols])

    pipeline.save(exp_dir / "preprocessing_pipeline.joblib")

    # ── 5. Build datasets & train ──
    model_type = config.model.model_type.lower()
    eval_policy = config.model.evaluation_policy

    if model_type == "svm":
        train_ds = TabularDataset.from_dataframe(train_pp, feature_cols, config.data.target_column)
        val_ds = TabularDataset.from_dataframe(val_pp, feature_cols, config.data.target_column)
        test_eval_pp = _test_frame_for_policy(
            test_pp,
            model_type=model_type,
            sequence_length=config.model.sequence_length,
            evaluation_policy=eval_policy,
        )
        test_ds = TabularDataset.from_dataframe(
            test_eval_pp, feature_cols, config.data.target_column
        )

        train_ds.save(exp_dir / "datasets" / "train")
        test_ds.save(exp_dir / "datasets" / "test")

        from solrad_correction.models.svm import SVMRegressor

        model = SVMRegressor.from_config(config.model)
        training_started = time.monotonic()
        model.fit(train_ds, val_ds, config.model)  # type: ignore
        training_duration_seconds = time.monotonic() - training_started
        model.save(exp_dir / "model.joblib")

        y_pred = model.predict(test_ds)  # type: ignore
        y_true = test_ds.y
        prediction_index = _prediction_index_for_policy(
            cast("pd.DatetimeIndex", test_pp.index),
            model_type=model_type,
            sequence_length=config.model.sequence_length,
            evaluation_policy=eval_policy,
        )

    elif model_type in ("lstm", "transformer"):
        from solrad_correction.datasets.sequence import WindowedSequenceDataset

        seq_len = config.model.sequence_length

        train_features = train_pp[feature_cols].to_numpy()
        train_target = train_pp[config.data.target_column].to_numpy()
        val_features = val_pp[feature_cols].to_numpy()
        val_target = val_pp[config.data.target_column].to_numpy()
        test_features = test_pp[feature_cols].to_numpy()
        test_target = test_pp[config.data.target_column].to_numpy()

        train_seq = WindowedSequenceDataset(train_features, train_target, seq_len)
        val_seq = WindowedSequenceDataset(val_features, val_target, seq_len)
        test_seq = WindowedSequenceDataset(test_features, test_target, seq_len)

        train_seq.save(exp_dir / "datasets" / "train", feature_names=feature_cols)
        test_seq.save(exp_dir / "datasets" / "test", feature_names=feature_cols)

        input_size = train_features.shape[1]

        if model_type == "lstm":
            from solrad_correction.models.lstm import LSTMRegressor

            model = LSTMRegressor.from_config(config.model, input_size)  # type: ignore
        else:
            from solrad_correction.models.transformer import TransformerRegressor

            model = TransformerRegressor.from_config(config.model, input_size)  # type: ignore

        training_started = time.monotonic()
        model.fit(train_seq, val_seq, config.model)  # type: ignore
        training_duration_seconds = time.monotonic() - training_started
        model.save(exp_dir / "model.pt")

        y_pred = model.predict(test_seq)  # type: ignore
        y_true = test_seq.target_values()
        prediction_index = _prediction_index_for_policy(
            cast("pd.DatetimeIndex", test_pp.index),
            model_type=model_type,
            sequence_length=seq_len,
            evaluation_policy=eval_policy,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ── 7. Evaluate ──
    # Inverse transform predictions and ground truth to original scale
    y_true_orig = pipeline.inverse_transform_column(y_true, config.data.target_column)
    y_pred_orig = pipeline.inverse_transform_column(y_pred, config.data.target_column)

    metrics = compute_regression_metrics(y_true_orig, y_pred_orig)

    report = ExperimentReport(
        experiment_name=config.name,
        model_name=model_type,
        metrics=metrics,
        config=config.to_dict(),
        train_history=getattr(model, "training_history", {}),
        metadata=collect_run_metadata(
            config=config,
            model=model,
            started_at=experiment_started,
            training_duration_seconds=training_duration_seconds,
        ),
    )

    save_experiment_results(report, y_true_orig, y_pred_orig, exp_dir, index=prediction_index)
    report.print_summary()

    return report
