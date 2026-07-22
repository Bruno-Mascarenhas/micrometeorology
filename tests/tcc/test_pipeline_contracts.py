"""End-to-end experiment artifact contracts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    RuntimeConfig,
    SplitConfig,
)
from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.datasets.sequence import WindowedSequenceDataset
from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.experiments.pipeline import build_datasets, build_features
from solrad_correction.experiments.results import LoadedData, PreprocessedSplits
from solrad_correction.experiments.runner import run_experiment


def _preprocessed_dataset_splits() -> PreprocessedSplits:
    index = pd.date_range("2024-01-01", periods=30, freq="1h")
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(30, dtype=np.float32),
            "feature_b": np.arange(100, 130, dtype=np.float32),
            "target": np.arange(200, 230, dtype=np.float32),
        },
        index=index,
    )
    training_frame = frame.iloc[:12]
    validation_frame = frame.iloc[12:20]
    test_frame = frame.iloc[20:]
    preprocessing_pipeline = PreprocessingPipeline(
        scaler_type="none",
        impute_strategy="drop",
        feature_columns=["feature_a", "feature_b"],
        target_column="target",
    )
    preprocessing_pipeline.fit(training_frame)
    return PreprocessedSplits(
        train=preprocessing_pipeline.transform(training_frame),
        val=preprocessing_pipeline.transform(validation_frame),
        test=preprocessing_pipeline.transform(test_frame),
        pipeline=preprocessing_pipeline,
        feature_cols=["feature_a", "feature_b"],
    )


def test_build_datasets_tabular_bundle_preserves_model_native_alignment() -> None:
    preprocessed_splits = _preprocessed_dataset_splits()
    config = ExperimentConfig(
        data=DataConfig(
            target_column="target",
            feature_columns=["feature_a", "feature_b"],
        ),
        model=ModelConfig(
            model_type="svm",
            sequence_length=3,
            evaluation_policy="model_native",
        ),
    )

    dataset_bundle = build_datasets(config, preprocessed_splits)

    assert isinstance(dataset_bundle.train, TabularDataset)
    assert isinstance(dataset_bundle.val, TabularDataset)
    assert isinstance(dataset_bundle.test, TabularDataset)
    assert dataset_bundle.input_size is None
    assert dataset_bundle.prediction_index is not None
    assert dataset_bundle.prediction_index.equals(preprocessed_splits.test.index)
    np.testing.assert_array_equal(
        dataset_bundle.y_true,
        preprocessed_splits.test["target"].to_numpy(dtype=np.float32),
    )


def test_build_datasets_sequence_bundle_aligns_targets_to_window_end() -> None:
    preprocessed_splits = _preprocessed_dataset_splits()
    config = ExperimentConfig(
        data=DataConfig(
            target_column="target",
            feature_columns=["feature_a", "feature_b"],
        ),
        model=ModelConfig(model_type="lstm", sequence_length=3),
    )

    dataset_bundle = build_datasets(config, preprocessed_splits)

    assert isinstance(dataset_bundle.train, WindowedSequenceDataset)
    assert isinstance(dataset_bundle.val, WindowedSequenceDataset)
    assert isinstance(dataset_bundle.test, WindowedSequenceDataset)
    assert dataset_bundle.input_size == 2
    assert dataset_bundle.prediction_index is not None
    assert dataset_bundle.prediction_index.equals(preprocessed_splits.test.index[2:])
    np.testing.assert_array_equal(
        dataset_bundle.y_true,
        preprocessed_splits.test["target"].to_numpy(dtype=np.float32)[2:],
    )


def test_build_features_keeps_requested_temporal_and_cyclic_columns() -> None:
    """Regression for finding 1: engineered features survive feature_columns."""
    index = pd.date_range("2024-06-01", periods=48, freq="1h")
    frame = pd.DataFrame(
        {
            "SWDOWN": np.arange(48, dtype=np.float32),
            "T2": np.arange(48, dtype=np.float32),
            "UNRELATED": np.arange(48, dtype=np.float32),
            "SW_dif": np.arange(48, dtype=np.float32),
        },
        index=index,
    )
    cfg = ExperimentConfig(
        data=DataConfig(target_column="SW_dif", feature_columns=["SWDOWN", "T2"]),
        features=FeatureConfig(add_temporal=True, cyclic_encoding=True, lag_steps=[1]),
    )

    features = build_features(LoadedData(frame=frame), cfg)

    expected_engineered = {
        "hour",
        "day_of_year",
        "month",
        "weekday",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "month_sin",
        "month_cos",
        "SWDOWN_lag_1",
        "T2_lag_1",
    }
    assert {"SWDOWN", "T2"}.issubset(features.feature_cols)
    assert expected_engineered.issubset(features.feature_cols)
    assert "SW_dif" not in features.feature_cols
    assert "UNRELATED" not in features.feature_cols


def test_model_and_preprocessing_persist_even_when_prediction_crashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for finding 7: a post-training crash must not discard the fit."""
    from solrad_correction.experiments import pipeline as pipeline_module

    scratch = Path("scratch") / "svm_crash_persistence_contract"
    data_path = scratch / "hourly.parquet"
    output_dir = scratch / "output"

    def _boom(*_args: object, **_kwargs: object) -> np.ndarray:
        raise RuntimeError("simulated prediction crash")

    try:
        scratch.mkdir(parents=True, exist_ok=True)
        index = pd.date_range("2024-01-01", periods=48, freq="1h")
        rng = np.random.default_rng(11)
        f1 = rng.normal(size=48).astype(np.float32)
        target = (0.5 * f1).astype(np.float32)
        pd.DataFrame({"f1": f1, "target": target}, index=index).to_parquet(data_path)

        cfg = ExperimentConfig(
            name="svm_crash",
            data=DataConfig(
                hourly_data_path=str(data_path),
                source_format="parquet",
                target_column="target",
                feature_columns=["f1"],
            ),
            split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
            features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
            model=ModelConfig(model_type="svm"),
            runtime=RuntimeConfig(device="cpu"),
            output_dir=str(output_dir),
        )
        monkeypatch.setattr(pipeline_module, "predict_model", _boom)

        with pytest.raises(RuntimeError, match="simulated prediction crash"):
            run_experiment(cfg)

        exp_dir = output_dir / "svm_crash"
        assert (exp_dir / "models" / "model.joblib").exists()
        assert (exp_dir / "preprocessing" / "preprocessing_pipeline.joblib").exists()
        assert (exp_dir / "metadata" / "preprocessing_state.json").exists()
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_svm_run_writes_canonical_artifact_layout_and_prediction_schema() -> None:
    scratch = Path("scratch") / "svm_artifact_contract"
    data_path = scratch / "hourly.parquet"
    output_dir = scratch / "output"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        index = pd.date_range("2024-01-01", periods=48, freq="1h")
        rng = np.random.default_rng(8)
        f1 = rng.normal(size=48).astype(np.float32)
        f2 = rng.normal(size=48).astype(np.float32)
        target = (0.5 * f1 + 0.3 * f2).astype(np.float32)
        pd.DataFrame({"f1": f1, "f2": f2, "target": target}, index=index).to_parquet(data_path)

        cfg = ExperimentConfig(
            name="svm_artifacts",
            data=DataConfig(
                hourly_data_path=str(data_path),
                source_format="parquet",
                target_column="target",
                feature_columns=["f1", "f2"],
                dtype_map={"f1": "float32", "f2": "float32", "target": "float32"},
            ),
            split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
            preprocess=PreprocessConfig(scaler_type="standard", impute_strategy="drop"),
            features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
            model=ModelConfig(model_type="svm", svm_c=1.0),
            runtime=RuntimeConfig(device="cpu", limit_rows=40),
            output_dir=str(output_dir),
        )

        report = run_experiment(cfg)
        exp_dir = output_dir / "svm_artifacts"
        predictions = pd.read_csv(
            exp_dir / "predictions" / "predictions.csv", index_col=0, parse_dates=True
        )
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))

        assert report.metrics["RMSE"] >= 0.0
        assert {"y_true", "y_pred"}.issubset(predictions.columns)
        assert len(predictions) == 8
        # Finding 15: model_native predictions must carry timestamps.
        assert isinstance(predictions.index, pd.DatetimeIndex)
        assert predictions.index.name == "timestamp"
        for relative in [
            "configs/config.yaml",
            "configs/config_resolved.json",
            "metrics/metrics.json",
            "predictions/predictions.csv",
            "models/model.joblib",
            "datasets/train/data.npz",
            "metadata/preprocessing_state.json",
            "preprocessing/preprocessing_pipeline.joblib",
        ]:
            assert (exp_dir / relative).exists()
            assert relative in manifest["artifacts"]
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_lstm_run_writes_lazy_sequence_artifacts_checkpoints_profile_and_manifest() -> None:
    scratch = Path("scratch") / "lstm_artifact_contract"
    data_path = scratch / "hourly.csv"
    output_dir = scratch / "output"
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        index = pd.date_range("2024-01-01", periods=80, freq="1h")
        rng = np.random.default_rng(42)
        f1 = rng.normal(size=80).astype(np.float32)
        f2 = rng.normal(size=80).astype(np.float32)
        target = (0.7 * f1 - 0.2 * f2 + rng.normal(scale=0.01, size=80)).astype(np.float32)
        pd.DataFrame({"f1": f1, "f2": f2, "target": target}, index=index).to_csv(data_path)

        cfg = ExperimentConfig(
            name="lstm_artifacts",
            data=DataConfig(
                hourly_data_path=str(data_path),
                target_column="target",
                feature_columns=["f1", "f2"],
            ),
            split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
            preprocess=PreprocessConfig(scaler_type="standard", impute_strategy="drop"),
            features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
            model=ModelConfig(
                model_type="lstm",
                lstm_hidden_size=4,
                lstm_num_layers=1,
                sequence_length=4,
                batch_size=8,
                max_epochs=1,
                patience=2,
            ),
            runtime=RuntimeConfig(device="cpu", num_workers=0, profile=True),
            output_dir=str(output_dir),
        )

        report = run_experiment(cfg)
        exp_dir = output_dir / "lstm_artifacts"
        profile = json.loads((exp_dir / "profiles" / "profile.json").read_text(encoding="utf-8"))
        metadata = json.loads((exp_dir / "metadata" / "metadata.json").read_text(encoding="utf-8"))
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(
            exp_dir / "predictions" / "predictions.csv", index_col=0, parse_dates=True
        )

        assert report.train_history["train_loss"]
        assert profile["schema_version"] == 1
        assert "load_data" in profile["stage_seconds"]
        assert "train_model" in profile["stage_seconds"]
        assert metadata["model"]["parameter_count"] > 0
        # Findings 9 + 15: one prediction per window, targeted at the window's
        # last row, each carrying its timestamp (16 test rows, 13 windows).
        assert isinstance(predictions.index, pd.DatetimeIndex)
        assert len(predictions) == 13
        for relative in [
            "checkpoints/best.pt",
            "checkpoints/last.pt",
            "datasets/train/windowed_sequences.npz",
            "metrics/training_history.csv",
            "models/model.pt",
            "profiles/profile.json",
        ]:
            assert (exp_dir / relative).exists()
            assert relative in manifest["artifacts"]
        assert not (exp_dir / "datasets" / "train" / "sequences.npz").exists()
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
