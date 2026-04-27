"""End-to-end artifact tests for synthetic solrad experiments."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    RuntimeConfig,
    SplitConfig,
)
from solrad_correction.experiments.runner import run_experiment


def test_lstm_run_writes_lazy_sequence_artifacts_history_and_metadata():
    scratch = Path("scratch") / "lstm_artifact_test"
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

        assert report.train_history["train_loss"]
        assert (exp_dir / "metrics" / "training_history.csv").exists()
        assert (exp_dir / "metadata" / "metadata.json").exists()
        assert (exp_dir / "profiles" / "profile.json").exists()
        assert (exp_dir / "checkpoints" / "best.pt").exists()
        assert (exp_dir / "checkpoints" / "last.pt").exists()
        assert (exp_dir / "datasets" / "train" / "windowed_sequences.npz").exists()
        assert (exp_dir / "datasets" / "val" / "windowed_sequences.npz").exists()
        assert not (exp_dir / "datasets" / "train" / "sequences.npz").exists()
        assert (exp_dir / "predictions" / "predictions.csv").exists()
        assert (exp_dir / "metrics" / "metrics.json").exists()
        assert (exp_dir / "metadata" / "preprocessing_state.json").exists()
        assert (exp_dir / "preprocessing" / "preprocessing_pipeline.joblib").exists()
        assert (exp_dir / "models" / "model.pt").exists()
        assert (exp_dir / "manifest.json").exists()

        metadata = json.loads((exp_dir / "metadata" / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["python"]["version"]
        assert "cuda_available" in metadata["device"]
        assert metadata["model"]["parameter_count"] > 0
        assert metadata["model"]["best_epoch"] == 1
        assert metadata["model"]["dataloader"]["device"] == "cpu"
        assert metadata["timing"]["training_duration_seconds"] is not None
        assert metadata["config_summary"]["model_type"] == "lstm"

        profile = json.loads((exp_dir / "profiles" / "profile.json").read_text(encoding="utf-8"))
        assert profile["schema_version"] == 1
        assert "load_data" in profile["stage_seconds"]
        assert "train_model" in profile["stage_seconds"]

        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["schema_version"] == 1
        assert "metrics/metrics.json" in manifest["artifacts"]
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_svm_run_writes_canonical_artifact_layout():
    scratch = Path("scratch") / "svm_artifact_test"
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

        assert report.metrics["RMSE"] >= 0.0
        assert (exp_dir / "configs" / "config.yaml").exists()
        assert (exp_dir / "configs" / "config_resolved.json").exists()
        assert (exp_dir / "models" / "model.joblib").exists()
        assert (exp_dir / "datasets" / "train" / "data.npz").exists()
        assert (exp_dir / "datasets" / "val" / "data.npz").exists()
        assert (exp_dir / "datasets" / "test" / "data.npz").exists()
        assert (exp_dir / "predictions" / "predictions.csv").exists()
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
