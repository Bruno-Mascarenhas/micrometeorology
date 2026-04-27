"""End-to-end experiment artifact contracts."""

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
        predictions = pd.read_csv(exp_dir / "predictions" / "predictions.csv")
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))

        assert report.metrics["RMSE"] >= 0.0
        assert {"y_true", "y_pred"}.issubset(predictions.columns)
        assert len(predictions) == 8
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

        assert report.train_history["train_loss"]
        assert profile["schema_version"] == 1
        assert "load_data" in profile["stage_seconds"]
        assert "train_model" in profile["stage_seconds"]
        assert metadata["model"]["parameter_count"] > 0
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
