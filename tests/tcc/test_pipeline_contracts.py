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
from solrad_correction.experiments.results import DatasetBundle, LoadedData, PreprocessedSplits
from solrad_correction.experiments.runner import run_experiment
from solrad_correction.models.base import BaseRegressorModel


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


def _gapped_preprocessed_dataset_splits(*, temporal_index: bool) -> PreprocessedSplits:
    """Splits whose test half has a 3-hour hole, as ``impute_strategy: drop`` leaves."""
    full_index = pd.date_range("2024-01-01", periods=34, freq="1h")
    kept_positions = [position for position in range(34) if position not in {24, 25}]
    frame = pd.DataFrame(
        {
            "feature_a": np.arange(34, dtype=np.float32)[kept_positions],
            "feature_b": np.arange(100, 134, dtype=np.float32)[kept_positions],
            "target": np.arange(200, 234, dtype=np.float32)[kept_positions],
        },
        index=full_index[kept_positions] if temporal_index else pd.RangeIndex(len(kept_positions)),
    )
    preprocessing_pipeline = PreprocessingPipeline(
        scaler_type="none",
        impute_strategy="drop",
        feature_columns=["feature_a", "feature_b"],
        target_column="target",
    )
    preprocessing_pipeline.fit(frame.iloc[:12])
    return PreprocessedSplits(
        train=preprocessing_pipeline.transform(frame.iloc[:12]),
        val=preprocessing_pipeline.transform(frame.iloc[12:20]),
        test=preprocessing_pipeline.transform(frame.iloc[20:]),
        pipeline=preprocessing_pipeline,
        feature_cols=["feature_a", "feature_b"],
    )


def _common_horizon_bundle(splits: PreprocessedSplits, model_type: str) -> DatasetBundle:
    return build_datasets(
        ExperimentConfig(
            data=DataConfig(target_column="target", feature_columns=["feature_a", "feature_b"]),
            model=ModelConfig(
                model_type=model_type,
                sequence_length=3,
                evaluation_policy="common_sequence_horizon",
            ),
        ),
        splits,
    )


def test_common_sequence_horizon_evaluates_both_model_kinds_on_the_same_rows() -> None:
    """Regression: a gapped test split must not evaluate SVM on rows the LSTM dropped."""
    splits = _gapped_preprocessed_dataset_splits(temporal_index=True)

    tabular_bundle = _common_horizon_bundle(splits, "svm")
    sequence_bundle = _common_horizon_bundle(splits, "lstm")

    assert len(splits.test) == 12
    # 10 windows start inside the split; the 2 straddling the 3-hour hole go.
    assert len(sequence_bundle.y_true) == 8
    assert len(tabular_bundle.y_true) == len(sequence_bundle.y_true)
    np.testing.assert_array_equal(tabular_bundle.y_true, sequence_bundle.y_true)
    assert tabular_bundle.prediction_index is not None
    assert sequence_bundle.prediction_index is not None
    assert tabular_bundle.prediction_index.equals(sequence_bundle.prediction_index)


def test_common_sequence_horizon_keeps_positional_rows_without_a_datetime_index() -> None:
    """A non-temporal index makes the sequence dataset gap-blind, so the trim stays positional."""
    splits = _gapped_preprocessed_dataset_splits(temporal_index=False)

    tabular_bundle = _common_horizon_bundle(splits, "svm")
    sequence_bundle = _common_horizon_bundle(splits, "lstm")

    assert len(sequence_bundle.y_true) == 10
    assert len(tabular_bundle.y_true) == 10
    np.testing.assert_array_equal(tabular_bundle.y_true, sequence_bundle.y_true)
    assert tabular_bundle.prediction_index is None
    assert sequence_bundle.prediction_index is None


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


def test_build_features_never_materializes_same_row_functions_of_the_target() -> None:
    """Regression: SW_dif_lag_1 + SW_dif_diff_1 reconstructs SW_dif exactly."""
    index = pd.date_range("2024-06-01", periods=48, freq="1h")
    frame = pd.DataFrame(
        {
            "SWDOWN": np.arange(48, dtype=np.float32),
            "SW_dif": np.arange(100, 148, dtype=np.float32),
        },
        index=index,
    )
    cfg = ExperimentConfig(
        data=DataConfig(target_column="SW_dif", feature_columns=["SWDOWN", "SW_dif"]),
        features=FeatureConfig(
            add_temporal=False,
            cyclic_encoding=False,
            lag_steps=[1],
            rolling_windows=[3],
            rolling_aggs=["mean"],
            add_diffs=True,
        ),
    )

    features = build_features(LoadedData(frame=frame), cfg)

    assert {"SWDOWN_lag_1", "SWDOWN_roll_mean_3", "SWDOWN_diff_1"}.issubset(features.feature_cols)
    # An autoregressive target lag stays; nothing reading the target's own row does.
    assert "SW_dif_lag_1" in features.feature_cols
    assert "SW_dif_diff_1" not in features.frame.columns
    assert "SW_dif_roll_mean_3" not in features.frame.columns
    assert "SW_dif" not in features.feature_cols


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


def test_run_writes_each_artifact_once_and_profiles_the_artifact_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for findings 7 + 8: no duplicated writes, and the final stage is profiled."""
    from solrad_correction.experiments.writer import ExperimentWriter

    scratch = Path("scratch") / "svm_single_write_contract"
    data_path = scratch / "hourly.parquet"
    output_dir = scratch / "output"
    write_counts = {"model": 0, "preprocessing": 0}
    write_model = ExperimentWriter.write_model
    write_preprocessing = ExperimentWriter.write_preprocessing

    def counting_write_model(
        writer: ExperimentWriter, config: ExperimentConfig, model: BaseRegressorModel
    ) -> None:
        write_counts["model"] += 1
        write_model(writer, config, model)

    def counting_write_preprocessing(
        writer: ExperimentWriter, pipeline: PreprocessingPipeline
    ) -> None:
        write_counts["preprocessing"] += 1
        write_preprocessing(writer, pipeline)

    try:
        scratch.mkdir(parents=True, exist_ok=True)
        index = pd.date_range("2024-01-01", periods=48, freq="1h")
        rng = np.random.default_rng(5)
        f1 = rng.normal(size=48).astype(np.float32)
        target = (0.5 * f1).astype(np.float32)
        pd.DataFrame({"f1": f1, "target": target}, index=index).to_parquet(data_path)

        cfg = ExperimentConfig(
            name="svm_single_write",
            data=DataConfig(
                hourly_data_path=str(data_path),
                source_format="parquet",
                target_column="target",
                feature_columns=["f1"],
            ),
            split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
            features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
            model=ModelConfig(model_type="svm"),
            runtime=RuntimeConfig(device="cpu", profile=True),
            output_dir=str(output_dir),
        )
        monkeypatch.setattr(ExperimentWriter, "write_model", counting_write_model)
        monkeypatch.setattr(ExperimentWriter, "write_preprocessing", counting_write_preprocessing)

        run_experiment(cfg)

        exp_dir = output_dir / "svm_single_write"
        profile = json.loads((exp_dir / "profiles" / "profile.json").read_text(encoding="utf-8"))
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))

        assert write_counts == {"model": 1, "preprocessing": 1}
        assert "write_experiment_results" in profile["stage_seconds"]
        assert profile["total_stage_seconds"] == sum(profile["stage_seconds"].values())
        for relative in [
            "models/model.joblib",
            "preprocessing/preprocessing_pipeline.joblib",
            "metadata/preprocessing_state.json",
        ]:
            assert (exp_dir / relative).exists()
            assert relative in manifest["artifacts"]
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)


def test_write_result_refuses_to_finalize_before_the_persist_stages(tmp_path: Path) -> None:
    """Regression for finding 7: the finalizer must never manifest a model-less run."""
    from solrad_correction.evaluation.reports import ExperimentReport
    from solrad_correction.experiments.pipeline import build_configured_model
    from solrad_correction.experiments.results import (
        EvaluationResult,
        ExperimentResult,
        PipelineProfile,
        PredictionOutput,
    )
    from solrad_correction.experiments.writer import ExperimentWriter

    preprocessed_splits = _preprocessed_dataset_splits()
    cfg = ExperimentConfig(
        name="svm_unpersisted",
        data=DataConfig(target_column="target", feature_columns=["feature_a", "feature_b"]),
        model=ModelConfig(model_type="svm"),
        runtime=RuntimeConfig(device="cpu"),
        output_dir=str(tmp_path),
    )
    dataset_bundle = build_datasets(cfg, preprocessed_splits)
    empty_predictions = np.zeros(0, dtype=np.float32)
    experiment_result = ExperimentResult(
        report=ExperimentReport(experiment_name=cfg.name, model_name="svm"),
        processed=preprocessed_splits,
        datasets=dataset_bundle,
        model=build_configured_model(cfg, dataset_bundle),
        predictions=PredictionOutput(
            y_true=empty_predictions, y_pred=empty_predictions, index=None
        ),
        evaluation=EvaluationResult(y_true=empty_predictions, y_pred=empty_predictions, metrics={}),
    )
    writer = ExperimentWriter.from_config(cfg)

    with pytest.raises(RuntimeError, match="persist_model"):
        writer.write_result(
            config=cfg,
            result=experiment_result,
            profile=PipelineProfile(stage_seconds={}),
        )

    assert not (cfg.experiment_dir / "manifest.json").exists()


def test_training_history_csv_keeps_a_single_absolute_epoch_column(tmp_path: Path) -> None:
    """Regression for finding 9: an epoch-carrying history must not duplicate the column."""
    from solrad_correction.evaluation.reports import ExperimentReport
    from solrad_correction.experiments.writer import ExperimentWriter

    writer = ExperimentWriter.from_config(
        ExperimentConfig(name="history", output_dir=str(tmp_path))
    )
    writer.prepare()

    writer.write_report(
        ExperimentReport(
            experiment_name="history",
            model_name="lstm",
            train_history={"epoch": [31, 32], "train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]},
        )
    )
    history = pd.read_csv(tmp_path / "history" / "metrics" / "training_history.csv")

    assert list(history.columns) == ["epoch", "train_loss", "val_loss"]
    assert history["epoch"].tolist() == [31, 32]


def test_a_resumed_run_merges_its_history_with_the_rows_already_on_disk(
    tmp_path: Path,
) -> None:
    """A resume used to relabel its epochs from 0 and overwrite the earlier curve."""
    from solrad_correction.evaluation.reports import ExperimentReport
    from solrad_correction.experiments.writer import ExperimentWriter

    writer = ExperimentWriter.from_config(
        ExperimentConfig(name="resumed", output_dir=str(tmp_path))
    )
    writer.prepare()
    history_path = tmp_path / "resumed" / "metrics" / "training_history.csv"

    writer.write_report(
        ExperimentReport(
            experiment_name="resumed",
            model_name="lstm",
            train_history={"epoch": [1, 2], "train_loss": [0.9, 0.7], "val_loss": [1.0, 0.8]},
        )
    )
    writer.write_report(
        ExperimentReport(
            experiment_name="resumed",
            model_name="lstm",
            train_history={"epoch": [3, 4], "train_loss": [0.5, 0.4], "val_loss": [0.6, 0.5]},
        ),
        merge_existing=True,
    )

    history = pd.read_csv(history_path)
    assert history["epoch"].tolist() == [1, 2, 3, 4]
    assert history["train_loss"].tolist() == [0.9, 0.7, 0.5, 0.4]


def test_a_rerun_of_the_same_epochs_replaces_them_rather_than_duplicating(
    tmp_path: Path,
) -> None:
    from solrad_correction.evaluation.reports import ExperimentReport
    from solrad_correction.experiments.writer import ExperimentWriter

    writer = ExperimentWriter.from_config(ExperimentConfig(name="rerun", output_dir=str(tmp_path)))
    writer.prepare()

    writer.write_report(
        ExperimentReport(
            experiment_name="rerun",
            model_name="lstm",
            train_history={"epoch": [1, 2], "train_loss": [0.9, 0.7]},
        )
    )
    writer.write_report(
        ExperimentReport(
            experiment_name="rerun",
            model_name="lstm",
            train_history={"epoch": [2, 3], "train_loss": [0.6, 0.5]},
        ),
        merge_existing=True,
    )

    history = pd.read_csv(tmp_path / "rerun" / "metrics" / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2, 3]
    assert history["train_loss"].tolist() == [0.9, 0.6, 0.5]


def test_the_trainer_records_absolute_epoch_numbers() -> None:
    """The history's ``epoch`` column is what makes a resume mergeable."""
    pytest.importorskip("torch")
    import torch

    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.sequence import WindowedSequenceDataset
    from solrad_correction.training.trainer import Trainer

    rng = np.random.default_rng(4)
    features = rng.normal(0, 1, (60, 3)).astype("float32")
    target = rng.normal(0, 1, 60).astype("float32")
    dataset = WindowedSequenceDataset(features, target, sequence_length=4)

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            prediction: torch.Tensor = self.linear(x[:, -1, :])
            return prediction

    config = ModelConfig(model_type="lstm", max_epochs=5, batch_size=16)
    trainer = Trainer(model=Tiny(), device="cpu", config=config, start_epoch=3)
    _model, history = trainer.train(dataset)

    # start_epoch=3 with max_epochs=5 trains epochs 4 and 5, not 1 and 2.
    assert history["epoch"] == [4.0, 5.0]
    assert len(history["train_loss"]) == 2


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
