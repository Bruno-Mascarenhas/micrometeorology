"""Centralized experiment artifact writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solrad_correction.config import ExperimentConfig
from solrad_correction.data.preprocessing import PreprocessingPipeline
from solrad_correction.evaluation.reports import ExperimentReport
from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
from solrad_correction.experiments.results import ExperimentResult, PipelineProfile
from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.models.registry import get_model_spec
from solrad_correction.utils.io import save_json, save_predictions


@dataclass(slots=True)
class ExperimentWriter:
    """Own all stable paths and artifact writes for one experiment."""

    layout: ArtifactLayout

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> ExperimentWriter:
        """Build a writer targeting ``config.experiment_dir``."""
        return cls(ArtifactLayout.from_experiment_dir(config.experiment_dir))

    def prepare(self) -> None:
        """Create the experiment directory tree (idempotent)."""
        self.layout.ensure_directories()

    def write_result(
        self,
        *,
        config: ExperimentConfig,
        result: ExperimentResult,
        profile: PipelineProfile,
    ) -> None:
        """Write the run-final artifacts and the manifest.

        The fitted preprocessing and the trained model are written by their own
        pipeline stages as soon as they exist, so this finalizer requires them
        already on disk instead of writing the identical bytes a second time.

        The artifact writes are timed as the ``write_experiment_results`` stage
        and the stage closes before ``profile.json`` is serialized, so the
        profile reports its own artifact-writing cost.
        """
        self._require_persisted_stages(config)
        with profile.stage("write_experiment_results"):
            self.prepare()
            self.write_config(config)
            self.write_datasets(result)
            self.write_report(result.report)
            self.write_predictions(result)
        self.write_profile(config, profile)
        self.write_manifest(config)

    def _require_persisted_stages(self, config: ExperimentConfig) -> None:
        """Fail loudly when preprocessing or the model was never persisted.

        Raises
        ------
        RuntimeError
            If any artifact written by the ``persist_preprocessing`` or
            ``persist_model`` stage is missing.
        """
        spec = get_model_spec(config.model.model_type)
        model_path = self.layout.model_pt if spec.kind == "sequence" else self.layout.model_joblib
        missing = [
            str(path)
            for path in (
                self.layout.preprocessing_joblib,
                self.layout.preprocessing_state,
                model_path,
            )
            if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                "write_result called before persist_preprocessing/persist_model; "
                f"missing artifacts: {', '.join(missing)}"
            )

    def write_config(self, config: ExperimentConfig) -> None:
        """Write the run's ``config.yaml``."""
        config.save(self.layout.config_yaml)

    def write_preprocessing(self, pipeline: PreprocessingPipeline) -> None:
        """Persist the fitted preprocessing state; safe to call before predictions."""
        self.prepare()
        pipeline.save(self.layout.preprocessing_joblib)
        pipeline.save_state_json(self.layout.preprocessing_state)

    def write_datasets(self, result: ExperimentResult) -> None:
        """Serialize the train/val/test datasets under ``datasets/`` (val optional)."""
        from solrad_correction.datasets.serialization import save_dataset

        feature_names = result.processed.feature_cols
        save_dataset(
            result.datasets.train, self.layout.datasets_dir / "train", feature_names=feature_names
        )
        if result.datasets.val is not None:
            save_dataset(
                result.datasets.val, self.layout.datasets_dir / "val", feature_names=feature_names
            )
        save_dataset(
            result.datasets.test, self.layout.datasets_dir / "test", feature_names=feature_names
        )

    def write_model(self, config: ExperimentConfig, model: BaseRegressorModel) -> None:
        """Persist the trained model; safe to call immediately after fit."""
        self.prepare()
        spec = get_model_spec(config.model.model_type)
        if spec.kind == "sequence":
            model.save(self.layout.model_pt)
        else:
            model.save(self.layout.model_joblib)

    def write_report(self, report: ExperimentReport) -> None:
        """Write metrics, resolved config, and (when present) training history and metadata.

        A history that already carries absolute ``epoch`` numbers uses them as
        the CSV index; one without them falls back to positional epochs, so the
        column is never written twice.
        """
        save_json(report.metrics, self.layout.metrics)
        save_json(report.config, self.layout.config_resolved)
        if report.train_history:
            import pandas as pd

            history_frame = pd.DataFrame(report.train_history)
            if "epoch" in history_frame.columns:
                history_frame.set_index("epoch").to_csv(self.layout.training_history)
            else:
                history_frame.to_csv(self.layout.training_history, index_label="epoch")
        if report.metadata:
            save_json(report.metadata, self.layout.metadata)

    def write_predictions(self, result: ExperimentResult) -> None:
        """Write the aligned ``y_true``/``y_pred`` table to ``predictions.csv``."""
        save_predictions(
            result.evaluation.y_true,
            result.evaluation.y_pred,
            self.layout.predictions,
            result.predictions.index,
        )

    def write_profile(self, config: ExperimentConfig, profile: PipelineProfile) -> None:
        """Write per-stage timing to ``profile.json`` — only when profiling is enabled."""
        if config.runtime.profile:
            save_json(
                {
                    "schema_version": 1,
                    "stage_seconds": profile.stage_seconds,
                    "total_stage_seconds": sum(profile.stage_seconds.values()),
                },
                self.layout.profile,
            )

    def write_manifest(self, config: ExperimentConfig) -> None:
        """Write ``manifest.json`` — the checksummed inventory of all artifacts."""
        write_manifest(self.layout, extra=self.manifest_extra(config))

    @staticmethod
    def manifest_extra(config: ExperimentConfig) -> dict[str, Any]:
        """Extra provenance fields (experiment name, model type) embedded in the manifest."""
        return {
            "experiment_name": config.name,
            "model_type": config.model.model_type.lower(),
            "profile_enabled": config.runtime.profile,
        }
