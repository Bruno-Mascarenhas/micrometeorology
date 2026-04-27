"""Centralized experiment artifact writer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
from solrad_correction.models.registry import get_model_spec
from solrad_correction.utils.io import save_json, save_predictions

if TYPE_CHECKING:
    from solrad_correction.config import ExperimentConfig
    from solrad_correction.evaluation.reports import ExperimentReport
    from solrad_correction.experiments.pipeline import PipelineProfile
    from solrad_correction.experiments.results import ExperimentResult


@dataclass(slots=True)
class ExperimentWriter:
    """Own all stable paths and artifact writes for one experiment."""

    layout: ArtifactLayout

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> ExperimentWriter:
        return cls(ArtifactLayout.from_experiment_dir(config.experiment_dir))

    def prepare(self) -> None:
        self.layout.ensure_directories()

    def write_result(
        self,
        *,
        config: ExperimentConfig,
        result: ExperimentResult,
        profile: PipelineProfile,
    ) -> None:
        """Write the full v2 artifact set and final manifest."""
        self.prepare()
        self.write_config(config)
        self.write_preprocessing(result)
        self.write_datasets(result)
        self.write_model(config, result)
        self.write_report(result.report)
        self.write_predictions(result)
        self.write_profile(config, profile)
        self.write_manifest(config)

    def write_config(self, config: ExperimentConfig) -> None:
        config.save(self.layout.config_yaml)

    def write_preprocessing(self, result: ExperimentResult) -> None:
        result.processed.pipeline.save(self.layout.preprocessing_joblib)
        result.processed.pipeline.save_state_json(self.layout.preprocessing_state)

    def write_datasets(self, result: ExperimentResult) -> None:
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

    def write_model(self, config: ExperimentConfig, result: ExperimentResult) -> None:
        spec = get_model_spec(config.model.model_type)
        if spec.kind == "sequence":
            result.model.save(self.layout.model_pt)
        else:
            result.model.save(self.layout.model_joblib)

    def write_report(self, report: ExperimentReport) -> None:
        save_json(report.metrics, self.layout.metrics)
        save_json(report.config, self.layout.config_resolved)
        if report.train_history:
            import pandas as pd

            pd.DataFrame(report.train_history).to_csv(
                self.layout.training_history,
                index_label="epoch",
            )
        if report.metadata:
            save_json(report.metadata, self.layout.metadata)

    def write_predictions(self, result: ExperimentResult) -> None:
        save_predictions(
            result.evaluation.y_true,
            result.evaluation.y_pred,
            self.layout.predictions,
            result.predictions.index,
        )

    def write_profile(self, config: ExperimentConfig, profile: PipelineProfile) -> None:
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
        write_manifest(self.layout, extra=self.manifest_extra(config))

    @staticmethod
    def manifest_extra(config: ExperimentConfig) -> dict[str, Any]:
        return {
            "experiment_name": config.name,
            "model_type": config.model.model_type.lower(),
            "profile_enabled": config.runtime.profile,
        }
