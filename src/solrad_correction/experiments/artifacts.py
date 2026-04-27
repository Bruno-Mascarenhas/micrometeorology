"""Experiment artifact layout and manifest helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solrad_correction.utils.io import save_json


@dataclass(frozen=True, slots=True)
class ArtifactLayout:
    """Canonical artifact locations for one experiment run."""

    root: Path

    @classmethod
    def from_experiment_dir(cls, experiment_dir: str | Path) -> ArtifactLayout:
        return cls(Path(experiment_dir))

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def preprocessing_dir(self) -> Path:
        return self.root / "preprocessing"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def datasets_dir(self) -> Path:
        return self.root / "datasets"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def profiles_dir(self) -> Path:
        return self.root / "profiles"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def config_yaml(self) -> Path:
        return self.configs_dir / "config.yaml"

    @property
    def config_resolved(self) -> Path:
        return self.configs_dir / "config_resolved.json"

    @property
    def metrics(self) -> Path:
        return self.metrics_dir / "metrics.json"

    @property
    def training_history(self) -> Path:
        return self.metrics_dir / "training_history.csv"

    @property
    def predictions(self) -> Path:
        return self.predictions_dir / "predictions.csv"

    @property
    def metadata(self) -> Path:
        return self.metadata_dir / "metadata.json"

    @property
    def preprocessing_state(self) -> Path:
        return self.metadata_dir / "preprocessing_state.json"

    @property
    def preprocessing_joblib(self) -> Path:
        return self.preprocessing_dir / "preprocessing_pipeline.joblib"

    @property
    def model_pt(self) -> Path:
        return self.models_dir / "model.pt"

    @property
    def model_joblib(self) -> Path:
        return self.models_dir / "model.joblib"

    @property
    def profile(self) -> Path:
        return self.profiles_dir / "profile.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    def ensure_directories(self) -> None:
        """Create the stable experiment directory tree."""
        for directory in [
            self.configs_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.metadata_dir,
            self.preprocessing_dir,
            self.models_dir,
            self.checkpoints_dir,
            self.datasets_dir / "train",
            self.datasets_dir / "val",
            self.datasets_dir / "test",
            self.logs_dir,
            self.profiles_dir,
            self.cache_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def write_manifest(layout: ArtifactLayout, *, extra: dict[str, Any] | None = None) -> None:
    """Write a manifest of existing artifacts with schema and checksums."""
    artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(p for p in layout.root.rglob("*") if p.is_file() and p != layout.manifest):
        relative = path.relative_to(layout.root).as_posix()
        artifacts[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    save_json(
        {
            "schema_version": 1,
            "artifact_layout": "solrad_correction_experiment_v2",
            "root": str(layout.root),
            "artifacts": artifacts,
            "extra": extra or {},
        },
        layout.manifest,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
