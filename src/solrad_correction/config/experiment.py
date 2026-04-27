"""Top-level experiment configuration and validation."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from solrad_correction.config.data import DataConfig
from solrad_correction.config.features import FeatureConfig
from solrad_correction.config.models import ModelConfig
from solrad_correction.config.preprocessing import PreprocessConfig
from solrad_correction.config.runtime import RuntimeConfig
from solrad_correction.config.split import SplitConfig


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""

    name: str = "unnamed"
    description: str = ""
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output_dir: str = "output/experiments"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        errors: list[str] = []

        from solrad_correction.models.registry import supported_model_names

        supported_models = supported_model_names()
        model_type = self.model.model_type.lower()
        if model_type not in supported_models:
            errors.append(f"model.model_type must be one of: {', '.join(supported_models)}")

        if self.data.source_format not in {"auto", "csv", "parquet"}:
            errors.append("data.source_format must be one of: auto, csv, parquet")
        if self.data.sensor_min_samples <= 0:
            errors.append("data.sensor_min_samples must be positive")

        if self.model.evaluation_policy not in {"model_native", "common_sequence_horizon"}:
            errors.append(
                "model.evaluation_policy must be one of: model_native, common_sequence_horizon"
            )

        split_total = self.split.train_ratio + self.split.val_ratio + self.split.test_ratio
        if abs(split_total - 1.0) > 1e-6:
            errors.append(f"split ratios must sum to 1.0, got {split_total:.4f}")
        if min(self.split.train_ratio, self.split.val_ratio, self.split.test_ratio) < 0:
            errors.append("split ratios must be non-negative")

        if self.model.sequence_length <= 0:
            errors.append("model.sequence_length must be positive")
        if self.model.batch_size <= 0:
            errors.append("model.batch_size must be positive")
        if self.model.max_epochs <= 0:
            errors.append("model.max_epochs must be positive")
        if self.model.tf_d_model <= 0:
            errors.append("model.tf_d_model must be positive")
        if self.model.tf_nhead <= 0:
            errors.append("model.tf_nhead must be positive")
        elif self.model.tf_d_model % self.model.tf_nhead != 0:
            errors.append("model.tf_d_model must be divisible by model.tf_nhead")

        if self.runtime.device not in {"auto", "cpu", "cuda"}:
            errors.append("runtime.device must be one of: auto, cpu, cuda")
        if self.runtime.num_workers is not None and self.runtime.num_workers < 0:
            errors.append("runtime.num_workers must be non-negative")
        if self.runtime.prefetch_factor is not None and self.runtime.prefetch_factor <= 0:
            errors.append("runtime.prefetch_factor must be positive when set")
        if self.runtime.num_workers == 0 and self.runtime.prefetch_factor is not None:
            errors.append("runtime.prefetch_factor requires runtime.num_workers > 0")
        if self.runtime.gradient_clip is not None and self.runtime.gradient_clip < 0:
            errors.append("runtime.gradient_clip must be non-negative when set")
        if self.runtime.checkpoint_every is not None and self.runtime.checkpoint_every <= 0:
            errors.append("runtime.checkpoint_every must be positive when set")
        if self.runtime.limit_rows is not None and self.runtime.limit_rows <= 0:
            errors.append("runtime.limit_rows must be positive when set")

        if errors:
            joined = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"Invalid experiment config:\n{joined}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.dump(self.to_dict(), handle, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            data=DataConfig(**data.get("data", {})),
            split=SplitConfig(**data.get("split", {})),
            preprocess=PreprocessConfig(**data.get("preprocess", {})),
            features=FeatureConfig(**data.get("features", {})),
            model=ModelConfig(**data.get("model", {})),
            runtime=RuntimeConfig(**data.get("runtime", {})),
            output_dir=data.get("output_dir", "output/experiments"),
        )
