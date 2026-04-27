"""Experiment and model configuration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore


@dataclass(slots=True)
class DataConfig:
    """Data loading and preparation settings."""

    sensor_data_path: str | None = None
    hourly_data_path: str | None = None
    wrf_data_path: str | None = None

    target_column: str = "SW_dif"
    feature_columns: list[str] = field(default_factory=list)

    # Temporal resolution
    use_raw: bool = False  # True = raw sensor data; False = hourly aggregated
    resample_freq: str | None = None  # Optional resampling (e.g. "30min")

    # Station coordinates (for WRF grid point extraction)
    station_lat: float = -12.95
    station_lon: float = -38.51


@dataclass(slots=True)
class SplitConfig:
    """Train / validation / test split settings."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False  # Default: chronological (no shuffle for time series)


@dataclass(slots=True)
class PreprocessConfig:
    """Preprocessing pipeline settings."""

    scaler_type: str = "standard"  # "standard", "minmax", "none"
    impute_strategy: str = "drop"  # "drop", "ffill", "mean", "interpolate"
    drop_na_threshold: float = 0.5  # Drop columns with > 50% NaN


@dataclass(slots=True)
class FeatureConfig:
    """Feature engineering settings."""

    lag_steps: list[int] = field(default_factory=list)  # e.g. [1, 2, 3, 6, 12, 24]
    rolling_windows: list[int] = field(default_factory=list)  # e.g. [3, 6, 12, 24]
    rolling_aggs: list[str] = field(default_factory=lambda: ["mean", "std"])
    add_temporal: bool = True  # hour, day_of_year, month
    cyclic_encoding: bool = True  # sin/cos for temporal features
    add_diffs: bool = False  # first differences


@dataclass(slots=True)
class ModelConfig:
    """Model-specific hyperparameters."""

    model_type: str = "svm"  # "svm", "lstm", "transformer"

    # Checkpointing / Logging
    pretrained_path: str | None = None
    log_dir: str | None = None

    # SVM
    svm_kernel: str = "rbf"
    svm_c: float = 1.0
    svm_epsilon: float = 0.1
    svm_gamma: str = "scale"

    # LSTM
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1

    # Transformer
    tf_d_model: int = 64
    tf_nhead: int = 4
    tf_num_encoder_layers: int = 2
    tf_dim_feedforward: int = 128
    tf_dropout: float = 0.1

    # Sequence models shared
    sequence_length: int = 24
    batch_size: int = 32

    # Training (neural)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10  # early stopping
    min_delta: float = 1e-4
    torch_compile: bool = False
    evaluation_policy: str = "model_native"  # "model_native" or "common_sequence_horizon"


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

    # Output
    output_dir: str = "output/experiments"

    @property
    def experiment_dir(self) -> Path:
        return Path(self.output_dir) / self.name

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved config as plain Python data."""
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Validate configuration values that can be checked before data loading."""
        errors: list[str] = []

        model_type = self.model.model_type.lower()
        if model_type not in {"svm", "lstm", "transformer"}:
            errors.append("model.model_type must be one of: svm, lstm, transformer")

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

        if errors:
            joined = "\n".join(f"- {error}" for error in errors)
            raise ValueError(f"Invalid experiment config:\n{joined}")

    def save(self, path: str | Path) -> None:
        """Save config to YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """Load config from YAML."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            seed=data.get("seed", 42),
            data=DataConfig(**data.get("data", {})),
            split=SplitConfig(**data.get("split", {})),
            preprocess=PreprocessConfig(**data.get("preprocess", {})),
            features=FeatureConfig(**data.get("features", {})),
            model=ModelConfig(**data.get("model", {})),
            output_dir=data.get("output_dir", "output/experiments"),
        )
