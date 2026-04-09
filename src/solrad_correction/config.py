"""Experiment and model configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
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


@dataclass
class SplitConfig:
    """Train / validation / test split settings."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False  # Default: chronological (no shuffle for time series)


@dataclass
class PreprocessConfig:
    """Preprocessing pipeline settings."""

    scaler_type: str = "standard"  # "standard", "minmax", "none"
    impute_strategy: str = "drop"  # "drop", "ffill", "mean", "interpolate"
    drop_na_threshold: float = 0.5  # Drop columns with > 50% NaN


@dataclass
class FeatureConfig:
    """Feature engineering settings."""

    lag_steps: list[int] = field(default_factory=list)  # e.g. [1, 2, 3, 6, 12, 24]
    rolling_windows: list[int] = field(default_factory=list)  # e.g. [3, 6, 12, 24]
    rolling_aggs: list[str] = field(default_factory=lambda: ["mean", "std"])
    add_temporal: bool = True  # hour, day_of_year, month
    cyclic_encoding: bool = True  # sin/cos for temporal features
    add_diffs: bool = False  # first differences


@dataclass
class ModelConfig:
    """Model-specific hyperparameters."""

    model_type: str = "svm"  # "svm", "lstm", "transformer"

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

    # Transfer learning
    pretrained_path: str | None = None  # Path to load weights from


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

    def save(self, path: str | Path) -> None:
        """Save config to YAML."""
        import dataclasses

        def _to_dict(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            return obj

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False, sort_keys=False)

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
