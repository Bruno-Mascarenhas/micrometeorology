"""Preprocessing configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class PreprocessConfig:
    """Preprocessing pipeline settings."""

    scaler_type: str = "standard"
    impute_strategy: str = "drop"
    drop_na_threshold: float = 0.5
