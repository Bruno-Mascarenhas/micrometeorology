"""Model registry and factory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from solrad_correction.config import ModelConfig
    from solrad_correction.models.base import BaseRegressorModel

ModelKind = Literal["tabular", "sequence"]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Registered model metadata."""

    name: str
    kind: ModelKind


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "svm": ModelSpec(name="svm", kind="tabular"),
    "lstm": ModelSpec(name="lstm", kind="sequence"),
    "transformer": ModelSpec(name="transformer", kind="sequence"),
}


def supported_model_names() -> tuple[str, ...]:
    """Return the supported public model names."""
    return tuple(sorted(MODEL_REGISTRY))


def get_model_spec(model_type: str) -> ModelSpec:
    """Return the registered spec for a model type."""
    key = model_type.lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model type: {model_type}. Available models: {available}"
        ) from exc


def build_model(
    config: ModelConfig,
    *,
    input_size: int | None = None,
    device: str | None = None,
) -> BaseRegressorModel:
    """Build a model from config using the registry."""
    spec = get_model_spec(config.model_type)

    if spec.name == "svm":
        from solrad_correction.models.svm import SVMRegressor

        return SVMRegressor.from_config(config)

    if input_size is None:
        raise ValueError(f"Model '{spec.name}' requires input_size")

    if spec.name == "lstm":
        from solrad_correction.models.lstm import LSTMRegressor

        return LSTMRegressor.from_config(config, input_size, device=device)

    if spec.name == "transformer":
        from solrad_correction.models.transformer import TransformerRegressor

        return TransformerRegressor.from_config(config, input_size, device=device)

    raise ValueError(f"Registered model '{spec.name}' has no factory")
