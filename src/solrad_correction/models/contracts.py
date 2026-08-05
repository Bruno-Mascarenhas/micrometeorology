"""Model name/kind table for solrad_correction.

This module is deliberately **torch-free**: it imports nothing but the standard
library, so config validation and evaluation row-alignment — which only need to
know the supported model names and whether a model is sequential — never pay
for (or require) a torch import. The factory that actually builds the models
lives in :mod:`solrad_correction.models.registry`.
"""

from dataclasses import dataclass
from typing import Literal

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
