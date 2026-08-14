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
    """Registered model metadata.

    Attributes
    ----------
    name:
        Canonical lowercase name a config selects the model by.
    kind:
        ``"tabular"`` for a model consuming independent rows, ``"sequence"``
        for one consuming sliding windows over the time axis. The kind is what
        decides how many test rows a model predicts, and therefore which rows
        two model families have in common at evaluation time.
    """

    name: str
    kind: ModelKind


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "svm": ModelSpec(name="svm", kind="tabular"),
    "lstm": ModelSpec(name="lstm", kind="sequence"),
    "transformer": ModelSpec(name="transformer", kind="sequence"),
}


def supported_model_names() -> tuple[str, ...]:
    """Return the supported public model names.

    Returns
    -------
    tuple of str
        Every registered name, alphabetically ordered so config validation
        messages and CLI choice lists stay stable between runs.
    """
    return tuple(sorted(MODEL_REGISTRY))


def get_model_spec(model_type: str) -> ModelSpec:
    """Return the registered spec for a model type.

    Parameters
    ----------
    model_type:
        Model name as written in the config, matched case-insensitively.

    Returns
    -------
    ModelSpec
        The registered spec.

    Raises
    ------
    ValueError
        If the name is not registered. The message lists the names that are.
    """
    key = model_type.lower()
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"Unknown model type: {model_type}. Available models: {available}"
        ) from exc
