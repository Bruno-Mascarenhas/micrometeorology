"""Model factory helpers.

The name/kind table lives in the torch-free
:mod:`solrad_correction.models.contracts` and is re-exported here so
``models.registry`` stays the single import site for callers that also build
models. Metadata-only callers should import from ``models.contracts`` directly:
building a model needs torch, looking one up does not.
"""

from solrad_correction.config import ModelConfig
from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.models.contracts import (
    MODEL_REGISTRY,
    ModelKind,
    ModelSpec,
    get_model_spec,
    supported_model_names,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelKind",
    "ModelSpec",
    "build_model",
    "get_model_spec",
    "supported_model_names",
]


def build_model(
    config: ModelConfig,
    *,
    input_size: int | None = None,
    device: str | None = None,
) -> BaseRegressorModel:
    """Build a model from config using the registry.

    Each concrete model module is imported inside the branch that builds it, so
    a tabular run never pays for a torch import.

    Parameters
    ----------
    config:
        Model section of the experiment config: ``model_type`` selects the
        model and the remaining fields supply that model's hyperparameters.
    input_size:
        Number of input features ``F`` per time step. Required by the sequence
        models, which size their first layer from it; unused by the tabular
        ones, which take their width from the fitted estimator.
    device:
        Torch device string the sequence models are moved to. ``None`` lets the
        model resolve the device itself.

    Returns
    -------
    BaseRegressorModel
        An untrained model instance.

    Raises
    ------
    ValueError
        If ``config.model_type`` is not registered, if a sequence model is
        requested without ``input_size``, or if a registered name reaches the
        end of this factory without a branch that builds it.
    """
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
