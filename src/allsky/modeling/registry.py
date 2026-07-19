"""Model registry: name -> builder, and :func:`build_model` from an experiment.

The registry maps the six experiment model names to builder callables that
translate an :class:`allsky.config.ExperimentConfig` (plus the discovered
feature count and, per mode, an embedding dimension or an image backbone) into
an ``nn.Module`` honouring the
:class:`allsky.modeling.contracts.MultimodalModel` contract:

- ``climatology`` -> :class:`ClimatologyModel`
- ``sensor_only`` -> :class:`SensorOnlyModel`
- ``image_only`` -> :class:`ImageOnlyModel`
- ``concat`` / ``film`` / ``cross_attention`` -> :class:`MultimodalNet`

Architecture hyper-parameters ride on the permissive
:class:`allsky.config.ExperimentModelConfig` (``extra="allow"``) and are read by
name with defaults.  ``extra="allow"`` is kept (unknown keys are preserved), but
:func:`build_model` logs a WARNING listing any ``model`` key the selected builder
does not recognise — cheap typo protection (e.g. ``droput`` silently ignored).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, cast

from torch import nn

from allsky.config import ExperimentConfig
from allsky.features.policy import resolve_feature_set
from allsky.modeling.baselines import ClimatologyModel, ImageOnlyModel, SensorOnlyModel
from allsky.modeling.multimodal import MultimodalNet
from allsky.modeling.visual_encoder import build_visual_encoder

logger = logging.getLogger(__name__)

__all__ = ["MODEL_BUILDERS", "build_model", "temporal_pooling_for_strategy"]

#: Builder signature:
#: ``(cfg, n_features, embedding_dim, image_backbone, temporal_pooling) -> nn.Module``.
ModelBuilder = Callable[
    [
        ExperimentConfig,
        int,
        int | None,
        nn.Module | None,
        Literal["mean", "attention"] | None,
    ],
    nn.Module,
]

# Per-builder recognised ``model`` hyper-parameter keys (``name`` excluded). A
# config key outside the selected model's set triggers a typo warning in
# :func:`build_model`; it is still kept (``extra="allow"``).
_COMMON_PARAMS = frozenset({"sensor_hidden", "trunk_hidden", "trunk_layers", "dropout"})
_VISUAL_PARAMS = frozenset(
    {"visual_out_dim", "backbone_frozen", "unfreeze_last_n", "temporal_pooling", "image_size"}
)
_CROSS_ATTENTION_PARAMS = frozenset({"num_heads", "token_dim"})
#: Model name -> the set of hyper-parameter keys that builder consumes.
KNOWN_MODEL_PARAMS: dict[str, frozenset[str]] = {
    "climatology": frozenset(),
    "sensor_only": _COMMON_PARAMS,
    "image_only": frozenset({"trunk_hidden", "trunk_layers", "dropout"}) | _VISUAL_PARAMS,
    "concat": _COMMON_PARAMS | _VISUAL_PARAMS,
    "film": _COMMON_PARAMS | _VISUAL_PARAMS,
    "cross_attention": _COMMON_PARAMS | _VISUAL_PARAMS | _CROSS_ATTENTION_PARAMS,
}


def _params(cfg: ExperimentConfig) -> dict[str, Any]:
    """Architecture hyper-parameters from the model config (drops ``name``)."""
    params = dict(cfg.model.model_dump())
    params.pop("name", None)
    return params


def _sensor_hidden(params: dict[str, Any]) -> tuple[int, ...]:
    """Sensor-encoder widths from *params* (default ``(64, 128)``)."""
    return tuple(params.get("sensor_hidden", (64, 128)))


def temporal_pooling_for_strategy(strategy: str) -> Literal["mean", "attention"]:
    """Visual temporal pooler implied by an alignment *strategy*.

    Only ``"attention_pooling"`` — whose dataset emits a padded ``embedding_seq``
    + ``frame_mask`` — uses the learned single-query attention pooler; every other
    strategy (``"center_frame"``, ``"mean_embedding"``) pools with the mask-aware
    mean, which is inert when the dataset emits a plain ``embedding``. The engine
    and evaluator pass this to :func:`build_model` so a windowed model is built —
    and reloaded on evaluate — with the matching pooler (an attention-pooled
    checkpoint carries the extra query/attention weights, so rebuilding it with
    ``"mean"`` would fail ``load_state_dict``).
    """
    return "attention" if strategy == "attention_pooling" else "mean"


def _temporal_pooling(
    params: dict[str, Any], override: Literal["mean", "attention"] | None = None
) -> Literal["mean", "attention"]:
    """Temporal pooling: *override* (engine/evaluator) wins, else the model param.

    The default is ``"mean"``; the value is validated downstream by
    :class:`~allsky.modeling.visual_encoder.PrecomputedEmbedding`.
    """
    if override is not None:
        return override
    return cast('Literal["mean", "attention"]', str(params.get("temporal_pooling", "mean")))


def _build_climatology(
    cfg: ExperimentConfig,
    n_features: int,  # noqa: ARG001 - uniform builder signature
    embedding_dim: int | None,  # noqa: ARG001 - uniform builder signature
    image_backbone: nn.Module | None,  # noqa: ARG001 - uniform builder signature
    temporal_pooling: Literal["mean", "attention"] | None,  # noqa: ARG001 - uniform signature
) -> nn.Module:
    return ClimatologyModel(cfg.targets)


def _build_sensor_only(
    cfg: ExperimentConfig,
    n_features: int,
    embedding_dim: int | None,  # noqa: ARG001 - uniform builder signature
    image_backbone: nn.Module | None,  # noqa: ARG001 - uniform builder signature
    temporal_pooling: Literal["mean", "attention"] | None,  # noqa: ARG001 - uniform signature
) -> nn.Module:
    params = _params(cfg)
    return SensorOnlyModel(
        n_features,
        cfg.targets,
        sensor_hidden=_sensor_hidden(params),
        trunk_hidden=int(params.get("trunk_hidden", 256)),
        trunk_layers=int(params.get("trunk_layers", 2)),
        dropout=float(params.get("dropout", 0.1)),
    )


def _build_image_only(
    cfg: ExperimentConfig,
    n_features: int,  # noqa: ARG001 - uniform builder signature
    embedding_dim: int | None,
    image_backbone: nn.Module | None,
    temporal_pooling: Literal["mean", "attention"] | None,
) -> nn.Module:
    params = _params(cfg)
    visual = build_visual_encoder(
        cfg.data.input_mode,
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        out_dim=params.get("visual_out_dim"),
        frozen=bool(params.get("backbone_frozen", False)),
        unfreeze_last_n=int(params.get("unfreeze_last_n", 0)),
        dropout=float(params.get("dropout", 0.1)),
        temporal_pooling=_temporal_pooling(params, temporal_pooling),
    )
    return ImageOnlyModel(
        visual,
        cfg.targets,
        trunk_hidden=int(params.get("trunk_hidden", 256)),
        trunk_layers=int(params.get("trunk_layers", 2)),
        dropout=float(params.get("dropout", 0.1)),
    )


def _multimodal_builder(fusion_name: str) -> ModelBuilder:
    """Return a builder that assembles a :class:`MultimodalNet` with *fusion_name*."""

    def builder(
        cfg: ExperimentConfig,
        n_features: int,  # noqa: ARG001 - width derived from the feature set
        embedding_dim: int | None,
        image_backbone: nn.Module | None,
        temporal_pooling: Literal["mean", "attention"] | None,
    ) -> nn.Module:
        params = _params(cfg)
        feature_set = cfg.features.feature_set
        return MultimodalNet(
            feature_columns=resolve_feature_set(feature_set),
            targets=cfg.targets,
            fusion_name=fusion_name,
            input_mode=cfg.data.input_mode,
            feature_set=feature_set,
            embedding_dim=embedding_dim,
            image_backbone=image_backbone,
            sensor_hidden=_sensor_hidden(params),
            visual_out_dim=params.get("visual_out_dim"),
            trunk_hidden=int(params.get("trunk_hidden", 256)),
            trunk_layers=int(params.get("trunk_layers", 2)),
            dropout=float(params.get("dropout", 0.1)),
            num_heads=int(params.get("num_heads", 4)),
            token_dim=params.get("token_dim"),
            backbone_frozen=bool(params.get("backbone_frozen", False)),
            unfreeze_last_n=int(params.get("unfreeze_last_n", 0)),
            temporal_pooling=_temporal_pooling(params, temporal_pooling),
            backbone_lr=cfg.train.backbone_lr,
        )

    return builder


#: Model name -> builder callable.
MODEL_BUILDERS: dict[str, ModelBuilder] = {
    "climatology": _build_climatology,
    "sensor_only": _build_sensor_only,
    "image_only": _build_image_only,
    "concat": _multimodal_builder("concat"),
    "film": _multimodal_builder("film"),
    "cross_attention": _multimodal_builder("cross_attention"),
}


def build_model(
    experiment_cfg: ExperimentConfig,
    n_features: int,
    embedding_dim: int | None = None,
    image_backbone: nn.Module | None = None,
    *,
    temporal_pooling: Literal["mean", "attention"] | None = None,
) -> nn.Module:
    """Build the model named by ``experiment_cfg.model.name``.

    Parameters
    ----------
    experiment_cfg:
        The full experiment config (its ``model``, ``targets``, ``features`` and
        ``data`` sections drive assembly).
    n_features:
        Number of engineered feature columns served to the model.
    embedding_dim:
        Visual-embedding dimension, required for ``input_mode='embedding'`` with
        a visual branch.
    image_backbone:
        Image backbone (``nn.Module`` with ``.dim``), required for
        ``input_mode='image'`` with a visual branch.
    temporal_pooling:
        Visual temporal pooler (``"mean"`` | ``"attention"``) for a windowed
        ``embedding_seq``. When ``None`` (direct callers) the model-config value
        applies (default ``"mean"``); the engine and evaluator pass the value
        implied by ``experiment_cfg.data.alignment.strategy`` via
        :func:`temporal_pooling_for_strategy` so training and evaluation agree.

    Raises
    ------
    ValueError
        If ``experiment_cfg.model.name`` is not a registered model; the message
        lists the available names.
    """
    name = experiment_cfg.model.name
    try:
        builder = MODEL_BUILDERS[name]
    except KeyError:
        available = ", ".join(sorted(MODEL_BUILDERS))
        raise ValueError(f"unknown model {name!r}; available: {available}") from None
    _warn_unknown_params(name, experiment_cfg)
    return builder(experiment_cfg, n_features, embedding_dim, image_backbone, temporal_pooling)


def _warn_unknown_params(name: str, experiment_cfg: ExperimentConfig) -> None:
    """Log a warning for any ``model`` key the *name* builder does not recognise.

    ``ExperimentModelConfig`` keeps unknown keys (``extra="allow"``); this catches
    typos (a mistyped hyper-parameter that would otherwise be silently ignored)
    without failing the run.
    """
    known = KNOWN_MODEL_PARAMS.get(name, frozenset())
    extras = sorted(set(_params(experiment_cfg)) - known)
    if extras:
        logger.warning(
            "model %r received unknown hyper-parameter(s) %s (not in known keys %s); "
            "kept via extra='allow' but ignored by the builder — check for typos",
            name,
            extras,
            sorted(known),
        )
