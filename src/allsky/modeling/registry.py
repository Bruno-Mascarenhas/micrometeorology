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

import logging
from collections.abc import Callable, Mapping
from typing import Any, Literal, cast

from torch import nn

from allsky.config import ExperimentConfig
from allsky.embeddings.backbone import Pooling
from allsky.features.policy import resolve_feature_set
from allsky.modeling.baselines import ClimatologyModel, ImageOnlyModel, SensorOnlyModel
from allsky.modeling.multimodal import MultimodalNet
from allsky.modeling.visual_encoder import build_visual_encoder
from allsky.training.errors import TrainingError

logger = logging.getLogger(__name__)

__all__ = [
    "MODEL_BUILDERS",
    "build_model",
    "default_image_backbone_builder",
    "restore_model",
    "temporal_pooling_for_strategy",
]

#: Builder signature (positional):
#: ``(cfg, n_features, embedding_dim, image_backbone, temporal_pooling) -> nn.Module``.
#: :func:`build_model` dispatches positionally, so every builder takes all five
#: slots; a builder that does not consume one names it with a leading underscore.
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
# Visual knobs the training/evaluation pipeline reads rather than the builder
# here — legitimate keys that must not warn: ``image_size`` in
# ``engine._build_datasets`` and the evaluator, ``backbone`` /
# ``backbone_pooling`` in :func:`default_image_backbone_builder` (which also
# runs on evaluate).  They are not in ``_COMMON_PARAMS``: ``sensor_only`` and
# ``climatology`` never build an image backbone, so those must keep warning.
_PIPELINE_VISUAL_PARAMS = frozenset({"image_size", "backbone", "backbone_pooling"})
_VISUAL_PARAMS = (
    frozenset({"visual_out_dim", "backbone_frozen", "unfreeze_last_n", "temporal_pooling"})
    | _PIPELINE_VISUAL_PARAMS
)
_CROSS_ATTENTION_PARAMS = frozenset({"num_heads", "token_dim"})
#: Model name -> the hyper-parameter keys that model consumes (the registry
#: builder plus the engine / evaluator paths the model runs under).
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

    Parameters
    ----------
    strategy:
        ``cfg.data.alignment.strategy`` (``"center_frame"``, ``"mean_embedding"``
        or ``"attention_pooling"``).

    Returns
    -------
    Literal["mean", "attention"]
        The pooler name to hand :func:`build_model`.
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
    _n_features: int,
    _embedding_dim: int | None,
    _image_backbone: nn.Module | None,
    _temporal_pooling: Literal["mean", "attention"] | None,
) -> nn.Module:
    return ClimatologyModel(cfg.targets)


def _build_sensor_only(
    cfg: ExperimentConfig,
    n_features: int,
    _embedding_dim: int | None,
    _image_backbone: nn.Module | None,
    _temporal_pooling: Literal["mean", "attention"] | None,
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
    _n_features: int,
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
        backbone_lr=cfg.train.backbone_lr,
    )


def _multimodal_builder(fusion_name: str) -> ModelBuilder:
    """Return a builder that assembles a :class:`MultimodalNet` with *fusion_name*."""

    def builder(
        cfg: ExperimentConfig,
        _n_features: int,
        embedding_dim: int | None,
        image_backbone: nn.Module | None,
        temporal_pooling: Literal["mean", "attention"] | None,
    ) -> nn.Module:
        params = _params(cfg)
        feature_set = cfg.features.feature_set
        return MultimodalNet(
            feature_columns=resolve_feature_set(feature_set, cfg.features.extra),
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

    Returns
    -------
    nn.Module
        A model honouring the
        :class:`allsky.modeling.contracts.MultimodalModel` contract.

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


def default_image_backbone_builder(cfg: ExperimentConfig, device: str) -> Callable[[], nn.Module]:
    """Build the default image backbone factory for ``input_mode='image'``.

    When no ``image_backbone_builder`` is injected, image-mode training must still
    build a real visual backbone or the shipped v6 config cannot run.  The factory
    constructs the backbone named by the model config (``model.backbone``, default
    ``dinov2_vits14``; ``model.backbone_pooling``, default ``cls``) on the run
    device via :func:`allsky.embeddings.backbone.build_backbone`, then coerces it
    into a trainable ``nn.Module`` (:func:`allsky.modeling.visual_encoder.coerce_image_backbone`
    wraps the DINOv2 extraction wrapper; an ``nn.Module`` — e.g. a test stub, or a
    monkeypatched ``build_backbone`` — passes straight through).  Construction is
    deferred to call time and any failure is re-raised with a message naming the
    config knobs to fix.  ``build_backbone`` is imported inside the factory, not
    at module scope, so a test that monkeypatches
    ``allsky.embeddings.backbone.build_backbone`` is honoured.
    """

    def build() -> nn.Module:
        from allsky.embeddings.backbone import build_backbone
        from allsky.modeling.visual_encoder import coerce_image_backbone

        params = dict(cfg.model.model_dump())
        name = str(params.get("backbone", "dinov2_vits14"))
        pooling = str(params.get("backbone_pooling", "cls"))
        try:
            backbone = build_backbone(name, pooling=_backbone_pooling(pooling), device=device)
            return coerce_image_backbone(backbone, pooling=pooling)
        except Exception as exc:
            raise TrainingError(
                "failed to construct the default image backbone for input_mode='image' "
                f"(model.backbone={name!r}, model.backbone_pooling={pooling!r}, "
                f"train.device={device!r}); fix those config knobs or inject an "
                f"image_backbone_builder. Cause: {exc}"
            ) from exc

    return build


def _backbone_pooling(value: str) -> Pooling:
    """Narrow the free-form ``model.backbone_pooling`` string to the backbone's literal.

    ``ExperimentModelConfig`` is ``extra="allow"``, so the knob arrives as an
    arbitrary string while :func:`allsky.embeddings.backbone.build_backbone` accepts
    only these three names.  Rejecting anything else here does not change what a bad
    value does to a run — it already ended as the ``RuntimeError``
    :func:`default_image_backbone_builder` raises, since the only backbone that
    reads ``pooling`` (``DinoV2Backbone``) rejects an unknown one, and the other
    (``fake``) is not an ``nn.Module`` and exposes no ``load_torch_module``, so
    ``coerce_image_backbone`` refuses it regardless of pooling.  It only moves the
    failure one call earlier, onto a message that names the knob.
    """
    if value == "cls":
        return "cls"
    if value == "mean":
        return "mean"
    if value == "cls+mean":
        return "cls+mean"
    raise ValueError(f"unknown backbone pooling {value!r}; expected 'cls', 'mean' or 'cls+mean'")


def restore_model(
    cfg: ExperimentConfig,
    checkpoint: Mapping[str, Any],
    n_features: int,
    *,
    embedding_dim: int | None,
    device: str,
    image_backbone_builder: Callable[[], nn.Module] | None = None,
) -> nn.Module:
    """Rebuild the model a checkpoint was trained as and load its weights into it.

    The two consumers of a trained checkpoint — offline evaluation and live
    snapshot scoring — must rebuild the SAME architecture the run trained, or
    ``load_state_dict`` rejects the weights: the temporal pooler is taken from
    ``cfg.data.alignment.strategy`` because an attention-pooled model carries
    extra weights a mean-pooled one has no slot for, and an ``input_mode="image"``
    checkpoint needs the visual backbone present for the same reason, even though
    nothing here is training it.

    Parameters
    ----------
    cfg:
        The checkpoint's own config, as validated from ``checkpoint["config"]``.
    checkpoint:
        Loaded checkpoint mapping; ``model_state`` is read and loaded strictly,
        so a mismatched architecture raises rather than silently dropping weights.
    n_features:
        Width of the tabular feature vector the run was fitted with.
    embedding_dim:
        Width of the stored embedding for ``input_mode="embedding"``; ``None``
        for image mode.
    device:
        Device to move the restored model to.
    image_backbone_builder:
        Injection hook for the visual backbone, so a test can supply a stub
        instead of downloading hub weights. ``None`` falls back to
        :func:`default_image_backbone_builder`, the backbone the config names.

    Returns
    -------
    torch.nn.Module
        The model in ``eval()`` mode on *device*, weights loaded.
    """
    image_backbone = None
    if cfg.data.input_mode == "image":
        builder = image_backbone_builder or default_image_backbone_builder(cfg, device)
        image_backbone = builder()
    model = build_model(
        cfg,
        n_features,
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        temporal_pooling=temporal_pooling_for_strategy(cfg.data.alignment.strategy),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model


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
