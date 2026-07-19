"""Configuration models for the all-sky pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class VideoConfig(BaseModel):
    """How all-sky videos map to wall-clock time.

    The camera produces one-day timelapse files named by date; each frame
    covers ``minutes_per_frame`` minutes of real time starting at
    ``start_time`` local time.
    """

    pattern: str = "data/all-sky/allsky-*.mp4"
    filename_date_format: str = "allsky-%Y%m%d"
    start_time: str = "06:00"
    minutes_per_frame: float = 1.0


class SensorConfig(BaseModel):
    """Radiation-sensor sources and column selection."""

    paths: list[str] = Field(default_factory=lambda: ["data/LBM_lenta_2025.dat"])
    ghi_column: str = "CM3Up_Wm2_Avg"
    # CMP21 is the station's primary diffuse pyranometer, but its W/m2 channel
    # is currently zero-filled by the logger program (only the raw CMP21_Avg mV
    # channel is live) — PSP is the working diffuse measurement. Switch to
    # "CMP21_Wm2_Avg" once the CR5000 program conversion is fixed; None falls
    # back to Erbs pseudo-targets (target_source="erbs_pseudo").
    diffuse_column: str | None = "PSP_Wm2_Avg"
    feature_columns: list[str] = Field(
        default_factory=lambda: [
            "CM3Up_Wm2_Avg",
            "CG3Up_Wm2_Avg",
            "CM3Dn_Wm2_Avg",
            "Net_Wm2_Avg",
            "CUV5_Wm2_Avg",
            "PAR_Wm2_Avg",
        ]
    )
    tolerance_minutes: float = 5.0


class SiteConfig(BaseModel):
    """Observation site (LabMiM/UFBA, Salvador-BA by default)."""

    latitude: float = -13.00
    longitude: float = -38.51


class LabelConfig(BaseModel):
    """Weak cloud-condition labels from the clearness index kt."""

    kt_clear: float = 0.65
    kt_overcast: float = 0.35
    min_solar_elevation_deg: float = 10.0
    # QC guard: kt above this is a sensor artifact (GHI spikes far beyond
    # clear-sky), not weather — such rows are dropped from the dataset.
    max_kt: float = 1.2


class ModelConfig(BaseModel):
    """SkyFusionNet architecture parameters."""

    image_size: int = 224
    backbone: str = "small"  # "small" (built-in conv net) or "resnet18"
    embed_dim: int = 128
    hidden_dim: int = 256
    n_classes: int = 3


class TrainConfig(BaseModel):
    """Training run parameters."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    device: str = "auto"  # auto -> cuda | mps | cpu
    # Automatic mixed precision on CUDA (Colab T4/L4/A100): ~2x throughput.
    amp: bool = True
    out_dir: str = "output/allsky"
    seed: int = 42
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0


class AllSkyConfig(BaseModel):
    """Root configuration for the all-sky pipeline."""

    video: VideoConfig = Field(default_factory=VideoConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


def load_config(path: str | Path | None = None) -> AllSkyConfig:
    """Load configuration from YAML, falling back to defaults when *path* is None."""
    if path is None:
        return AllSkyConfig()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AllSkyConfig.model_validate(raw)


# ---------------------------------------------------------------------------
# Wave C1b — multimodal experiment / prepare configs (new stack).
#
# These models describe the NEW pipeline (portable manifest, embeddings, model
# zoo, experiment engine). They are strict (``extra="forbid"``) so a typo in a
# YAML key fails loudly rather than being silently ignored — unlike the legacy
# ``AllSkyConfig`` tree above, which stays permissive for backward
# compatibility. YAML files compose via an ``extends:`` list resolved by
# :func:`load_experiment_config` / :func:`load_prepare_config`.
# ---------------------------------------------------------------------------


class AlignmentConfig(BaseModel):
    """Image <-> sensor temporal alignment for a sample window.

    ``strategy`` selects an :class:`allsky.data.alignment.AlignmentStrategy`
    (``center_frame`` picks the frame nearest the window centre at
    manifest-build time; windowed poolers act at the dataset level).
    ``window_minutes`` is the full width of the alignment window.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: str = "center_frame"
    window_minutes: float = 10.0


class DataSourceConfig(BaseModel):
    """Where a training experiment reads its data from.

    ``input_mode`` chooses between end-to-end image training (``image``) and
    training on precomputed visual embeddings (``embedding``); the latter
    additionally uses ``embeddings_dir``. Paths are resolved by the data layer;
    image paths inside the manifest are relative POSIX paths against
    ``data_root``.

    ``embeddings_preload`` (default ``True``) loads every embedding shard once
    into one resident ``(N, dim)`` array for training/eval, instead of the small
    LRU of open shards that thrashes under shuffled access; set it ``False`` to
    keep the lazy LRU path (e.g. when the store does not fit in memory).
    """

    model_config = ConfigDict(extra="forbid")

    manifest: str = "manifest.parquet"
    data_root: str = "."
    embeddings_dir: str | None = None
    embeddings_preload: bool = True
    split_artifact: str = "splits.json"
    input_mode: Literal["image", "embedding"] = "image"
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)


class FeaturesConfig(BaseModel):
    """Sensor feature policy selector.

    ``set`` (``safe`` | ``extended``) maps to
    :data:`allsky.features.policy.SAFE_FEATURES` / ``EXTENDED_FEATURES``. The
    extended set adds ablation-only radiometric auxiliaries and is never
    selected silently. The Python attribute is ``feature_set`` (``set`` is the
    YAML key, exposed via alias) to avoid shadowing the builtin.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    feature_set: Literal["safe", "extended"] = Field(default="safe", alias="set")


class DHITargetConfig(BaseModel):
    """Diffuse horizontal irradiance target head."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    loss: Literal["mse", "mae", "huber", "heteroscedastic"] = "huber"
    weight: float = 1.0


class KIndexTargetConfig(BaseModel):
    """Clearness / clear-sky index target head.

    ``kind`` selects k* (``kstar``, GHI over Haurwitz clear-sky GHI) or the
    clearness index k_t (``kt``).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kind: Literal["kstar", "kt"] = "kstar"
    loss: Literal["mse", "mae", "huber"] = "huber"
    weight: float = 1.0


class SkyClassTargetConfig(BaseModel):
    """Sky-condition classification head (clear / partial / overcast)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    weight: float = 1.0


class CloudFractionTargetConfig(BaseModel):
    """Cloud-fraction regression head (sigmoid, [0, 1])."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    weight: float = 1.0


class TargetsConfig(BaseModel):
    """Which prediction heads the experiment trains and how they are weighted."""

    model_config = ConfigDict(extra="forbid")

    dhi: DHITargetConfig = Field(default_factory=DHITargetConfig)
    kindex: KIndexTargetConfig = Field(default_factory=KIndexTargetConfig)
    sky: SkyClassTargetConfig = Field(default_factory=SkyClassTargetConfig)
    cloud_fraction: CloudFractionTargetConfig = Field(default_factory=CloudFractionTargetConfig)


class ExperimentModelConfig(BaseModel):
    """Model architecture selector.

    ``name`` keys into the model registry (``climatology``, ``sensor_only``,
    ``image_only``, ``concat``, ``film``, ``cross_attention``). Architecture
    hyper-parameters are architecture-specific and passed through verbatim, so
    this model is permissive (``extra="allow"``); unknown keys are kept and
    consumed by the model builder in a later wave.
    """

    model_config = ConfigDict(extra="allow")

    name: str = "concat"


class SchedulerConfig(BaseModel):
    """Learning-rate scheduler selector with pass-through params."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["none", "cosine", "plateau"] = "none"
    params: dict[str, Any] = Field(default_factory=dict)


class AMPConfig(BaseModel):
    """Automatic mixed-precision settings (GradScaler only for fp16 + CUDA)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    dtype: Literal["fp16", "bf16"] = "fp16"


class EarlyStoppingConfig(BaseModel):
    """Early-stopping controller (monitor a validation metric)."""

    model_config = ConfigDict(extra="forbid")

    patience: int = 10
    min_delta: float = 0.0
    monitor: str = "val_loss"


class ExperimentTrainConfig(BaseModel):
    """Optimisation / engine settings for an experiment run.

    ``backbone_lr`` (when set) drives a separate parameter group for the visual
    backbone; ``out_subdir`` is the run directory created under
    ``ExperimentConfig.output_dir``.
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-4
    backbone_lr: float | None = None
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    amp: AMPConfig = Field(default_factory=AMPConfig)
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    num_workers: int = 2
    device: str = "auto"
    out_subdir: str = "run"


class ExperimentConfig(BaseModel):
    """Root config for a multimodal training experiment (new stack).

    The optional top-level ``experiment: true`` marker (see
    :func:`is_experiment_config`) routes the ``train`` CLI to the new engine in
    a later wave; it is accepted here so strict validation does not reject it.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: bool = False
    name: str = "experiment"
    seed: int = 42
    output_dir: str = "output/allsky/experiments"
    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    targets: TargetsConfig = Field(default_factory=TargetsConfig)
    model: ExperimentModelConfig = Field(default_factory=ExperimentModelConfig)
    train: ExperimentTrainConfig = Field(default_factory=ExperimentTrainConfig)


# ---------------------------------------------------------------------------
# PrepareConfig tree — drives prepare-local / validate-dataset /
# precompute-embeddings / export-colab-bundle.
# ---------------------------------------------------------------------------


class MaskConfig(BaseModel):
    """Static horizon/obstruction mask. ``threshold=None`` selects an auto value."""

    model_config = ConfigDict(extra="forbid")

    path: str | None = None
    threshold: float | None = None


class CropConfig(BaseModel):
    """Optional pixel crop applied before resize (``height``/``width`` None = full)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    top: int = 0
    left: int = 0
    height: int | None = None
    width: int | None = None


class NightFilterConfig(BaseModel):
    """Drop frames whose solar elevation is below ``min_solar_elevation_deg``."""

    model_config = ConfigDict(extra="forbid")

    min_solar_elevation_deg: float = 5.0


class PrepareSensorConfig(BaseModel):
    """Meteorological sensor sources and source-column -> engineered-name mapping."""

    model_config = ConfigDict(extra="forbid")

    paths: list[str] = Field(default_factory=lambda: ["data/LBM_lenta_2025.dat"])
    #: Global-horizontal-irradiance logger column driving the k-index and the
    #: Erbs pseudo-target (never a model feature — it lives in
    #: :data:`allsky.features.policy.FORBIDDEN_FEATURES`).
    ghi_column: str = "CM3Up_Wm2_Avg"
    column_map: dict[str, str] = Field(default_factory=dict)
    tolerance_minutes: float = 5.0


class PrepareTargetsConfig(BaseModel):
    """Target derivation: diffuse column, k-index kind and sky-class thresholds."""

    model_config = ConfigDict(extra="forbid")

    diffuse_column: str | None = "PSP_Wm2_Avg"
    kindex_kind: Literal["kstar", "kt"] = "kstar"
    class_clear: float = 0.65
    class_overcast: float = 0.35


class DatasetOutputConfig(BaseModel):
    """Where the prepared dataset (manifest + frames) is written."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str = "output/allsky/dataset"
    dataset_version: str = "2"


class EmbeddingsConfig(BaseModel):
    """Visual-embedding precompute settings (DINOv2 by default)."""

    model_config = ConfigDict(extra="forbid")

    backbone: str = "dinov2_vits14"
    revision: str = "main"
    pooling: Literal["cls", "mean", "cls+mean"] = "cls"
    batch_size: int = 32
    device: str = "auto"
    shard_size: int = 1024
    dtype: Literal["fp16", "fp32"] = "fp16"


class SplitsConfig(BaseModel):
    """Day-based train/val/test split fractions and seed."""

    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.2
    test_fraction: float = 0.1
    seed: int = 42


class PrepareConfig(BaseModel):
    """Root config for dataset preparation, embeddings and export.

    ``video`` and ``site`` reuse the permissive legacy sections
    (:class:`VideoConfig` / :class:`SiteConfig`); every prepare-specific section
    is strict so typos fail loudly.
    """

    model_config = ConfigDict(extra="forbid")

    video: VideoConfig = Field(default_factory=VideoConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)
    #: Sensor feature policy (``safe`` | ``extended``) baked into the manifest at
    #: build time; the ``set`` YAML key aliases the ``feature_set`` attribute.
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    resize: int | None = None
    night_filter: NightFilterConfig = Field(default_factory=NightFilterConfig)
    sensor: PrepareSensorConfig = Field(default_factory=PrepareSensorConfig)
    targets: PrepareTargetsConfig = Field(default_factory=PrepareTargetsConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    output: DatasetOutputConfig = Field(default_factory=DatasetOutputConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)


# ---------------------------------------------------------------------------
# YAML composition (``extends:``) + loaders.
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Merge rules (``override`` wins): nested dicts are merged key-by-key; scalars
    and lists overwrite wholesale (a shorter list replaces a longer one). Inputs
    are never mutated.
    """
    result: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _load_yaml_with_extends(path: str | Path, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML mapping, resolving an optional ``extends:`` list depth-first.

    ``extends`` is a path (or list of paths) relative to the including file. Each
    parent is fully resolved (its own ``extends`` first), deep-merged in list
    order, then the including file's own keys are merged on top (later wins).
    A cyclic ``extends`` reference raises :class:`ValueError` naming the chain.
    """
    resolved = Path(path).resolve()
    if resolved in _stack:
        chain = " -> ".join(str(node) for node in (*_stack, resolved))
        raise ValueError(f"Cyclic 'extends' reference detected: {chain}")
    with open(resolved, encoding="utf-8") as handle:
        loaded: Any = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"config {resolved} must be a YAML mapping, got {type(loaded).__name__}")
    raw: dict[str, Any] = dict(loaded)
    extends = raw.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    merged: dict[str, Any] = {}
    for relative in extends:
        parent = _load_yaml_with_extends(resolved.parent / relative, (*_stack, resolved))
        merged = _deep_merge(merged, parent)
    return _deep_merge(merged, raw)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an :class:`ExperimentConfig`, resolving ``extends:``."""
    return ExperimentConfig.model_validate(_load_yaml_with_extends(path))


def load_prepare_config(path: str | Path) -> PrepareConfig:
    """Load and validate a :class:`PrepareConfig`, resolving ``extends:``."""
    return PrepareConfig.model_validate(_load_yaml_with_extends(path))


def is_experiment_config(path_or_dict: str | Path | dict[str, Any]) -> bool:
    """Return True when the config declares the top-level ``experiment: true`` marker.

    Accepts an already-parsed mapping or a path (whose ``extends:`` chain is
    resolved first, so the marker is honoured wherever it is set). Legacy
    ``AllSkyConfig`` YAML lacks the key and returns False, so the ``train`` CLI
    can dispatch legacy vs. experiment runs from this alone.
    """
    data = path_or_dict if isinstance(path_or_dict, dict) else _load_yaml_with_extends(path_or_dict)
    return data.get("experiment") is True
