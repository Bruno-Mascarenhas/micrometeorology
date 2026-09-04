"""Configuration models for the all-sky pipeline.

Two roots sit at the top of the tree. :class:`ExperimentConfig` describes a
multimodal training run — portable manifest, embeddings, model zoo, experiment
engine. :class:`PrepareConfig` describes dataset preparation and drives
``prepare-local``, ``validate-dataset``, ``precompute-embeddings`` and
``export-colab-bundle``. Both reuse the :class:`VideoConfig` and
:class:`SiteConfig` sections verbatim; ``VideoConfig`` is the one permissive
section left, every other one is strict (``extra="forbid"``), so a typo in a
YAML key fails loudly instead of being ignored.

YAML files compose through an ``extends:`` list that
:func:`load_experiment_config` and :func:`load_prepare_config` resolve
depth-first before validation.
"""

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from labmim_core.site import SiteConfig

#: Fixed UTC offset of the LabMiM camera and datalogger clocks. Pinned rather
#: than read from the host TZ: a UTC-configured container would otherwise shift
#: every capture time by three hours while the instruments keep stamping local
#: time. A fixed offset is correct here because America/Bahia has observed no
#: DST since 2019.
SITE_UTC_OFFSET_HOURS = -3
SITE_TZ_NAME = "America/Bahia"
SITE_TZ = dt.timezone(dt.timedelta(hours=SITE_UTC_OFFSET_HOURS))


class VideoConfig(BaseModel):
    """How all-sky videos map to wall-clock time.

    ``timestamps: overlay`` (the default) reads the capture time the camera
    burns into every frame. ``timestamps: modelled`` instead places frame *N*
    at ``start_time + N * minutes_per_frame``, which the Planetário da UFBA
    camera does not follow — its capture interval changes between day and night
    and its videos do not all start at the same hour, so the modelled mapping
    mislabels frames by up to two and a half hours. See
    ``docs/allsky-archive.md``; ``start_time`` and ``minutes_per_frame`` are
    read only in ``modelled`` mode.
    """

    pattern: str = "data/all-sky/allsky-*.mp4"
    filename_date_format: str = "allsky-%Y%m%d"
    timestamps: Literal["overlay", "modelled"] = "overlay"
    start_time: str = "06:00"
    minutes_per_frame: float = 1.0


#: The :class:`VideoConfig` fields that decide *which capture* a given frame is,
#: as opposed to which days are in scope.  Every resume gate that must survive a
#: day being added to ``pattern`` — but not a clock change — hashes exactly
#: these, so the frame extractor and the embedding store cannot drift apart on
#: what makes a frame a different artifact.  ``filename_date_format`` belongs
#: here because it decides which day a video file covers, and therefore the date
#: half of every frame's timestamp and filename under both clocks.
VIDEO_TIME_FIELDS = (
    "timestamps",
    "filename_date_format",
    "start_time",
    "minutes_per_frame",
)

#: The :class:`PrepareConfig` sections that decide which PIXELS an extracted
#: frame holds, as opposed to the encoder that later reads them.  Beside
#: :data:`VIDEO_TIME_FIELDS` for the same reason: the frame extractor's resume
#: gate and the embedding store's both hash this, and two copies of the tuple
#: would let one start covering a section the other does not — resuming an
#: embedding store onto frames it was never extracted from.
FRAME_PIXEL_SECTIONS = ("mask", "crop", "pad", "resize")

#: Filenames a prepared dataset is published under. The bundle writer and the
#: prepare CLI both name these; separate copies let a rename land in one and not
#: the other, and the bundle reader would then look for a member the writer no
#: longer produces. Declared here rather than in ``allsky.data.contracts``
#: because importing that package pulls pandas, which the CLI modules must not.
DATASET_MANIFEST_FILENAME = "manifest.parquet"

#: The provenance sidecar sits beside the manifest parquet under the parquet's
#: own name plus this suffix. The writer and every reader address it through
#: :func:`manifest_meta_path`, so they cannot disagree about where it is.
MANIFEST_META_SUFFIX = ".meta.json"


def manifest_meta_path(manifest_path: Path) -> Path:
    """Path of the provenance sidecar beside *manifest_path*.

    Parameters
    ----------
    manifest_path:
        Path to the manifest parquet.

    Returns
    -------
    pathlib.Path
        ``<manifest name>.meta.json`` in the same directory.
    """
    return manifest_path.with_name(f"{manifest_path.name}{MANIFEST_META_SUFFIX}")


DATASET_SPLIT_FILENAME = "splits.json"


#: The window modes a dataset can build. This module is the single owner of the
#: name set: it is a leaf (stdlib + yaml + pydantic only), so both the config
#: and :mod:`allsky.data.datasets` can read it without an import cycle, and a
#: typo such as ``centre_frame`` fails at ``load_experiment_config`` time rather
#: than deep inside dataset construction — or, in image mode, not at all.
AlignmentStrategyName = Literal["center_frame", "mean_embedding", "attention_pooling"]


class AlignmentConfig(BaseModel):
    """Image <-> sensor temporal alignment for a sample window.

    ``strategy`` is the window mode: ``center_frame`` picks the frame nearest
    the window centre at manifest-build time, while ``mean_embedding`` and
    ``attention_pooling`` pool every frame in the window at dataset level.
    ``mean_embedding`` works in both input modes; ``attention_pooling`` needs a
    learned pooler that only the embedding source has (see
    :class:`DataSourceConfig`). ``window_minutes`` is the full width of the
    alignment window.
    """

    model_config = ConfigDict(extra="forbid")

    strategy: AlignmentStrategyName = "center_frame"
    window_minutes: float = 10.0
    #: Cap on frames per window in IMAGE mode, evenly subsampled keeping the
    #: ends. The embedding path ignores it: an embedding is a 384-float read,
    #: while a frame is a JPEG decode plus a backbone forward, so a ten-minute
    #: window at this camera's one-frame-per-minute cadence would be eleven
    #: forwards per sample.
    max_frames: int = 5


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

    A windowed ``alignment.strategy`` works in both modes — the image dataset
    stacks the frames and the encoder mean-pools them — except
    ``attention_pooling``, whose learned pooler exists only on the embedding
    source and is rejected rather than silently replaced by the mean.
    """

    model_config = ConfigDict(extra="forbid")

    manifest: str = "manifest.parquet"
    data_root: str = "."
    embeddings_dir: str | None = None
    embeddings_preload: bool = True
    split_artifact: str = "splits.json"
    input_mode: Literal["image", "embedding"] = "image"
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)

    @model_validator(mode="after")
    def _learned_pooling_needs_embedding_mode(self) -> DataSourceConfig:
        """Refuse ``attention_pooling`` in image mode, where no pooler learns.

        The mean-pooled window IS available for images: the dataset stacks the
        frames and :class:`~allsky.modeling.visual_encoder.ImageEncoder` folds
        them into the batch, runs the backbone once and takes the masked mean.
        What image mode has no equivalent of is the LEARNED pooler — the
        single-query attention that ``attention_pooling`` selects lives on
        :class:`~allsky.modeling.visual_encoder.PrecomputedEmbedding`, and asking
        for it here would train a model whose pooling is not the one the config
        names.
        """
        if self.input_mode == "image" and self.alignment.strategy == "attention_pooling":
            raise ValueError(
                "alignment.strategy 'attention_pooling' uses a learned pooler that exists "
                "only on the precomputed-embedding source; image mode pools its window with "
                "the mask-aware mean. Use input_mode: embedding, or strategy: "
                "mean_embedding / center_frame."
            )
        return self


class FeaturesConfig(BaseModel):
    """Sensor feature policy selector.

    ``set`` (``bare`` | ``minimal`` | ``safe`` | ``extended``) maps to
    :data:`allsky.features.policy.BARE_FEATURES` / ``MINIMAL_FEATURES`` /
    ``SAFE_FEATURES`` / ``EXTENDED_FEATURES``. The minimal set drops the
    thermohygrometer channels for periods where that instrument is down and the
    bare set drops the barometer too; the extended set adds
    ablation-only radiometric auxiliaries and is never selected silently. The
    Python attribute is ``feature_set`` (``set`` is the YAML key, exposed via
    alias) to avoid shadowing the builtin.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Spelled out rather than imported from allsky.features.policy, which owns
    # the tiers: allsky.features.__init__ eagerly imports engineering, which
    # imports this module, so the import that would remove the copy closes a
    # cycle. tests/allsky/test_config.py pins the two spellings equal.
    feature_set: Literal["bare", "minimal", "safe", "extended"] = Field(default="safe", alias="set")

    #: Feature names appended verbatim to the resolved set, for ablations over a
    #: column the tiers do not name. The manifest must already carry them: the
    #: engine resolves feature columns from here and fails on any it lacks.
    extra: list[str] = Field(default_factory=list)


#: How the DHI head's target is expressed. ``raw`` fits W m-2 directly.
#: ``clearsky_index`` fits ``DHI / DHI_clearsky`` and multiplies the prediction
#: back by each row's own clear-sky DHI, so the metrics stay in W m-2 while the
#: network no longer spends capacity on the deterministic solar-geometry
#: envelope — an envelope that DRIFTS between this dataset's chronological
#: splits (the DHI-vs-elevation slope moves 20 % from train to test).
DHIParameterization = Literal["raw", "clearsky_index"]


class DHITargetConfig(BaseModel):
    """Diffuse horizontal irradiance target head."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    loss: Literal["mse", "mae", "huber", "heteroscedastic"] = "huber"
    weight: float = 1.0
    parameterization: DHIParameterization = "raw"


class KIndexTargetConfig(BaseModel):
    """Clearness / clear-sky index target head.

    ``kind`` does **not** select the target: the ``target_kindex`` column is
    baked at prepare time from :attr:`PrepareTargetsConfig.kindex_kind` and the
    head trains on that column verbatim. ``kind`` only *asserts* which of k*
    (``kstar``, GHI over Haurwitz clear-sky GHI) or the clearness index k_t
    (``kt``) the manifest was built with, and the two must match —
    :func:`allsky.evaluation.evaluator.evaluate_checkpoint` compares it against
    the manifest's ``kindex_kind``, warning by default and raising under
    ``strict``, and surfaces the verdict as ``kindex_kind_ok``.

    The assertion is gated on ``enabled``: ``sky_class`` is derived from the same
    k-index array, but an experiment with only the sky head enabled is
    deliberately not checked, since ``kind`` defaults to ``kstar`` and an
    unconditional check would reject every default experiment against a
    kt-prepared dataset.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    kind: Literal["kstar", "kt"] = "kstar"
    loss: Literal["mse", "mae", "huber"] = "huber"
    weight: float = 1.0


class SkyClassTargetConfig(BaseModel):
    """Sky-condition classification head over the four published Kt conditions."""

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


#: How the burned-in timestamp band is handled; see
#: :func:`allsky.preprocessing.remove_timestamp_band`.
OverlayPolicy = Literal["keep", "fill", "inpaint", "crop"]

#: Fraction of the frame height the camera's burned-in overlay occupies.
#: Measured over eight frames spanning 2026-05-08..08-01: the text reaches
#: y=75 of 512 (0.1465) at its tallest. 0.16 adds a margin. It lives here, in
#: the leaf module, so the config default and
#: :mod:`allsky.preprocessing` cannot drift apart.
TIMESTAMP_BAND_FRACTION = 0.16

#: Side of the square the visual backbone is fed, in pixels. DINOv2's patch grid
#: is 14 px, and 224 = 16 x 14 is the resolution its weights were trained at.
#: The training engine, the evaluator and the live snapshot all read the default
#: from here, so a run whose config omits ``model.image_size`` is trained,
#: evaluated and served at the same resolution.
DEFAULT_IMAGE_SIZE = 224


class PreprocessingConfig(BaseModel):
    """Deterministic image preprocessing, applied to EVERY split.

    Unlike :class:`AugmentationConfig`, which is random and training-only, these
    transforms must be identical at training and inference, so they travel in
    ``checkpoint["config"]`` and every path that turns a frame into model input
    rebuilds the pipeline from there.

    ``overlay="keep"`` and no ROI is the historical behaviour and the default.
    """

    model_config = ConfigDict(extra="forbid")

    overlay: OverlayPolicy = "keep"
    band_fraction: float = TIMESTAMP_BAND_FRACTION
    roi_radius_fraction: float | None = None


class AugmentationConfig(BaseModel):
    """Image augmentation, applied to the TRAINING split only.

    Every probability defaults to ``0.0``, so an experiment that does not
    mention this section trains on exactly the pixels it trained on before.
    The transforms and the physical argument for each live in
    :mod:`allsky.augmentation`; flips and frame-centred rotations are absent on
    purpose, because they move the sun while the geometry features stay put.
    """

    model_config = ConfigDict(extra="forbid")

    p_exposure: float = 0.0
    exposure_log2: float = 0.35
    p_noise: float = 0.0
    noise_sigma: float = 0.01
    p_translate: float = 0.0
    translate_px: int = 4
    p_erase: float = 0.0


class ExperimentModelConfig(BaseModel):
    """Model architecture selector.

    ``name`` keys into the model registry (``climatology``, ``sensor_only``,
    ``image_only``, ``concat``, ``film``, ``cross_attention``). Hyper-parameters
    are architecture-specific and passed through verbatim, so this model is
    permissive (``extra="allow"``): unknown keys are kept for the model builder.
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
    """Early-stopping controller (monitor a validation metric).

    ``patience`` is the number of non-improving epochs tolerated before the run
    stops; ``patience=1`` already means "stop at the first non-improving epoch",
    so ``0`` (which would fire before any epoch could improve) is rejected. A
    negative ``min_delta`` would make a *worsening* metric count as an
    improvement — resetting the patience counter forever and overwriting
    ``best.ckpt`` with a strictly worse model — so it is rejected too.
    """

    model_config = ConfigDict(extra="forbid")

    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=0.0, ge=0)
    monitor: str = "val_loss"


class ExperimentTrainConfig(BaseModel):
    """Optimisation / engine settings for an experiment run.

    ``backbone_lr`` (when set) drives a separate parameter group for the visual
    backbone; ``out_subdir`` is the run directory created under
    ``ExperimentConfig.output_dir``.

    ``epochs`` must be at least 1: both checkpoint writes live inside the epoch
    loop, so ``epochs: 0`` would exit 0 while advertising ``last.ckpt`` /
    ``best.ckpt`` paths that were never written. (Resuming a finished run — where
    the loop body is skipped because the start epoch already equals ``epochs`` —
    is unaffected; those checkpoints are already on disk.)
    """

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=20, ge=1)
    batch_size: int = 32
    lr: float = 3e-4
    backbone_lr: float | None = None
    weight_decay: float = 1e-4
    # AdamW is the only algorithm allsky.training.engine builds. Declared as the
    # literal so a config naming another one is refused when it is loaded, rather
    # than after the seed, the dataset and the model of a whole run.
    optimizer: Literal["adamw"] = "adamw"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    amp: AMPConfig = Field(default_factory=AMPConfig)
    grad_accum_steps: int = 1
    grad_clip_norm: float | None = None
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    num_workers: int = 2
    device: str = "auto"
    out_subdir: str = "run"


class ExperimentConfig(BaseModel):
    """Root config for a multimodal training experiment.

    The optional top-level ``experiment: true`` marker (see
    :func:`is_experiment_config`) routes the ``train`` CLI to the experiment
    engine; it is accepted here so strict validation does not reject it.
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
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


def model_param(cfg: ExperimentConfig, key: str, default: Any) -> Any:
    """Read an architecture hyper-parameter off the permissive model config.

    ``ExperimentModelConfig`` accepts unknown keys on purpose, so a missing one
    falls back rather than raising. It lives here, beside the config it reads,
    because the training engine, the evaluator and the live snapshot all need it
    and only one of them should own it.
    """
    return dict(cfg.model.model_dump()).get(key, default)


def image_size_of(cfg: ExperimentConfig) -> int:
    """Side of the square frame *cfg* feeds the visual backbone, in pixels.

    The training engine, the evaluator and the live snapshot must all resize to
    the same number or a checkpoint scores frames at a resolution it was never
    fitted on, which no error reports.
    """
    return int(model_param(cfg, "image_size", DEFAULT_IMAGE_SIZE))


def geometry_channels_of(cfg: ExperimentConfig) -> tuple[str, ...]:
    """Per-pixel solar geometry maps *cfg* appends to each frame, in stack order.

    Accepts ``true`` for every map, ``false`` (or nothing) for none, or a list
    naming a subset. Like :func:`image_size_of`, this is read by the training
    engine, the evaluator, the registry and the dataset alike: the frame the
    loader emits must carry exactly the channels the patch projection was widened
    for, and a checkpoint trained with the maps cannot be reloaded into a model
    built without them.
    """
    from allsky.geometry import resolve_geometry_channels

    return resolve_geometry_channels(model_param(cfg, "geometry_channels", False))


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


class PadConfig(BaseModel):
    """Pixel padding applied after the crop and before the resize.

    The four sides are given separately because the sky disc is not concentric
    with the sensor: making it concentric with the output square takes unequal
    padding, and the amounts come from a lens calibration rather than from the
    frame's own dimensions.

    ``fill`` is the level written into the padded rows and columns, on the same
    0-255 scale as the frame. It is sky the camera does not image, not sky that
    is dark, so nothing downstream should read it as a measurement.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    top: int = Field(default=0, ge=0)
    bottom: int = Field(default=0, ge=0)
    left: int = Field(default=0, ge=0)
    right: int = Field(default=0, ge=0)
    fill: int = Field(default=0, ge=0, le=255)


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
    #: Minutes added to every logger stamp before a frame is paired to it.
    #: The CR5000 END-stamps its averages: the row written at ``t`` is the mean
    #: over ``(t - 5min, t]``, verified against the 1-minute ``LBM_solar_2024``
    #: table over their 2024-03..07 overlap (RMS 0.083 W/m2, r = 1.000000,
    #: n = 35,492, against 64.79 W/m2 for the begin-stamped reading). Pairing a
    #: frame to the raw stamp therefore labels it with an average whose time
    #: centroid sits 2.5 min earlier. Measured on 79,860 daylight 1-minute GHI
    #: samples, the raw-stamp join carries 94.00 W/m2 of label noise against
    #: 74.99 W/m2 for ``-2.5`` (-20.2%). Left at 0.0 so an existing dataset
    #: rebuilds unchanged; it is part of the manifest resume hash, so a change
    #: invalidates the manifest instead of being silently reused.
    timestamp_offset_minutes: float = 0.0


class PrepareTargetsConfig(BaseModel):
    """Target derivation: the diffuse column and which k-index to record.

    ``diffuse_column`` names the logger channel holding measured diffuse
    horizontal irradiance in W m-2; rows built from it are flagged
    ``target_source="measured"``. Setting it to ``None`` switches the whole
    dataset to Erbs pseudo-targets derived from GHI
    (:func:`allsky.erbs.pseudo_diffuse`, ``target_source="erbs_pseudo"``).

    The sky-condition bins are not configurable: they are the published bounds
    of :data:`allsky.data.sky.SKY_CLASS_KT_UPPER_BOUNDS`, always applied
    to Kt. A config carrying the retired ``class_clear``/``class_overcast`` keys
    is rejected rather than silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    diffuse_column: str | None = "PSP_Wm2_Avg"
    kindex_kind: Literal["kstar", "kt"] = "kstar"


class DatasetOutputConfig(BaseModel):
    """Where the prepared dataset (manifest + frames) is written."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str = "output/allsky/dataset"
    dataset_version: str = "2"


class EmbeddingsConfig(BaseModel):
    """Visual-embedding precompute settings (DINOv2 by default).

    .. warning::
       ``revision`` is currently **inert**: nothing reads it.
       :class:`allsky.embeddings.backbone.DinoV2Backbone` pins
       ``DINOV2_REVISION`` from a module constant, and that pinned SHA — not this
       value — is what ``embeddings.meta.json`` records.  Editing it changes no
       embedding, only the section hash that gates ``precompute-embeddings``
       resume.  Do not treat a value written here as the provenance of a store.
    """

    model_config = ConfigDict(extra="forbid")

    backbone: str = "dinov2_vits14"
    revision: str = "main"
    pooling: Literal["cls", "mean", "cls+mean"] = "cls"
    batch_size: int = 32
    device: str = "auto"
    shard_size: int = 1024
    dtype: Literal["fp16", "fp32"] = "fp16"


class SplitsConfig(BaseModel):
    """Day-based train/val/test partition.

    ``strategy`` defaults to ``chronological``: the earliest days train, the
    latest test, and ``gap_days`` are dropped at each boundary so no training
    day is adjacent to an evaluation day. ``random`` is available for studies
    that estimate a quantity from a simultaneous image, and has to be asked for
    by name — chosen silently it reports a forecasting skill the model does not
    have.
    """

    model_config = ConfigDict(extra="forbid")

    val_fraction: float = 0.2
    test_fraction: float = 0.1
    seed: int = 42
    strategy: Literal["chronological", "random"] = "chronological"
    gap_days: int = 1


class PrepareConfig(BaseModel):
    """Root config for dataset preparation, embeddings and export.

    ``video`` and ``site`` reuse the shared sections (:class:`VideoConfig`, still
    permissive, and the strict :class:`SiteConfig`); every prepare-specific
    section is strict so typos fail loudly. ``features`` selects the sensor feature
    policy baked into the manifest at build time — its YAML key is ``set``,
    aliasing the ``feature_set`` attribute.
    """

    model_config = ConfigDict(extra="forbid")

    video: VideoConfig = Field(default_factory=VideoConfig)
    site: SiteConfig = Field(default_factory=SiteConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    mask: MaskConfig = Field(default_factory=MaskConfig)
    crop: CropConfig = Field(default_factory=CropConfig)
    pad: PadConfig = Field(default_factory=PadConfig)
    resize: int | None = None
    night_filter: NightFilterConfig = Field(default_factory=NightFilterConfig)
    sensor: PrepareSensorConfig = Field(default_factory=PrepareSensorConfig)
    targets: PrepareTargetsConfig = Field(default_factory=PrepareTargetsConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    output: DatasetOutputConfig = Field(default_factory=DatasetOutputConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)


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
    resolved first, so the marker is honoured wherever it is set). A config that
    lacks the key returns False, so the ``train`` CLI can reject non-experiment
    YAML with a clear error from this check alone.
    """
    data = path_or_dict if isinstance(path_or_dict, dict) else _load_yaml_with_extends(path_or_dict)
    return data.get("experiment") is True
