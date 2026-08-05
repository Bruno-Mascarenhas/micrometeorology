"""Multi-task training engine for the multimodal all-sky experiments.

:func:`run_experiment` drives one experiment end to end from an
:class:`allsky.config.ExperimentConfig`:

#. seed everything (:func:`solrad_correction.utils.seeds.set_global_seed`) and
   resolve the device (clear error when ``cuda`` is requested but unavailable);
#. load the v2 manifest parquet + meta sidecar and the persisted day split, then
   slice train/val rows by ``day_id`` (val required, test ignored here);
#. fit the :class:`~allsky.features.normalization.FeatureNormalizer` and the
   per-target :class:`~allsky.features.normalization.TargetNormalizer` mapping on
   the **train** rows only;
#. build the image or embedding dataset per ``cfg.data.input_mode`` and their
   DataLoaders;
#. build the model via :func:`allsky.modeling.registry.build_model` (climatology
   is fit from the train targets and skips gradient steps), an AdamW optimizer
   over ``model.param_groups`` when available, an optional scheduler and AMP;
#. run per-epoch train/val passes computing loss components **and** physical-unit
   metrics (denormalized DHI/kindex MAE, sky accuracy), logging to TensorBoard,
   ``metrics.csv`` (appended) and ``metrics.json`` (atomically rewritten).  Each
   reported ``loss_<head>`` is the mean over the rows that head actually had a
   target for, and the reported ``loss`` is the configured weighted sum of those
   means, so the monitored quantity does not move with ``train.batch_size``;
#. checkpoint ``last.ckpt`` every epoch and ``best.ckpt`` on monitor improvement
   (resume-safe best seeding), with early stopping;
#. resume fully from ``last.ckpt`` (``resume="auto"`` or a path), restoring
   model / optimizer / scheduler / scaler / epoch / global_step / best / RNG —
   but only after the checkpoint's dataset provenance matches this run (a rebuilt
   manifest is refused, not silently re-scaled), with the stored best discarded
   when the early-stopping monitor changed, a restored cosine horizon re-pointed
   at the current epoch budget, and nothing trained at all when the stopping rule
   was already satisfied.

The engine imports torch at module scope and is therefore only ever imported
lazily (from the CLI or via :func:`allsky.training.__getattr__`), so
``import allsky`` / ``import allsky.cli`` stay torch-free.
"""

import contextlib
import csv
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from allsky.atomic import atomic_write, atomic_write_json
from allsky.config import ExperimentConfig, SchedulerConfig, TargetsConfig
from allsky.data.datasets import EmbeddingReader
from allsky.data.loading import (
    default_embedding_reader,
    load_manifest,
    load_split,
    resolve_against_root,
)
from allsky.embeddings.backbone import Pooling
from allsky.features.normalization import TargetNormalizer
from allsky.features.policy import active_feature_groups, resolve_feature_set
from allsky.modeling.baselines import ClimatologyModel
from allsky.training.checkpointing import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    capture_rng_state,
    code_version,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from solrad_correction.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

__all__ = ["resolve_run_device", "run_experiment"]


def resolve_run_device(requested: str) -> str:
    """Resolve *requested* to a concrete device, erroring on unavailable cuda.

    Delegates ``"auto"`` resolution to
    :func:`allsky.training.device.resolve_device`, then raises a clear
    :class:`RuntimeError` when ``"cuda"`` is asked for but no CUDA device is
    available (rather than failing opaquely deep inside the first ``.to("cuda")``).
    """
    from allsky.training.device import resolve_device

    device = resolve_device(requested)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device 'cuda' was requested but no CUDA device is available; "
            "use device='cpu' or device='auto'"
        )
    return device


def run_experiment(
    cfg: ExperimentConfig,
    *,
    data_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str | None = None,
    amp: bool | None = None,
    resume: str | Path | None = None,
    trust_checkpoint: bool = False,
    image_backbone_builder: Callable[[], nn.Module] | None = None,
    embedding_reader: EmbeddingReader | None = None,
) -> dict[str, Any]:
    """Run one multimodal experiment and return a summary dict.

    Parameters
    ----------
    cfg:
        The experiment configuration.
    data_root:
        Overrides ``cfg.data.data_root`` (the root that the manifest, split and
        embeddings paths, and the manifest's relative ``image_path`` values,
        resolve against).
    output_dir:
        Overrides the run directory (default ``cfg.output_dir/cfg.train.out_subdir``).
    device:
        Overrides ``cfg.train.device`` (``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"``).
    amp:
        Overrides ``cfg.train.amp.enabled``.
    resume:
        ``"auto"`` loads ``<run_dir>/last.ckpt`` if present; a path loads that
        checkpoint; ``None`` starts fresh.
    trust_checkpoint:
        Read the resumed checkpoint with the unrestricted unpickler. Off by
        default: a checkpoint that travelled through Colab or a shared Drive is
        an untrusted input, so it is read under torch's restricted reader and a
        payload outside the allowlist is refused rather than executed.
    image_backbone_builder:
        Zero-arg factory returning the image backbone for ``input_mode="image"``
        visual models.  The DINOv2 backbone is only used when explicitly injected
        here (tests inject a tiny stub); no backbone is downloaded implicitly.
    embedding_reader:
        Injected :class:`~allsky.data.datasets.EmbeddingReader` for
        ``input_mode="embedding"`` (tests pass a dict-backed fake); defaults to a
        :class:`allsky.embeddings.storage.SafetensorsEmbeddingReader` over
        ``cfg.data.embeddings_dir``.

    Returns
    -------
    dict
        ``{best_metric, epochs_ran, epoch, global_step, final_val_metrics,
        output_dir, checkpoint_last, checkpoint_best, wall_seconds}``.
    """
    started = time.monotonic()
    set_global_seed(cfg.seed)
    resolved_device = resolve_run_device(device if device is not None else cfg.train.device)
    logger.info("device: %s (requested %r)", resolved_device, device or cfg.train.device)

    root = Path(data_root) if data_root is not None else Path(cfg.data.data_root)
    run_dir = (
        Path(output_dir) if output_dir is not None else Path(cfg.output_dir) / cfg.train.out_subdir
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---------------------------------------------------------------
    manifest_path = resolve_against_root(cfg.data.manifest, root)
    manifest, meta = load_manifest(manifest_path)
    split = load_split(resolve_against_root(cfg.data.split_artifact, root))
    train_df, val_df = _select_splits(manifest, split)
    logger.info("split %s: %d train / %d val rows", split.split_id[:12], len(train_df), len(val_df))

    feature_columns = resolve_feature_set(cfg.features.feature_set)
    target_normalizers = _fit_target_normalizers(train_df)
    train_ds, val_ds, embedding_dim = _build_datasets(
        cfg,
        train_df,
        val_df,
        feature_columns,
        root=root,
        embedding_reader=embedding_reader,
    )
    feature_normalizer = train_ds.stats
    batch_size = int(cfg.train.batch_size)
    # Dedicated CPU generator for the train RandomSampler: re-seeded per epoch
    # below so the batch order is deterministic in (seed, epoch) and identical
    # whether the epoch is reached in one run or after a resume (see finding 1).
    train_sampler_generator = torch.Generator()
    train_loader = _make_loader(
        train_ds,
        cfg,
        resolved_device,
        batch_size,
        shuffle=True,
        sampler_generator=train_sampler_generator,
    )
    val_loader = _make_loader(val_ds, cfg, resolved_device, batch_size, shuffle=False)

    # --- model / optimizer / scheduler / amp --------------------------------
    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy

    image_backbone = None
    if cfg.data.input_mode == "image":
        # Production path (finding F6): with no injected builder, construct the
        # backbone the config names (default DINOv2) so a shipped image-mode
        # experiment (v6) actually runs. The injection hook still wins for tests.
        builder = image_backbone_builder or _default_image_backbone_builder(cfg, resolved_device)
        image_backbone = builder()
    # The windowed-alignment strategy selects the visual temporal pooler so the
    # model matches what the dataset emits (attention_pooling -> learned attention
    # over embedding_seq; center_frame / mean_embedding -> mask-aware mean).
    temporal_pooling = temporal_pooling_for_strategy(cfg.data.alignment.strategy)
    model = build_model(
        cfg,
        len(feature_columns),
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        temporal_pooling=temporal_pooling,
    ).to(resolved_device)

    # Bind the narrowed model rather than a bare bool: only ClimatologyModel carries
    # fit_from_targets, and `climatology is not None` is the same predicate the
    # isinstance check produced.
    climatology = model if isinstance(model, ClimatologyModel) else None
    is_climatology = climatology is not None
    if climatology is not None:
        _fit_climatology(climatology, cfg, train_df, target_normalizers)

    optimizer, lr_labels = _build_optimizer(model, cfg)
    monitor_key = _monitor_key(cfg.train.early_stopping.monitor)
    monitor_mode = "max" if "acc" in monitor_key else "min"
    scheduler, scheduler_is_plateau = _build_scheduler(
        cfg.train.scheduler, optimizer, cfg.train.epochs, monitor_mode
    )

    amp_enabled = amp if amp is not None else bool(cfg.train.amp.enabled)
    autocast_device, autocast_dtype, scaler = _build_amp(
        amp_enabled, cfg.train.amp.dtype, resolved_device
    )

    from allsky.training.losses import MultitaskLoss

    loss_fn = MultitaskLoss(cfg.targets, target_normalizers).to(resolved_device)

    # --- resume -------------------------------------------------------------
    fields = _csv_fields(cfg)
    start_epoch = 0
    global_step = 0
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []
    resume_path = _resume_path(resume, run_dir)
    if resume_path is not None:
        # Load once, then check the payload against this run before any of it is
        # applied: a refused resume must leave the model, the optimizer and the
        # metrics files exactly as it found them.
        checkpoint = load_checkpoint(
            resume_path, map_location=resolved_device, trust_pickle=trust_checkpoint
        )
        _check_resume_provenance(
            checkpoint,
            path=resume_path,
            split_id=split.split_id,
            meta=meta,
            feature_columns=feature_columns,
        )
        stored_monitor = (checkpoint.get("best_metric") or {}).get("name")
        monitor_changed = stored_monitor is not None and str(stored_monitor) != monitor_key
        start_epoch, global_step, best_value, best_epoch, restored_no_improve = _restore(
            checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            skip_scheduler_state=monitor_changed and scheduler_is_plateau,
        )
        # Restore the patience counter; a pre-field checkpoint yields a safe lower
        # bound (epochs since the recorded best), never a negative value.
        epochs_no_improve = (
            restored_no_improve
            if restored_no_improve is not None
            else (max(0, start_epoch - best_epoch) if best_epoch is not None else 0)
        )
        if monitor_changed:
            # The stored best belongs to another metric — another scale, and often
            # the other comparison direction — so seeding from it would make every
            # improvement test meaningless.
            logger.warning(
                "resume: early-stopping monitor changed %r -> %r; discarding the stored "
                "best (%s=%s @ epoch %s) and the patience counter — only epochs from here "
                "on are candidates for best.ckpt under the new monitor",
                stored_monitor,
                monitor_key,
                stored_monitor,
                best_value,
                best_epoch,
            )
            best_value, best_epoch, epochs_no_improve = None, None, 0
            _rotate_stale_best(run_dir)
        _reconcile_cosine_horizon(scheduler, optimizer, cfg)
        # Crash-window dedupe: metrics.csv/json are written before last.ckpt each
        # epoch, so a crash in that gap can leave rows for an epoch the checkpoint
        # never completed. Drop rows past the resumed epoch and rewrite both files
        # from the truncated history before the loop appends again (finding 3).
        history = _truncate_metrics(run_dir, fields, start_epoch)
        logger.info(
            "resumed from %s at epoch %d (global_step %d, epochs_no_improve %d)",
            resume_path,
            start_epoch,
            global_step,
            epochs_no_improve,
        )
    else:
        # Fresh start into a possibly-reused run dir: stale metrics.csv/json from
        # an earlier run would otherwise be appended to (finding F10). Rotate them
        # aside so this run's metrics files contain exactly this run's epochs.
        _reset_stale_metrics(run_dir)

    # --- epoch loop ---------------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter

    dhi_mean, dhi_std = _stats_or_identity(target_normalizers, "dhi")
    kindex_mean, kindex_std = _stats_or_identity(target_normalizers, "kindex")
    component_weights = _loss_component_weights(cfg.targets)
    epochs_ran = 0
    last_val_metrics: dict[str, float] = {}
    patience = int(cfg.train.early_stopping.patience)
    min_delta = float(cfg.train.early_stopping.min_delta)
    if resume_path is not None and epochs_no_improve >= patience:
        # The stopping rule is only re-tested at the END of an epoch, so a resume
        # into an already-early-stopped checkpoint would train one more epoch per
        # invocation — and a cron re-running `--resume auto` would walk the whole
        # remaining budget past the declared stop.
        logger.info(
            "early stopping already satisfied at epoch %d (no %s improvement for %d >= "
            "patience %d); nothing to train — raise train.early_stopping.patience to continue",
            start_epoch,
            monitor_key,
            epochs_no_improve,
            patience,
        )
    else:
        writer = SummaryWriter(log_dir=str(run_dir / "runs"))
        try:
            for epoch in range(start_epoch, cfg.train.epochs):
                # Deterministic, resume-stable batch order: order = f(seed, epoch),
                # independent of the resume point, persistent_workers, or global-RNG
                # draw count (finding 1).
                train_sampler_generator.manual_seed(cfg.seed * 100003 + epoch)
                # Read the rates BEFORE the epoch trains and before scheduler.step():
                # what is logged must be the rate this epoch actually ran at, per
                # named group (param_groups[0] is the backbone whenever backbone_lr
                # is set, not the rate driving the trunk and heads).
                lrs = _current_lrs(optimizer, lr_labels)
                train_metrics, global_step = _train_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    device=resolved_device,
                    autocast_device=autocast_device,
                    autocast_dtype=autocast_dtype,
                    scaler=scaler,
                    grad_accum_steps=max(1, int(cfg.train.grad_accum_steps)),
                    grad_clip_norm=cfg.train.grad_clip_norm,
                    skip_optimization=is_climatology,
                    global_step=global_step,
                    target_stats=(dhi_mean, dhi_std, kindex_mean, kindex_std),
                    component_weights=component_weights,
                )
                val_metrics = _eval_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    device=resolved_device,
                    autocast_device=autocast_device,
                    autocast_dtype=autocast_dtype,
                    amp_enabled=amp_enabled,
                    target_stats=(dhi_mean, dhi_std, kindex_mean, kindex_std),
                    component_weights=component_weights,
                )
                last_val_metrics = val_metrics

                monitor_value = val_metrics.get(monitor_key)
                if monitor_value is None:
                    raise KeyError(
                        f"early-stopping monitor {cfg.train.early_stopping.monitor!r} "
                        f"resolves to {monitor_key!r}, absent from val metrics "
                        f"{sorted(val_metrics)}"
                    )
                if scheduler is not None:
                    scheduler.step(monitor_value) if scheduler_is_plateau else scheduler.step()

                improved = _improved(monitor_value, best_value, monitor_mode, min_delta)
                if improved:
                    best_value, best_epoch, epochs_no_improve = monitor_value, epoch + 1, 0
                else:
                    epochs_no_improve += 1

                _log_epoch(writer, epoch, lrs, train_metrics, val_metrics)
                row = _epoch_row(fields, epoch + 1, lrs, train_metrics, val_metrics)
                _append_csv(run_dir / "metrics.csv", fields, row)
                history.append(row)
                atomic_write_json(run_dir / "metrics.json", history)

                best_metric = {"name": monitor_key, "value": best_value, "epoch": best_epoch}
                common = _checkpoint_common(
                    cfg=cfg,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    feature_normalizer=feature_normalizer,
                    target_normalizers=target_normalizers,
                    feature_columns=feature_columns,
                    meta=meta,
                    split_id=split.split_id,
                    image_backbone=image_backbone,
                )
                rng_state = capture_rng_state()
                save_checkpoint(
                    run_dir / LAST_CHECKPOINT,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                    rng_state=rng_state,
                    epochs_no_improve=epochs_no_improve,
                    **common,
                )
                if improved:
                    save_checkpoint(
                        run_dir / BEST_CHECKPOINT,
                        epoch=epoch + 1,
                        global_step=global_step,
                        best_metric=best_metric,
                        rng_state=rng_state,
                        epochs_no_improve=epochs_no_improve,
                        **common,
                    )
                epochs_ran += 1
                logger.info(
                    "epoch %d/%d — train loss %.4f | val loss %.4f | %s %.4f (best %.4f @ %s)",
                    epoch + 1,
                    cfg.train.epochs,
                    train_metrics.get("loss", float("nan")),
                    val_metrics.get("loss", float("nan")),
                    monitor_key,
                    monitor_value,
                    best_value if best_value is not None else float("nan"),
                    best_epoch,
                )
                if epochs_no_improve >= patience:
                    logger.info(
                        "early stopping at epoch %d (no %s improvement for %d)",
                        epoch + 1,
                        monitor_key,
                        patience,
                    )
                    break
        finally:
            writer.close()

    return {
        "best_metric": {"name": monitor_key, "value": best_value, "epoch": best_epoch},
        "epochs_ran": epochs_ran,
        "epoch": start_epoch + epochs_ran,
        "global_step": global_step,
        "final_val_metrics": last_val_metrics,
        "output_dir": str(run_dir),
        "checkpoint_last": str(run_dir / LAST_CHECKPOINT),
        "checkpoint_best": str(run_dir / BEST_CHECKPOINT),
        "wall_seconds": time.monotonic() - started,
    }


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _select_splits(manifest: pd.DataFrame, split: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train/val manifest rows by ``day_id`` (validation split required)."""
    day_ids = manifest["day_id"].astype(str)
    train_days = set(split.days_for("train"))
    val_days = set(split.days_for("val"))
    if not val_days:
        raise ValueError("split has no validation days; a val split is required for training")
    train_df = manifest.loc[day_ids.isin(train_days)].reset_index(drop=True)
    val_df = manifest.loc[day_ids.isin(val_days)].reset_index(drop=True)
    if train_df.empty:
        raise ValueError("no train rows: the split's train days are absent from the manifest")
    if val_df.empty:
        raise ValueError("no val rows: the split's validation days are absent from the manifest")
    return train_df, val_df


def _fit_target_normalizers(train_df: pd.DataFrame) -> dict[str, TargetNormalizer]:
    """Fit ``dhi`` / ``kindex`` target normalizers on the train rows only."""
    from allsky.features.normalization import fit_target_normalizers

    raw = fit_target_normalizers(train_df, ["target_dhi", "target_kindex"])
    return {"dhi": raw["target_dhi"], "kindex": raw["target_kindex"]}


def _build_datasets(
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    root: Path,
    embedding_reader: EmbeddingReader | None,
) -> tuple[Any, Any, int | None]:
    """Build the train/val datasets for the configured input mode.

    Returns ``(train_ds, val_ds, embedding_dim)`` where ``embedding_dim`` is the
    reader dimension in embedding mode and ``None`` in image mode.
    """
    from allsky.data.datasets import MultimodalEmbeddingDataset, MultimodalImageDataset

    if cfg.data.input_mode == "embedding":
        reader = (
            embedding_reader
            if embedding_reader is not None
            else default_embedding_reader(cfg, root)
        )
        _validate_embedding_coverage(reader, train_df, val_df)
        # Wire the alignment strategy end to end: center_frame (default) keeps the
        # single-frame embedding; mean_embedding / attention_pooling resolve each
        # row's same-day co-frame window (dataset-level) and pool accordingly.
        # The config's AlignmentStrategyName IS the dataset's WindowMode, so the
        # two can no longer disagree about which modes exist.
        window = cfg.data.alignment.strategy
        window_minutes = float(cfg.data.alignment.window_minutes)
        train_ds = MultimodalEmbeddingDataset(
            train_df,
            feature_columns,
            embedding_reader=reader,
            train=True,
            window=window,
            window_minutes=window_minutes,
        )
        val_ds = MultimodalEmbeddingDataset(
            val_df,
            feature_columns,
            embedding_reader=reader,
            train=False,
            stats=train_ds.stats,
            window=window,
            window_minutes=window_minutes,
        )
        embedding_dim = int(getattr(reader, "dim", 0)) or int(train_ds.embedding_dim)
        return train_ds, val_ds, embedding_dim

    image_size = int(_model_param(cfg, "image_size", 224))
    image_train = MultimodalImageDataset(
        train_df, feature_columns, data_root=root, image_size=image_size, train=True
    )
    image_val = MultimodalImageDataset(
        val_df,
        feature_columns,
        data_root=root,
        image_size=image_size,
        train=False,
        stats=image_train.stats,
    )
    return image_train, image_val, None


def _validate_embedding_coverage(
    reader: EmbeddingReader, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    """Fail up front if any needed ``sample_id`` has no embedding (readers that expose them)."""
    lister = getattr(reader, "sample_ids", None)
    if not callable(lister):
        return
    available = {str(s) for s in lister()}
    needed = {str(s) for s in pd.concat([train_df["sample_id"], val_df["sample_id"]])}
    missing = sorted(needed - available)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"embeddings are missing {len(missing)} required sample_id(s): {preview}"
            + (" ..." if len(missing) > 10 else "")
        )


def _make_loader(
    dataset: Any,
    cfg: ExperimentConfig,
    device: str,
    batch_size: int,
    *,
    shuffle: bool,
    sampler_generator: torch.Generator | None = None,
) -> DataLoader[Any]:
    """Build a DataLoader with resume-stable, RNG-isolated batch ordering.

    Determinism relies on two dedicated generators, never on the global RNG:

    - the shuffled (train) loader uses an explicit
      :class:`~torch.utils.data.RandomSampler` bound to *sampler_generator*, which
      :func:`run_experiment` re-seeds per epoch to ``seed * 100003 + epoch``.  The
      permutation is therefore a pure function of ``(seed, epoch)`` and identical
      whether an epoch is reached in one run or after a resume — including with
      ``persistent_workers`` on, where the sampler is re-drawn every epoch;
    - a per-loader ``generator`` (seeded from ``cfg.seed``) feeds the worker
      ``base_seed`` draw so that draw consumes this generator, **not** the global
      RNG.  Otherwise the base_seed (drawn only on a loader's first iteration)
      would perturb the global RNG that drives dropout — differently for a resumed
      run whose loader is created mid-schedule than for an uninterrupted one.

    No ``worker_init_fn`` is set: the datasets do no worker-side random
    augmentation (they read fixed features/embeddings), so worker RNG never
    influences a batch; add one here if augmentation is introduced.
    """
    num_workers = int(cfg.train.num_workers)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg.seed))
    sampler: RandomSampler | None = None
    if shuffle:
        if sampler_generator is None:
            raise ValueError("a shuffled loader requires a sampler_generator")
        sampler = RandomSampler(dataset, generator=sampler_generator)
    return DataLoader(
        cast("Dataset[Any]", dataset),
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
        generator=loader_generator,
    )


# ---------------------------------------------------------------------------
# model / optimizer / scheduler / amp helpers
# ---------------------------------------------------------------------------


def _model_param(cfg: ExperimentConfig, key: str, default: Any) -> Any:
    """Read an architecture hyper-parameter off the permissive model config."""
    return dict(cfg.model.model_dump()).get(key, default)


def _default_image_backbone_builder(cfg: ExperimentConfig, device: str) -> Callable[[], nn.Module]:
    """Build the default image backbone factory for ``input_mode='image'`` (finding F6).

    When no ``image_backbone_builder`` is injected, image-mode training must still
    build a real visual backbone or the shipped v6 config cannot run.  The factory
    constructs the backbone named by the model config (``model.backbone``, default
    ``dinov2_vits14``; ``model.backbone_pooling``, default ``cls``) on the run
    device via :func:`allsky.embeddings.backbone.build_backbone`, then coerces it
    into a trainable ``nn.Module`` (:func:`allsky.modeling.visual_encoder.coerce_image_backbone`
    wraps the DINOv2 extraction wrapper; an ``nn.Module`` — e.g. a test stub, or a
    monkeypatched ``build_backbone`` — passes straight through).  Construction is
    deferred to call time and any failure is re-raised with a message naming the
    config knobs to fix.
    """

    def build() -> nn.Module:
        # build_backbone is imported here, not at module scope, so a test that
        # monkeypatches allsky.embeddings.backbone.build_backbone is honoured.
        from allsky.embeddings.backbone import build_backbone
        from allsky.modeling.visual_encoder import coerce_image_backbone

        params = dict(cfg.model.model_dump())
        name = str(params.get("backbone", "dinov2_vits14"))
        pooling = str(params.get("backbone_pooling", "cls"))
        try:
            backbone = build_backbone(name, pooling=_backbone_pooling(pooling), device=device)
            return coerce_image_backbone(backbone, pooling=pooling)
        except Exception as exc:
            raise RuntimeError(
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
    :func:`_default_image_backbone_builder` raises, since the only backbone that
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


def _fit_climatology(
    model: ClimatologyModel,
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    target_normalizers: Mapping[str, TargetNormalizer],
) -> None:
    """Fit the constant-prediction climatology model from raw train targets."""
    model.fit_from_targets(
        dhi=train_df["target_dhi"].to_numpy() if cfg.targets.dhi.enabled else None,
        kindex=train_df["target_kindex"].to_numpy() if cfg.targets.kindex.enabled else None,
        cloud_fraction=(
            train_df["cloud_fraction"].to_numpy() if cfg.targets.cloud_fraction.enabled else None
        ),
        sky_class=train_df["sky_class"].to_numpy() if cfg.targets.sky.enabled else None,
        target_normalizers=target_normalizers,
    )


def _build_optimizer(
    model: nn.Module, cfg: ExperimentConfig
) -> tuple[torch.optim.Optimizer, list[str]]:
    """AdamW over ``model.param_groups(backbone_lr)`` when available, else parameters.

    Also returns one metrics label per parameter group, captured here while the
    per-group override is still visible — AdamW fills ``lr`` into *every* group, so
    after construction the backbone group is no longer distinguishable.  A group
    carrying its own ``lr`` is the image backbone (the only group
    :meth:`MultimodalModel.param_groups` sets it on); everything else runs at
    ``train.lr`` and is labelled ``lr``.
    """
    param_groups_fn = getattr(model, "param_groups", None)
    if callable(param_groups_fn):
        params: Any = param_groups_fn(cfg.train.backbone_lr)
        labels = ["lr_backbone" if "lr" in group else "lr" for group in params]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        labels = ["lr"]
    if cfg.train.backbone_lr is not None and "lr_backbone" not in labels:
        # Either the model has no param_groups at all or its backbone contributed
        # no trainable parameter: nothing runs at backbone_lr, which is only
        # obvious from the metrics much later (an lr_backbone column left blank).
        logger.warning(
            "train.backbone_lr=%s is set but model %r produced no separate backbone "
            "parameter group; every trainable parameter runs at train.lr=%s",
            cfg.train.backbone_lr,
            cfg.model.name,
            cfg.train.lr,
        )
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return optimizer, labels


def _current_lrs(optimizer: torch.optim.Optimizer, labels: list[str]) -> dict[str, float]:
    """The learning rate in effect per labelled parameter group.

    ``strict=True`` is safe across a resume: ``load_state_dict`` already raises on
    a group-count mismatch, so the labels stay index-aligned with the groups.
    """
    return {
        label: float(group["lr"])
        for label, group in zip(labels, optimizer.param_groups, strict=True)
    }


def _build_scheduler(
    sched_cfg: SchedulerConfig, optimizer: torch.optim.Optimizer, epochs: int, mode: str
) -> tuple[Any | None, bool]:
    """Build the scheduler; returns ``(scheduler_or_none, is_plateau)``."""
    name = sched_cfg.name
    if name == "none":
        return None, False
    params = dict(sched_cfg.params)
    if name == "cosine":
        t_max = int(params.pop("T_max", epochs))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **params), False
    if name == "plateau":
        params.setdefault("mode", mode)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params), True
    raise ValueError(f"unknown scheduler {name!r}; expected 'none', 'cosine' or 'plateau'")


def _build_amp(amp_enabled: bool, dtype: str, device: str) -> tuple[str | None, Any, Any | None]:
    """Resolve the autocast device/dtype and GradScaler for the AMP config.

    fp16 requires CUDA and a GradScaler; bf16 uses autocast on CUDA or CPU with
    no scaler.  Returns ``(autocast_device, autocast_dtype, scaler)``; when AMP is
    off, ``autocast_device`` is ``None``.
    """
    if not amp_enabled:
        return None, None, None
    if dtype == "fp16":
        if device != "cuda":
            raise RuntimeError("amp dtype 'fp16' requires a CUDA device; use 'bf16' on CPU")
        return "cuda", torch.float16, torch.amp.GradScaler("cuda")
    autocast_device = "cuda" if device == "cuda" else "cpu"
    return autocast_device, torch.bfloat16, None


# ---------------------------------------------------------------------------
# epoch passes
# ---------------------------------------------------------------------------


def _autocast(device: str | None, dtype: Any) -> Any:
    """Autocast context for AMP, or a null context when AMP is off."""
    if device is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device, dtype=dtype)


def _move(batch: dict[str, Any], device: str) -> dict[str, Any]:
    """Move tensor values of *batch* to *device* (non-tensors pass through)."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def _train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    autocast_device: str | None,
    autocast_dtype: Any,
    scaler: Any | None,
    grad_accum_steps: int,
    grad_clip_norm: float | None,
    skip_optimization: bool,
    global_step: int,
    target_stats: tuple[float, float, float, float],
    component_weights: Mapping[str, float],
) -> tuple[dict[str, float], int]:
    """Run one training epoch with grad accumulation/clipping; return metrics + step."""
    model.train()
    accumulator = _MetricAccumulator(target_stats, component_weights)
    n_batches = len(loader)
    pending = 0
    optimizer.zero_grad(set_to_none=True)
    for i, raw in enumerate(loader):
        batch = _move(raw, device)
        with _autocast(autocast_device, autocast_dtype):
            outputs = model(batch)
            losses = loss_fn(outputs, batch)
            loss = losses["loss"]
        if not skip_optimization:
            scaled = loss / grad_accum_steps
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            pending += 1
            if pending == grad_accum_steps or (i + 1) == n_batches:
                if grad_clip_norm is not None:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                global_step += 1
        accumulator.update(outputs, batch, losses)
    return accumulator.result(), global_step


def _eval_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    loss_fn: nn.Module,
    device: str,
    autocast_device: str | None,
    autocast_dtype: Any,
    amp_enabled: bool,
    target_stats: tuple[float, float, float, float],
    component_weights: Mapping[str, float],
) -> dict[str, float]:
    """Run one validation epoch (no grad); return loss components + physical metrics."""
    model.eval()
    accumulator = _MetricAccumulator(target_stats, component_weights)
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with _autocast(autocast_device if amp_enabled else None, autocast_dtype):
                outputs = model(batch)
                losses = loss_fn(outputs, batch)
            accumulator.update(outputs, batch, losses)
    return accumulator.result()


#: Loss-component key -> the batch column whose *present* rows that component was
#: reduced over.  ``MultitaskLoss`` averages every component over its own mask
#: (finite target, or ``sky_class >= 0``), so the epoch average has to weight each
#: batch by those counts and not by the batch size.
_COMPONENT_TARGET_COLUMNS = {
    "loss_dhi": "dhi",
    "loss_kindex": "kindex",
    "loss_cloud_fraction": "cloud_fraction",
    "loss_sky": "sky_class",
}


def _loss_component_weights(targets: TargetsConfig) -> dict[str, float]:
    """Per-head weights, keyed by loss-component name, for the epoch loss total.

    :meth:`allsky.training.losses.MultitaskLoss.forward` sums its per-head
    components with these same configured weights; the epoch metrics rebuild that
    weighted sum from the per-head epoch means, so both must read the one
    :class:`~allsky.config.TargetsConfig`.
    """
    return {
        "loss_dhi": float(targets.dhi.weight),
        "loss_kindex": float(targets.kindex.weight),
        "loss_sky": float(targets.sky.weight),
        "loss_cloud_fraction": float(targets.cloud_fraction.weight),
    }


def _present_target_count(batch: Mapping[str, Tensor], column: str) -> int:
    """Rows of *column* a loss component counted: finite value, or class ``>= 0``."""
    target = batch[column]
    mask = target >= 0 if column == "sky_class" else torch.isfinite(target)
    return int(mask.sum())


class _MetricAccumulator:
    """Accumulate loss components and physical-unit quick metrics over an epoch.

    Every component :class:`~allsky.training.losses.MultitaskLoss` returns is a
    mean over the rows whose target is present, so each batch is folded in
    weighted by *its own* count of present rows and divided by the epoch's count
    for that head — not by the batch row count, which made the epoch metrics
    depend on ``train.batch_size`` whenever labels were missing (an all-missing
    batch contributed an exact 0 with full weight).  The reported ``loss`` is
    likewise rebuilt from the per-head epoch means and *component_weights* — the
    same weighted sum the loss module computes per batch — instead of averaging
    the per-batch totals.

    Only the reported metrics changed: ``losses["loss"]`` still drives
    ``backward()`` per batch, unweighted by anything here.

    A head with no present row in the whole epoch keeps its ``loss_<head>`` key
    (metrics.csv has a column for it and it is a legal early-stopping monitor)
    with value ``0.0``, contributes nothing to the total, and is named in a
    warning.
    """

    def __init__(
        self,
        target_stats: tuple[float, float, float, float],
        component_weights: Mapping[str, float],
    ) -> None:
        self._dhi_mean, self._dhi_std, self._kindex_mean, self._kindex_std = target_stats
        self._component_weights = dict(component_weights)
        self._component_sums: dict[str, float] = {}
        self._component_counts: dict[str, int] = {}
        self._dhi_abs = 0.0
        self._dhi_n = 0
        self._kindex_abs = 0.0
        self._kindex_n = 0
        self._sky_correct = 0
        self._sky_n = 0

    def update(
        self, outputs: Mapping[str, Tensor], batch: dict[str, Tensor], losses: Mapping[str, Tensor]
    ) -> None:
        """Fold one batch's outputs/targets/losses into the running sums."""
        size = int(batch["features"].shape[0])
        for key, value in losses.items():
            if key == "loss":
                continue
            column = _COMPONENT_TARGET_COLUMNS.get(key)
            # An unrecognised (future) component falls back to the batch row count.
            count = size if column is None else _present_target_count(batch, column)
            self._component_sums.setdefault(key, 0.0)
            self._component_counts.setdefault(key, 0)
            if count == 0:
                continue
            self._component_sums[key] += float(value.detach()) * count
            self._component_counts[key] += count
        if "dhi" in outputs:
            self._dhi_abs, self._dhi_n = _mae_accumulate(
                outputs["dhi"],
                batch["dhi"],
                self._dhi_mean,
                self._dhi_std,
                self._dhi_abs,
                self._dhi_n,
            )
        if "kindex" in outputs:
            self._kindex_abs, self._kindex_n = _mae_accumulate(
                outputs["kindex"],
                batch["kindex"],
                self._kindex_mean,
                self._kindex_std,
                self._kindex_abs,
                self._kindex_n,
            )
        if "sky_logits" in outputs:
            predicted = outputs["sky_logits"].detach().argmax(dim=-1)
            mask = batch["sky_class"] >= 0
            if bool(mask.any()):
                self._sky_correct += int((predicted[mask] == batch["sky_class"][mask]).sum())
                self._sky_n += int(mask.sum())

    def result(self) -> dict[str, float]:
        """Finalize the epoch: per-head masked means and their weighted total."""
        components: dict[str, float] = {}
        total = 0.0
        heads_without_labels: list[str] = []
        for key, summed in self._component_sums.items():
            count = self._component_counts[key]
            if count == 0:
                components[key] = 0.0
                heads_without_labels.append(key)
                continue
            components[key] = summed / count
            total += self._component_weights.get(key, 1.0) * components[key]
        if heads_without_labels:
            logger.warning(
                "epoch metrics: %s had no valid target row in the whole split; reported as 0.0 "
                "and excluded from the total loss — check the head is meant to be enabled",
                ", ".join(sorted(heads_without_labels)),
            )
        metrics: dict[str, float] = {"loss": total, **components}
        if self._dhi_n:
            metrics["dhi_mae"] = self._dhi_abs / self._dhi_n
        if self._kindex_n:
            metrics["kindex_mae"] = self._kindex_abs / self._kindex_n
        if self._sky_n:
            metrics["sky_acc"] = self._sky_correct / self._sky_n
        return metrics


def _mae_accumulate(
    pred: Tensor, target: Tensor, mean: float, std: float, abs_sum: float, count: int
) -> tuple[float, int]:
    """Add the physical-unit absolute error of *pred* vs *target* over finite rows."""
    physical = pred.detach().float() * std + mean
    truth = target.detach().float()
    mask = torch.isfinite(truth)
    if bool(mask.any()):
        abs_sum += float((physical[mask] - truth[mask]).abs().sum())
        count += int(mask.sum())
    return abs_sum, count


# ---------------------------------------------------------------------------
# resume / checkpoint payload helpers
# ---------------------------------------------------------------------------


def _resume_path(resume: str | Path | None, run_dir: Path) -> Path | None:
    """Resolve the checkpoint to resume from (``"auto"`` finds ``last.ckpt``)."""
    if resume is None:
        return None
    if isinstance(resume, str) and resume == "auto":
        candidate = run_dir / LAST_CHECKPOINT
        if candidate.exists():
            return candidate
        logger.info("resume='auto' but %s does not exist; starting fresh", candidate)
        return None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def _check_resume_provenance(
    checkpoint: Mapping[str, Any],
    *,
    path: Path,
    split_id: str,
    meta: Mapping[str, Any],
    feature_columns: list[str],
) -> None:
    """Refuse a resume whose checkpoint was trained against a different dataset.

    ``run_experiment`` re-fits the feature and target normalizers on whatever is on
    disk *before* restoring, so resuming into a rebuilt manifest would silently
    reinterpret converged weights in a re-scaled target space — and compare
    ``best.ckpt`` across two different validation day sets.  Every recorded field
    is compared exactly; a field the checkpoint does not carry is skipped so
    pre-provenance checkpoints still resume, mirroring the evaluator's
    ``_check_split_id``.  Unlike an evaluation, a resume has no legitimate reason
    to continue on a mismatch.
    """
    mismatches: list[str] = []
    for field, stored, current in (
        ("split_id", checkpoint.get("split_id"), split_id),
        ("manifest_sha256", checkpoint.get("manifest_sha256"), meta.get("manifest_sha256")),
        ("dataset_version", checkpoint.get("dataset_version"), _dataset_version(meta)),
    ):
        if stored is None:
            logger.info("resume: %s is not recorded in %s; skipping that check", field, path)
        elif str(stored) != str(current):
            mismatches.append(
                f"{field}: checkpoint {str(stored)[:12]} vs current {str(current)[:12]}"
            )
    stored_columns = checkpoint.get("feature_columns")
    if stored_columns is None:
        logger.info("resume: feature_columns is not recorded in %s; skipping that check", path)
    elif list(stored_columns) != feature_columns:
        mismatches.append(
            f"feature_columns: checkpoint {list(stored_columns)} vs current {feature_columns}"
        )
    if mismatches:
        raise RuntimeError(
            f"refusing to resume from {path}: it was trained against a different dataset "
            + "; ".join(mismatches)
            + ". Run without --resume (or point --out-dir at a fresh run directory) to "
            "train from scratch on the current dataset."
        )


def _restore(
    checkpoint: Mapping[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    *,
    skip_scheduler_state: bool = False,
) -> tuple[int, int, float | None, int | None, int | None]:
    """Restore all training state from an already-loaded *checkpoint*.

    :func:`run_experiment` loads the payload itself so the provenance and monitor
    checks can run before any state is applied.  *skip_scheduler_state* drops the
    stored scheduler state, which is what a ``ReduceLROnPlateau`` needs when the
    monitor changed: its ``mode``/``best``/``num_bad_epochs`` belong to the old
    metric, while the reduced learning rate itself survives in the optimizer's
    param groups.

    Returns ``(epoch, global_step, best_value, best_epoch, epochs_no_improve)``;
    ``epochs_no_improve`` is ``None`` on pre-field checkpoints (the caller then
    reconstructs a lower bound from ``epoch``/``best_epoch``).
    """
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if (
        scheduler is not None
        and not skip_scheduler_state
        and checkpoint.get("scheduler_state") is not None
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint["rng_state"])
    best = checkpoint.get("best_metric") or {}
    stored_no_improve = checkpoint.get("epochs_no_improve")
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["global_step"]),
        best.get("value"),
        best.get("epoch"),
        None if stored_no_improve is None else int(stored_no_improve),
    )


def _reconcile_cosine_horizon(
    scheduler: Any | None, optimizer: torch.optim.Optimizer, cfg: ExperimentConfig
) -> None:
    """Re-point a restored cosine schedule at the run's *current* epoch budget.

    ``scheduler.load_state_dict`` brings back the checkpoint's ``T_max``, so a
    resume that extends the budget keeps annealing against the old horizon: past
    ``T_max`` the cosine turns around and the learning rate climbs back to its
    initial value over the second half of the run.  A ``T_max`` pinned explicitly
    in ``train.scheduler.params`` is the operator's contract and is honoured.

    The rate is re-derived in closed form rather than left to the scheduler:
    ``CosineAnnealingLR.get_lr`` is a multiplicative recursion off each group's
    current rate, which the old horizon may already have annealed to ``eta_min``.
    """
    sched_cfg = cfg.train.scheduler
    if scheduler is None or sched_cfg.name != "cosine" or "T_max" in sched_cfg.params:
        return
    budget = int(cfg.train.epochs)
    if int(scheduler.T_max) == budget:
        return
    logger.warning(
        "resume: cosine T_max %d from the checkpoint reconciled to the current budget %d; "
        "the learning rate is re-derived from the closed-form cosine at epoch %d",
        scheduler.T_max,
        budget,
        scheduler.last_epoch,
    )
    scheduler.T_max = budget
    lrs = [
        scheduler.eta_min
        + (base_lr - scheduler.eta_min)
        * (1 + math.cos(math.pi * scheduler.last_epoch / budget))
        / 2
        for base_lr in scheduler.base_lrs
    ]
    for group, lr in zip(optimizer.param_groups, lrs, strict=True):
        group["lr"] = lr
    scheduler._last_lr = list(lrs)


def _rotate_stale_best(run_dir: Path) -> None:
    """Move an existing ``best.ckpt`` aside when a monitor change invalidates it.

    Mirrors :func:`_reset_stale_metrics`: once the stored best is discarded the
    first resumed epoch unconditionally improves on ``None`` and rewrites
    ``best.ckpt``, which would destroy the previous monitor's best weights with no
    record of them.
    """
    path = run_dir / BEST_CHECKPOINT
    if not path.exists():
        return
    backup = path.with_name(f"{BEST_CHECKPOINT}.stale")
    os.replace(path, backup)
    logger.warning(
        "resume: rotated %s aside to %s (it was selected under the previous monitor)",
        path,
        backup.name,
    )


def _checkpoint_common(
    *,
    cfg: ExperimentConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    feature_normalizer: Any,
    target_normalizers: Mapping[str, TargetNormalizer],
    feature_columns: list[str],
    meta: Mapping[str, Any],
    split_id: str,
    image_backbone: nn.Module | None,
) -> dict[str, Any]:
    """Assemble the checkpoint fields shared by last.ckpt and best.ckpt."""
    normalizers = {
        "feature_normalizer": feature_normalizer.to_dict(),
        "target_normalizers": {k: v.to_dict() for k, v in target_normalizers.items()},
    }
    backbone_info = None
    if cfg.data.input_mode == "image" and image_backbone is not None:
        backbone_info = {
            "name": getattr(image_backbone, "name", type(image_backbone).__name__),
            "revision": getattr(image_backbone, "revision", None),
            "pooling": getattr(image_backbone, "pooling", None),
            "dim": getattr(image_backbone, "dim", None),
            "frozen": bool(_model_param(cfg, "backbone_frozen", False)),
        }
    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "config": cfg.model_dump(),
        "normalizers": normalizers,
        "feature_columns": feature_columns,
        "feature_groups": active_feature_groups(cfg.features.feature_set),
        "dataset_version": _dataset_version(meta),
        "split_id": split_id,
        "manifest_sha256": meta.get("manifest_sha256"),
        "backbone_info": backbone_info,
        "code_version_info": code_version(),
    }


def _dataset_version(meta: Mapping[str, Any]) -> str:
    """The dataset version a checkpoint records for *meta* (the code's when absent)."""
    from allsky.data.contracts import DATASET_VERSION

    return str(meta.get("dataset_version", DATASET_VERSION))


# ---------------------------------------------------------------------------
# metrics logging helpers
# ---------------------------------------------------------------------------


def _monitor_key(monitor: str) -> str:
    """Normalize an early-stopping monitor string to a val-metric key."""
    for prefix in ("val/", "val_"):
        if monitor.startswith(prefix):
            return monitor[len(prefix) :]
    return monitor


def _improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    """True when *current* improves on *best* by more than *min_delta*."""
    if best is None:
        return True
    if mode == "min":
        return current < best - min_delta
    return current > best + min_delta


def _stats_or_identity(
    normalizers: Mapping[str, TargetNormalizer], key: str
) -> tuple[float, float]:
    """Return ``(mean, std)`` for *key*, or ``(0.0, 1.0)`` when absent."""
    normalizer = normalizers.get(key)
    if normalizer is None:
        return 0.0, 1.0
    return float(normalizer.mean), float(normalizer.std)


def _csv_fields(cfg: ExperimentConfig) -> list[str]:
    """Stable, config-derived CSV column order (identical across resumes).

    ``lr_backbone`` is always emitted rather than made to depend on the optimizer's
    group count: the header must not change mid-run, and a run without a separate
    backbone rate simply leaves the cell blank.
    """
    fields = ["epoch", "lr", "lr_backbone"]
    for split in ("train", "val"):
        fields.append(f"{split}_loss")
        if cfg.targets.dhi.enabled:
            fields += [f"{split}_loss_dhi", f"{split}_dhi_mae"]
        if cfg.targets.kindex.enabled:
            fields += [f"{split}_loss_kindex", f"{split}_kindex_mae"]
        if cfg.targets.sky.enabled:
            fields += [f"{split}_loss_sky", f"{split}_sky_acc"]
        if cfg.targets.cloud_fraction.enabled:
            fields.append(f"{split}_loss_cloud_fraction")
    return fields


def _epoch_row(
    fields: list[str],
    epoch: int,
    lrs: Mapping[str, float],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Build a metrics row keyed by the canonical *fields* (missing -> empty)."""
    row: dict[str, Any] = dict.fromkeys(fields, "")
    row["epoch"] = epoch
    row.update({key: value for key, value in lrs.items() if key in row})
    for key, value in train_metrics.items():
        field = f"train_{key}"
        if field in row:
            row[field] = value
    for key, value in val_metrics.items():
        field = f"val_{key}"
        if field in row:
            row[field] = value
    return row


def _log_epoch(
    writer: Any,
    epoch: int,
    lrs: Mapping[str, float],
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    """Write per-epoch TensorBoard scalars."""
    for name, value in lrs.items():
        writer.add_scalar(name, value, epoch)
    for key, value in train_metrics.items():
        writer.add_scalar(f"train/{key}", value, epoch)
    for key, value in val_metrics.items():
        writer.add_scalar(f"val/{key}", value, epoch)


def _append_csv(path: Path, fields: list[str], row: Mapping[str, Any]) -> None:
    """Append *row* to the metrics CSV (writing the header only when new)."""
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _rewrite_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite the metrics CSV as *fields* header + *rows*."""

    def _write(tmp: Path) -> None:
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(path, _write)


def _truncate_metrics(run_dir: Path, fields: list[str], resumed_epoch: int) -> list[dict[str, Any]]:
    """Drop metrics rows past *resumed_epoch* and rewrite CSV + JSON from the rest.

    ``metrics.csv``/``metrics.json`` are written before ``last.ckpt`` each epoch,
    so a crash in that gap can leave rows for an epoch the resumed checkpoint never
    completed.  Only rows with ``epoch <= resumed_epoch`` (completed epochs) are
    kept; both files are atomically rewritten from them and the truncated history
    is returned for the loop to keep appending to.  ``metrics.json`` is the source
    of truth (it is always present once a checkpoint exists); if it is somehow
    absent the files are left untouched rather than risking data loss.
    """
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_json.exists():
        if metrics_csv.exists():
            logger.warning(
                "resume: metrics.json is missing but metrics.csv is present; leaving the "
                "metrics files untouched (cannot safely truncate without the JSON history)"
            )
        return []
    loaded = json.loads(metrics_json.read_text(encoding="utf-8"))
    history = [row for row in loaded if int(row.get("epoch", 0)) <= resumed_epoch]
    dropped = len(loaded) - len(history)
    if dropped:
        logger.info("resume: dropped %d stale metrics row(s) past epoch %d", dropped, resumed_epoch)
    _rewrite_csv(metrics_csv, fields, history)
    atomic_write_json(metrics_json, history)
    return history


def _reset_stale_metrics(run_dir: Path) -> None:
    """Rotate any pre-existing ``metrics.csv`` / ``metrics.json`` aside on a fresh run.

    A fresh (non-resume) run into a reused run directory must not append to the
    previous run's metrics (finding F10).  Each stale file is renamed to
    ``<name>.stale`` (replacing an older backup) rather than deleted, so the prior
    run's numbers are still recoverable; the fresh run then re-creates the files
    from scratch.
    """
    for name in ("metrics.csv", "metrics.json"):
        path = run_dir / name
        if path.exists():
            backup = path.with_name(f"{name}.stale")
            os.replace(path, backup)
            logger.warning(
                "fresh run: rotated stale %s aside to %s (a previous run wrote it)",
                path,
                backup.name,
            )
