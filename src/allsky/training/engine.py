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
   (resume-safe best seeding), with early stopping.  A run in which every epoch
   left the monitor non-finite writes no ``best.ckpt`` and raises rather than
   returning a summary naming an artifact that was never written;
#. resume fully from ``last.ckpt`` (``resume="auto"`` or a path), restoring
   model / optimizer / scheduler / scaler / epoch / global_step / best / RNG —
   but only after the checkpoint's dataset provenance matches this run (a rebuilt
   manifest is refused, not silently re-scaled), with the stored best discarded
   when the early-stopping monitor changed — it belongs to another metric, on
   another scale and often in the other comparison direction, so seeding from it
   would make every improvement test meaningless — a restored cosine horizon
   re-pointed at the current epoch budget, and nothing trained at all when the
   stopping rule was already satisfied.

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
from allsky.augmentation import AugmentationPipeline
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
from allsky.preprocessing import PreprocessingPipeline
from allsky.training.checkpointing import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    capture_rng_state,
    code_version,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from allsky.training.errors import TrainingError
from solrad_correction.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

__all__ = ["resolve_run_device", "run_experiment"]

MONITOR_CHANGE_SUFFIX = ".stale-monitor"
STALE_RUN_SUFFIX = ".stale"


def resolve_run_device(requested: str) -> str:
    """Resolve *requested* to a concrete device, erroring on unavailable cuda.

    Delegates ``"auto"`` resolution to
    :func:`allsky.training.device.resolve_device`, then raises a clear
    :class:`RuntimeError` when ``"cuda"`` is asked for but no CUDA device is
    available (rather than failing opaquely deep inside the first ``.to("cuda")``).

    Parameters
    ----------
    requested:
        ``"auto"``, or an explicit device string from ``cfg.train.device`` / the
        CLI override.

    Returns
    -------
    str
        The concrete device the run will use.

    Raises
    ------
    RuntimeError
        If ``"cuda"`` was requested but no CUDA device is available.
    """
    from allsky.training.device import resolve_device

    device = resolve_device(requested)
    if device == "cuda" and not torch.cuda.is_available():
        raise TrainingError(
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
        checkpoint; ``None`` starts fresh.  A resume into a checkpoint whose
        early-stopping rule is already satisfied trains nothing: the rule is
        otherwise only re-tested at the END of an epoch, so each invocation
        would train one more epoch and a cron re-running ``--resume auto`` would
        walk the whole remaining budget past the declared stop.
    trust_checkpoint:
        Read the resumed checkpoint with the unrestricted unpickler. Off by
        default: a checkpoint that travelled through Colab or a shared Drive is
        an untrusted input, so it is read under torch's restricted reader and a
        payload outside the allowlist is refused rather than executed.
    image_backbone_builder:
        Zero-arg factory returning the image backbone for ``input_mode="image"``
        visual models.  When omitted, :func:`_default_image_backbone_builder`
        constructs the backbone the model config names (``model.backbone``,
        default ``dinov2_vits14``) on the run device, which may download hub
        weights; tests inject a tiny stub to avoid that.
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
        ``checkpoint_best`` is the path only when that file exists on disk and
        ``None`` otherwise (a resume that trained nothing into a new run
        directory), so the summary never names an artifact a caller cannot read.

    Raises
    ------
    RuntimeError
        If epochs ran and every one of them left the monitor non-finite: no
        ``best.ckpt`` exists, and the run reports the divergence instead of
        exiting as a success.
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

    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy

    image_backbone = None
    if cfg.data.input_mode == "image":
        builder = image_backbone_builder or _default_image_backbone_builder(cfg, resolved_device)
        image_backbone = builder()
    temporal_pooling = temporal_pooling_for_strategy(cfg.data.alignment.strategy)
    model = build_model(
        cfg,
        len(feature_columns),
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        temporal_pooling=temporal_pooling,
    ).to(resolved_device)

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

    fields = _csv_fields(cfg)
    start_epoch = 0
    global_step = 0
    best_value: float | None = None
    best_epoch: int | None = None
    epochs_no_improve = 0
    history: list[dict[str, Any]] = []
    resume_path = _resume_path(resume, run_dir)
    if resume_path is not None:
        checkpoint = load_checkpoint(
            resume_path, map_location=resolved_device, trust_pickle=trust_checkpoint
        )
        _check_resume_provenance(
            checkpoint,
            path=resume_path,
            split_id=split.split_id,
            meta=meta,
            feature_columns=feature_columns,
            cfg=cfg,
        )
        _warn_optimizer_knob_drift(checkpoint, cfg)
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
        epochs_no_improve = (
            restored_no_improve
            if restored_no_improve is not None
            else (max(0, start_epoch - best_epoch) if best_epoch is not None else 0)
        )
        if monitor_changed:
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
        history = _truncate_metrics(run_dir, fields, start_epoch)
        logger.info(
            "resumed from %s at epoch %d (global_step %d, epochs_no_improve %d)",
            resume_path,
            start_epoch,
            global_step,
            epochs_no_improve,
        )
    else:
        _reset_stale_run_artifacts(run_dir)
    superseded_best_pending = resume_path is None and (run_dir / BEST_CHECKPOINT).exists()

    from torch.utils.tensorboard import SummaryWriter

    dhi_mean, dhi_std = _stats_or_identity(target_normalizers, "dhi")
    kindex_mean, kindex_std = _stats_or_identity(target_normalizers, "kindex")
    component_weights = _loss_component_weights(cfg.targets)
    epochs_ran = 0
    last_val_metrics: dict[str, float] = {}
    patience = int(cfg.train.early_stopping.patience)
    min_delta = float(cfg.train.early_stopping.min_delta)
    if resume_path is not None and epochs_no_improve >= patience:
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
                train_sampler_generator.manual_seed(cfg.seed * 100003 + epoch)
                # Augmentation seeds on (seed, epoch, idx); without advancing
                # this, every epoch would replay the identical draw per sample.
                if hasattr(train_ds, "epoch"):
                    train_ds.epoch = epoch
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
                    target_stats=(dhi_mean, dhi_std, kindex_mean, kindex_std),
                    component_weights=component_weights,
                )
                last_val_metrics = val_metrics

                monitor_value = val_metrics.get(monitor_key)
                if monitor_value is None:
                    raise TrainingError(
                        f"early-stopping monitor {cfg.train.early_stopping.monitor!r} "
                        f"resolves to {monitor_key!r}, absent from val metrics "
                        f"{sorted(val_metrics)}"
                    )
                if scheduler is not None:
                    scheduler.step(monitor_value) if scheduler_is_plateau else scheduler.step()

                if not math.isfinite(monitor_value):
                    logger.warning(
                        "epoch %d: monitor %s is %s (the epoch diverged); it is not a "
                        "candidate for best.ckpt and counts as no improvement",
                        epoch + 1,
                        monitor_key,
                        monitor_value,
                    )
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
                    if superseded_best_pending:
                        _rotate_superseded_best(run_dir)
                        superseded_best_pending = False
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
        if epochs_ran and best_value is None:
            raise TrainingError(
                f"no best checkpoint: {monitor_key} was non-finite in every one of the "
                f"{epochs_ran} epoch(s) that ran, so no epoch was ever a candidate for "
                f"{BEST_CHECKPOINT}. The metrics and the diverged weights are in {run_dir} "
                f"({LAST_CHECKPOINT}); lower train.lr, enable train.grad_clip_norm or turn "
                "AMP off before re-running"
            )

    best_checkpoint = run_dir / BEST_CHECKPOINT
    return {
        "best_metric": {"name": monitor_key, "value": best_value, "epoch": best_epoch},
        "epochs_ran": epochs_ran,
        "epoch": start_epoch + epochs_ran,
        "global_step": global_step,
        "final_val_metrics": last_val_metrics,
        "output_dir": str(run_dir),
        "checkpoint_last": str(run_dir / LAST_CHECKPOINT),
        "checkpoint_best": str(best_checkpoint) if best_checkpoint.exists() else None,
        "wall_seconds": time.monotonic() - started,
    }


def _select_splits(manifest: pd.DataFrame, split: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice train/val manifest rows by ``day_id`` (validation split required)."""
    day_ids = manifest["day_id"].astype(str)
    train_days = set(split.days_for("train"))
    val_days = set(split.days_for("val"))
    if not val_days:
        raise TrainingError("split has no validation days; a val split is required for training")
    train_df = manifest.loc[day_ids.isin(train_days)].reset_index(drop=True)
    val_df = manifest.loc[day_ids.isin(val_days)].reset_index(drop=True)
    if train_df.empty:
        raise TrainingError("no train rows: the split's train days are absent from the manifest")
    if val_df.empty:
        raise TrainingError("no val rows: the split's validation days are absent from the manifest")
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

    The alignment strategy is wired end to end: ``center_frame`` (default) keeps
    each row's single-frame embedding, while ``mean_embedding`` /
    ``attention_pooling`` make the dataset resolve the row's same-day co-frame
    window and pool over it.  The config's ``AlignmentStrategyName`` IS the
    dataset's ``WindowMode``, so the two cannot disagree about which modes exist.

    Returns
    -------
    tuple[Any, Any, int | None]
        ``(train_ds, val_ds, embedding_dim)`` where ``embedding_dim`` is the
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
    # Field names match one-for-one on both sides, so a new config key reaches
    # the pipeline instead of being silently dropped by a hand-written mapping.
    pipeline = AugmentationPipeline(**cfg.augmentation.model_dump())
    if pipeline.enabled:
        logger.info("augmentation: %s", pipeline)
    preprocess = PreprocessingPipeline(**cfg.preprocessing.model_dump())
    if preprocess.enabled:
        logger.info("preprocessing %s: %s", preprocess.identity, preprocess)
    image_train = MultimodalImageDataset(
        train_df,
        feature_columns,
        data_root=root,
        image_size=image_size,
        train=True,
        augment=pipeline,
        preprocess=preprocess,
        seed=cfg.seed,
    )
    image_val = MultimodalImageDataset(
        val_df,
        feature_columns,
        data_root=root,
        image_size=image_size,
        train=False,
        stats=image_train.stats,
        preprocess=preprocess,
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
        raise TrainingError(
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


def _model_param(cfg: ExperimentConfig, key: str, default: Any) -> Any:
    """Read an architecture hyper-parameter off the permissive model config."""
    return dict(cfg.model.model_dump()).get(key, default)


def _default_image_backbone_builder(cfg: ExperimentConfig, device: str) -> Callable[[], nn.Module]:
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

    ``train.optimizer`` is not read: AdamW is the only algorithm this engine
    builds, and :class:`~allsky.config.TrainConfig` declares the field as that one
    literal, so a config naming another one is refused when it is loaded rather
    than after a run has already seeded, loaded its dataset and built its model.
    """
    param_groups_fn = getattr(model, "param_groups", None)
    if callable(param_groups_fn):
        params: Any = param_groups_fn(cfg.train.backbone_lr)
        labels = ["lr_backbone" if "lr" in group else "lr" for group in params]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        labels = ["lr"]
    if cfg.train.backbone_lr is not None and "lr_backbone" not in labels:
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

    Call this before the epoch trains and before ``scheduler.step()``: what is
    logged must be the rate the epoch actually ran at.  The labels matter as much
    as the values — ``param_groups[0]`` is the image backbone whenever
    ``backbone_lr`` is set, not the rate driving the trunk and heads.

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

    fp16 requires CUDA and a GradScaler; bf16 autocasts on the run's own device,
    with no scaler.  Returns ``(autocast_device, autocast_dtype, scaler)``; when
    AMP is off, ``autocast_device`` is ``None``.

    Autocast is per device type, so the run's device is passed through rather than
    folded into ``"cpu"``: a CPU autocast context leaves every op on another
    device in fp32, i.e. AMP asked for, recorded in the checkpoint's config, and
    never applied.  A device torch has no autocast for is refused as loudly as
    fp16 off CUDA.
    """
    if not amp_enabled:
        return None, None, None
    if dtype == "fp16":
        if device != "cuda":
            raise TrainingError("amp dtype 'fp16' requires a CUDA device; use 'bf16' on CPU")
        return "cuda", torch.float16, torch.amp.GradScaler("cuda")
    if not torch.amp.is_autocast_available(device):
        raise TrainingError(
            f"amp has no autocast for device {device!r}; set train.amp.enabled=false "
            "or run on a device torch can autocast"
        )
    return device, torch.bfloat16, None


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
    target_stats: tuple[float, float, float, float],
    component_weights: Mapping[str, float],
) -> dict[str, float]:
    """Run one validation epoch (no grad); return loss components + physical metrics."""
    model.eval()
    accumulator = _MetricAccumulator(target_stats, component_weights)
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with _autocast(autocast_device, autocast_dtype):
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


def _present_target_count(batch: Mapping[str, Tensor], column: str) -> Tensor:
    """Rows of *column* a loss component counted: finite value, or class ``>= 0``.

    A zero-dim ``int64`` tensor on the batch's device, so the count folds into the
    epoch without reading anything back to the host.
    """
    target = batch[column]
    mask = target >= 0 if column == "sky_class" else torch.isfinite(target)
    return mask.sum()


def _accumulation_dtype(sample: Tensor) -> torch.dtype:
    """The dtype an epoch's running sums use for tensors living where *sample* does.

    float64, so summing one term per batch over a whole epoch does not drift the
    reported metrics; mps has no float64, so there the sums stay float32.
    """
    return torch.float32 if sample.device.type == "mps" else torch.float64


def _fold(running: Tensor | None, term: Tensor) -> Tensor:
    """Add *term* to a running total, seeded on *term*'s device by the first batch."""
    return term if running is None else running + term


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

    Every running total is a zero-dim tensor on the batch's own device: folding a
    batch in must not read a scalar back to the host, so the whole epoch costs the
    handful of device syncs :meth:`result` does at the end.
    """

    def __init__(
        self,
        target_stats: tuple[float, float, float, float],
        component_weights: Mapping[str, float],
    ) -> None:
        self._dhi_mean, self._dhi_std, self._kindex_mean, self._kindex_std = target_stats
        self._component_weights = dict(component_weights)
        self._component_sums: dict[str, Tensor] = {}
        self._component_counts: dict[str, Tensor] = {}
        self._physical_sums: dict[str, Tensor] = {}
        self._physical_counts: dict[str, Tensor] = {}

    def update(
        self, outputs: Mapping[str, Tensor], batch: dict[str, Tensor], losses: Mapping[str, Tensor]
    ) -> None:
        """Fold one batch's outputs/targets/losses into the running sums.

        A loss component with no entry in :data:`_COMPONENT_TARGET_COLUMNS` — an
        unrecognised, later-added head — falls back to being weighted by the
        batch row count.

        A head with no labelled row in this batch is dropped from both the sum and
        the count, whatever its loss value: that loss is a masked ``(pred * 0)``
        reduction, so an overflowed activation makes it NaN, and weighting it by a
        zero count would carry that NaN into every later batch of the head.  A
        non-finite loss on a head that *does* have rows is a real divergence and
        still propagates.
        """
        size = int(batch["features"].shape[0])
        for key, value in losses.items():
            if key == "loss":
                continue
            column = _COMPONENT_TARGET_COLUMNS.get(key)
            count = (
                _present_target_count(batch, column)
                if column is not None
                else torch.full((), size, dtype=torch.long, device=value.device)
            )
            scaled = value.detach().to(_accumulation_dtype(value)) * count
            # Dropped with ``where`` rather than by a Python ``if count == 0``:
            # branching on the count would read it back to the host, one device
            # sync per head per batch.
            weighted = torch.where(count > 0, scaled, torch.zeros_like(scaled))
            self._component_sums[key] = _fold(self._component_sums.get(key), weighted)
            self._component_counts[key] = _fold(self._component_counts.get(key), count)
        if "dhi" in outputs:
            self._fold_physical(
                "dhi_mae",
                *_mae_terms(outputs["dhi"], batch["dhi"], self._dhi_mean, self._dhi_std),
            )
        if "kindex" in outputs:
            self._fold_physical(
                "kindex_mae",
                *_mae_terms(
                    outputs["kindex"], batch["kindex"], self._kindex_mean, self._kindex_std
                ),
            )
        if "sky_logits" in outputs:
            predicted = outputs["sky_logits"].detach().argmax(dim=-1)
            mask = batch["sky_class"] >= 0
            hits = (predicted == batch["sky_class"]) & mask
            self._fold_physical("sky_acc", hits.sum(), mask.sum())

    def _fold_physical(self, key: str, summed: Tensor, count: Tensor) -> None:
        """Fold one batch's ``(sum, row count)`` for the physical-unit metric *key*."""
        self._physical_sums[key] = _fold(self._physical_sums.get(key), summed)
        self._physical_counts[key] = _fold(self._physical_counts.get(key), count)

    def result(self) -> dict[str, float]:
        """Finalize the epoch: per-head masked means and their weighted total."""
        components: dict[str, float] = {}
        total = 0.0
        heads_without_labels: list[str] = []
        for key, summed in self._component_sums.items():
            count = int(self._component_counts[key])
            if count == 0:
                components[key] = 0.0
                heads_without_labels.append(key)
                continue
            components[key] = float(summed) / count
            total += self._component_weights.get(key, 1.0) * components[key]
        if heads_without_labels:
            logger.warning(
                "epoch metrics: %s had no valid target row in the whole split; reported as 0.0 "
                "and excluded from the total loss — check the head is meant to be enabled",
                ", ".join(sorted(heads_without_labels)),
            )
        metrics: dict[str, float] = {"loss": total, **components}
        for key, summed in self._physical_sums.items():
            count = int(self._physical_counts[key])
            if count:
                metrics[key] = float(summed) / count
        return metrics


def _mae_terms(pred: Tensor, target: Tensor, mean: float, std: float) -> tuple[Tensor, Tensor]:
    """One batch's physical-unit absolute-error sum and its finite-target row count.

    Both are zero-dim tensors on *pred*'s device: the sum of
    ``|pred * std + mean - target|`` over the rows whose target is finite (in
    :func:`_accumulation_dtype`), and how many rows those were (``int64``).
    """
    physical = pred.detach().float() * std + mean
    truth = target.detach().float()
    mask = torch.isfinite(truth)
    # Masked with ``where`` rather than ``physical[mask]``: boolean-mask indexing
    # lowers to nonzero/masked_select, whose output shape is data-dependent, so it
    # synchronizes the device to the host once per head per batch. Zeroing the
    # excluded rows keeps the shape static and the sum identical.
    error = torch.where(mask, (physical - truth).abs(), torch.zeros_like(truth))
    return error.to(_accumulation_dtype(error)).sum(), mask.sum()


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
    cfg: Any,
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

    The alignment fields are checked alongside the dataset ones because they
    decide what every sample's embedding IS — the pooling window's width, and
    centre-frame against mean — while leaving no architectural trace: the
    attention pooler's ``(1, 1, D)`` query is sequence-length independent and
    centre-frame shares the mean pooler, so ``load_state_dict`` would accept the
    old weights and the run would continue on differently-pooled inputs, with
    ``best.ckpt`` selected across two regimes.
    """
    stored_cfg = checkpoint.get("config") or {}
    stored_data = stored_cfg.get("data") or {}
    stored_alignment = stored_data.get("alignment") or {}

    mismatches: list[str] = []
    for field, stored, current in (
        ("split_id", checkpoint.get("split_id"), split_id),
        ("manifest_sha256", checkpoint.get("manifest_sha256"), meta.get("manifest_sha256")),
        ("dataset_version", checkpoint.get("dataset_version"), _dataset_version(meta)),
        ("input_mode", stored_data.get("input_mode"), cfg.data.input_mode),
        ("alignment.strategy", stored_alignment.get("strategy"), cfg.data.alignment.strategy),
        (
            "alignment.window_minutes",
            stored_alignment.get("window_minutes"),
            cfg.data.alignment.window_minutes,
        ),
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
        raise TrainingError(
            f"refusing to resume from {path}: it was trained against a different dataset "
            + "; ".join(mismatches)
            + ". Run without --resume (or point --out-dir at a fresh run directory) to "
            "train from scratch on the current dataset."
        )


_OPTIMIZER_KNOB_REL_TOL = 1e-12


def _knob_differs(stored: float | None, current: float | None) -> bool:
    """True when an optimizer knob was edited between the two invocations."""
    if stored is None or current is None:
        return (stored is None) != (current is None)
    return not math.isclose(float(stored), float(current), rel_tol=_OPTIMIZER_KNOB_REL_TOL)


def _warn_optimizer_knob_drift(checkpoint: Mapping[str, Any], cfg: ExperimentConfig) -> None:
    """Warn when an edited optimizer knob is about to lose to the restored state.

    ``optimizer.load_state_dict`` rebuilds every parameter group from the stored
    ``lr`` / ``weight_decay``, which is what continuing a run means — a plateau
    reduction and a mid-cosine rate both live there, so re-seeding from the config
    would undo them.  The edit is still discarded, though, and
    :func:`_checkpoint_common` writes the *edited* config into the new checkpoint,
    so the drift is named here instead of being absorbed.  A knob the checkpoint
    does not record is skipped, mirroring :func:`_check_resume_provenance`.
    """
    stored_train = (checkpoint.get("config") or {}).get("train") or {}
    for field, current in (
        ("lr", cfg.train.lr),
        ("weight_decay", cfg.train.weight_decay),
        ("backbone_lr", cfg.train.backbone_lr),
    ):
        if field not in stored_train:
            continue
        stored = stored_train[field]
        if _knob_differs(stored, current):
            logger.warning(
                "resume: train.%s=%s differs from the checkpoint's %s; the restored "
                "optimizer state wins, so the resumed epochs train at %s while the new "
                "checkpoint records %s",
                field,
                current,
                stored,
                stored,
                current,
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

    Mirrors :func:`_reset_stale_run_artifacts`: once the stored best is discarded the
    first resumed epoch unconditionally improves on ``None`` and rewrites
    ``best.ckpt``, which would destroy the previous monitor's best weights with no
    record of them.

    The destination is ``best.ckpt.stale-monitor``, distinct from the
    ``best.ckpt.stale`` a fresh run into the same directory writes: sharing one
    name would let that fresh run replace the only surviving copy of the previous
    monitor's weights, the one artifact in the run directory nothing can recompute.
    A second monitor change gets its own numbered destination for the same reason:
    every monitor a run directory has been through keeps its own best.
    """
    path = run_dir / BEST_CHECKPOINT
    if not path.exists():
        return
    backup = _free_rotation_destination(path.with_name(f"{BEST_CHECKPOINT}{MONITOR_CHANGE_SUFFIX}"))
    os.replace(path, backup)
    logger.warning(
        "resume: rotated %s aside to %s (it was selected under the previous monitor)",
        path,
        backup.name,
    )


def _free_rotation_destination(preferred: Path) -> Path:
    """*preferred* if free, else the first ``<preferred>.<n>`` (n from 2) that is.

    A rotation destination is never overwritten: the weights it holds were selected
    under a monitor no later run recomputes.
    """
    destination = preferred
    ordinal = 2
    while destination.exists():
        destination = preferred.with_name(f"{preferred.name}.{ordinal}")
        ordinal += 1
    return destination


def _rotate_superseded_best(run_dir: Path) -> None:
    """Move the previous run's ``best.ckpt`` aside now that a fresh one replaces it.

    Deferred from :func:`_reset_stale_run_artifacts` to the first epoch that
    actually improves: rotating at the start of a fresh run leaves the directory
    with no ``best.ckpt`` at all for a run that then dies (an unresolvable monitor,
    a divergent first epoch), which is when the previous best matters most.

    The destination is taken through :func:`_free_rotation_destination` for the
    same reason a monitor change is: two fresh runs in one directory would
    otherwise rotate onto the same name, and the second would delete the weights
    the first preserved.
    """
    path = run_dir / BEST_CHECKPOINT
    backup = _free_rotation_destination(path.with_name(f"{BEST_CHECKPOINT}{STALE_RUN_SUFFIX}"))
    os.replace(path, backup)
    logger.warning(
        "fresh run: rotated stale %s aside to %s (a previous run wrote it)", path, backup.name
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
    elif cfg.data.input_mode == "embedding":
        backbone_info = _embedding_recipe(cfg)
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


def _embedding_recipe(cfg: ExperimentConfig) -> dict[str, Any] | None:
    """The encoding recipe of the store this embedding-mode run reads, if it can be read.

    Embedding-mode training never builds a backbone, so without this the
    checkpoint records no encoder identity at all and predicting a live frame
    from it needs the training machine's store still mounted at the path baked
    into ``data.data_root``.  Copying the recipe in at save time makes the
    checkpoint portable; None when the store cannot be read, which leaves
    prediction to fall back on the store exactly as before rather than on a
    guess.
    """
    from allsky.data.loading import resolve_against_root
    from allsky.snapshot import embedding_recipe_of

    if cfg.data.embeddings_dir is None:
        return None
    store = resolve_against_root(cfg.data.embeddings_dir, Path(cfg.data.data_root))
    return embedding_recipe_of(store)


def _dataset_version(meta: Mapping[str, Any]) -> str:
    """The dataset version a checkpoint records for *meta* (the code's when absent)."""
    from allsky.data.contracts import DATASET_VERSION

    return str(meta.get("dataset_version", DATASET_VERSION))


def _monitor_key(monitor: str) -> str:
    """Normalize an early-stopping monitor string to a val-metric key."""
    for prefix in ("val/", "val_"):
        if monitor.startswith(prefix):
            return monitor[len(prefix) :]
    return monitor


def _improved(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    """True when *current* improves on *best* by more than *min_delta*.

    A non-finite *current* never improves: taken as the best it would pin
    ``best.ckpt`` to the diverged epoch and, since every later comparison against
    a NaN best is False, silently starve the patience counter until the run stops
    looking converged.  A non-finite *best* — restored from a checkpoint written
    before this guard — is treated as no baseline at all, so the first finite
    epoch reclaims ``best.ckpt``.
    """
    if not math.isfinite(current):
        return False
    if best is None or not math.isfinite(best):
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


def _reset_stale_run_artifacts(run_dir: Path) -> None:
    """Rotate a previous run's metrics and checkpoints aside on a fresh run.

    A fresh (non-resume) run into a reused run directory must not append to the
    previous run's metrics.  Each stale file is renamed to
    ``<name>.stale`` (replacing an older backup) rather than deleted, so the prior
    run's numbers are still recoverable; the fresh run then re-creates the files
    from scratch.

    ``last.ckpt`` is rotated with them: it is overwritten at the end of epoch 1,
    which would leave the preserved metrics describing weights that no longer
    exist anywhere.  ``best.ckpt`` is *not* rotated here —
    :func:`_rotate_superseded_best` does it at the first epoch that improves, so a
    fresh run that dies before producing a replacement leaves the previous best
    where it is instead of emptying the directory.

    Only these three names are rotated, and only onto their own ``.stale``
    destination: the ``best.ckpt.stale-monitor`` :func:`_rotate_stale_best` wrote
    under an earlier monitor is left untouched.
    """
    for name in ("metrics.csv", "metrics.json", LAST_CHECKPOINT):
        path = run_dir / name
        if path.exists():
            backup = path.with_name(f"{name}{STALE_RUN_SUFFIX}")
            os.replace(path, backup)
            logger.warning(
                "fresh run: rotated stale %s aside to %s (a previous run wrote it)",
                path,
                backup.name,
            )
