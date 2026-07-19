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
   ``metrics.csv`` (appended) and ``metrics.json`` (atomically rewritten);
#. checkpoint ``last.ckpt`` every epoch and ``best.ckpt`` on monitor improvement
   (resume-safe best seeding), with early stopping;
#. resume fully from ``last.ckpt`` (``resume="auto"`` or a path), restoring
   model / optimizer / scheduler / scaler / epoch / global_step / best / RNG.

The engine imports torch at module scope and is therefore only ever imported
lazily (from the CLI or via :func:`allsky.training.__getattr__`), so
``import allsky`` / ``import allsky.cli`` stay torch-free.
"""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

from allsky.features.policy import active_feature_groups, resolve_feature_set
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

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor, nn

    from allsky.config import ExperimentConfig, SchedulerConfig
    from allsky.data.datasets import EmbeddingReader, WindowMode
    from allsky.features.normalization import TargetNormalizer

logger = logging.getLogger(__name__)

__all__ = ["resolve_run_device", "run_experiment"]


def resolve_run_device(requested: str) -> str:
    """Resolve *requested* to a concrete device, erroring on unavailable cuda.

    Delegates ``"auto"`` resolution to
    :func:`allsky.training.legacy.resolve_device`, then raises a clear
    :class:`RuntimeError` when ``"cuda"`` is asked for but no CUDA device is
    available (rather than failing opaquely deep inside the first ``.to("cuda")``).
    """
    from allsky.training.legacy import resolve_device

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
    manifest_path = _resolve(cfg.data.manifest, root)
    manifest, meta = _load_manifest(manifest_path)
    split = _load_split(_resolve(cfg.data.split_artifact, root))
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
    from allsky.modeling.baselines import ClimatologyModel
    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy

    image_backbone = None
    if cfg.data.input_mode == "image" and image_backbone_builder is not None:
        image_backbone = image_backbone_builder()
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

    is_climatology = isinstance(model, ClimatologyModel)
    if is_climatology:
        _fit_climatology(model, cfg, train_df, target_normalizers)

    optimizer = _build_optimizer(model, cfg)
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
        start_epoch, global_step, best_value, best_epoch, restored_no_improve = _restore(
            resume_path, resolved_device, model, optimizer, scheduler, scaler
        )
        # Restore the patience counter; a pre-field checkpoint yields a safe lower
        # bound (epochs since the recorded best), never a negative value.
        epochs_no_improve = (
            restored_no_improve
            if restored_no_improve is not None
            else (max(0, start_epoch - best_epoch) if best_epoch is not None else 0)
        )
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

    # --- epoch loop ---------------------------------------------------------
    from torch.utils.tensorboard import SummaryWriter

    dhi_mean, dhi_std = _stats_or_identity(target_normalizers, "dhi")
    kindex_mean, kindex_std = _stats_or_identity(target_normalizers, "kindex")
    writer = SummaryWriter(log_dir=str(run_dir / "runs"))
    epochs_ran = 0
    last_val_metrics: dict[str, float] = {}
    patience = int(cfg.train.early_stopping.patience)
    min_delta = float(cfg.train.early_stopping.min_delta)
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            # Deterministic, resume-stable batch order: order = f(seed, epoch),
            # independent of the resume point, persistent_workers, or global-RNG
            # draw count (finding 1).
            train_sampler_generator.manual_seed(cfg.seed * 100003 + epoch)
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
            )
            last_val_metrics = val_metrics

            monitor_value = val_metrics.get(monitor_key)
            if monitor_value is None:
                raise KeyError(
                    f"early-stopping monitor {cfg.train.early_stopping.monitor!r} "
                    f"resolves to {monitor_key!r}, absent from val metrics {sorted(val_metrics)}"
                )
            if scheduler is not None:
                scheduler.step(monitor_value) if scheduler_is_plateau else scheduler.step()

            improved = _improved(monitor_value, best_value, monitor_mode, min_delta)
            if improved:
                best_value, best_epoch, epochs_no_improve = monitor_value, epoch + 1, 0
            else:
                epochs_no_improve += 1

            current_lr = float(optimizer.param_groups[0]["lr"])
            _log_epoch(writer, epoch, current_lr, train_metrics, val_metrics)
            row = _epoch_row(fields, epoch + 1, current_lr, train_metrics, val_metrics)
            _append_csv(run_dir / "metrics.csv", fields, row)
            history.append(row)
            _atomic_write_json(run_dir / "metrics.json", history)

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


def _resolve(path: str | Path, root: Path) -> Path:
    """Resolve *path* against *root* unless it is already absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the manifest parquet and its ``<name>.meta.json`` sidecar (if any)."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest parquet not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    meta_path = manifest_path.with_name(manifest_path.name + ".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        logger.warning("no manifest meta sidecar at %s; provenance fields will be null", meta_path)
    return manifest, meta


def _load_split(path: Path) -> Any:
    """Load the day-split artifact from *path*."""
    from allsky.data.splits import load_split_artifact

    if not path.exists():
        raise FileNotFoundError(f"split artifact not found: {path}")
    return load_split_artifact(path)


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
        reader = embedding_reader if embedding_reader is not None else _default_reader(cfg, root)
        _validate_embedding_coverage(reader, train_df, val_df)
        # Wire the alignment strategy end to end: center_frame (default) keeps the
        # single-frame embedding; mean_embedding / attention_pooling resolve each
        # row's same-day co-frame window (dataset-level) and pool accordingly. The
        # dataset validates the mode against its supported WindowMode set.
        window = cast("WindowMode", cfg.data.alignment.strategy)
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


def _default_reader(cfg: ExperimentConfig, root: Path) -> EmbeddingReader:
    """Build the safetensors embedding reader from ``cfg.data.embeddings_dir``."""
    from allsky.embeddings.storage import SafetensorsEmbeddingReader

    if cfg.data.embeddings_dir is None:
        raise ValueError("input_mode='embedding' requires cfg.data.embeddings_dir")
    reader: EmbeddingReader = SafetensorsEmbeddingReader(_resolve(cfg.data.embeddings_dir, root))
    return reader


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


def _fit_climatology(
    model: nn.Module,
    cfg: ExperimentConfig,
    train_df: pd.DataFrame,
    target_normalizers: Mapping[str, TargetNormalizer],
) -> None:
    """Fit the constant-prediction climatology model from raw train targets."""
    model.fit_from_targets(  # type: ignore[operator]
        dhi=train_df["target_dhi"].to_numpy() if cfg.targets.dhi.enabled else None,
        kindex=train_df["target_kindex"].to_numpy() if cfg.targets.kindex.enabled else None,
        cloud_fraction=(
            train_df["cloud_fraction"].to_numpy() if cfg.targets.cloud_fraction.enabled else None
        ),
        sky_class=train_df["sky_class"].to_numpy() if cfg.targets.sky.enabled else None,
        target_normalizers=target_normalizers,
    )


def _build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> torch.optim.Optimizer:
    """AdamW over ``model.param_groups(backbone_lr)`` when available, else parameters."""
    param_groups_fn = getattr(model, "param_groups", None)
    if callable(param_groups_fn):
        params: Any = param_groups_fn(cfg.train.backbone_lr)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


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
) -> tuple[dict[str, float], int]:
    """Run one training epoch with grad accumulation/clipping; return metrics + step."""
    model.train()
    accumulator = _MetricAccumulator(target_stats)
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
) -> dict[str, float]:
    """Run one validation epoch (no grad); return loss components + physical metrics."""
    model.eval()
    accumulator = _MetricAccumulator(target_stats)
    with torch.no_grad():
        for raw in loader:
            batch = _move(raw, device)
            with _autocast(autocast_device if amp_enabled else None, autocast_dtype):
                outputs = model(batch)
                losses = loss_fn(outputs, batch)
            accumulator.update(outputs, batch, losses)
    return accumulator.result()


class _MetricAccumulator:
    """Accumulate loss components and physical-unit quick metrics over an epoch."""

    def __init__(self, target_stats: tuple[float, float, float, float]) -> None:
        self._dhi_mean, self._dhi_std, self._kindex_mean, self._kindex_std = target_stats
        self._n = 0
        self._loss_sum = 0.0
        self._component_sums: dict[str, float] = {}
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
        self._n += size
        self._loss_sum += float(losses["loss"].detach()) * size
        for key, value in losses.items():
            if key == "loss":
                continue
            self._component_sums[key] = (
                self._component_sums.get(key, 0.0) + float(value.detach()) * size
            )
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
        """Finalize the sample-weighted averages for the epoch."""
        denom = max(self._n, 1)
        metrics: dict[str, float] = {"loss": self._loss_sum / denom}
        for key, value in self._component_sums.items():
            metrics[key] = value / denom
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


def _restore(
    path: Path,
    device: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
) -> tuple[int, int, float | None, int | None, int | None]:
    """Restore all training state from *path*.

    Returns ``(epoch, global_step, best_value, best_epoch, epochs_no_improve)``;
    ``epochs_no_improve`` is ``None`` on pre-field checkpoints (the caller then
    reconstructs a lower bound from ``epoch``/``best_epoch``).
    """
    checkpoint = load_checkpoint(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state") is not None:
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
    from allsky.data.contracts import DATASET_VERSION

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
        "dataset_version": str(meta.get("dataset_version", DATASET_VERSION)),
        "split_id": split_id,
        "manifest_sha256": meta.get("manifest_sha256"),
        "backbone_info": backbone_info,
        "code_version_info": code_version(),
    }


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
    """Stable, config-derived CSV column order (identical across resumes)."""
    fields = ["epoch", "lr"]
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
    lr: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> dict[str, Any]:
    """Build a metrics row keyed by the canonical *fields* (missing -> empty)."""
    row: dict[str, Any] = dict.fromkeys(fields, "")
    row["epoch"] = epoch
    row["lr"] = lr
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
    lr: float,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
) -> None:
    """Write per-epoch TensorBoard scalars."""
    writer.add_scalar("lr", lr, epoch)
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
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


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
    _atomic_write_json(metrics_json, history)
    return history


def _atomic_write_json(path: Path, obj: Any) -> None:
    """Atomically (temp + ``os.replace``) rewrite *path* as JSON."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, default=str)
    os.replace(tmp, path)
