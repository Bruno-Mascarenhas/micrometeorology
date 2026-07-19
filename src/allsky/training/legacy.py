"""Training loop for :class:`allsky.models.SkyFusionNet`.

Design notes
------------
- torch, tqdm, tensorboard, and ``allsky.dataset`` are imported lazily inside
  functions so this module (notably the pure :func:`split_days`) stays
  importable in torch-free environments.
- Train/validation are split by calendar DAY, never by row: consecutive frames
  of the same day are near-duplicates, so a row-level split would leak
  validation information into training. :func:`split_days` enforces this and
  :func:`train` re-checks it.
- Diffuse targets may be Erbs pseudo-targets (``target_source="erbs_pseudo"``
  in the index) — regression metrics then measure agreement with the Erbs
  decomposition, not with a real shaded pyranometer.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from allsky.config import AllSkyConfig
from solrad_correction.utils.metadata import collect_run_metadata
from solrad_correction.utils.seeds import set_global_seed

logger = logging.getLogger(__name__)

#: torch types at runtime. Kept as aliases so importing this module — and thus
#: ``allsky.training`` — stays torch-free (torch is imported lazily in the funcs).
type TorchModule = Any
type TorchOptimizer = Any
type TorchDataLoader = Any
type TorchDataset = Any


def split_days(
    index_df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    timestamp_column: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an index into train/val by calendar day (leakage guard).

    Whole days are assigned to either split, so no two frames from the same
    day can end up on opposite sides. The day list is shuffled with
    ``numpy.random.default_rng(seed)``; validation receives
    ``round(n_days * val_fraction)`` days (at least 1, at most ``n_days - 1``).

    Raises ``ValueError`` if ``val_fraction`` is outside ``(0, 1)`` or the
    index spans fewer than two distinct days.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    timestamps = pd.to_datetime(index_df[timestamp_column])
    day = timestamps.dt.normalize()
    unique_days = day.drop_duplicates().sort_values().to_numpy()
    if len(unique_days) < 2:
        raise ValueError(f"day-based split needs at least 2 distinct days, got {len(unique_days)}")
    shuffled = np.random.default_rng(seed).permutation(unique_days)
    n_val = min(max(1, round(len(unique_days) * val_fraction)), len(unique_days) - 1)
    val_mask = day.isin(shuffled[:n_val])
    train_df = index_df.loc[~val_mask]
    val_df = index_df.loc[val_mask]
    overlap = np.intersect1d(day[~val_mask].unique(), day[val_mask].unique())
    if overlap.size:  # pragma: no cover - impossible by construction
        raise RuntimeError(f"day-split leakage guard failed: shared days {overlap!r}")
    return train_df, val_df


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``"auto"`` to the best available device: cuda -> mps -> cpu."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _build_loaders(
    cfg: AllSkyConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    device: str,
) -> tuple[TorchDataLoader, TorchDataLoader]:
    """Build train/val DataLoaders from index slices.

    Narrow protocol with :mod:`allsky.dataset` (imported lazily, so this
    module works without torch until training actually starts):
    ``AllSkyDataset(index_df, model_cfg, train=bool, stats=...)`` yielding the
    batch dict documented in :mod:`allsky.models`. Standardization statistics
    come from the TRAIN split only and are handed to the validation dataset
    (``stats=train_ds.stats``) so no validation data leaks into them.
    """
    from torch.utils.data import DataLoader

    from allsky.dataset import AllSkyDataset

    # Pin feature columns to the config so the dataset feature vector always
    # matches the model input size (train() builds SkyFusionNet from the same
    # cfg.sensor.feature_columns).
    feature_columns = list(cfg.sensor.feature_columns)
    train_ds = AllSkyDataset(train_df, cfg.model, train=True, feature_columns=feature_columns)
    val_ds = AllSkyDataset(
        val_df, cfg.model, train=False, feature_columns=feature_columns, stats=train_ds.stats
    )
    common: dict[str, Any] = {
        "batch_size": cfg.train.batch_size,
        "num_workers": cfg.train.num_workers,
        "pin_memory": device == "cuda",
        # Keep workers alive across epochs — worker re-fork dominates epoch
        # startup on Colab when the dataset is image-heavy.
        "persistent_workers": cfg.train.num_workers > 0,
    }
    return (
        DataLoader(cast("TorchDataset", train_ds), shuffle=True, **common),
        DataLoader(cast("TorchDataset", val_ds), shuffle=False, **common),
    )


def _run_epoch(
    model: TorchModule,
    loader: TorchDataLoader,
    device: str,
    *,
    w_cls: float,
    w_reg: float,
    optimizer: TorchOptimizer | None = None,
    desc: str = "train",
    amp: bool = False,
    scaler: Any | None = None,
) -> dict[str, float]:
    """Run one epoch; trains when *optimizer* is given, evaluates otherwise.

    With ``amp`` (CUDA only) the forward/loss run under ``torch.autocast``
    and gradients are scaled — roughly 2x throughput on Colab T4/L4 GPUs.
    MAE/RMSE are reported in raw W/m2 (model output scale) — only the loss
    uses the /100 normalization documented in :func:`allsky.models.multitask_loss`.
    """
    import torch
    from tqdm.auto import tqdm

    from allsky.models import multitask_loss

    training = optimizer is not None
    model.train(training)
    n = 0
    correct = 0
    loss_sum = 0.0
    abs_err_sum = 0.0
    sq_err_sum = 0.0
    with torch.set_grad_enabled(training):
        for raw_batch in tqdm(loader, desc=desc, leave=False):
            batch = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in raw_batch.items()
            }
            with torch.autocast(device_type="cuda", enabled=amp):
                outputs = model(batch["image"], batch["features"])
                losses = multitask_loss(outputs, batch, w_cls=w_cls, w_reg=w_reg)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(losses["loss"]).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    optimizer.step()
            batch_size = int(batch["cloud_class"].shape[0])
            n += batch_size
            loss_sum += float(losses["loss"].detach()) * batch_size
            predicted = outputs["logits"].detach().argmax(dim=1)
            correct += int((predicted == batch["cloud_class"]).sum())
            err = (outputs["diffuse"].detach() - batch["diffuse"]).float()
            abs_err_sum += float(err.abs().sum())
            sq_err_sum += float((err**2).sum())
    if n == 0:
        raise ValueError(f"empty DataLoader in {desc!r} epoch — check the index split")
    return {
        "loss": loss_sum / n,
        "accuracy": correct / n,
        "mae_wm2": abs_err_sum / n,
        "rmse_wm2": math.sqrt(sq_err_sum / n),
    }


def train(
    cfg: AllSkyConfig,
    *,
    index_path: str | Path | None = None,
    resume: str | Path | None = None,
    val_fraction: float = 0.2,
) -> dict[str, float]:
    """Train SkyFusionNet from a built index parquet; return best val metrics.

    Writes into ``cfg.train.out_dir``: ``last.pt`` (every epoch, resumable),
    ``best.pt`` (lowest validation loss), ``config.json``, ``metadata.json``
    (git commit, package versions, timing), and TensorBoard events under
    ``runs/``. Pass ``resume=<last.pt>`` to continue an interrupted run.
    """
    started = time.monotonic()
    set_global_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info("Device: %s (requested %r)", device, cfg.train.device)

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = Path(index_path) if index_path is not None else out_dir / "index.parquet"
    index_df = pd.read_parquet(index_file)
    logger.info("Index: %d rows from %s", len(index_df), index_file)

    # Pre-check the day count instead of catching split_days' ValueError: only a
    # genuinely single-day index falls back to train==val (finding F9); every
    # other error — notably a bad val_fraction outside (0, 1) — propagates.
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    n_days = pd.to_datetime(index_df["timestamp"]).dt.normalize().nunique()
    if n_days < 2:
        logger.warning(
            "Index spans a single calendar day: validation REUSES the training day. "
            "Metrics are not leakage-free — smoke/debug runs only."
        )
        train_df = val_df = index_df
    else:
        train_df, val_df = split_days(index_df, val_fraction=val_fraction, seed=cfg.train.seed)
    logger.info("Split: %d train rows / %d val rows (by day)", len(train_df), len(val_df))
    train_loader, val_loader = _build_loaders(cfg, train_df, val_df, device)

    import torch
    from torch.utils.tensorboard import SummaryWriter

    from allsky.models import SkyFusionNet

    amp = bool(cfg.train.amp) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp else None
    if device == "cuda":
        # Fixed input sizes: let cuDNN pick the fastest kernels once.
        torch.backends.cudnn.benchmark = True
    logger.info("AMP mixed precision: %s", "on" if amp else "off")

    model = SkyFusionNet(cfg.model, n_features=len(cfg.sensor.feature_columns)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    start_epoch = 0
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        logger.info("Resumed from %s at epoch %d", resume, start_epoch)

    (out_dir / "config.json").write_text(
        json.dumps(cfg.model_dump(), indent=2, default=str), encoding="utf-8"
    )
    writer = SummaryWriter(log_dir=str(out_dir / "runs"))
    w_cls, w_reg = cfg.train.cls_loss_weight, cfg.train.reg_loss_weight
    best_val_loss = math.inf
    best_metrics: dict[str, float] = {}
    try:
        for epoch in range(start_epoch, cfg.train.epochs):
            train_metrics = _run_epoch(
                model,
                train_loader,
                device,
                w_cls=w_cls,
                w_reg=w_reg,
                optimizer=optimizer,
                desc=f"epoch {epoch + 1}/{cfg.train.epochs}",
                amp=amp,
                scaler=scaler,
            )
            val_metrics = _run_epoch(
                model, val_loader, device, w_cls=w_cls, w_reg=w_reg, desc="val", amp=amp
            )
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            logger.info(
                "epoch %d/%d — train loss %.4f | val loss %.4f acc %.3f "
                "MAE %.1f W/m2 RMSE %.1f W/m2",
                epoch + 1,
                cfg.train.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["mae_wm2"],
                val_metrics["rmse_wm2"],
            )
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_metrics": val_metrics,
                "config": cfg.model_dump(),
            }
            torch.save(checkpoint, out_dir / "last.pt")
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_metrics = {**val_metrics, "epoch": float(epoch + 1)}
                torch.save(checkpoint, out_dir / "best.pt")
    finally:
        writer.close()

    metadata = collect_run_metadata(
        config=cfg,
        model=model,
        started_at=started,
        training_duration_seconds=time.monotonic() - started,
    )
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Best validation metrics: %s", best_metrics or "<no epochs ran>")
    return best_metrics
