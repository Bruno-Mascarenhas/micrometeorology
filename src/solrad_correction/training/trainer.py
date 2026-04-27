"""High-level Trainer for PyTorch models."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solrad_correction.training.callbacks import EarlyStopping
from solrad_correction.training.dataloaders import DataLoaderSettings, resolve_dataloader_settings
from solrad_correction.training.loops import evaluate_epoch, train_one_epoch
from solrad_correction.training.progress import TrainingProgress

if TYPE_CHECKING:
    from solrad_correction.config import ModelConfig, RuntimeConfig
    from solrad_correction.datasets.sequence import SequenceDataset, WindowedSequenceDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the PyTorch training loop.

    Features:
    - Progress display with batch % and ETA
    - Early stopping
    - Best-model checkpointing
    - Transfer learning (resume from start_epoch)
    - Device management (CPU/CUDA auto-detection)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        config: ModelConfig | None = None,
        runtime: RuntimeConfig | None = None,
        start_epoch: int = 0,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        scaler_state: dict[str, Any] | None = None,
        checkpoint_config: dict[str, Any] | None = None,
    ) -> None:
        self.model: nn.Module = model  # type: ignore
        self.device = device
        self.config = config
        self.runtime = runtime
        self.start_epoch = start_epoch
        self._resume_optimizer_state = optimizer_state
        self._resume_scheduler_state = scheduler_state
        self._resume_scaler_state = scaler_state
        self._checkpoint_config = checkpoint_config

        # Defaults from config
        self.lr = config.learning_rate if config else 1e-3
        self.weight_decay = config.weight_decay if config else 1e-5
        self.max_epochs = config.max_epochs if config else 100
        self.batch_size = config.batch_size if config else 32
        self.patience = config.patience if config else 10
        self.min_delta = config.min_delta if config else 1e-4
        self.completed_epochs = start_epoch
        self.optimizer_state: dict[str, Any] | None = None
        self.scheduler_state: dict[str, Any] | None = None
        self.scaler_state: dict[str, Any] | None = None
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self.dataloader_settings: DataLoaderSettings | None = None

    def train(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
    ) -> tuple[nn.Module, dict]:
        """Run the full training loop.

        Returns ``(trained_model, history)`` where history contains
        per-epoch losses.
        """
        settings = (
            resolve_dataloader_settings(self.runtime)
            if self.runtime is not None
            else DataLoaderSettings(
                device=self.device,
                num_workers=0,
                pin_memory=self.device != "cpu",
                persistent_workers=False,
                prefetch_factor=None,
                amp="cuda" in self.device,
                torch_compile=False,
                gradient_clip=1.0,
            )
        )
        self.dataloader_settings = settings
        self.device = settings.device
        self.model.to(self.device)

        # Optimize with torch.compile if supported (PyTorch 2.0+)
        if settings.torch_compile and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore
                logger.info("Successfully applied torch.compile to the model")
            except Exception as e:
                logger.debug("torch.compile not supported or failed: %s", e)

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory,
            persistent_workers=settings.persistent_workers,
            prefetch_factor=settings.prefetch_factor,
        )

        val_loader = None
        if val_data:
            val_loader = DataLoader(
                val_data,
                batch_size=self.batch_size,
                num_workers=settings.num_workers,
                pin_memory=settings.pin_memory,
                persistent_workers=settings.persistent_workers,
                prefetch_factor=settings.prefetch_factor,
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Automatic Mixed Precision (AMP)
        use_amp = settings.amp
        scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

        if self._resume_optimizer_state is not None:
            try:
                optimizer.load_state_dict(self._resume_optimizer_state)
                logger.info("Restored optimizer state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible optimizer state: %s", exc)

        if self._resume_scheduler_state is not None:
            try:
                scheduler.load_state_dict(self._resume_scheduler_state)
                logger.info("Restored scheduler state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible scheduler state: %s", exc)

        if scaler is not None and self._resume_scaler_state is not None:
            try:
                scaler.load_state_dict(self._resume_scaler_state)
                logger.info("Restored AMP scaler state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible AMP scaler state: %s", exc)

        # TensorBoard Tracking
        writer = None
        if self.config and self.config.log_dir:
            writer = SummaryWriter(log_dir=self.config.log_dir)
            logger.info("TensorBoard tracking enabled at %s", self.config.log_dir)

        progress = TrainingProgress(
            total_epochs=self.max_epochs,
            start_epoch=self.start_epoch,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # In-Memory Best Checkpointing
        best_loss = float("inf")
        best_state_dict = copy.deepcopy(self.model.state_dict())
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
        checkpoint_dir = (
            Path(self.runtime.checkpoint_dir)
            if self.runtime is not None and self.runtime.checkpoint_dir
            else None
        )
        checkpoint_every = (
            self.runtime.checkpoint_every
            if self.runtime is not None and self.runtime.checkpoint_every is not None
            else 1
        )
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        total_epochs = self.start_epoch + self.max_epochs
        for epoch in range(self.start_epoch, total_epochs):
            self.completed_epochs = epoch + 1
            progress.start_epoch(epoch)

            # Train
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                optimizer,
                criterion,
                self.device,
                scaler=scaler,
                clip_val=settings.gradient_clip,
                progress_callback=progress.update_batch,
            )
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = evaluate_epoch(
                    self.model,
                    val_loader,
                    criterion,
                    self.device,
                    amp_enabled=use_amp,
                )
                history["val_loss"].append(val_loss)

            # TensorBoard logging
            if writer:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                if val_loss is not None:
                    writer.add_scalar("Loss/Validation", val_loss, epoch)
                    writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

            monitor = val_loss if val_loss is not None else train_loss

            # LR Scheduler step
            scheduler.step(monitor)

            # In-Memory Checkpoint
            if monitor < best_loss:
                best_loss = monitor
                self.best_metric = best_loss
                self.best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(self.model.state_dict())
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(scheduler.state_dict())
                best_scaler_state = (
                    copy.deepcopy(scaler.state_dict()) if scaler is not None else None
                )
                if checkpoint_dir is not None:
                    self._save_checkpoint(
                        checkpoint_dir / "best.pt",
                        epoch=epoch + 1,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        metric=monitor,
                        kind="best",
                    )

            if checkpoint_dir is not None and (epoch + 1) % checkpoint_every == 0:
                self._save_checkpoint(
                    checkpoint_dir / "last.pt",
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metric=monitor,
                    kind="last",
                )

            extra = ""
            # Early stopping
            if early_stop(monitor):
                extra = " [EARLY STOP]"
                progress.end_epoch(train_loss, val_loss, extra)
                break

            progress.end_epoch(train_loss, val_loss, extra)

        progress.finish()
        if writer:
            writer.close()

        # Restore best weights before returning
        logger.info("Restoring best model weights (loss=%.6f)", best_loss)
        self.model.load_state_dict(best_state_dict)
        optimizer.load_state_dict(best_optimizer_state)
        scheduler.load_state_dict(best_scheduler_state)
        if scaler is not None and best_scaler_state is not None:
            scaler.load_state_dict(best_scaler_state)

        self.optimizer_state = optimizer.state_dict()
        self.scheduler_state = scheduler.state_dict()
        self.scaler_state = scaler.state_dict() if scaler is not None else None

        return self.model, history

    def _save_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        scaler: torch.cuda.amp.GradScaler | None,
        metric: float,
        kind: str,
    ) -> None:
        from solrad_correction.utils.serialization import save_torch_checkpoint

        save_torch_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=self._checkpoint_config,
            epoch=epoch,
            path=path,
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if scaler is not None else None,
            metadata={
                "checkpoint_kind": kind,
                "monitor_metric": metric,
                "dataloader": self.dataloader_settings.to_dict()
                if self.dataloader_settings is not None
                else {},
            },
        )
