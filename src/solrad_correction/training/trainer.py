"""High-level Trainer for PyTorch models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from solrad_correction.training.callbacks import EarlyStopping
from solrad_correction.training.checkpoints import CheckpointManager
from solrad_correction.training.dataloaders import DataLoaderSettings, resolve_dataloader_settings
from solrad_correction.training.factories import (
    create_criterion,
    create_optimizer,
    create_scheduler,
    create_summary_writer,
)
from solrad_correction.training.loops import evaluate_epoch, train_one_epoch
from solrad_correction.training.progress import TrainingProgress
from solrad_correction.training.state import BestModelState, TrainingPlan, TrainingState

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
        self.plan = TrainingPlan.from_config(config)
        self.state = TrainingState(completed_epochs=start_epoch)

        # Defaults from config
        self.lr = self.plan.learning_rate
        self.weight_decay = self.plan.weight_decay
        self.max_epochs = self.plan.max_epochs
        self.batch_size = self.plan.batch_size
        self.patience = self.plan.patience
        self.min_delta = self.plan.min_delta
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

        train_loader = self._build_loader(train_data, settings=settings, shuffle=True)
        val_loader = (
            self._build_loader(val_data, settings=settings, shuffle=False) if val_data else None
        )

        optimizer = create_optimizer(self.model, self.plan)
        criterion = create_criterion()
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Automatic Mixed Precision (AMP)
        use_amp = settings.amp
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

        # Learning Rate Scheduler
        scheduler = create_scheduler(optimizer)

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
        writer = create_summary_writer(self.config.log_dir if self.config else None)
        if writer and self.config:
            logger.info("TensorBoard tracking enabled at %s", self.config.log_dir)

        progress = TrainingProgress(
            total_epochs=self.max_epochs,
            start_epoch=self.start_epoch,
        )

        history = self.state.history

        best = BestModelState()
        checkpoint_manager = CheckpointManager.from_runtime(
            self.runtime,
            checkpoint_config=self._checkpoint_config,
        )

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

            # Best-model tracking keeps only CPU model weights in memory.
            if best.capture_if_better(self.model, monitor, epoch + 1):
                self.best_metric = best.metric
                self.best_epoch = best.epoch
                if checkpoint_manager.enabled:
                    checkpoint_manager.save_best(
                        epoch=epoch + 1,
                        model=self.model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        metric=monitor,
                        dataloader_settings=self.dataloader_settings,
                    )

            if checkpoint_manager.should_save_last(epoch + 1):
                checkpoint_manager.save_last(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    metric=monitor,
                    dataloader_settings=self.dataloader_settings,
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
        logger.info("Restoring best model weights (loss=%.6f)", best.metric)
        best.restore(self.model)

        self.optimizer_state = optimizer.state_dict()
        self.scheduler_state = scheduler.state_dict()
        self.scaler_state = scaler.state_dict() if scaler is not None else None
        self.state.completed_epochs = self.completed_epochs
        self.state.best_metric = self.best_metric
        self.state.best_epoch = self.best_epoch
        self.state.optimizer_state = self.optimizer_state
        self.state.scheduler_state = self.scheduler_state
        self.state.scaler_state = self.scaler_state

        return self.model, history

    def _build_loader(
        self,
        dataset: SequenceDataset | WindowedSequenceDataset,
        *,
        settings: DataLoaderSettings,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory,
            persistent_workers=settings.persistent_workers,
            prefetch_factor=settings.prefetch_factor,
        )
