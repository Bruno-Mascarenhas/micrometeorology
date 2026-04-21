"""High-level Trainer for PyTorch models."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from solrad_correction.training.callbacks import EarlyStopping
from solrad_correction.training.loops import evaluate_epoch, train_one_epoch
from solrad_correction.training.progress import TrainingProgress
from solrad_correction.utils.seeds import set_global_seed

if TYPE_CHECKING:
    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.sequence import SequenceDataset

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
        start_epoch: int = 0,
    ) -> None:
        self.model: nn.Module = model  # type: ignore
        self.device = device
        self.config = config
        self.start_epoch = start_epoch

        # Defaults from config
        self.lr = config.learning_rate if config else 1e-3
        self.weight_decay = config.weight_decay if config else 1e-5
        self.max_epochs = config.max_epochs if config else 100
        self.batch_size = config.batch_size if config else 32
        self.patience = config.patience if config else 10
        self.min_delta = config.min_delta if config else 1e-4
        self.seed = 42

    def train(
        self,
        train_data: SequenceDataset,
        val_data: SequenceDataset | None = None,
    ) -> tuple[nn.Module, dict]:
        """Run the full training loop.

        Returns ``(trained_model, history)`` where history contains
        per-epoch losses.
        """
        set_global_seed(self.seed)
        self.model.to(self.device)

        # Optimize with torch.compile if supported (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore
                logger.info("Successfully applied torch.compile to the model")
            except Exception as e:
                logger.debug("torch.compile not supported or failed: %s", e)

        # High-performance dataloading
        num_workers = min(4, torch.get_num_threads())
        pin_memory = self.device != "cpu"

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

        val_loader = None
        if val_data:
            val_loader = DataLoader(
                val_data,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                prefetch_factor=2 if num_workers > 0 else None,
            )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        # Automatic Mixed Precision (AMP)
        use_amp = "cuda" in self.device
        scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None

        # Learning Rate Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )

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

        total_epochs = self.start_epoch + self.max_epochs
        for epoch in range(self.start_epoch, total_epochs):
            progress.start_epoch(epoch)

            # Train
            train_loss = train_one_epoch(
                self.model,
                train_loader,
                optimizer,
                criterion,
                self.device,
                scaler=scaler,
                progress_callback=progress.update_batch,
            )
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = evaluate_epoch(self.model, val_loader, criterion, self.device)
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
                best_state_dict = copy.deepcopy(self.model.state_dict())

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

        return self.model, history
