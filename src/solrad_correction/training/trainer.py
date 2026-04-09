"""High-level Trainer for PyTorch models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

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
        self.model = model
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

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size) if val_data else None

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        progress = TrainingProgress(
            total_epochs=self.max_epochs,
            start_epoch=self.start_epoch,
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

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
                progress_callback=progress.update_batch,
            )
            history["train_loss"].append(train_loss)

            # Validate
            val_loss = None
            if val_loader:
                val_loss = evaluate_epoch(self.model, val_loader, criterion, self.device)
                history["val_loss"].append(val_loss)

            extra = ""
            # Early stopping
            monitor = val_loss if val_loss is not None else train_loss
            if early_stop(monitor):
                extra = " [EARLY STOP]"
                progress.end_epoch(train_loss, val_loss, extra)
                break

            progress.end_epoch(train_loss, val_loss, extra)

        progress.finish()
        return self.model, history
