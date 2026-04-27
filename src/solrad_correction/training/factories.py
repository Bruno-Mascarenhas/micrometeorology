"""Replaceable factories for PyTorch training components."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from solrad_correction.training.state import TrainingPlan


def create_optimizer(model: nn.Module, plan: TrainingPlan) -> torch.optim.Optimizer:
    """Create the default optimizer for neural solrad models."""
    return torch.optim.Adam(
        model.parameters(),
        lr=plan.learning_rate,
        weight_decay=plan.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create the default validation-loss scheduler."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )


def create_criterion() -> nn.Module:
    """Create the default regression loss."""
    return nn.MSELoss()


def create_summary_writer(log_dir: str | None) -> SummaryWriter | None:
    """Create an optional TensorBoard writer."""
    if not log_dir:
        return None
    return SummaryWriter(log_dir=str(Path(log_dir)))
