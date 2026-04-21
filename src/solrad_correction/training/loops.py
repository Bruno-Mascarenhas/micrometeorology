"""Low-level training and evaluation loops."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None = None,
    clip_val: float | None = 1.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    progress_callback:
        Called with ``(batch_idx, total_batches)`` for progress display.

    Returns the average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = len(dataloader)

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)

        optimizer.zero_grad(set_to_none=True)

        device_type = "cuda" if "cuda" in device else "cpu"
        with torch.autocast(device_type=device_type, enabled=scaler is not None):
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

        if scaler is not None:
            scaler.scale(loss).backward()

            # Gradient clipping requires unscaling first
            if clip_val is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_val is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()

        total_loss += loss.item()

        if progress_callback:
            progress_callback(batch_idx + 1, n_batches)

    return total_loss / max(n_batches, 1)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """Run one evaluation pass.

    Returns the average loss.
    """
    model.eval()
    total_loss = 0.0
    n_batches = len(dataloader)

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)

            device_type = "cuda" if "cuda" in device else "cpu"
            with torch.autocast(device_type=device_type, enabled="cuda" in device):
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

            total_loss += loss.item()

    return total_loss / max(n_batches, 1)
