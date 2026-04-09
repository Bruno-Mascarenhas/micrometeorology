"""Low-level training and evaluation loops."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    progress_callback: callable | None = None,
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
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(-1)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
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
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(-1)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()

    return total_loss / max(n_batches, 1)
