"""Low-level training and evaluation loops."""

import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.amp.GradScaler | None = None,
    clip_val: float | None = 1.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model:
        Module in training mode by the end of the call; its parameters are
        updated in place.
    dataloader:
        Yields ``(x, y)`` pairs with ``x`` of shape ``(B, T, F)`` and ``y`` of
        shape ``(B,)``, both ``float32`` and in the pipeline's scaled
        space. ``y`` is
        unsqueezed to ``(B, 1)`` to match the models' output.
    optimizer:
        Stepped once per batch.
    criterion:
        Must reduce with ``reduction='mean'`` over the batch dimension: the
        returned value re-weights each batch by its sample count, which is only
        correct for a batch-mean loss.
    device:
        Device the batches are moved to; also selects the autocast device type.
    scaler:
        AMP gradient scaler. Passing one enables autocast for the forward pass;
        ``None`` trains in full precision.
    clip_val:
        Max gradient norm, or ``None`` to skip clipping.
    progress_callback:
        Called with ``(batch_idx, total_batches)`` for progress display.

    Returns
    -------
    float
        The sample-weighted average of the per-batch losses over the epoch, in
        scaled target units squared, so a partial tail batch counts in
        proportion to its size. This is not the dataset MSE: the parameters
        change after every optimizer step, so it is a running average over a
        moving model.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
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

        batch_samples = y_batch.shape[0]
        total_loss += loss.item() * batch_samples
        total_samples += batch_samples

        if progress_callback:
            progress_callback(batch_idx + 1, n_batches)

    return total_loss / max(total_samples, 1)


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    *,
    amp_enabled: bool | None = None,
) -> float:
    """Run one evaluation pass.

    Parameters
    ----------
    model:
        Module switched to eval mode; no gradients are taken and no parameter
        changes.
    dataloader:
        Yields ``(x, y)`` pairs with ``x`` of shape ``(B, T, F)`` and ``y`` of
        shape ``(B,)``, both ``float32`` and in the pipeline's scaled space.
    criterion:
        Must reduce with ``reduction='mean'`` over the batch dimension: the
        returned value re-weights each batch by its sample count, which is only
        correct for a batch-mean loss.
    device:
        Device the batches are moved to; also selects the autocast device type.
    amp_enabled:
        Whether to run the forward pass under autocast. ``None`` falls back to
        "autocast on CUDA"; the trainer passes the run's resolved AMP setting so
        the validation loss is computed at the same precision as the training
        loss it is compared against.

    Returns
    -------
    float
        The sample-weighted mean loss over the whole dataset, in scaled target
        units squared. With a batch-mean criterion and a fixed model this is
        exactly the dataset-level loss (for ``nn.MSELoss``, the dataset MSE),
        independent of how the samples are split into batches.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).unsqueeze(-1)

            device_type = "cuda" if "cuda" in device else "cpu"
            use_amp = "cuda" in device if amp_enabled is None else amp_enabled
            with torch.autocast(device_type=device_type, enabled=use_amp):
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)

            batch_samples = y_batch.shape[0]
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples

    return total_loss / max(total_samples, 1)
