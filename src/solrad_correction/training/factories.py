"""Replaceable factories for PyTorch training components.

Every default the training loop depends on — optimizer, scheduler, loss and
TensorBoard writer — is built here, so a run can be re-pointed at a different
optimizer or loss without touching the loop itself.

``SummaryWriter`` is an alias for the runtime type
``torch.utils.tensorboard.SummaryWriter``: the annotation has to resolve without
importing tensorboard at module load, and the real import stays deferred inside
``create_summary_writer``.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from solrad_correction.training.state import TrainingPlan

type SummaryWriter = Any


def create_optimizer(model: nn.Module, plan: TrainingPlan) -> torch.optim.Optimizer:
    """Create the default optimizer for neural solrad models.

    Adam over every parameter of ``model``, with the learning rate and weight
    decay the plan resolved from the experiment config.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=plan.learning_rate,
        weight_decay=plan.weight_decay,
    )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create the default validation-loss scheduler.

    Halves the learning rate after five epochs without an improvement in the
    monitored loss, down to a floor of ``1e-6``. Its patience is fixed here
    rather than read from the config, and sits below the default
    early-stopping patience of ten, so a plateau gets a smaller step to try
    before the run is abandoned.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )


def create_criterion() -> nn.Module:
    """Create the default regression loss.

    Mean squared error, reduced over the batch. The loops rely on that mean
    reduction: they re-weight each batch by its sample count to recover a
    dataset-level loss.
    """
    return nn.MSELoss()


def create_summary_writer(log_dir: str | None) -> SummaryWriter | None:
    """Create an optional TensorBoard writer.

    The import is deferred so tensorboard stays an optional dependency: it is
    only required when ``log_dir`` is actually configured.

    Parameters
    ----------
    log_dir:
        Event-file directory. Empty or ``None`` disables TensorBoard logging.

    Returns
    -------
    SummaryWriter or None
        ``None`` when logging is disabled, which every call site treats as
        "skip the scalar writes".
    """
    if not log_dir:
        return None
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(Path(log_dir)))
