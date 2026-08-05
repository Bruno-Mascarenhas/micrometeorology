"""Training plan and best-state helpers for PyTorch regressors."""

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from solrad_correction.config import ModelConfig


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Resolved hyperparameters for one training call."""

    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    min_delta: float = 1e-4

    @classmethod
    def from_config(cls, config: ModelConfig | None) -> TrainingPlan:
        """Extract training hyperparameters from a model config (defaults if ``None``)."""
        if config is None:
            return cls()
        return cls(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_epochs=config.max_epochs,
            batch_size=config.batch_size,
            patience=config.patience,
            min_delta=config.min_delta,
        )


@dataclass(slots=True)
class TrainingState:
    """Mutable training state exposed back to model wrappers."""

    completed_epochs: int
    # ``epoch`` holds the ABSOLUTE 1-based epoch number, so a resumed run's rows
    # keep the numbering of the run they continue instead of restarting at 0.
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"epoch": [], "train_loss": [], "val_loss": []}
    )
    best_metric: float | None = None
    best_epoch: int | None = None
    epochs_no_improve: int = 0
    optimizer_state: dict | None = None
    scheduler_state: dict | None = None
    scaler_state: dict | None = None


@dataclass(slots=True)
class BestModelState:
    """Memory-conscious best-model state tracker.

    Only model tensors are copied, and they are moved to CPU immediately. Best
    optimizer/scheduler/scaler states are persisted in ``best.pt`` by the
    checkpoint manager instead of being kept in memory.
    """

    metric: float = float("inf")
    epoch: int = 0
    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)

    def capture_if_better(self, model: nn.Module, metric: float, epoch: int) -> bool:
        """Snapshot ``model``'s weights when ``metric`` strictly improves on the best.

        On improvement the tensors are detached, cloned and moved to CPU, and the
        method returns ``True``; otherwise nothing is stored and it returns
        ``False``. Lower ``metric`` is treated as better.

        A non-finite ``metric`` is never an improvement: every comparison with
        ``nan`` is False, so a single diverged epoch would otherwise be captured
        as best and leave ``self.metric`` permanently ``nan``, making every
        later epoch look better than the genuine best.
        """
        if not math.isfinite(metric) or metric >= self.metric:
            return False
        self.metric = metric
        self.epoch = epoch
        self.state_dict = {
            key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        """Load the captured best weights into ``model`` (no-op if none captured)."""
        if self.state_dict:
            model.load_state_dict(self.state_dict)
