"""Training callbacks: early stopping and model checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    mode:
        ``"min"`` for loss-like metrics (lower is better),
        ``"max"`` for accuracy-like metrics (higher is better).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        """Check if training should stop.

        Returns True if training should stop.
        """
        if self.best_score is None:
            self.best_score = metric
            return False

        improved = (
            metric < self.best_score - self.min_delta
            if self.mode == "min"
            else metric > self.best_score + self.min_delta
        )

        if improved:
            self.best_score = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    "Early stopping at patience %d (best=%.6f)", self.patience, self.best_score
                )
                return True

        return False


class ModelCheckpoint:
    """Save the best model based on a monitored metric.

    Parameters
    ----------
    path:
        File path for the checkpoint.
    mode:
        ``"min"`` or ``"max"`` — which direction is better.
    """

    def __init__(self, path: str | Path, mode: str = "min") -> None:
        self.path = Path(path)
        self.mode = mode
        self.best_score: float | None = None

    def __call__(self, metric: float, save_fn: Callable) -> bool:
        """Check and save if this is the best model so far.

        Returns True if the model was saved.
        """
        if self.best_score is None:
            self.best_score = metric
            save_fn(self.path)
            logger.info("Checkpoint saved: %.6f → %s", metric, self.path)
            return True

        improved = metric < self.best_score if self.mode == "min" else metric > self.best_score

        if improved:
            self.best_score = metric
            save_fn(self.path)
            logger.info("Checkpoint updated: %.6f → %s", metric, self.path)
            return True

        return False
