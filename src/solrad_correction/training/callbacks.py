"""Training callbacks: early stopping and model checkpointing."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Uses ``__slots__`` for reduced memory overhead.

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

    __slots__ = ("best_score", "counter", "min_delta", "mode", "patience", "should_stop")

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
