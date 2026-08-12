"""Training callbacks: early stopping and model checkpointing."""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of consecutive epochs with no improvement before stopping.
    min_delta:
        Minimum change, in the monitored metric's own units, that counts as an
        improvement. It rules out stopping being deferred by numerical noise.
    mode:
        ``"min"`` for loss-like metrics (lower is better),
        ``"max"`` for accuracy-like metrics (higher is better).

    Attributes
    ----------
    best_score:
        Best metric seen so far, or ``None`` before the first call. A resume
        seeds it with the previous run's best, so improvement is measured
        against the whole training history rather than from scratch.
    counter:
        Epochs since the last improvement. Seeded on a resume too, so patience
        is not silently reset.
    should_stop:
        Latched once the counter reaches ``patience``.
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
        """Record one epoch's metric and report whether training should stop.

        The first call only establishes the baseline; it never stops. A
        non-finite metric fails both comparisons, so a diverged epoch counts as
        no improvement rather than as a new best.

        Parameters
        ----------
        metric:
            The monitored value for the epoch just finished, in the units the
            criterion produces.

        Returns
        -------
        bool
            ``True`` on the epoch where patience runs out, ``False`` otherwise.
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
