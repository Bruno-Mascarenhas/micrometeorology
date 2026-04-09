"""Training progress display with percentage and ETA."""

from __future__ import annotations

import sys
import time


class TrainingProgress:
    """Display training progress with batch %, epoch %, and ETA.

    Usage::

        progress = TrainingProgress(total_epochs=100)
        for epoch in range(100):
            progress.start_epoch(epoch)
            # ... training loop calls progress.update_batch(i, total) ...
            progress.end_epoch(train_loss=0.5, val_loss=0.4)
        progress.finish()
    """

    def __init__(self, total_epochs: int, start_epoch: int = 0) -> None:
        self.total_epochs = total_epochs
        self._initial_epoch = start_epoch
        self.epoch_start_time = 0.0
        self.training_start_time = time.time()
        self.current_epoch = start_epoch

    def start_epoch(self, epoch: int) -> None:
        """Mark the beginning of an epoch."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()

    def update_batch(self, batch: int, total_batches: int) -> None:
        """Update progress within an epoch (batch-level)."""
        pct = batch / total_batches * 100
        elapsed = time.time() - self.epoch_start_time
        eta_epoch = elapsed / batch * (total_batches - batch) if batch > 0 else 0.0

        # Overall progress
        epochs_done = self.current_epoch - self._initial_epoch + batch / total_batches
        total_to_do = self.total_epochs - self._initial_epoch
        overall_pct = epochs_done / total_to_do * 100 if total_to_do > 0 else 0

        sys.stdout.write(
            f"\r  Epoch {self.current_epoch + 1}/{self._initial_epoch + self.total_epochs} "
            f"[{pct:5.1f}%] "
            f"ETA epoch: {self._fmt_time(eta_epoch)} | "
            f"Overall: {overall_pct:5.1f}%"
        )
        sys.stdout.flush()

    def end_epoch(
        self,
        train_loss: float,
        val_loss: float | None = None,
        extra: str = "",
    ) -> None:
        """Print epoch summary."""
        elapsed = time.time() - self.epoch_start_time
        total_elapsed = time.time() - self.training_start_time

        # ETA for remaining epochs
        epochs_done = self.current_epoch - self._initial_epoch + 1
        total_to_do = self.total_epochs - self._initial_epoch
        if epochs_done > 0:
            eta_total = total_elapsed / epochs_done * (total_to_do - epochs_done)
        else:
            eta_total = 0.0

        val_str = f"  val_loss={val_loss:.6f}" if val_loss is not None else ""
        sys.stdout.write(
            f"\r  Epoch {self.current_epoch + 1}/{self._initial_epoch + self.total_epochs} "
            f"— train_loss={train_loss:.6f}{val_str} "
            f"({self._fmt_time(elapsed)}/epoch, ETA: {self._fmt_time(eta_total)})"
            f"{extra}\n"
        )
        sys.stdout.flush()

    def finish(self) -> None:
        """Print final training summary."""
        total = time.time() - self.training_start_time
        print(f"\n✓ Training complete in {self._fmt_time(total)}")

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format seconds as human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            m, s = divmod(int(seconds), 60)
            return f"{m}m{s:02d}s"
        else:
            h, remainder = divmod(int(seconds), 3600)
            m, s = divmod(remainder, 60)
            return f"{h}h{m:02d}m{s:02d}s"
