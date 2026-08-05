"""Checkpoint management for PyTorch training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from solrad_correction.training.dataloaders import DataLoaderSettings
from solrad_correction.utils.serialization import capture_rng_state, save_torch_checkpoint


@dataclass(slots=True)
class CheckpointManager:
    """Own best/last checkpoint paths and serialization metadata."""

    directory: Path | None
    every: int = 1
    config: dict[str, Any] | None = None
    #: Digest of the preprocessing transform the saved weights were trained
    #: under, so a later resume can refuse a refit that changed it.
    preprocessing_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        checkpoint_config: dict[str, Any] | None = None,
        preprocessing_fingerprint: str | None = None,
    ) -> CheckpointManager:
        """Build a manager from a runtime config; disabled when it has no checkpoint dir."""
        directory = (
            Path(runtime.checkpoint_dir) if runtime is not None and runtime.checkpoint_dir else None
        )
        every = (
            runtime.checkpoint_every
            if runtime is not None and runtime.checkpoint_every is not None
            else 1
        )
        return cls(
            directory=directory,
            every=every,
            config=checkpoint_config,
            preprocessing_fingerprint=preprocessing_fingerprint,
        )

    @property
    def enabled(self) -> bool:
        """Whether a checkpoint directory is configured (writes are no-ops otherwise)."""
        return self.directory is not None

    def should_save_last(self, epoch: int) -> bool:
        """Whether ``last.pt`` is due this epoch (enabled and on the ``every`` cadence)."""
        return self.enabled and epoch % self.every == 0

    def save_best(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        scaler: torch.amp.GradScaler | None,
        metric: float,
        dataloader_settings: DataLoaderSettings | None,
        best_metric: float | None = None,
        best_epoch: int | None = None,
        epochs_no_improve: int = 0,
    ) -> None:
        """Write ``best.pt``; ``best_metric``/``best_epoch`` default to this call's values."""
        self.save(
            "best.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metric=metric,
            kind="best",
            dataloader_settings=dataloader_settings,
            best_metric=best_metric if best_metric is not None else metric,
            best_epoch=best_epoch if best_epoch is not None else epoch,
            epochs_no_improve=epochs_no_improve,
        )

    def save_last(
        self,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        scaler: torch.amp.GradScaler | None,
        metric: float,
        dataloader_settings: DataLoaderSettings | None,
        best_metric: float | None = None,
        best_epoch: int | None = None,
        epochs_no_improve: int = 0,
    ) -> None:
        """Write ``last.pt`` for resume, carrying the best metric/epoch seen so far."""
        self.save(
            "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metric=metric,
            kind="last",
            dataloader_settings=dataloader_settings,
            best_metric=best_metric,
            best_epoch=best_epoch,
            epochs_no_improve=epochs_no_improve,
        )

    def save(
        self,
        filename: str,
        *,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        scaler: torch.amp.GradScaler | None,
        metric: float,
        kind: str,
        dataloader_settings: DataLoaderSettings | None,
        best_metric: float | None = None,
        best_epoch: int | None = None,
        epochs_no_improve: int = 0,
    ) -> None:
        """Serialize model/optimizer/scheduler/scaler state to ``filename``.

        A no-op when no checkpoint directory is configured. Resume-critical
        metadata (kind, monitored metric, best metric/epoch, early-stopping
        no-improvement counter, DataLoader settings, RNG state) is embedded
        alongside the tensors. The RNG snapshot is what makes a resumed run
        reproducible: the shuffle order, dropout masks and any initialization
        drawn after the resume point continue the interrupted sequence instead
        of restarting from the seed.
        """
        if self.directory is None:
            return
        save_torch_checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=self.config,
            epoch=epoch,
            path=self.directory / filename,
            scheduler_state=scheduler.state_dict(),
            scaler_state=scaler.state_dict() if scaler is not None else None,
            metadata={
                "checkpoint_kind": kind,
                "monitor_metric": metric,
                # Best metric across the whole run so far; resume reads this
                # to seed best-model tracking and early stopping.
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                # Early-stopping no-improvement counter at this epoch; resume
                # restores it so patience is not silently reset to zero.
                "epochs_no_improve": epochs_no_improve,
                "dataloader": dataloader_settings.to_dict()
                if dataloader_settings is not None
                else {},
                # Python/numpy/torch RNG snapshot, in weights_only-loadable
                # types (see capture_rng_state): a resume that reseeds from
                # scratch replays the first epochs' shuffle order instead of
                # continuing the stream, so the resumed run is not the run it
                # claims to continue.
                "rng_state": capture_rng_state(),
                # Digest of the transform these weights were trained under; a
                # resume compares it against the freshly refitted one.
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
            },
        )
