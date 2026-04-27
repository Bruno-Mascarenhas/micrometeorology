"""Checkpoint management for PyTorch training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from solrad_correction.utils.serialization import save_torch_checkpoint

if TYPE_CHECKING:
    import torch
    from torch import nn

    from solrad_correction.training.dataloaders import DataLoaderSettings


@dataclass(slots=True)
class CheckpointManager:
    """Own best/last checkpoint paths and serialization metadata."""

    directory: Path | None
    every: int = 1
    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime(
        cls,
        runtime: Any,
        *,
        checkpoint_config: dict[str, Any] | None = None,
    ) -> CheckpointManager:
        directory = (
            Path(runtime.checkpoint_dir) if runtime is not None and runtime.checkpoint_dir else None
        )
        every = (
            runtime.checkpoint_every
            if runtime is not None and runtime.checkpoint_every is not None
            else 1
        )
        return cls(directory=directory, every=every, config=checkpoint_config)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def should_save_last(self, epoch: int) -> bool:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
                "dataloader": dataloader_settings.to_dict()
                if dataloader_settings is not None
                else {},
            },
        )
