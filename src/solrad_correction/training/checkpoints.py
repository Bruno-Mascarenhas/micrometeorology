"""Checkpoint management for PyTorch training."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from solrad_correction.config import RuntimeConfig
from solrad_correction.training.dataloaders import DataLoaderSettings
from solrad_correction.utils.serialization import capture_rng_state, save_torch_checkpoint

#: Names the checkpoint manager writes under. The reader in
#: :mod:`solrad_correction.models.torch_base` and the trainer's best-model
#: reload both address these files, so the strings live here, next to the writer.
BEST_CHECKPOINT = "best.pt"
LAST_CHECKPOINT = "last.pt"


@dataclass(slots=True)
class CheckpointManager:
    """Own best/last checkpoint paths and serialization metadata.

    Attributes
    ----------
    directory:
        Where ``best.pt`` and ``last.pt`` are written; created on construction.
        ``None`` disables checkpointing and turns every write into a no-op, so
        the trainer needs no branch of its own.
    every:
        Cadence in epochs for ``last.pt``. ``best.pt`` ignores it and is written
        whenever the monitored metric improves.
    config:
        Architecture arguments stored alongside the weights, so a checkpoint
        can be rebuilt into the module it came from.
    preprocessing_fingerprint:
        Digest of the preprocessing transform the saved weights were trained
        under, so a later resume can refuse a refit that changed it.
    """

    directory: Path | None
    every: int = 1
    config: dict[str, Any] | None = None
    preprocessing_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_runtime(
        cls,
        runtime: RuntimeConfig | None,
        *,
        checkpoint_config: dict[str, Any] | None = None,
        preprocessing_fingerprint: str | None = None,
    ) -> CheckpointManager:
        """Build a manager from a runtime config.

        Checkpointing is disabled when the runtime is absent or carries no
        checkpoint directory, so a smoke run writes no weights at all.
        """
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
        """Write ``best.pt`` for the epoch that just improved.

        ``best_metric``/``best_epoch`` default to this call's own values,
        which is correct precisely because this file is only written on an
        improvement.
        """
        self.save(
            BEST_CHECKPOINT,
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
        """Write ``last.pt`` for resume, carrying the best metric/epoch seen so far.

        Unlike ``best.pt``, this file records the epoch just finished whether
        or not it improved, so the best values have to be passed in explicitly:
        they belong to some earlier epoch.
        """
        self.save(
            LAST_CHECKPOINT,
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

        A no-op when no checkpoint directory is configured.

        Resume-critical metadata is embedded alongside the tensors:

        ``best_metric``/``best_epoch``
            Best monitored loss across the whole run so far; a resume reads it
            to seed best-model tracking and early stopping.
        ``epochs_no_improve``
            Early-stopping counter at this epoch, restored on resume so
            patience is not silently reset to zero.
        ``dataloader``
            The run's resolved DataLoader/AMP settings.
        ``rng_state``
            Python/numpy/torch RNG snapshot, in weights_only-loadable types
            (see ``capture_rng_state``). This is what makes a resumed run
            reproducible: the shuffle order, dropout masks and any
            initialization drawn after the resume point continue the
            interrupted sequence instead of replaying it from the seed.
        ``preprocessing_fingerprint``
            Digest of the transform these weights were trained under; a resume
            compares it against the freshly refitted one.

        Parameters
        ----------
        filename:
            File name inside the checkpoint directory, ``best.pt`` or
            ``last.pt``.
        epoch:
            Absolute 1-based epoch the state belongs to.
        model:
            Plain (uncompiled) module, so state_dict keys carry no
            ``_orig_mod.`` prefix.
        optimizer, scheduler, scaler:
            Training state needed to continue rather than restart; ``scaler``
            is ``None`` outside AMP runs.
        metric:
            Monitored loss at this epoch, in scaled target units.
        kind:
            ``"best"`` or ``"last"``, recorded so a resume can tell whether the
            file it loaded is itself the best one.
        dataloader_settings:
            Resolved runtime settings, serialized as a plain dict.
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
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "epochs_no_improve": epochs_no_improve,
                "dataloader": dataloader_settings.to_dict()
                if dataloader_settings is not None
                else {},
                "rng_state": capture_rng_state(),
                "preprocessing_fingerprint": self.preprocessing_fingerprint,
            },
        )
