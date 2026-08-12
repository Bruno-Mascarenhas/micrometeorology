"""High-level Trainer for PyTorch models."""

import logging
import math
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from solrad_correction.config import ModelConfig, RuntimeConfig
from solrad_correction.datasets.sequence import (
    SequenceDataset,
    WindowedSequenceDataset,
    collate_sequence_batch,
)
from solrad_correction.training.callbacks import EarlyStopping
from solrad_correction.training.checkpoints import CheckpointManager
from solrad_correction.training.dataloaders import DataLoaderSettings, resolve_dataloader_settings
from solrad_correction.training.factories import (
    create_criterion,
    create_optimizer,
    create_scheduler,
    create_summary_writer,
)
from solrad_correction.training.loops import evaluate_epoch, train_one_epoch
from solrad_correction.training.progress import TrainingProgress
from solrad_correction.training.state import BestModelState, TrainingPlan, TrainingState
from solrad_correction.utils.serialization import load_torch_checkpoint, restore_rng_state

logger = logging.getLogger(__name__)


def _unwrap_compiled(module: nn.Module) -> nn.Module:
    """Return the original module underneath a ``torch.compile`` wrapper.

    Compiled modules (``OptimizedModule``) expose the wrapped module as
    ``_orig_mod`` and share its parameters. Persisting the unwrapped module
    keeps checkpoint/state-dict keys free of the ``_orig_mod.`` prefix so
    they load back into plain (uncompiled) modules.
    """
    return getattr(module, "_orig_mod", module)


class Trainer:
    """Orchestrates the PyTorch training loop.

    Features:
    - Progress display with batch % and ETA
    - Early stopping
    - Best-model checkpointing
    - Transfer learning (resume from start_epoch)
    - Device management (CPU/CUDA auto-detection)

    Parameters
    ----------
    model:
        Module to train, moved to the resolved device by ``train``.
    device:
        Fallback device, used only when no runtime config is supplied;
        otherwise the runtime's resolved device wins.
    config:
        Hyperparameters for the run. ``None`` falls back to
        :class:`TrainingPlan` defaults.
    runtime:
        Runtime settings: device, workers, AMP, ``torch.compile``, gradient
        clipping and the checkpoint directory.
    start_epoch:
        Absolute epoch to start from. Non-zero on a resume, and measured
        against the same ``max_epochs`` budget.
    optimizer_state, scheduler_state, scaler_state:
        State dicts recovered from a resume checkpoint. Each is applied
        best-effort: an incompatible one is logged and skipped rather than
        aborting the run.
    checkpoint_config:
        Architecture arguments stored inside every checkpoint this run writes.
    best_metric, best_epoch:
        Seeded from a resume checkpoint so a resumed run never overwrites a
        better model persisted by the previous run.
    epochs_no_improve:
        Early-stopping counter carried across a resume so patience is not
        silently reset to zero.
    rng_state:
        Python/numpy/torch RNG snapshot from the resume checkpoint. Applied
        once the module and optimizer exist, since both draw from the global
        streams.
    preprocessing_fingerprint:
        Digest of the transform this run's features were scaled with, baked
        into every checkpoint so a later resume can refuse a changed scaler.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        config: ModelConfig | None = None,
        runtime: RuntimeConfig | None = None,
        start_epoch: int = 0,
        optimizer_state: dict[str, Any] | None = None,
        scheduler_state: dict[str, Any] | None = None,
        scaler_state: dict[str, Any] | None = None,
        checkpoint_config: dict[str, Any] | None = None,
        best_metric: float | None = None,
        best_epoch: int | None = None,
        epochs_no_improve: int = 0,
        rng_state: dict[str, Any] | None = None,
        preprocessing_fingerprint: str | None = None,
    ) -> None:
        self.model: nn.Module = model
        self.device = device
        self.config = config
        self.runtime = runtime
        self.start_epoch = start_epoch
        self._resume_optimizer_state = optimizer_state
        self._resume_scheduler_state = scheduler_state
        self._resume_scaler_state = scaler_state
        self._checkpoint_config = checkpoint_config
        self.plan = TrainingPlan.from_config(config)
        self.state = TrainingState(completed_epochs=start_epoch)

        self.lr = self.plan.learning_rate
        self.weight_decay = self.plan.weight_decay
        self.max_epochs = self.plan.max_epochs
        self.batch_size = self.plan.batch_size
        self.patience = self.plan.patience
        self.min_delta = self.plan.min_delta
        self.completed_epochs = start_epoch
        self.optimizer_state: dict[str, Any] | None = None
        self.scheduler_state: dict[str, Any] | None = None
        self.scaler_state: dict[str, Any] | None = None
        self.best_metric: float | None = best_metric
        self.best_epoch: int | None = best_epoch
        self.epochs_no_improve = epochs_no_improve
        self._resume_rng_state = rng_state or {}
        self._preprocessing_fingerprint = preprocessing_fingerprint
        self.dataloader_settings: DataLoaderSettings | None = None

    def train(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
    ) -> tuple[nn.Module, dict]:
        """Run the full training loop.

        ``config.max_epochs`` is the total epoch budget of the run: a resumed
        run trains only the remaining ``max_epochs - start_epoch`` epochs.

        The monitored metric driving ``scheduler.step``, early stopping and
        best-model selection is the sample-weighted mean loss returned by
        ``evaluate_epoch`` (``train_one_epoch`` when there is no val set), i.e.
        the val-set MSE in standard-scaled target units. Checkpoints predating
        that scale carry ``best_metric``/``monitor_metric`` as a mean of
        per-batch means, which over-weights the partial tail batch by
        ``n_samples / n_batches``: such a run must not be resumed here, because
        the seeded best would be compared against a differently-scaled value.

        Parameters
        ----------
        train_data:
            Windowed training set yielding ``(B, T, F)`` features and ``(B,)``
            targets, ``float32`` and in the pipeline's scaled space.
        val_data:
            Optional validation set in the same space. Without it the training
            loss becomes the monitored metric, which makes early stopping and
            best-model selection measure fit rather than generalization.

        Returns
        -------
        tuple of (nn.Module, dict)
            The model carrying the best weights of the run — plain, never a
            ``torch.compile`` wrapper, so downstream ``save``/``state_dict``
            keys stay free of the ``_orig_mod.`` prefix — and the history dict
            of per-epoch losses.

        Notes
        -----
        ``torch.compile`` is typed as returning a bare callable, so module
        identity is re-established with an ``isinstance`` check before the rest
        of the loop (``.to``, ``.parameters()``, ``_unwrap_compiled``) uses it;
        anything else it returns leaves the eager model in place.

        The TensorBoard writer is released in a ``finally`` so a raise
        mid-training still flushes the queued scalars: its event-file thread is
        a daemon with no atexit hook, so an unclosed writer loses exactly the
        epochs leading up to the failure. ``progress.finish()`` stays on the
        success path — it prints "Training complete".

        The early-stopping counter is updated BEFORE any checkpoint is
        persisted, so a file written this epoch records the current
        no-improvement count and a later resume restores it instead of
        resetting patience to zero.

        When no epoch of a resumed run beats the seeded best metric, nothing is
        captured in memory and the on-disk ``best.pt`` is what keeps the
        returned model the best across all runs. The ``self.best_metric is not
        None`` guard is what confines that fallback to a resume: without it a
        FRESH run reusing an existing checkpoint directory would, whenever no
        epoch produced a finite improving metric (divergence, an all-NaN
        feature column), load and score the previous run's weights under this
        run's config.
        """
        if self.start_epoch >= self.max_epochs:
            logger.warning(
                "start_epoch (%d) >= max_epochs (%d): nothing left to train. "
                "Increase model.max_epochs to continue this run.",
                self.start_epoch,
                self.max_epochs,
            )
            self.optimizer_state = self._resume_optimizer_state
            self.scheduler_state = self._resume_scheduler_state
            self.scaler_state = self._resume_scaler_state
            self.state.best_metric = self.best_metric
            self.state.best_epoch = self.best_epoch
            self.state.epochs_no_improve = self.epochs_no_improve
            self.state.optimizer_state = self.optimizer_state
            self.state.scheduler_state = self.scheduler_state
            self.state.scaler_state = self.scaler_state
            return self.model, self.state.history

        settings = (
            resolve_dataloader_settings(self.runtime)
            if self.runtime is not None
            else DataLoaderSettings(
                device=self.device,
                num_workers=0,
                pin_memory=self.device != "cpu",
                persistent_workers=False,
                prefetch_factor=None,
                amp="cuda" in self.device,
                torch_compile=False,
                gradient_clip=1.0,
            )
        )
        self.dataloader_settings = settings
        self.device = settings.device
        self.model.to(self.device)

        if settings.torch_compile and hasattr(torch, "compile"):
            try:
                compiled = torch.compile(self.model)
            except Exception as exc:  # noqa: BLE001 - optional speedup; any backend fault falls back
                logger.debug("torch.compile not supported or failed: %s", exc)
            else:
                if isinstance(compiled, nn.Module):
                    self.model = compiled
                    logger.info("Successfully applied torch.compile to the model")
                else:
                    logger.debug(
                        "torch.compile returned a %s rather than a module; keeping the eager model",
                        type(compiled).__name__,
                    )

        plain_model = _unwrap_compiled(self.model)

        if self._resume_rng_state:
            restore_rng_state(self._resume_rng_state)
            logger.info("Restored RNG state from the resume checkpoint")

        train_loader = self._build_loader(train_data, settings=settings, shuffle=True)
        val_loader = (
            self._build_loader(val_data, settings=settings, shuffle=False) if val_data else None
        )

        optimizer = create_optimizer(self.model, self.plan)
        criterion = create_criterion()
        early_stop = EarlyStopping(patience=self.patience, min_delta=self.min_delta)
        if self.best_metric is not None:
            early_stop.best_score = self.best_metric
        early_stop.counter = self.epochs_no_improve

        use_amp = settings.amp
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

        scheduler = create_scheduler(optimizer)

        if self._resume_optimizer_state is not None:
            try:
                optimizer.load_state_dict(self._resume_optimizer_state)
                logger.info("Restored optimizer state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible optimizer state: %s", exc)

        if self._resume_scheduler_state is not None:
            try:
                scheduler.load_state_dict(self._resume_scheduler_state)
                logger.info("Restored scheduler state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible scheduler state: %s", exc)

        if scaler is not None and self._resume_scaler_state is not None:
            try:
                scaler.load_state_dict(self._resume_scaler_state)
                logger.info("Restored AMP scaler state from checkpoint")
            except (RuntimeError, ValueError) as exc:
                logger.warning("Skipping incompatible AMP scaler state: %s", exc)

        writer = create_summary_writer(self.config.log_dir if self.config else None)
        if writer and self.config:
            logger.info("TensorBoard tracking enabled at %s", self.config.log_dir)

        progress = TrainingProgress(
            total_epochs=self.max_epochs,
            start_epoch=self.start_epoch,
        )

        history = self.state.history

        best = BestModelState()
        if self.best_metric is not None:
            best.metric = self.best_metric
            best.epoch = self.best_epoch or 0
        checkpoint_manager = CheckpointManager.from_runtime(
            self.runtime,
            checkpoint_config=self._checkpoint_config,
            preprocessing_fingerprint=self._preprocessing_fingerprint,
        )

        try:
            for epoch in range(self.start_epoch, self.max_epochs):
                self.completed_epochs = epoch + 1
                progress.start_epoch(epoch)

                train_loss = train_one_epoch(
                    self.model,
                    train_loader,
                    optimizer,
                    criterion,
                    self.device,
                    scaler=scaler,
                    clip_val=settings.gradient_clip,
                    progress_callback=progress.update_batch,
                )
                history["epoch"].append(float(epoch + 1))
                history["train_loss"].append(train_loss)

                val_loss = None
                if val_loader:
                    val_loss = evaluate_epoch(
                        self.model,
                        val_loader,
                        criterion,
                        self.device,
                        amp_enabled=use_amp,
                    )
                    history["val_loss"].append(val_loss)

                if writer:
                    writer.add_scalar("Loss/Train", train_loss, epoch)
                    if val_loss is not None:
                        writer.add_scalar("Loss/Validation", val_loss, epoch)
                        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

                monitor = val_loss if val_loss is not None else train_loss
                if not math.isfinite(monitor):
                    logger.warning(
                        "Epoch %d monitored loss is non-finite (%r): training has diverged and "
                        "best-model tracking ignores this epoch",
                        epoch + 1,
                        monitor,
                    )

                scheduler.step(monitor)

                stop = early_stop(monitor)

                if best.capture_if_better(plain_model, monitor, epoch + 1):
                    self.best_metric = best.metric
                    self.best_epoch = best.epoch
                    if checkpoint_manager.enabled:
                        checkpoint_manager.save_best(
                            epoch=epoch + 1,
                            model=plain_model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            metric=monitor,
                            dataloader_settings=self.dataloader_settings,
                            best_metric=self.best_metric,
                            best_epoch=self.best_epoch,
                            epochs_no_improve=early_stop.counter,
                        )

                if checkpoint_manager.should_save_last(epoch + 1):
                    checkpoint_manager.save_last(
                        epoch=epoch + 1,
                        model=plain_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        metric=monitor,
                        dataloader_settings=self.dataloader_settings,
                        best_metric=self.best_metric,
                        best_epoch=self.best_epoch,
                        epochs_no_improve=early_stop.counter,
                    )

                extra = ""
                if stop:
                    extra = " [EARLY STOP]"
                    progress.end_epoch(train_loss, val_loss, extra)
                    break

                progress.end_epoch(train_loss, val_loss, extra)
        finally:
            if writer:
                writer.close()

        progress.finish()

        if best.state_dict:
            logger.info("Restoring best model weights (loss=%.6f)", best.metric)
            best.restore(plain_model)
        elif (
            self.best_metric is not None
            and checkpoint_manager.enabled
            and checkpoint_manager.directory is not None
        ):
            best_path = checkpoint_manager.directory / "best.pt"
            if best_path.exists():
                checkpoint = load_torch_checkpoint(best_path)
                plain_model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(
                    "No epoch improved on the resumed best (loss=%.6f); "
                    "restored best model weights from %s",
                    best.metric,
                    best_path,
                )

        self.model = plain_model

        self.epochs_no_improve = early_stop.counter
        self.optimizer_state = optimizer.state_dict()
        self.scheduler_state = scheduler.state_dict()
        self.scaler_state = scaler.state_dict() if scaler is not None else None
        self.state.completed_epochs = self.completed_epochs
        self.state.best_metric = self.best_metric
        self.state.best_epoch = self.best_epoch
        self.state.epochs_no_improve = self.epochs_no_improve
        self.state.optimizer_state = self.optimizer_state
        self.state.scheduler_state = self.scheduler_state
        self.state.scaler_state = self.scaler_state

        return self.model, history

    def _build_loader(
        self,
        dataset: SequenceDataset | WindowedSequenceDataset,
        *,
        settings: DataLoaderSettings,
        shuffle: bool,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=settings.num_workers,
            pin_memory=settings.pin_memory,
            persistent_workers=settings.persistent_workers,
            prefetch_factor=settings.prefetch_factor,
            collate_fn=collate_sequence_batch,
        )
