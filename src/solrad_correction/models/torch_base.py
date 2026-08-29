"""Base class for PyTorch-based regressors with transfer learning support."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from solrad_correction.config import ModelConfig
from solrad_correction.datasets.sequence import (
    SequenceDataset,
    WindowedSequenceDataset,
    collate_sequence_batch,
)
from solrad_correction.models.base import SequenceRegressorModel, TrainingResult
from solrad_correction.training.checkpoints import BEST_CHECKPOINT
from solrad_correction.training.dataloaders import DataLoaderSettings
from solrad_correction.utils.device import resolve_device
from solrad_correction.utils.memory import assert_array_size
from solrad_correction.utils.serialization import (
    load_torch_checkpoint,
    save_torch_checkpoint,
    unwrap_compiled,
)

logger = logging.getLogger(__name__)


class TorchRegressorModel(SequenceRegressorModel):
    """Base for PyTorch sequential models (LSTM, Transformer, etc.).

    Subclasses must:
    1. Set ``self._module`` (a ``nn.Module``) in ``__init__``.
    2. Override the ``name`` property.
    3. Override ``load`` to rebuild their architecture from a checkpoint, and
       ``save`` to write the architecture arguments that rebuild needs.

    Supports full training resume via ``RuntimeConfig.resume``.

    Attributes
    ----------
    _start_epoch:
        Epoch the next ``fit`` starts from: ``0`` for a fresh model, and for a
        resumed one the number of epochs the previous run completed. It is an
        absolute count, so ``config.max_epochs`` remains the budget of the
        whole run rather than of each resume.
    """

    _module: nn.Module
    _device: str
    _start_epoch: int

    def __init__(self, device: str | None = None) -> None:
        self._device = device or resolve_device("auto")
        self._start_epoch = 0
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._scaler_state: dict[str, Any] | None = None
        self._best_metric: float | None = None
        self._best_epoch: int | None = None
        self._epochs_no_improve: int = 0
        self._rng_state: dict[str, Any] = {}
        self._preprocessing_fingerprint: str | None = None
        self._resume_preprocessing_fingerprint: str | None = None
        self._dataloader_settings: DataLoaderSettings | None = None
        logger.info("Device: %s", self._device)

    def _build_module(self, **kwargs: Any) -> nn.Module:
        """Extension hook for building the architecture from keyword arguments.

        No shipped subclass uses it: both build their module directly in
        ``__init__`` and rebuild it in ``load``. It stays as the seam a future
        model can hook into.
        """
        raise NotImplementedError

    def _load_resume_checkpoint(self, path: str) -> None:
        """Load a full training checkpoint for resumed training.

        Weights, optimizer, scheduler and scaler states are applied here, along
        with the best-metric and early-stopping counters recovered from the
        checkpoint metadata. The RNG snapshot is only carried, not applied: it
        has to be restored AFTER the module and optimizer are built, since both
        draw from the global streams, which is why ``Trainer`` applies it at the
        top of ``train()``.
        """
        checkpoint = load_torch_checkpoint(path)
        self._module.load_state_dict(checkpoint["model_state_dict"])
        self._start_epoch = checkpoint.get("epoch", 0)
        self._optimizer_state = checkpoint.get("optimizer_state_dict")
        self._scheduler_state = checkpoint.get("scheduler_state_dict")
        self._scaler_state = checkpoint.get("scaler_state_dict")
        self._best_metric, self._best_epoch = self._resolve_resume_best(checkpoint, path)
        self._epochs_no_improve = self._resolve_resume_epochs_no_improve(
            checkpoint, self._best_epoch, self._start_epoch
        )
        metadata = checkpoint.get("metadata") or {}
        self._rng_state = metadata.get("rng_state") or {}
        self._resume_preprocessing_fingerprint = metadata.get("preprocessing_fingerprint")
        logger.info("Loaded resume checkpoint from %s (epoch %d)", path, self._start_epoch)

    @staticmethod
    def _resolve_resume_best(
        checkpoint: dict[str, Any], path: str
    ) -> tuple[float | None, int | None]:
        """Recover the previous run's best monitor metric for resume.

        Preference order: the ``best_metric`` persisted in the checkpoint
        metadata, the checkpoint's own metric when it *is* the best
        checkpoint, then a sibling ``best.pt`` written by the checkpoint
        manager (covers checkpoints written before ``best_metric`` existed).
        """
        metadata = checkpoint.get("metadata") or {}
        best_metric = metadata.get("best_metric")
        best_epoch = metadata.get("best_epoch")
        if best_metric is not None:
            return best_metric, best_epoch
        if metadata.get("checkpoint_kind") == "best":
            return metadata.get("monitor_metric"), checkpoint.get("epoch")

        best_path = Path(path).parent / BEST_CHECKPOINT
        if best_path.exists():
            best_checkpoint = load_torch_checkpoint(best_path)
            best_metadata = best_checkpoint.get("metadata") or {}
            metric = best_metadata.get("best_metric")
            if metric is None:
                metric = best_metadata.get("monitor_metric")
            if metric is not None:
                return metric, best_checkpoint.get("epoch")
        return None, None

    @staticmethod
    def _resolve_resume_epochs_no_improve(
        checkpoint: dict[str, Any], best_epoch: int | None, start_epoch: int
    ) -> int:
        """Recover the early-stopping no-improvement counter for resume.

        Prefers the ``epochs_no_improve`` value persisted in the checkpoint
        metadata. Older checkpoints predate that field: derive it from the gap
        between the completed epoch and the best epoch (epochs elapsed since the
        last improvement), falling back to ``0`` when neither is available.
        """
        metadata = checkpoint.get("metadata") or {}
        persisted = metadata.get("epochs_no_improve")
        if persisted is not None:
            return int(persisted)
        if best_epoch is not None:
            return max(0, int(start_epoch) - int(best_epoch))
        return 0

    def _assert_preprocessing_unchanged(self, *, allow_change: bool, path: str) -> None:
        """Refuse a resume whose scaler no longer matches the trained weights.

        A resume refits the preprocessing from whatever is on disk now and never
        loads the checkpoint's own transform, so a changed feature set, target,
        scaler type or fitted mean/scale silently feeds the restored weights a
        differently-scaled input — the loss looks like a fresh run's and every
        metric derived from it is meaningless. Checkpoints written before the
        fingerprint existed carry none and are resumed with a warning.
        """
        stored = self._resume_preprocessing_fingerprint
        current = self._preprocessing_fingerprint
        if stored is None or current is None:
            logger.warning(
                "Resuming %s without a preprocessing fingerprint on %s: the refitted "
                "scaler is NOT verified against the one these weights were trained under",
                path,
                "the checkpoint" if stored is None else "this run",
            )
            return
        if stored == current:
            return
        message = (
            f"the preprocessing refitted for this run ({current[:12]}) does not match the "
            f"one {path} was trained under ({stored[:12]}): the restored weights would "
            "receive differently-scaled inputs. Re-run without --resume, or pass "
            "--allow-preprocessing-change to accept the re-scaling."
        )
        if allow_change:
            logger.warning("Resuming anyway: %s", message)
            return
        raise ValueError(message)

    def fit(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
        config: ModelConfig | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        """Train using the standard training loop with progress display.

        Parameters
        ----------
        train_data:
            Windowed training set yielding ``(B, T, F)`` feature batches and
            ``(B,)`` targets, both ``float32`` and in the pipeline's scaled
            space.
        val_data:
            Optional validation set, in the same scaled space, driving the
            scheduler, early stopping and best-model selection.
        config:
            Hyperparameters for this run (learning rate, epoch budget, batch
            size, patience).
        **kwargs:
            ``runtime`` (a ``RuntimeConfig``) selects device, AMP, DataLoader
            settings and resume; ``preprocessing_fingerprint`` identifies the
            transform the features were scaled with and is checked against the
            checkpoint's before a resume is allowed to continue.

        Returns
        -------
        TrainingResult
            This model, carrying the best weights of the run, and the per-epoch
            history.
        """
        from solrad_correction.training.trainer import Trainer

        runtime = kwargs.get("runtime")
        self._preprocessing_fingerprint = kwargs.get("preprocessing_fingerprint")

        if runtime and runtime.resume:
            self._load_resume_checkpoint(runtime.resume)
            self._assert_preprocessing_unchanged(
                allow_change=bool(getattr(runtime, "allow_preprocessing_change", False)),
                path=runtime.resume,
            )

        trainer = Trainer(
            model=self._module,
            device=self._device,
            config=config,
            runtime=runtime,
            start_epoch=self._start_epoch,
            optimizer_state=self._optimizer_state,
            scheduler_state=self._scheduler_state,
            scaler_state=self._scaler_state,
            checkpoint_config=getattr(self, "_config_kwargs", None),
            preprocessing_fingerprint=self._preprocessing_fingerprint,
            best_metric=self._best_metric,
            best_epoch=self._best_epoch,
            epochs_no_improve=self._epochs_no_improve,
            rng_state=self._rng_state,
        )
        self._module, history = trainer.train(train_data, val_data)
        self._start_epoch = trainer.completed_epochs
        self._optimizer_state = trainer.optimizer_state
        self._scheduler_state = trainer.scheduler_state
        self._scaler_state = trainer.scaler_state
        self._best_metric = trainer.best_metric
        self._best_epoch = trainer.best_epoch
        self._epochs_no_improve = trainer.epochs_no_improve
        self._dataloader_settings = trainer.dataloader_settings
        self._history = history
        self._config = config
        self._runtime = runtime
        return TrainingResult(model=self, history=history)

    @property
    def training_history(self) -> dict[str, list[float]]:
        """Per-epoch curves from the latest fit call.

        Keyed by ``"epoch"`` (absolute 1-based epoch numbers), ``"train_loss"``
        and ``"val_loss"``; losses are in scaled target units. Empty before the
        first fit.
        """
        return getattr(self, "_history", {})

    @property
    def best_metric(self) -> float | None:
        """Best monitored loss seen so far, in scaled target units.

        Spans a resumed run's whole history, not only the epochs since the
        resume. ``None`` when no epoch has produced a finite metric yet.
        """
        return getattr(self, "_best_metric", None)

    @property
    def best_epoch(self) -> int | None:
        """Absolute 1-based epoch at which ``best_metric`` was reached."""
        return getattr(self, "_best_epoch", None)

    @property
    def dataloader_settings(self) -> DataLoaderSettings | None:
        """Resolved DataLoader settings from the latest fit call."""
        return getattr(self, "_dataloader_settings", None)

    def predict(self, data: SequenceDataset | WindowedSequenceDataset | np.ndarray) -> np.ndarray:
        """Generate predictions using a batched DataLoader to prevent OOM.

        Inference always runs in full float32 precision (no autocast) and
        returns a float32 array: AMP is a training-time optimization, and
        half-precision predictions (~3 significant digits) would make saved
        predictions and metrics differ between CUDA and CPU for the same
        checkpoint.

        Parameters
        ----------
        data:
            Windowed dataset, or a feature array of shape ``(N, T, F)``,
            already in the scaled feature space the model was fitted on.

        Returns
        -------
        numpy.ndarray
            Predictions of shape ``(N,)``, ``float32``, in scaled target units,
            in dataset order — the loader never shuffles, so row ``i`` of the
            output belongs to window ``i`` of the input.

        Notes
        -----
        The batch size comes from the config ``fit`` stored, falling back to
        256. ``_config`` is absent entirely on an instance built by ``load()``,
        which bypasses ``__init__`` — hence the ``getattr`` — and is otherwise
        the ``ModelConfig | None`` that ``fit()`` stored, so "not None" is
        exactly the case that carries a batch size.
        """
        from torch.utils.data import DataLoader, Dataset, TensorDataset

        self._module.eval()
        self._module.to(self._device)

        dataset: Dataset
        if isinstance(data, Dataset):
            dataset = data
        else:
            arr = np.asarray(data)
            assert_array_size(arr.shape, np.float32, context="torch prediction input array")
            x_input = torch.as_tensor(arr, dtype=torch.float32)
            dataset = TensorDataset(x_input)

        config = getattr(self, "_config", None)
        batch_size = 256 if config is None else config.batch_size

        settings = self._dataloader_settings
        if settings is not None and settings.num_workers > 0:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=settings.num_workers,
                pin_memory=settings.pin_memory,
                persistent_workers=settings.persistent_workers,
                prefetch_factor=settings.prefetch_factor,
                collate_fn=collate_sequence_batch,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate_sequence_batch,
            )
        all_preds = []

        with torch.inference_mode():
            for batch in loader:
                batch_x = batch[0] if isinstance(batch, list | tuple) else batch
                batch_x = batch_x.to(self._device, non_blocking=True)
                preds = self._module(batch_x)
                all_preds.append(preds.float().cpu().numpy().flatten())

        return np.concatenate(all_preds)

    def _checkpoint_config(self) -> Any:
        """What travels with the weights as ``config``.

        Subclasses that rebuild their module from architecture arguments
        override this to persist those instead: their ``load`` reads
        ``checkpoint["config"]["input_size"]`` and friends, which the experiment
        config cannot supply.
        """
        import dataclasses

        if getattr(self, "_config", None) is None:
            return None
        return (
            dataclasses.asdict(self._config)
            if dataclasses.is_dataclass(self._config)
            else self._config
        )

    def _restore_training_state(self, checkpoint: dict) -> None:
        """Adopt the optimiser/scheduler/scaler states and epoch a checkpoint carries.

        Every torch model restores the same five fields the same way; writing it
        per subclass is how one of them silently stops resuming.
        """
        self._start_epoch = checkpoint.get("epoch", 0)
        self._optimizer_state = checkpoint.get("optimizer_state_dict")
        self._scheduler_state = checkpoint.get("scheduler_state_dict")
        self._scaler_state = checkpoint.get("scaler_state_dict")
        self._best_metric = None
        self._best_epoch = None
        self._dataloader_settings = None

    def save(self, path: str | Path) -> None:
        """Save model checkpoint (state_dict + config for transfer learning).

        The plain module is what gets persisted: ``torch.compile`` wrappers
        prefix state_dict keys with ``_orig_mod.``, which plain modules cannot
        load back.
        """
        config_dict = self._checkpoint_config()
        module = unwrap_compiled(self._module)
        save_torch_checkpoint(
            model_state=module.state_dict(),
            optimizer_state=getattr(self, "_optimizer_state", None),
            config=config_dict,
            epoch=getattr(self, "_start_epoch", 0),
            path=path,
            scheduler_state=getattr(self, "_scheduler_state", None),
            scaler_state=getattr(self, "_scaler_state", None),
        )

    @classmethod
    def load(cls, path: str | Path) -> TorchRegressorModel:
        """Load model from checkpoint.

        Restores the training state only. The module itself is left unbuilt,
        so subclasses must override this to reconstruct their architecture and
        load the weights into it before the instance can predict.
        """
        checkpoint = load_torch_checkpoint(path)
        instance = cls.__new__(cls)
        instance._device = resolve_device("auto")
        instance._start_epoch = checkpoint.get("epoch", 0)
        instance._optimizer_state = checkpoint.get("optimizer_state_dict")
        instance._scheduler_state = checkpoint.get("scheduler_state_dict")
        instance._scaler_state = checkpoint.get("scaler_state_dict")
        instance._best_metric = None
        instance._best_epoch = None
        instance._dataloader_settings = None
        return instance
