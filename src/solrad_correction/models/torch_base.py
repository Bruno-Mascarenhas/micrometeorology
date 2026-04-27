"""Base class for PyTorch-based regressors with transfer learning support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn

from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.utils.seeds import get_device
from solrad_correction.utils.serialization import load_torch_checkpoint, save_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.sequence import SequenceDataset, WindowedSequenceDataset

logger = logging.getLogger(__name__)


class TorchRegressorModel(BaseRegressorModel):
    """Base for PyTorch sequential models (LSTM, Transformer, etc.).

    Subclasses must:
    1. Set ``self._module`` (a ``nn.Module``) in ``__init__``.
    2. Override ``_build_module(**kwargs)`` to construct the architecture.
    3. Override ``name`` property.

    Supports transfer learning via ``pretrained_path`` in config.
    """

    _module: nn.Module
    _device: str
    _start_epoch: int  # for transfer learning: resume from this epoch

    def __init__(self, device: str | None = None) -> None:
        self._device = device or get_device()
        self._start_epoch = 0
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._scaler_state: dict[str, Any] | None = None
        logger.info("Device: %s", self._device)

    def _build_module(self, **kwargs: Any) -> nn.Module:
        """Construct the nn.Module. Subclasses must override this."""
        raise NotImplementedError

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained weights for transfer learning / resumed training."""
        checkpoint = load_torch_checkpoint(path)
        self._module.load_state_dict(checkpoint["model_state_dict"])
        self._start_epoch = checkpoint.get("epoch", 0)
        self._optimizer_state = checkpoint.get("optimizer_state_dict")
        self._scheduler_state = checkpoint.get("scheduler_state_dict")
        self._scaler_state = checkpoint.get("scaler_state_dict")
        logger.info("Loaded pretrained weights from %s (epoch %d)", path, self._start_epoch)

    def fit(
        self,
        train_data: SequenceDataset | WindowedSequenceDataset,
        val_data: SequenceDataset | WindowedSequenceDataset | None = None,
        config: ModelConfig | None = None,
    ) -> TorchRegressorModel:
        """Train using the standard training loop with progress display."""
        from solrad_correction.training.trainer import Trainer

        if config and config.pretrained_path:
            self._load_pretrained(config.pretrained_path)

        trainer = Trainer(
            model=self._module,
            device=self._device,
            config=config,
            start_epoch=self._start_epoch,
            optimizer_state=self._optimizer_state,
            scheduler_state=self._scheduler_state,
            scaler_state=self._scaler_state,
        )
        self._module, history = trainer.train(train_data, val_data)
        self._start_epoch = trainer.completed_epochs
        self._optimizer_state = trainer.optimizer_state
        self._scheduler_state = trainer.scheduler_state
        self._scaler_state = trainer.scaler_state
        self._history = history
        self._config = config
        return self

    @property
    def training_history(self) -> dict[str, list[float]]:
        """Training history from the latest fit call."""
        return getattr(self, "_history", {})

    def predict(self, data: SequenceDataset | WindowedSequenceDataset | np.ndarray) -> np.ndarray:
        """Generate predictions using a batched DataLoader to prevent OOM."""
        from torch.utils.data import DataLoader, Dataset, TensorDataset

        self._module.eval()
        self._module.to(self._device)

        dataset: Dataset
        if self._is_torch_dataset(data):
            dataset = cast("Dataset", data)
        else:
            x_input = torch.tensor(np.asarray(data), dtype=torch.float32)
            dataset = TensorDataset(x_input)

        # Batch size defaults to a reasonable number if not specified in config
        batch_size = getattr(self, "_config", None)
        bs = batch_size.batch_size if hasattr(batch_size, "batch_size") else 256  # type: ignore

        loader = DataLoader(dataset, batch_size=bs, shuffle=False)
        all_preds = []

        with torch.inference_mode():
            device_type = "cuda" if "cuda" in self._device else "cpu"
            for batch in loader:
                batch_x = batch[0] if isinstance(batch, list | tuple) else batch
                batch_x = batch_x.to(self._device, non_blocking=True)
                with torch.autocast(device_type=device_type, enabled="cuda" in self._device):
                    preds = self._module(batch_x)
                all_preds.append(preds.cpu().numpy().flatten())

        return np.concatenate(all_preds)

    @staticmethod
    def _is_torch_dataset(data: object) -> bool:
        from torch.utils.data import Dataset

        return isinstance(data, Dataset)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint (state_dict + config for transfer learning)."""
        import dataclasses

        config_dict = None
        if hasattr(self, "_config") and self._config is not None:
            config_dict = (
                dataclasses.asdict(self._config)  # type: ignore
                if dataclasses.is_dataclass(self._config)
                else self._config
            )

        save_torch_checkpoint(
            model_state=self._module.state_dict(),
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

        Subclasses should override to properly reconstruct the module.
        """
        checkpoint = load_torch_checkpoint(path)
        instance = cls.__new__(cls)
        instance._device = get_device()
        instance._start_epoch = checkpoint.get("epoch", 0)
        instance._optimizer_state = checkpoint.get("optimizer_state_dict")
        instance._scheduler_state = checkpoint.get("scheduler_state_dict")
        instance._scaler_state = checkpoint.get("scaler_state_dict")
        # Subclass must call _build_module and load_state_dict
        return instance
