"""Base class for PyTorch-based regressors with transfer learning support."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.utils.seeds import get_device
from solrad_correction.utils.serialization import load_torch_checkpoint, save_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig
    from solrad_correction.datasets.sequence import SequenceDataset

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
        logger.info("Device: %s", self._device)

    def _build_module(self, **kwargs: Any) -> nn.Module:
        """Construct the nn.Module. Subclasses must override this."""
        raise NotImplementedError

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained weights for transfer learning / resumed training."""
        checkpoint = load_torch_checkpoint(path)
        self._module.load_state_dict(checkpoint["model_state_dict"])
        self._start_epoch = checkpoint.get("epoch", 0)
        logger.info("Loaded pretrained weights from %s (epoch %d)", path, self._start_epoch)

    def fit(
        self,
        train_data: SequenceDataset,
        val_data: SequenceDataset | None = None,
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
        )
        self._module, history = trainer.train(train_data, val_data)
        return self

    def predict(self, data: SequenceDataset | np.ndarray) -> np.ndarray:
        """Generate predictions in eval mode."""
        self._module.eval()
        self._module.to(self._device)

        if hasattr(data, "X"):
            x_input = data.X if isinstance(data.X, torch.Tensor) else torch.tensor(data.X, dtype=torch.float32)
        else:
            x_input = torch.tensor(np.asarray(data), dtype=torch.float32)

        with torch.no_grad():
            preds = self._module(x_input.to(self._device))

        return preds.cpu().numpy().flatten()

    def save(self, path: str | Path) -> None:
        """Save model checkpoint (state_dict + config for transfer learning)."""
        import dataclasses

        config_dict = None
        if hasattr(self, "_config") and self._config is not None:
            config_dict = dataclasses.asdict(self._config) if dataclasses.is_dataclass(self._config) else self._config

        save_torch_checkpoint(
            model_state=self._module.state_dict(),
            optimizer_state=None,
            config=config_dict,
            epoch=getattr(self, "_start_epoch", 0),
            path=path,
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
        # Subclass must call _build_module and load_state_dict
        return instance
