"""LSTM model for time-series regression."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from solrad_correction.models.torch_base import TorchRegressorModel
from solrad_correction.utils.serialization import load_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """LSTM architecture for time-series regression.

    Structure::

        Input (seq_len, input_size)
            → LSTM layers
            → Last hidden state
            → Linear → ReLU → Linear → output (scalar)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Any:
        """Forward pass.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, input_size)``.

        Returns
        -------
        Tensor of shape ``(batch, 1)``.
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Take last time step
        return self.head(last_hidden)


class LSTMRegressor(TorchRegressorModel):
    """LSTM regressor with transfer learning support.

    Example::

        model = LSTMRegressor(input_size=10, hidden_size=64)
        model.fit(train_dataset, val_dataset, config=config)

        # Resume training with more epochs:
        config.pretrained_path = "output/experiments/lstm_v1/model.pt"
        config.max_epochs = 50
        model2 = LSTMRegressor(input_size=10, hidden_size=64)
        model2.fit(train_dataset, val_dataset, config=config)
    """

    @property
    def name(self) -> str:
        return "LSTM"

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self._module = LSTMNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self._device)
        self._config_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }

    @classmethod
    def from_config(cls, config: ModelConfig, input_size: int) -> LSTMRegressor:
        """Create from experiment config."""
        return cls(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
        )

    @classmethod
    def load(cls, path: str | Path) -> LSTMRegressor:
        """Load LSTM from checkpoint."""
        checkpoint = load_torch_checkpoint(path)
        cfg = checkpoint.get("config", {})

        instance = cls(
            input_size=cfg.get("input_size", 1),
            hidden_size=cfg.get("hidden_size", 64),
            num_layers=cfg.get("num_layers", 2),
            dropout=cfg.get("dropout", 0.1),
        )
        instance._module.load_state_dict(checkpoint["model_state_dict"])
        instance._start_epoch = checkpoint.get("epoch", 0)
        instance._optimizer_state = checkpoint.get("optimizer_state_dict")
        instance._scheduler_state = checkpoint.get("scheduler_state_dict")
        instance._scaler_state = checkpoint.get("scaler_state_dict")
        return instance

    def save(self, path: str | Path) -> None:
        """Save LSTM with architecture config for reconstruction."""
        from solrad_correction.utils.serialization import save_torch_checkpoint

        save_torch_checkpoint(
            model_state=self._module.state_dict(),
            optimizer_state=getattr(self, "_optimizer_state", None),
            config=self._config_kwargs,
            epoch=self._start_epoch,
            path=path,
            scheduler_state=getattr(self, "_scheduler_state", None),
            scaler_state=getattr(self, "_scaler_state", None),
        )
