"""Transformer model for time-series regression."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from solrad_correction.models.torch_base import TorchRegressorModel
from solrad_correction.utils.serialization import load_torch_checkpoint

if TYPE_CHECKING:
    from pathlib import Path

    from solrad_correction.config import ModelConfig

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> Any:
        """Add positional encoding to input.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.pe[:, : x.size(1)]  # type: ignore
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for time-series regression.

    Structure::

        Input (seq_len, input_size)
            → Linear projection → d_model
            → Positional encoding
            → TransformerEncoder (N layers)
            → Mean pooling over sequence
            → Linear → ReLU → Linear → output (scalar)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Mean pooling over the sequence dimension
        x = x.mean(dim=1)
        return self.head(x)


class TransformerRegressor(TorchRegressorModel):
    """Transformer regressor with transfer learning support.

    Example::

        model = TransformerRegressor(input_size=10, d_model=64, nhead=4)
        model.fit(train_dataset, val_dataset, config=config)
    """

    @property
    def name(self) -> str:
        return "Transformer"

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self._module = TimeSeriesTransformer(
            input_size=input_size,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(self._device)
        self._config_kwargs = {
            "input_size": input_size,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
        }

    @classmethod
    def from_config(cls, config: ModelConfig, input_size: int) -> TransformerRegressor:
        """Create from experiment config."""
        return cls(
            input_size=input_size,
            d_model=config.tf_d_model,
            nhead=config.tf_nhead,
            num_encoder_layers=config.tf_num_encoder_layers,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
        )

    @classmethod
    def load(cls, path: str | Path) -> TransformerRegressor:
        """Load Transformer from checkpoint."""
        checkpoint = load_torch_checkpoint(path)
        cfg = checkpoint.get("config", {})

        instance = cls(
            input_size=cfg.get("input_size", 1),
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_encoder_layers=cfg.get("num_encoder_layers", 2),
            dim_feedforward=cfg.get("dim_feedforward", 128),
            dropout=cfg.get("dropout", 0.1),
        )
        instance._module.load_state_dict(checkpoint["model_state_dict"])
        instance._start_epoch = checkpoint.get("epoch", 0)
        instance._optimizer_state = checkpoint.get("optimizer_state_dict")
        instance._scheduler_state = checkpoint.get("scheduler_state_dict")
        instance._scaler_state = checkpoint.get("scaler_state_dict")
        return instance

    def save(self, path: str | Path) -> None:
        """Save Transformer with architecture config."""
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
