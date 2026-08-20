"""Transformer model for time-series regression."""

import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from solrad_correction.config import ModelConfig
from solrad_correction.models.torch_base import TorchRegressorModel
from solrad_correction.utils.serialization import load_torch_checkpoint

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Alternating sine and cosine over geometrically spaced wavelengths, as in
    Vaswani et al. 2017, section 3.5: even channels carry
    ``sin(pos / 10000^(2i/d_model))`` and odd channels the matching cosine.

    Parameters
    ----------
    d_model:
        Embedding width the encoding is added to.
    max_len:
        Longest sequence the table covers. Sequences longer than this index
        past the end of the buffer, so it bounds ``sequence_length``.
    dropout:
        Dropout probability applied after the encoding is added.

    Attributes
    ----------
    pe:
        Registered (non-parameter) buffer of shape ``(1, max_len, d_model)``,
        ``float32``, broadcast over the batch when added to the input. The
        class-level annotation exists for the type checker:
        ``nn.Module.__getattr__`` is typed ``Tensor | Module``, so without it
        the slice in ``forward`` is "not indexable". Annotating without a value
        binds nothing at runtime, which leaves ``register_buffer`` the sole
        owner of the attribute.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> Any:
        """Add the positional encoding to the input and apply dropout.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, T, d_model)``, ``float32``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, T, d_model)``, ``float32``: the input plus
            the first ``T`` rows of the encoding table.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder for time-series regression.

    Structure::

        Input (seq_len, input_size)
            -> Linear projection -> d_model
            -> Positional encoding
            -> TransformerEncoder (N layers)
            -> Mean pooling over sequence
            -> Linear -> ReLU -> Linear -> output (scalar)

    Parameters
    ----------
    input_size:
        Number of features ``F`` per time step, projected to ``d_model``.
    d_model:
        Width of the encoder's embedding space; must be divisible by ``nhead``.
    nhead:
        Number of attention heads per encoder layer.
    num_encoder_layers:
        Number of stacked encoder layers.
    dim_feedforward:
        Width of the position-wise feedforward network inside each layer.
    dropout:
        Dropout probability, shared by the positional encoding, the encoder
        layers and the head.
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
        """Encode the window and regress from its mean-pooled representation.

        Every step of the window contributes equally to the pooled vector, in
        contrast with the LSTM, which regresses from the final step alone.

        Parameters
        ----------
        x:
            Tensor of shape ``(B, T, F)``, ``float32``, in the scaled feature
            space the model was fitted on.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, 1)``, ``float32``, in scaled target units.
        """
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
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
    def from_config(
        cls,
        config: ModelConfig,
        input_size: int,
        device: str | None = None,
    ) -> TransformerRegressor:
        """Create an untrained regressor from the ``tf_*`` fields of a config.

        ``input_size`` comes from the built dataset rather than the config: it
        is the number of feature columns that survived preprocessing.
        """
        return cls(
            input_size=input_size,
            d_model=config.tf_d_model,
            nhead=config.tf_nhead,
            num_encoder_layers=config.tf_num_encoder_layers,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
            device=device,
        )

    @classmethod
    def load(cls, path: str | Path) -> TransformerRegressor:
        """Rebuild a Transformer from a checkpoint, architecture included.

        The architecture arguments travel inside the checkpoint, so the module
        is reconstructed at the width it was trained at. Optimizer, scheduler
        and AMP scaler states are carried over as well, which is what lets the
        loaded model continue a run rather than only score one; the best-metric
        and early-stopping counters, in contrast, start empty here and are
        recovered from checkpoint metadata by the resume path in
        ``TorchRegressorModel``.

        Raises
        ------
        KeyError
            When the checkpoint carries no ``config`` section, or one missing an
            architecture argument. Falling back to defaults would rebuild the
            module at a width nobody chose — and ``dropout`` never reaches the
            ``state_dict``, so ``load_state_dict`` would not object either: the
            resumed run would simply carry a regularization the original did not.
        """
        checkpoint = load_torch_checkpoint(path)
        cfg = checkpoint["config"]

        instance = cls(
            input_size=cfg["input_size"],
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_encoder_layers=cfg["num_encoder_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        )
        instance._module.load_state_dict(checkpoint["model_state_dict"])
        instance._start_epoch = checkpoint.get("epoch", 0)
        instance._optimizer_state = checkpoint.get("optimizer_state_dict")
        instance._scheduler_state = checkpoint.get("scheduler_state_dict")
        instance._scaler_state = checkpoint.get("scaler_state_dict")
        instance._best_metric = None
        instance._best_epoch = None
        instance._dataloader_settings = None
        return instance

    def save(self, path: str | Path) -> None:
        """Save weights plus the architecture arguments ``load`` needs.

        The ``_config_kwargs`` captured at construction — not the experiment
        config — are what travel with the weights, so the checkpoint always
        describes the module that was actually built.
        """
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
