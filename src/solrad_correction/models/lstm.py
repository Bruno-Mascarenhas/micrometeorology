"""LSTM model for time-series regression."""

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from solrad_correction.config import ModelConfig
from solrad_correction.models.torch_base import TorchRegressorModel
from solrad_correction.utils.serialization import load_torch_checkpoint

logger = logging.getLogger(__name__)


class LSTMNet(nn.Module):
    """LSTM architecture for time-series regression.

    Structure::

        Input (seq_len, input_size)
            -> LSTM layers
            -> Last hidden state
            -> Linear -> ReLU -> Linear -> output (scalar)

    Parameters
    ----------
    input_size:
        Number of features ``F`` per time step.
    hidden_size:
        Width of the LSTM hidden state; the head halves it before the output.
    num_layers:
        Number of stacked LSTM layers.
    dropout:
        Dropout probability inside the head, and between the LSTM layers. Torch
        applies inter-layer dropout only when there is more than one layer, so
        it is passed as ``0.0`` for a single layer rather than letting torch
        warn about a setting it would ignore.
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
        """Run the encoder and regress from the final time step.

        Only the last step's hidden state reaches the head: the window is
        summarized by where it ends, which is also the row the target belongs
        to.

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
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.head(last_hidden)


class LSTMRegressor(TorchRegressorModel):
    """LSTM regressor with transfer learning support.

    Example::

        model = LSTMRegressor(input_size=10, hidden_size=64)
        model.fit(train_dataset, val_dataset, config=config)

        # Resume training through RuntimeConfig.resume:
        # runtime.resume = "output/experiments/lstm_v1/checkpoints/last.pt"
        # model2.fit(train_dataset, val_dataset, config=config, runtime=runtime)
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
    def from_config(
        cls,
        config: ModelConfig,
        input_size: int,
        device: str | None = None,
    ) -> LSTMRegressor:
        """Create an untrained regressor from the ``lstm_*`` fields of a config.

        ``input_size`` comes from the built dataset rather than the config: it
        is the number of feature columns that survived preprocessing.
        """
        return cls(
            input_size=input_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            device=device,
        )

    @classmethod
    def load(cls, path: str | Path) -> LSTMRegressor:
        """Rebuild an LSTM from a checkpoint, architecture included.

        The architecture arguments travel inside the checkpoint, so the module
        is reconstructed at the width it was trained at. Optimizer, scheduler
        and AMP scaler states are carried over as well, which is what lets the
        loaded model continue a run rather than only score one; the best-metric
        and early-stopping counters, in contrast, start empty here and are
        recovered from checkpoint metadata by the resume path in
        ``TorchRegressorModel``.
        """
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
