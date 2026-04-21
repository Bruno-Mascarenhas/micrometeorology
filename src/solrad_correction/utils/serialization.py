"""Model serialization utilities — dispatch to joblib or torch."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def save_sklearn_model(model: object, path: str | Path) -> None:
    """Save a scikit-learn model via joblib."""
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("Saved sklearn model: %s", p)


def load_sklearn_model(path: str | Path) -> object:
    """Load a scikit-learn model via joblib."""
    import joblib

    return joblib.load(path)


def save_torch_checkpoint(
    model_state: dict,
    optimizer_state: dict | None,
    config: dict | None,
    epoch: int,
    path: str | Path,
) -> None:
    """Save a PyTorch checkpoint."""
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model_state,
        "epoch": epoch,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, p)
    logger.info("Saved checkpoint: %s (epoch %d)", p, epoch)


def load_torch_checkpoint(path: str | Path) -> dict:
    """Load a PyTorch checkpoint securely."""
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore
