"""Model serialization utilities dispatching to joblib or torch."""

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
    *,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
    metadata: dict | None = None,
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
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if scaler_state is not None:
        checkpoint["scaler_state_dict"] = scaler_state
    if config is not None:
        checkpoint["config"] = config
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, p)
    logger.info("Saved checkpoint: %s (epoch %d)", p, epoch)


def _strip_compiled_prefix(state: dict) -> dict:
    """Strip ``torch.compile`` key prefixes from a model state_dict.

    Checkpoints written from a compiled module carry ``_orig_mod.``-prefixed
    keys that cannot be loaded into a plain module. New checkpoints are saved
    unwrapped; this keeps previously written ones loadable.
    """
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state):
        return state
    logger.info("Normalizing torch.compile-prefixed state_dict keys")
    return {key.removeprefix(prefix): value for key, value in state.items()}


def load_torch_checkpoint(path: str | Path) -> dict:
    """Load a PyTorch checkpoint securely."""
    import torch

    checkpoint: dict = torch.load(path, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model_state_dict")
    if isinstance(model_state, dict):
        checkpoint["model_state_dict"] = _strip_compiled_prefix(model_state)
    return checkpoint
