"""Training subpackage for the all-sky pipeline.

The pre-refactor SkyFusionNet training loop lives verbatim in
:mod:`allsky.training.legacy`; the new multi-task engine, losses and
checkpointing modules land in :mod:`allsky.training.engine`,
:mod:`allsky.training.losses` and :mod:`allsky.training.checkpointing`. Every
public name from the legacy module is re-exported eagerly so
``from allsky.training import train`` (and ``split_days`` / ``resolve_device``)
keeps working for existing importers (``allsky.cli.legacy``, the dataset/CLI
tests).

Importing this package does NOT pull torch: the legacy re-exports keep their
heavy imports lazy, and the new torch-heavy names (``run_experiment``,
``MultitaskLoss``, the checkpoint helpers) are resolved lazily through
:func:`__getattr__`, so the pure :func:`split_days` / :func:`resolve_device`
helpers remain usable in a torch-free environment.
"""

from __future__ import annotations

from typing import Any

from allsky.training.legacy import logger, resolve_device, split_days, train

__all__ = [
    "MultitaskLoss",
    "capture_rng_state",
    "load_checkpoint",
    "logger",
    "resolve_device",
    "restore_rng_state",
    "run_experiment",
    "save_checkpoint",
    "split_days",
    "train",
]

#: Lazily resolved names -> the submodule that defines them (keeps torch out of
#: ``import allsky.training``; the submodule is imported only on first access).
_LAZY: dict[str, str] = {
    "run_experiment": "allsky.training.engine",
    "MultitaskLoss": "allsky.training.losses",
    "save_checkpoint": "allsky.training.checkpointing",
    "load_checkpoint": "allsky.training.checkpointing",
    "capture_rng_state": "allsky.training.checkpointing",
    "restore_rng_state": "allsky.training.checkpointing",
}


def __getattr__(name: str) -> Any:
    """Resolve the torch-heavy engine/loss/checkpoint names on first access."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
