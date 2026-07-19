"""Training subpackage for the all-sky pipeline.

The multi-task engine, losses and checkpointing modules live in
:mod:`allsky.training.engine`, :mod:`allsky.training.losses` and
:mod:`allsky.training.checkpointing`; the torch-free device resolver lives in
:mod:`allsky.training.device` and is re-exported here so
``from allsky.training import resolve_device`` keeps working for its importers
(the embedding backbone, the engine).

Importing this package does NOT pull torch: :func:`resolve_device` imports
torch lazily, and the torch-heavy names (``run_experiment``, ``MultitaskLoss``,
the checkpoint helpers) are resolved lazily through :func:`__getattr__`, so the
device helper remains usable in a torch-free environment.
"""

from __future__ import annotations

from typing import Any

from allsky.training.device import resolve_device

__all__ = [
    "MultitaskLoss",
    "capture_rng_state",
    "load_checkpoint",
    "resolve_device",
    "restore_rng_state",
    "run_experiment",
    "save_checkpoint",
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
