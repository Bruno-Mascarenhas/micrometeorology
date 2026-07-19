"""Torch-free device resolution for the all-sky training stack.

:func:`resolve_device` maps the ``"auto"`` sentinel to the best available
backend (cuda -> mps -> cpu) and passes any explicit request through
unchanged. torch is imported lazily inside the function so this module — and
thus ``import allsky.training`` — stays importable in a torch-free environment.
"""

from __future__ import annotations


def resolve_device(requested: str = "auto") -> str:
    """Resolve ``"auto"`` to the best available device: cuda -> mps -> cpu."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
