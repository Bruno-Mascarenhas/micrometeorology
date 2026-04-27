"""Runtime preparation before importing PyTorch."""

from __future__ import annotations

import os
import sys

_CONFIGURED = False
_DLL_HANDLES: list[object] = []


def configure_torch_runtime() -> None:
    """Prepare Windows/conda DLL paths and conservative torch defaults."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    if sys.platform != "win32":
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_bin = os.path.join(conda_prefix, "Library", "bin")
    if os.path.isdir(lib_bin):
        _DLL_HANDLES.append(os.add_dll_directory(lib_bin))


def preload_torch() -> object:
    """Import torch after runtime preparation and return the module."""
    configure_torch_runtime()
    import torch

    return torch
