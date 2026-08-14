"""Runtime preparation before importing PyTorch.

Everything here has to run before the first ``import torch`` of the process, so
the package's own ``__init__`` calls :func:`configure_torch_runtime` at import
time and code that needs the module goes through :func:`preload_torch`.
"""

import os
import sys

_CONFIGURED = False
_DLL_HANDLES: list[object] = []


def configure_torch_runtime() -> None:
    """Prepare Windows/conda DLL paths and conservative torch defaults.

    Idempotent and a no-op off ``win32``: Linux and macOS resolve torch's shared
    libraries through the normal loader path and need nothing set up.

    On Windows under a conda prefix, ``Library\\bin`` is added to the DLL search
    path so torch finds the MKL/OpenMP libraries conda installs there. Two
    environment defaults go with it, both set only if the caller has not already
    chosen: ``KMP_DUPLICATE_LIB_OK`` tolerates the duplicate OpenMP runtime that
    conda layout produces, which otherwise aborts the process on import, and
    ``TORCHDYNAMO_DISABLE`` keeps ``torch.compile`` out of the way on the
    platform where it is least dependable.

    The returned DLL-directory handles are kept in a module-level list; dropping
    them would remove the search path again.
    """
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
    """Import torch after runtime preparation and return the module.

    The single supported way to reach torch from a module that must not import
    it at load time (torch is an optional extra and costs seconds to import).

    Returns
    -------
    object
        The ``torch`` module, typed loosely so importing this helper does not
        require torch to be installed.
    """
    configure_torch_runtime()
    import torch

    return torch
