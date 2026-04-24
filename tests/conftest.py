"""Root conftest -- shared fixtures and Windows DLL-path workaround.

On Windows + conda, ``torch`` may fail to load ``shm.dll`` when imported
for the first time inside pytest's test process due to DLL search path
differences.  Eagerly importing ``torch`` in ``pytest_configure``
(before test collection) avoids the issue.
"""

from __future__ import annotations

import contextlib
import os
import sys


def pytest_configure(_config):
    """Register extra DLL search paths and pre-load torch on Windows."""
    if sys.platform == "win32":
        # Prevent OMP: Error #15 (dual OpenMP runtime init)
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # Disable torch.compile's lazy JIT (requires cl.exe / MSVC Build
        # Tools which may not be installed).  This does not affect model
        # correctness, only performance optimization.
        os.environ["TORCHDYNAMO_DISABLE"] = "1"

        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            lib_bin = os.path.join(conda_prefix, "Library", "bin")
            if os.path.isdir(lib_bin):
                os.add_dll_directory(lib_bin)

        # Eagerly import torch so that its DLL loading happens before
        # pytest's collector changes the process DLL state.
        with contextlib.suppress(Exception):
            import torch  # noqa: F401
