"""The collection gate's own contract: the list of torch-backed modules.

Four modules imported torch at module scope while absent from the tuple, which
on a dev install without the ``allsky`` extra is a collection ImportError rather
than the skip the gate promises. A hand-kept list drifts; this reads the files.
"""

import re
from pathlib import Path

from tests.allsky.conftest import _TORCH_BACKED

_MODULE_SCOPE_TORCH = re.compile(r"^(?:import torch|from torch)\b", re.MULTILINE)


def test_every_module_importing_torch_at_module_scope_is_gated():
    directory = Path(__file__).parent

    importers = {
        path.name
        for path in sorted(directory.glob("test_*.py"))
        if _MODULE_SCOPE_TORCH.search(path.read_text(encoding="utf-8"))
    }

    assert importers <= set(_TORCH_BACKED), sorted(importers - set(_TORCH_BACKED))


def test_every_gated_module_exists():
    """A renamed module left in the tuple silently stops running for anyone
    without the extra, which is the failure the gate cannot report itself.
    """
    directory = Path(__file__).parent

    assert [name for name in _TORCH_BACKED if not (directory / name).is_file()] == []
