"""Atomic file writes: temp file in the same directory + ``os.replace``.

Every artifact the allsky pipeline writes — parquet manifests and their meta
sidecars, embedding shards + index + meta, training checkpoints, metrics
CSV/JSON, evaluation reports and Colab bundles — is written through
:func:`atomic_write`.  The payload is written to a hidden temp file *in the
destination directory* (``.<name>.tmp-<pid>``) and then ``os.replace``-d onto
the final path, so a crash mid-write never leaves a half-written artifact in
place; the temp file is removed if the writer raises.

Same-directory placement is deliberate: ``os.replace`` is only atomic within a
single filesystem, so the temp file must never live in a system tempdir.

Pure stdlib: importing this module never pulls torch (callers that need torch —
e.g. checkpoint saving — import it lazily inside the writer callable).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["atomic_write", "atomic_write_json"]


def atomic_write(path: str | Path, writer: Callable[[Path], Any]) -> Path:
    """Atomically write *path* via *writer* (temp file + ``os.replace``).

    *writer* is a callable that receives the temp :class:`~pathlib.Path` and
    writes the payload to it (``lambda tmp: frame.to_parquet(tmp)``,
    ``lambda tmp: torch.save(payload, tmp)``, ...).  The temp file lives in the
    destination directory as ``.<name>.tmp-<pid>`` and is ``os.replace``-d onto
    *path* only after *writer* returns; if *writer* raises, the temp file is
    removed so a failed write leaves no debris.  Parent directories are created
    as needed.  Returns the destination path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    ok = False
    try:
        writer(tmp)
        os.replace(tmp, out)
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)
    return out


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    """Atomically write *obj* to *path* as indented UTF-8 JSON.

    Uses ``indent=2, ensure_ascii=False, default=str`` — the canonical encoding
    for the pipeline's JSON sidecars (manifest/embedding meta, metrics history,
    report payloads).
    """

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)

    return atomic_write(path, _write)
