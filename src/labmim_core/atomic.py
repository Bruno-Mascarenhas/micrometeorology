"""Atomic file writes: temp file in the same directory + ``os.replace``.

Every artifact this project publishes goes through :func:`atomic_write` — the
all-sky parquet manifests and their meta sidecars, embedding shards, training
checkpoints, metrics CSV/JSON, evaluation reports and Colab bundles, and on the
micrometeorology side the site's JSON and the operational record.  The payload is written to a hidden temp file *in the
destination directory* (``.<name>.tmp-<pid>``) and then ``os.replace``-d onto
the final path, so a crash mid-write never leaves a half-written artifact in
place; the temp file is removed if the writer raises.

Same-directory placement is deliberate: ``os.replace`` is only atomic within a
single filesystem, so the temp file must never live in a system tempdir.

Pure stdlib, and it imports nothing from this project: both packages write
through it, so it has to sit under both. Callers that need torch — checkpoint
saving — import it lazily inside the writer callable.
"""

import json
import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

__all__ = ["atomic_write", "atomic_write_json", "atomic_write_strict_json"]


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
        # ``os.replace`` makes the DIRECTORY ENTRY swap atomic; it does not order
        # the payload's blocks ahead of the rename. Without these two fsyncs the
        # guarantee holds only for a process crash, not for a host reset or power
        # loss, where a first-ever write can come back zero-length or holed while
        # its meta sidecar records the checksum of the whole file.
        _fsync_path(tmp)
        os.replace(tmp, out)
        _fsync_directory(out.parent)
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)
    return out


def _fsync_path(path: Path) -> None:
    """Flush a file's own blocks to disk, best-effort."""
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    """Flush a directory entry to disk, best-effort.

    Not every platform allows opening a directory for this (Windows does not),
    and where it fails the rename is still atomic — only its durability across a
    host crash is weaker, which is exactly the pre-existing behaviour.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    """Atomically write *obj* to *path* as indented UTF-8 JSON, non-finite floats allowed.

    Uses ``indent=2, ensure_ascii=False, default=str`` — the canonical encoding
    for the pipeline's JSON sidecars (manifest/embedding meta, metrics history,
    report payloads).  ``default=str`` only stringifies objects the encoder
    cannot serialize at all; it is NOT a guard against non-finite floats, which
    this writer emits as the bare ``NaN``/``Infinity``/``-Infinity`` tokens of
    Python's JSON extension.  Those tokens are deliberate here: a diverged
    epoch's ``float("nan")`` loss is real telemetry that the training metrics
    history must keep.

    For anything whose bytes can reach the public site, use
    :func:`atomic_write_strict_json` instead — a browser's ``response.json()``
    rejects those tokens and loses the whole payload.
    """

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)

    return atomic_write(path, _write)


def _strict_json_default(value: Any) -> Any:
    unwrap = getattr(value, "item", None)
    if not callable(unwrap):
        return str(value)
    scalar = unwrap()
    if isinstance(scalar, float) and not math.isfinite(scalar):
        raise ValueError(f"Out of range float values are not JSON compliant: {scalar!r}")
    return scalar


def atomic_write_strict_json(path: str | Path, obj: Any) -> Path:
    """Atomically write *obj* to *path* as RFC-compliant indented UTF-8 JSON.

    Same encoding and signature as :func:`atomic_write_json` plus
    ``allow_nan=False``, so a non-finite float raises :class:`ValueError` and
    nothing is published: the destination keeps its previous contents and the
    temp file is removed.  A strict ``response.json()`` fails the ENTIRE
    document on a bare ``NaN``/``Infinity`` token — indistinguishable, to a
    visitor, from a page that was never deployed — so a missing measurement
    must reach this writer already encoded as ``None``.

    This is the ``indent=2`` writer, for the all-sky artifacts a person reads
    by eye. The site's own JSON has a DIFFERENT byte contract — compact
    separators, no indentation — and is written by
    :func:`micrometeorology.common.site_json.write_json`. The two are not
    interchangeable: swapping one for the other rewrites every byte of the
    published payload.

    A numpy scalar is unwrapped to its Python value first, so a
    ``numpy.float32`` NaN is refused like a Python one rather than quoted as
    the string ``"nan"``; every other unknown type is written as ``str``.
    """

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                obj,
                handle,
                indent=2,
                ensure_ascii=False,
                default=_strict_json_default,
                allow_nan=False,
            )

    return atomic_write(path, _write)
