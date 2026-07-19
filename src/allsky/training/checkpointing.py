"""Atomic checkpoint save/load for the multimodal training engine.

A checkpoint is a single ``torch.save`` payload carrying everything needed to
resume a run bit-for-bit and to reconstruct the exact feature/target contract
the model was trained against:

- model / optimizer / scheduler / GradScaler state dicts;
- ``epoch`` (number of completed epochs) and ``global_step`` (optimizer steps);
- ``best_metric`` ``{name, value, epoch}`` so a resumed run never clobbers a
  better ``best.ckpt`` (the resume-safe best seeding pattern from
  ``solrad_correction``);
- ``epochs_no_improve`` — the early-stopping patience counter, so a resumed run
  restores its exact position on the patience curve (optional; ``None`` on old
  checkpoints, from which the engine reconstructs a lower bound);
- the full experiment ``config`` dump;
- ``normalizers`` (train-split ``feature_normalizer`` + ``target_normalizers``
  as JSON-able dicts), the ordered ``feature_columns`` and ``feature_groups``;
- dataset provenance: ``dataset_version``, ``split_id``, ``manifest_sha256``;
- ``backbone`` info (image mode only), ``code_version`` (package + git commit)
  and ``rng_state`` (python / numpy / torch / cuda) for deterministic resume.

Every write is atomic (temp file in the same directory + ``os.replace``), so a
crash never leaves a half-written checkpoint.  :func:`load_checkpoint` reads with
``weights_only=False`` — these files are our own, locally written, trusted
artifacts that legitimately contain non-tensor Python objects (config dicts,
RNG state, numpy arrays); they are never fetched from an untrusted source.
``torch.compile``'s ``_orig_mod.`` key prefixes are stripped on load so a
compiled-then-checkpointed model loads back into a plain module.
"""

from __future__ import annotations

import os
import random
import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

#: ``torch.nn.Module`` / ``torch.optim.Optimizer`` at runtime. Kept as aliases so
#: importing this module stays torch-free (torch is imported lazily in the funcs).
type TorchModule = Any
type TorchOptimizer = Any

__all__ = [
    "BEST_CHECKPOINT",
    "LAST_CHECKPOINT",
    "capture_rng_state",
    "code_version",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
]

#: Canonical checkpoint filenames written under a run directory.
LAST_CHECKPOINT = "last.ckpt"
BEST_CHECKPOINT = "best.ckpt"

#: ``torch.compile`` state-dict key prefix stripped on load.
_COMPILE_PREFIX = "_orig_mod."


def capture_rng_state() -> dict[str, Any]:
    """Snapshot the python / numpy / torch (and cuda) RNG state.

    The cuda entry is included only when a CUDA device is present, matching the
    ``rng_state`` contract (``cuda`` key "when present").
    """
    import torch

    state: dict[str, Any] = {
        "python": random.getstate(),
        # Snapshot the *global* numpy RNG (what set_global_seed / np.random.seed
        # drive); the Generator API is a different, unshared stream here.
        "numpy": np.random.get_state(),  # noqa: NPY002 - global RNG snapshot for resume
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG generators from a :func:`capture_rng_state` snapshot."""
    import torch

    python_state = state.get("python")
    if python_state is not None:
        version, internal, gauss = python_state
        random.setstate((version, tuple(internal), gauss))
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)  # noqa: NPY002 - restore global RNG snapshot
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(_as_uint8_tensor(torch_state))
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_as_uint8_tensor(item) for item in cuda_state])


def save_checkpoint(
    path: str | Path,
    *,
    model: TorchModule,
    optimizer: TorchOptimizer,
    scheduler: Any | None,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_metric: Mapping[str, Any],
    config: Mapping[str, Any],
    normalizers: Mapping[str, Any],
    feature_columns: Sequence[str],
    feature_groups: Mapping[str, Sequence[str]],
    dataset_version: str | None,
    split_id: str | None,
    manifest_sha256: str | None,
    backbone_info: Mapping[str, Any] | None = None,
    code_version_info: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
    epochs_no_improve: int | None = None,
) -> Path:
    """Atomically write a full training checkpoint to *path*.

    All state dicts, provenance and RNG state are packed into one payload and
    saved via a same-directory temp file + ``os.replace`` (no partial file on
    failure).  ``rng_state`` / ``code_version_info`` are captured on the spot
    when not supplied.  Returns the written path.
    """
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "epochs_no_improve": (None if epochs_no_improve is None else int(epochs_no_improve)),
        "best_metric": dict(best_metric),
        "config": dict(config),
        "normalizers": dict(normalizers),
        "feature_columns": list(feature_columns),
        "feature_groups": {key: list(value) for key, value in feature_groups.items()},
        "dataset_version": dataset_version,
        "split_id": split_id,
        "manifest_sha256": manifest_sha256,
        "backbone": dict(backbone_info) if backbone_info is not None else None,
        "code_version": dict(code_version_info)
        if code_version_info is not None
        else code_version(),
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
    }
    out = Path(path)
    _atomic_torch_save(payload, out)
    return out


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Uses ``weights_only=False`` (trusted, locally written file — see the module
    docstring) and strips any ``_orig_mod.`` compile prefixes from the model
    state so it loads into a plain module.
    """
    import torch

    checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    model_state = checkpoint.get("model_state")
    if isinstance(model_state, dict):
        checkpoint["model_state"] = _strip_compiled_prefix(model_state)
    return checkpoint


def code_version() -> dict[str, str | None]:
    """Package version plus a best-effort git commit (reproducibility stamp)."""
    try:
        version: str | None = importlib_metadata.version("labmim-micrometeorology")
    except importlib_metadata.PackageNotFoundError:
        version = None
    return {"package_version": version, "git_commit": _git_commit()}


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _atomic_torch_save(payload: dict[str, Any], out: Path) -> None:
    """``torch.save`` to a same-directory temp file, then ``os.replace`` onto *out*."""
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    ok = False
    try:
        torch.save(payload, tmp)
        os.replace(tmp, out)
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)


def _strip_compiled_prefix(state: dict[str, Any]) -> dict[str, Any]:
    """Strip ``torch.compile`` ``_orig_mod.`` key prefixes from a state dict."""
    if not any(key.startswith(_COMPILE_PREFIX) for key in state):
        return state
    return {key.removeprefix(_COMPILE_PREFIX): value for key, value in state.items()}


def _as_uint8_tensor(value: Any) -> Any:
    """Coerce *value* to a CPU ``uint8`` tensor for ``torch.set_rng_state``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.uint8, device="cpu")
    return torch.as_tensor(value, dtype=torch.uint8)


def _git_commit() -> str | None:
    """Current git commit hash, or None when unavailable (best-effort)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - git resolved from PATH
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
