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
crash never leaves a half-written checkpoint.  A checkpoint is **not** a trusted
local artifact: the documented Colab workflow trains on a third-party runtime and
copies the run directory to a shared Drive folder, from which ``allsky evaluate``
and ``allsky train --resume`` read it back.  :func:`load_checkpoint` therefore
reads under torch's restricted unpickler (``weights_only=True``) extended with
:func:`_safe_checkpoint_globals` — the non-tensor globals a real payload
legitimately contains — and only unpickles freely when the caller opts in with
``trust_pickle=True``.  ``torch.compile``'s ``_orig_mod.`` key prefixes are
stripped on load so a compiled-then-checkpointed model loads back into a plain
module.
"""

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from allsky.atomic import atomic_write
from allsky.provenance import code_version

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

    Returns
    -------
    dict[str, Any]
        ``{"python", "numpy", "torch"}`` plus ``"cuda"`` only when a CUDA device
        is present, matching the ``rng_state`` contract (``cuda`` key "when
        present").
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
    """Restore RNG generators from a :func:`capture_rng_state` snapshot.

    Parameters
    ----------
    state:
        The snapshot mapping.  Every key is optional: an absent generator is
        left untouched, and ``"cuda"`` is skipped when no CUDA device is present
        (so a GPU checkpoint resumes on CPU).
    """
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
    failure).

    Parameters
    ----------
    path:
        Destination file, typically ``<run_dir>/last.ckpt`` or ``best.ckpt``.
    model, optimizer, scheduler, scaler:
        Objects whose ``state_dict()`` is stored; *scheduler* and *scaler* may
        be ``None`` (stored as ``None``).
    epoch, global_step:
        Completed epochs and optimizer steps at the time of writing.
    best_metric:
        ``{name, value, epoch}`` of the best monitored value so far, so a
        resumed run never clobbers a better ``best.ckpt``.
    config:
        The full experiment config dump.
    normalizers:
        Train-split ``feature_normalizer`` and ``target_normalizers`` as
        JSON-able dicts — the scaling the stored weights were trained under.
    feature_columns, feature_groups:
        The ordered engineered feature names and their group membership.
    dataset_version, split_id, manifest_sha256:
        Dataset provenance, checked before a resume is allowed.
    backbone_info:
        Image-mode backbone identity (name / revision / pooling / dim / frozen);
        ``None`` in embedding mode.
    code_version_info:
        Package and git-commit provenance; captured on the spot when ``None``.
    rng_state:
        A :func:`capture_rng_state` snapshot; captured on the spot when ``None``.
    epochs_no_improve:
        Early-stopping patience counter, so a resumed run restores its exact
        position on the patience curve.

    Returns
    -------
    Path
        The written path.
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


def _safe_checkpoint_globals() -> list[Any]:
    """The non-tensor globals a :func:`save_checkpoint` payload legitimately contains.

    ``numpy`` entries cover the legacy MT19937 state array that
    :func:`capture_rng_state` snapshots.  ``datetime`` entries cover an unquoted
    ISO date in a free-form config section (``SchedulerConfig.params`` is
    ``dict[str, Any]``): ``yaml.safe_load`` turns it into a ``date``/``datetime``
    that reaches ``checkpoint["config"]`` through ``cfg.model_dump()``.

    ``numpy._core.multiarray._reconstruct`` is private and has moved once already
    (``np.core`` -> ``np._core``), so it is resolved defensively — a numpy that
    moves it again degrades to :func:`load_checkpoint`'s actionable error instead
    of an ``AttributeError`` on the load path.
    """
    import datetime

    allowed: list[Any] = [
        np.ndarray,
        np.dtype,
        np.dtypes.UInt32DType,
        datetime.date,
        datetime.datetime,
    ]
    numpy_core = getattr(np, "_core", None) or getattr(np, "core", None)
    reconstruct = getattr(getattr(numpy_core, "multiarray", None), "_reconstruct", None)
    if reconstruct is not None:
        allowed.append(reconstruct)
    return allowed


def load_checkpoint(
    path: str | Path, *, map_location: str = "cpu", trust_pickle: bool = False
) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Reads under torch's restricted unpickler (``weights_only=True``) extended with
    :func:`_safe_checkpoint_globals`, so a checkpoint that travelled through Colab
    or a shared Drive cannot execute code on load; a payload carrying anything
    outside the allowlist raises :class:`ValueError` without unpickling it.  Pass
    ``trust_pickle=True`` to fall back to the unrestricted reader for a file you
    produced yourself.  Any ``_orig_mod.`` compile prefixes are stripped from the
    model state so it loads into a plain module.

    Parameters
    ----------
    path:
        The checkpoint file.
    map_location:
        Device the tensors are mapped onto, as ``torch.load`` understands it.
    trust_pickle:
        Read with the unrestricted unpickler instead of the allowlist.

    Returns
    -------
    dict[str, Any]
        The payload :func:`save_checkpoint` wrote.

    Raises
    ------
    ValueError
        If the payload carries a Python object outside the allowlist; it is
        refused rather than unpickled.
    """
    import pickle

    import torch

    if trust_pickle:
        checkpoint: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=False)
    else:
        try:
            with torch.serialization.safe_globals(_safe_checkpoint_globals()):
                checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        except pickle.UnpicklingError as exc:
            raise ValueError(
                f"{path} carries a Python object outside the checkpoint allowlist and was "
                "NOT unpickled — this is what a poisoned checkpoint looks like. If the file "
                f"is one you produced yourself, re-read it with trust_pickle=True. Cause: {exc}"
            ) from exc
    model_state = checkpoint.get("model_state")
    if isinstance(model_state, dict):
        checkpoint["model_state"] = _strip_compiled_prefix(model_state)
    return checkpoint


def _atomic_torch_save(payload: dict[str, Any], out: Path) -> None:
    """``torch.save`` to a same-directory temp file, then ``os.replace`` onto *out*."""
    import torch

    atomic_write(out, lambda tmp: torch.save(payload, tmp))


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
