"""Model serialization utilities dispatching to joblib or torch."""

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

#: Attribute a ``torch.compile`` wrapper exposes its inner module under, and the
#: prefix that wrapper puts on every ``state_dict`` key. Both are torch's
#: convention rather than ours; five places used to know them independently.
COMPILED_ATTRIBUTE = "_orig_mod"
COMPILED_PREFIX = f"{COMPILED_ATTRIBUTE}."


class ModelIntegrityError(RuntimeError):
    """Raised when a pickled artifact fails its manifest checksum verification."""


def save_sklearn_model(model: object, path: str | Path) -> None:
    """Save a scikit-learn model via joblib, creating parent directories.

    The artifact is a pickle, so it is only as trustworthy as the checksum the
    experiment manifest records for it — :func:`load_sklearn_model` is what
    checks that on the way back in.
    """
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info("Saved sklearn model: %s", p)


MANIFEST_FILENAME = "manifest.json"
"""Name of the checksummed inventory an experiment directory carries.

The writer in :mod:`solrad_correction.experiments.artifacts` and the upward
walk here both address this file; a rename that reached only one would leave the
verifier reporting "no manifest.json found" for every run.
"""


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes — the one digest the manifest is built on."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256_file(path)


class ManifestLookup(NamedTuple):
    """Outcome of the upward walk for an experiment ``manifest.json``.

    ``found`` is the nearest manifest that parses to a JSON object.
    ``unusable`` is the nearest manifest file that exists but could not be
    parsed, with the reason: a truncated or unreadable checksum store must not
    be reported as "no manifest.json found".
    """

    found: tuple[Path, dict[str, Any]] | None
    unusable: tuple[Path, str] | None


def _find_manifest(path: Path) -> ManifestLookup:
    """Locate the nearest ancestor ``manifest.json`` and parse its contents.

    Walks upward from ``path``; the experiment manifest written by
    ``experiments.artifacts.write_manifest`` lives at the experiment root and
    covers every file beneath it. An unusable manifest is recorded and the walk
    continues — the walk reaches ``/``, so an unrelated or unreadable
    ``manifest.json`` in an outer directory must not mask a usable one.
    """
    unusable: tuple[Path, str] | None = None
    for parent in path.resolve().parents:
        candidate = parent / MANIFEST_FILENAME
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # json.JSONDecodeError is a ValueError; so is the UnicodeDecodeError
            # a manifest truncated mid multi-byte sequence raises.
            unusable = unusable or (candidate, f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(data, dict):
            return ManifestLookup(found=(candidate, data), unusable=unusable)
        unusable = unusable or (
            candidate,
            f"top-level JSON value is {type(data).__name__}, not an object",
        )
    return ManifestLookup(found=None, unusable=unusable)


def verify_pickle_integrity(path: str | Path) -> None:
    """Verify a pickled artifact against a reachable experiment manifest.

    When an ancestor ``manifest.json`` (see
    :func:`solrad_correction.experiments.artifacts.write_manifest`) records a
    sha256 for the artifact, the file is hashed and compared before it is
    unpickled; a mismatch raises :class:`ModelIntegrityError`. When no manifest
    covers the file, a warning is logged that an unverified pickle is being
    loaded — pickles execute arbitrary code on load, so the absence of a
    checksum is surfaced rather than hidden. A manifest that exists but cannot
    be parsed (a run killed mid-write leaves a truncated one) is reported as
    such instead of as a missing manifest.

    Parameters
    ----------
    path:
        Artifact about to be unpickled. Resolved before it is looked up in the
        manifest, so a relative or symlinked spelling matches the same entry.

    Raises
    ------
    ModelIntegrityError
        If the manifest records a sha256 for the artifact and the file does not
        hash to it.
    """
    p = Path(path)
    lookup = _find_manifest(p)
    if lookup.found is None:
        if lookup.unusable is not None:
            candidate, reason = lookup.unusable
            logger.warning(
                "Loading unverified pickle (manifest %s is present but unusable: %s): %s",
                candidate,
                reason,
                p,
            )
        else:
            logger.warning("Loading unverified pickle (no manifest.json found): %s", p)
        return

    manifest_path, data = lookup.found
    artifacts = data.get("artifacts", {})
    try:
        relative = p.resolve().relative_to(manifest_path.parent).as_posix()
    except ValueError:
        relative = None
    entry = artifacts.get(relative) if relative is not None else None
    expected = entry.get("sha256") if isinstance(entry, dict) else None
    if expected is None:
        logger.warning(
            "Loading unverified pickle (not covered by manifest %s): %s", manifest_path, p
        )
        return

    actual = _sha256_file(p)
    if actual != expected:
        raise ModelIntegrityError(
            f"Integrity check failed for {p}: manifest {manifest_path} records sha256 "
            f"{expected} but the file hashes to {actual}"
        )
    logger.debug("Verified pickle integrity via manifest %s: %s", manifest_path, p)


def load_sklearn_model(path: str | Path) -> object:
    """Load a scikit-learn model via joblib after verifying its integrity.

    The artifact is checked against a reachable experiment ``manifest.json``
    (raising :class:`ModelIntegrityError` on a checksum mismatch); when no
    manifest covers the file an unverified load is logged. See
    :func:`verify_pickle_integrity`.

    Raises
    ------
    ModelIntegrityError
        If the manifest's recorded sha256 does not match the file.
    """
    import joblib

    verify_pickle_integrity(path)
    return joblib.load(path)


def save_torch_checkpoint(
    model_state: dict,
    optimizer_state: dict | None,
    config: dict | None,
    epoch: int,
    path: str | Path,
    *,
    scheduler_state: dict | None = None,
    scaler_state: dict | None = None,
    metadata: dict | None = None,
) -> None:
    """Save a PyTorch checkpoint, creating parent directories.

    Only the pieces that exist are written, so a checkpoint from a run without a
    scheduler or without AMP carries no key for them and
    :func:`load_torch_checkpoint` reads it back unchanged.

    Parameters
    ----------
    model_state:
        ``state_dict`` of the module, written verbatim; any ``_orig_mod.`` key
        prefix a ``torch.compile`` wrapper left on it is normalized away by
        :func:`load_torch_checkpoint`, not here.
    optimizer_state, scheduler_state, scaler_state:
        ``state_dict`` of the optimizer, LR scheduler and AMP grad scaler;
        ``None`` omits the entry.
    config:
        Serialized experiment config, so the checkpoint says what produced it.
    epoch:
        Epoch counter stored with the weights; a resume restarts training from
        it, so it is a count of completed epochs rather than an index.
    path:
        Destination file.
    metadata:
        Extra values to carry along, typically the RNG snapshot from
        :func:`capture_rng_state` and the preprocessing fingerprint a resume is
        checked against. Everything in it must survive
        ``torch.load(..., weights_only=True)``.
    """
    import torch

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model_state,
        "epoch": epoch,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler_state_dict"] = scheduler_state
    if scaler_state is not None:
        checkpoint["scaler_state_dict"] = scaler_state
    if config is not None:
        checkpoint["config"] = config
    if metadata is not None:
        checkpoint["metadata"] = metadata
    torch.save(checkpoint, p)
    logger.info("Saved checkpoint: %s (epoch %d)", p, epoch)


def capture_rng_state() -> dict[str, Any]:
    """Snapshot the python / numpy / torch RNG states for a resumable checkpoint.

    Every value is a type ``torch.load(..., weights_only=True)`` accepts, which
    is what :func:`load_torch_checkpoint` uses: tensors, ints, and the tuple of
    ints Python's Mersenne Twister state reduces to. In particular numpy's state
    is stored as its ``MT19937`` key array (a uint32 tensor) plus ``pos``, NOT as
    the ``np.random.get_state()`` tuple — that tuple is refused by the
    restricted unpickler and would make every checkpoint unloadable.

    Returns
    -------
    dict
        Mapping consumed only by :func:`restore_rng_state`. The CUDA entry is
        present only when a device was visible at capture time, so a checkpoint
        written on GPU still restores on a CPU-only machine.
    """
    import random

    import numpy as np
    import torch

    python_version, python_keys, python_gauss = random.getstate()
    # legacy=False returns a dict of primitives + one array, which is both typed
    # and trivial to reduce to weights_only-safe values.
    numpy_state = np.random.get_state(legacy=False)  # noqa: NPY002 - global RNG snapshot
    numpy_keys = np.asarray(numpy_state["state"]["key"], dtype=np.uint32)
    state: dict[str, Any] = {
        "python_version": int(python_version),
        "python_keys": torch.tensor(python_keys, dtype=torch.int64),
        "python_gauss": None if python_gauss is None else float(python_gauss),
        "numpy_bit_generator": str(numpy_state["bit_generator"]),
        "numpy_keys": torch.from_numpy(numpy_keys.copy()),
        "numpy_pos": int(numpy_state["state"]["pos"]),
        "numpy_has_gauss": int(numpy_state.get("has_gauss", 0)),
        "numpy_cached_gaussian": float(numpy_state.get("gauss", 0.0)),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore the RNG states captured by :func:`capture_rng_state`.

    Silently tolerates a checkpoint written before RNG state was persisted, so
    an older ``last.pt`` still resumes (just not bit-reproducibly).

    Restoring rather than reseeding is what lets a resumed run continue the
    stream it was interrupted in: reseeding would replay the shuffle order and
    dropout masks of the first epochs.

    Parameters
    ----------
    state:
        A :func:`capture_rng_state` snapshot, or an empty mapping to leave the
        generators alone. The CUDA entry is applied only when a device is
        available now, so a GPU checkpoint restores on CPU without failing.
    """
    import random

    import numpy as np
    import torch

    if not state:
        return
    random.setstate(
        (
            int(state["python_version"]),
            tuple(int(key) for key in state["python_keys"].tolist()),
            state["python_gauss"],
        )
    )
    np.random.set_state(  # noqa: NPY002 - restore global RNG snapshot
        {
            "bit_generator": str(state["numpy_bit_generator"]),
            "state": {
                "key": state["numpy_keys"].numpy().astype(np.uint32, copy=False),
                "pos": int(state["numpy_pos"]),
            },
            "has_gauss": int(state["numpy_has_gauss"]),
            "gauss": float(state["numpy_cached_gaussian"]),
        }
    )
    torch.set_rng_state(state["torch"].to(dtype=torch.uint8))
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([tensor.to(dtype=torch.uint8) for tensor in cuda_state])


def _strip_compiled_prefix(state: dict) -> dict:
    """Strip ``torch.compile`` key prefixes from a model state_dict.

    Checkpoints written from a compiled module carry ``_orig_mod.``-prefixed
    keys that cannot be loaded into a plain module. New checkpoints are saved
    unwrapped; this keeps previously written ones loadable.
    """
    if not any(key.startswith(COMPILED_PREFIX) for key in state):
        return state
    logger.info("Normalizing torch.compile-prefixed state_dict keys")
    return {key.removeprefix(COMPILED_PREFIX): value for key, value in state.items()}


def unwrap_compiled(module: Any) -> Any:
    """Return the plain module underneath a ``torch.compile`` wrapper.

    ``OptimizedModule`` exposes the wrapped module as ``_orig_mod`` and shares
    its parameters, so persisting the unwrapped one keeps checkpoint keys free
    of :data:`COMPILED_PREFIX` and loadable into an uncompiled module. The
    attribute name is torch's, not ours, which is exactly why it should be
    written down once.
    """
    return getattr(module, COMPILED_ATTRIBUTE, module)


def load_torch_checkpoint(path: str | Path) -> dict:
    """Load a PyTorch checkpoint securely.

    Read with ``weights_only=True``, so the restricted unpickler admits only
    tensors and plain primitives and a tampered checkpoint cannot execute code
    on load — this is why :func:`capture_rng_state` stores the RNG snapshot in
    those types. Mapped to CPU, leaving placement to the caller's device config.

    Returns
    -------
    dict
        The saved mapping, with ``torch.compile``'s ``_orig_mod.`` key prefixes
        already stripped from ``model_state_dict`` so it loads into a plain
        module.
    """
    import torch

    checkpoint: dict = torch.load(path, map_location="cpu", weights_only=True)
    model_state = checkpoint.get("model_state_dict")
    if isinstance(model_state, dict):
        checkpoint["model_state_dict"] = _strip_compiled_prefix(model_state)
    return checkpoint
