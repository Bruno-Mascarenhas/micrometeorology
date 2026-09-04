"""Shared artifact loaders for the training engine and the evaluator.

Both :func:`allsky.training.run_experiment` and
:func:`allsky.evaluation.evaluate_checkpoint` resolve the same v2 artifacts
against a data root before running: the manifest parquet + its meta sidecar,
the persisted day split, and (in embedding mode) the safetensors embedding
reader.  These loaders are the single implementation of that resolution so the
training and evaluation entry points cannot drift.

Importing this module is torch-free: the safetensors reader (and torch itself)
are imported lazily inside :func:`default_embedding_reader`.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from allsky.config import ExperimentConfig, manifest_meta_path
from allsky.data.datasets import EmbeddingReader
from allsky.data.splits import DaySplit, load_split_artifact
from allsky.provenance import content_sha256

logger = logging.getLogger(__name__)

__all__ = [
    "default_embedding_reader",
    "load_manifest",
    "load_split",
    "resolve_against_root",
]


def resolve_against_root(path: str | Path, root: Path) -> Path:
    """Resolve *path* against *root* unless it is already absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def load_manifest(manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read the manifest parquet and its ``<name>.meta.json`` sidecar (if any).

    The sidecar's ``manifest_sha256`` is re-derived from the parquet that was
    actually read, here and nowhere else.  The two files are written by two
    independent atomic writes with no transaction linking them, so a crash
    between them leaves a new parquet beside a stale sidecar; and both hash
    guards downstream (the evaluator's and the resume provenance check) compare
    a checkpoint's stored string against the sidecar's stored string, never
    against the bytes.  Verifying once at the door makes both of them mean what
    they say.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The manifest and its sidecar meta.  *meta* is an empty dict when the
        sidecar is absent, in which case a warning is logged: the provenance
        fields it carries (``manifest_sha256`` for the hash check, ``split_id``,
        ``dataset_version``) are then unavailable to callers.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    ValueError
        If the sidecar's ``manifest_sha256`` does not describe the parquet.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest parquet not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    meta_path = manifest_meta_path(manifest_path)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        recorded = meta.get("manifest_sha256")
        if recorded is not None:
            actual = content_sha256(manifest)
            if actual != recorded:
                raise ValueError(
                    f"{meta_path.name} records manifest_sha256={recorded[:12]} but "
                    f"{manifest_path.name} hashes to {actual[:12]}: the two were written "
                    "by separate atomic writes and one of them is from another build"
                )
    else:
        logger.warning(
            "no manifest meta sidecar at %s; provenance fields are null and the "
            "manifest-hash / split-id checks are skipped",
            meta_path,
        )
    return manifest, meta


def load_split(path: Path) -> DaySplit:
    """Load the persisted day-split artifact from *path*.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the artifact's stored ``split_id`` does not match its assignment, or
        the assignment leaks a day across splits (see
        :func:`allsky.data.splits.load_split_artifact`).
    """
    if not path.exists():
        raise FileNotFoundError(f"split artifact not found: {path}")
    return load_split_artifact(path)


def default_embedding_reader(cfg: ExperimentConfig, root: Path) -> EmbeddingReader:
    """Build the safetensors embedding reader from ``cfg.data.embeddings_dir``.

    Preloads every shard into one resident array by default: shuffled training
    makes the shard LRU thrash, so the whole store is loaded once unless
    ``cfg.data.embeddings_preload`` is False.  The training engine and the
    evaluator share this loader so evaluation reads embeddings exactly as
    training did.

    Raises
    ------
    ValueError
        If ``cfg.data.embeddings_dir`` is unset, which embedding-mode runs
        require.
    """
    from allsky.embeddings.storage import SafetensorsEmbeddingReader

    if cfg.data.embeddings_dir is None:
        raise ValueError("input_mode='embedding' requires cfg.data.embeddings_dir")
    store = resolve_against_root(cfg.data.embeddings_dir, root)
    # Nothing here can refuse a store encoded by another backbone — the run's own
    # config names no encoder, only a directory — so the identity the store
    # records is at least written into the run's log, which is the only place a
    # reader of the results can later see which vectors the model was fitted on.
    _log_store_identity(store)
    reader: EmbeddingReader = SafetensorsEmbeddingReader(
        store,
        preload=cfg.data.embeddings_preload,
    )
    return reader


def _log_store_identity(store: Path) -> None:
    """Log the encoding recipe *store* records, or that it records none."""
    from allsky.embeddings.storage import META_FILENAME

    meta_path = store / META_FILENAME
    if not meta_path.is_file():
        logger.warning(
            "embedding store %s has no %s: its encoder is unrecorded", store, META_FILENAME
        )
        return
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("embedding store meta %s is unreadable (%s)", meta_path, exc)
        return
    recorded = {
        key: meta.get(key)
        for key in ("backbone", "revision", "pooling", "dim", "dtype", "pixel_config_sha256")
    }
    logger.info("embedding store %s encoded by %s", store, recorded)
