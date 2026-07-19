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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from allsky.config import ExperimentConfig
from allsky.data.datasets import EmbeddingReader

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

    Returns ``(manifest, meta)``; *meta* is an empty dict when the sidecar is
    absent, in which case a warning is logged because the provenance fields it
    carries (``manifest_sha256`` for the hash check, ``split_id``,
    ``dataset_version``) are then unavailable to callers.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest parquet not found: {manifest_path}")
    manifest = pd.read_parquet(manifest_path)
    meta_path = manifest_path.with_name(manifest_path.name + ".meta.json")
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        logger.warning(
            "no manifest meta sidecar at %s; provenance fields are null and the "
            "manifest-hash / split-id checks are skipped",
            meta_path,
        )
    return manifest, meta


def load_split(path: Path) -> Any:
    """Load the persisted day-split artifact from *path*."""
    from allsky.data.splits import load_split_artifact

    if not path.exists():
        raise FileNotFoundError(f"split artifact not found: {path}")
    return load_split_artifact(path)


def default_embedding_reader(cfg: ExperimentConfig, root: Path) -> EmbeddingReader:
    """Build the safetensors embedding reader from ``cfg.data.embeddings_dir``.

    Preloads every shard into one resident array by default (finding F7):
    shuffled training makes the shard LRU thrash, so the whole store is loaded
    once unless ``cfg.data.embeddings_preload`` is False.  The training engine
    and the evaluator share this loader so evaluation reads embeddings exactly
    as training did.
    """
    from allsky.embeddings.storage import SafetensorsEmbeddingReader

    if cfg.data.embeddings_dir is None:
        raise ValueError("input_mode='embedding' requires cfg.data.embeddings_dir")
    reader: EmbeddingReader = SafetensorsEmbeddingReader(
        resolve_against_root(cfg.data.embeddings_dir, root),
        preload=cfg.data.embeddings_preload,
    )
    return reader
