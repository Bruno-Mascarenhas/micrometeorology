"""On-disk layout for precomputed embeddings: safetensors shards + parquet index.

An embeddings directory holds three kinds of artifact:

- ``embeddings-{i:05d}.safetensors`` shards — each a single fp16 tensor under
  the key :data:`EMBEDDINGS_TENSOR_KEY` of shape ``(N_i, dim)``;
- ``index.parquet`` — one row per sample: ``sample_id`` (string), ``shard``
  (int, which shard file) and ``row`` (int, which row within it);
- ``embeddings.meta.json`` — provenance: backbone, revision, pooling, dim,
  transform description, config hash, sample count and storage dtype.

Everything is written **atomically** (temp file in the same directory + an
``os.replace``), so a crashed run never leaves a half-written shard or index.

:class:`SafetensorsEmbeddingReader` reads embeddings back one ``sample_id`` at a
time — satisfying the :class:`allsky.data.datasets.EmbeddingReader` protocol —
with a small LRU of open shards so sequential access touches each shard once.

This module is deliberately ``torch``-free: safetensors is read/written through
its numpy API, so importing it (and the reader) never pulls a heavy framework.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.atomic import atomic_write

logger = logging.getLogger(__name__)

__all__ = [
    "EMBEDDINGS_TENSOR_KEY",
    "INDEX_FILENAME",
    "META_FILENAME",
    "EmbeddingValidationReport",
    "SafetensorsEmbeddingReader",
    "load_shard",
    "read_index",
    "read_meta",
    "save_shard",
    "shard_filename",
    "shard_path",
    "validate_embeddings",
    "write_index",
    "write_meta",
]

#: Tensor key inside every safetensors shard.
EMBEDDINGS_TENSOR_KEY = "embeddings"
#: Parquet index filename inside an embeddings directory.
INDEX_FILENAME = "index.parquet"
#: Provenance sidecar filename inside an embeddings directory.
META_FILENAME = "embeddings.meta.json"


def shard_filename(shard_index: int) -> str:
    """Filename for shard *shard_index* (``embeddings-{i:05d}.safetensors``)."""
    return f"embeddings-{shard_index:05d}.safetensors"


def shard_path(embeddings_dir: str | Path, shard_index: int) -> Path:
    """Full path to shard *shard_index* inside *embeddings_dir*."""
    return Path(embeddings_dir) / shard_filename(shard_index)


def save_shard(path: str | Path, embeddings: np.ndarray) -> Path:
    """Atomically write *embeddings* as an fp16 safetensors shard.

    The array is cast to ``float16`` and stored contiguously under
    :data:`EMBEDDINGS_TENSOR_KEY`.  A partially written shard is impossible: the
    file appears in place only after a successful ``os.replace``.
    """
    from safetensors.numpy import save_file

    arr = np.ascontiguousarray(np.asarray(embeddings), dtype=np.float16)
    if arr.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, dim), got shape {arr.shape}")
    out = Path(path)
    atomic_write(out, lambda tmp: save_file({EMBEDDINGS_TENSOR_KEY: arr}, str(tmp)))
    return out


def load_shard(path: str | Path) -> np.ndarray:
    """Load the fp16 ``(N, dim)`` embedding matrix from a safetensors shard."""
    from safetensors.numpy import load_file

    return load_file(str(path))[EMBEDDINGS_TENSOR_KEY]


def write_index(embeddings_dir: str | Path, index: pd.DataFrame) -> Path:
    """Atomically write the parquet index into *embeddings_dir*."""
    out = Path(embeddings_dir) / INDEX_FILENAME
    atomic_write(out, lambda tmp: index.to_parquet(tmp, index=False))
    return out


def read_index(embeddings_dir: str | Path) -> pd.DataFrame | None:
    """Read the parquet index, or ``None`` when it does not exist yet."""
    path = Path(embeddings_dir) / INDEX_FILENAME
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_meta(embeddings_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Atomically write the ``embeddings.meta.json`` provenance sidecar."""
    out = Path(embeddings_dir) / META_FILENAME

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2, ensure_ascii=False, default=str)

    atomic_write(out, _write)
    return out


def read_meta(embeddings_dir: str | Path) -> dict[str, Any]:
    """Read the ``embeddings.meta.json`` provenance sidecar."""
    path = Path(embeddings_dir) / META_FILENAME
    with open(path, encoding="utf-8") as handle:
        loaded: dict[str, Any] = json.load(handle)
    return loaded


@dataclass(frozen=True)
class EmbeddingValidationReport:
    """Coverage report of an embedding index against a manifest.

    Attributes
    ----------
    missing:
        Manifest ``sample_id`` values with no embedding (index order preserved,
        de-duplicated).
    duplicate:
        ``sample_id`` values appearing more than once in the index (sorted).
    """

    missing: list[str]
    duplicate: list[str]

    @property
    def ok(self) -> bool:
        """True when nothing is missing and nothing is duplicated."""
        return not self.missing and not self.duplicate

    def raise_if_failed(self) -> None:
        """Raise :class:`ValueError` describing the first problem, if any."""
        if self.ok:
            return
        problems = []
        if self.missing:
            problems.append(f"{len(self.missing)} sample_id(s) missing embeddings: {self.missing}")
        if self.duplicate:
            problems.append(f"{len(self.duplicate)} duplicate sample_id(s): {self.duplicate}")
        raise ValueError("; ".join(problems))


def validate_embeddings(
    index: pd.DataFrame,
    manifest_sample_ids: Sequence[str],
) -> EmbeddingValidationReport:
    """Check an embedding *index* covers *manifest_sample_ids* exactly once.

    Parameters
    ----------
    index:
        Parquet index DataFrame (must have a ``sample_id`` column).
    manifest_sample_ids:
        The ``sample_id`` values the manifest expects embeddings for.

    Returns
    -------
    EmbeddingValidationReport
        ``missing`` sample ids (in manifest, absent from index) and
        ``duplicate`` sample ids (indexed more than once).
    """
    index_ids = [str(s) for s in index["sample_id"]]
    counts = Counter(index_ids)
    duplicate = sorted(sid for sid, n in counts.items() if n > 1)
    index_set = set(index_ids)
    missing: list[str] = []
    seen: set[str] = set()
    for raw in manifest_sample_ids:
        sid = str(raw)
        if sid not in index_set and sid not in seen:
            missing.append(sid)
            seen.add(sid)
    return EmbeddingValidationReport(missing=missing, duplicate=duplicate)


@dataclass(frozen=True, slots=True)
class _EmbeddingLocation:
    """Named position of one embedding inside the sharded store."""

    shard_index: int
    row_index: int


class SafetensorsEmbeddingReader:
    """``sample_id -> (dim,) float32`` reader over safetensors shards.

    Implements the :class:`allsky.data.datasets.EmbeddingReader` protocol
    (callable + ``dim``).  Two access modes:

    - **LRU** (default) — shards are memory-loaded on demand and kept in a small
      LRU (default 4), so a *sequential* pass over samples grouped by shard reads
      each shard file exactly once.  Under **shuffled** training the working set
      spans every shard, so an LRU of 4 thrashes (a year of 2048-row shards is
      hundreds of shards, giving a ~1% hit rate).
    - **Preload** (``preload=True``) — every shard is loaded once at construction
      into a single contiguous ``(N, dim)`` fp32 array with a ``sample_id -> row``
      map (the resident size is logged).  ``__call__`` is then a pure array index
      with no per-sample file I/O; this is what the training/eval engine uses by
      default (see ``data.embeddings_preload``).  The resident array is marked
      **read-only**, so a returned vector is a zero-copy but immutable view and
      cannot be used to mutate the shared store.  Keep the LRU path for ad-hoc
      access to stores that do not fit in memory.

    Parameters
    ----------
    embeddings_dir:
        Directory holding the shards, ``index.parquet`` and
        ``embeddings.meta.json``.
    cache_size:
        Maximum number of shard arrays held open in the LRU (>= 1); ignored when
        ``preload`` is True.
    preload:
        Load all shards into one resident array up front (see above).

    Raises
    ------
    FileNotFoundError
        If the index parquet is absent.
    """

    def __init__(
        self, embeddings_dir: str | Path, *, cache_size: int = 4, preload: bool = False
    ) -> None:
        if cache_size < 1:
            raise ValueError(f"cache_size must be >= 1, got {cache_size}")
        self._embeddings_dir = Path(embeddings_dir)
        embedding_index = read_index(self._embeddings_dir)
        if embedding_index is None:
            raise FileNotFoundError(
                f"no embedding index ({INDEX_FILENAME}) found in {self._embeddings_dir}"
            )
        self._location_by_sample_id: dict[str, _EmbeddingLocation] = {
            str(sample_id): _EmbeddingLocation(
                shard_index=int(shard_index), row_index=int(row_index)
            )
            for sample_id, shard_index, row_index in zip(
                embedding_index["sample_id"],
                embedding_index["shard"],
                embedding_index["row"],
                strict=True,
            )
        }
        self.meta = read_meta(self._embeddings_dir)
        self._embedding_dim = int(self.meta["dim"])
        self._shard_cache_size = cache_size
        self._shard_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        #: True when all shards are resident in ``self._preloaded_embeddings``.
        self.preloaded = False
        self._preloaded_embeddings: np.ndarray | None = None
        self._preloaded_row_by_sample_id: dict[str, int] = {}
        if preload:
            self._preload_all()

    @property
    def dim(self) -> int:
        """Embedding dimension (from ``embeddings.meta.json``)."""
        return self._embedding_dim

    def sample_ids(self) -> list[str]:
        """All ``sample_id`` values the reader can serve (index order)."""
        return list(self._location_by_sample_id)

    def _preload_all(self) -> None:
        """Load every shard once into one contiguous, read-only ``(N, dim)`` fp32 array.

        The resident matrix is marked read-only (``setflags(write=False)``) once
        populated: ``__call__`` returns zero-copy row *views* into it, so making
        the base immutable stops a caller mutating the shared store through a
        returned vector (the views inherit the read-only flag).
        """
        embedding_count = len(self._location_by_sample_id)
        preloaded_embeddings = np.empty((embedding_count, self._embedding_dim), dtype=np.float32)
        preloaded_row_by_sample_id: dict[str, int] = {}
        sample_ids_by_shard: dict[int, list[str]] = {}
        for sample_id, location in self._location_by_sample_id.items():
            sample_ids_by_shard.setdefault(location.shard_index, []).append(sample_id)

        preloaded_row_index = 0
        for shard_index in sorted(sample_ids_by_shard):
            shard_embeddings = load_shard(shard_path(self._embeddings_dir, shard_index))
            for sample_id in sample_ids_by_shard[shard_index]:
                location = self._location_by_sample_id[sample_id]
                preloaded_embeddings[preloaded_row_index] = shard_embeddings[location.row_index]
                preloaded_row_by_sample_id[sample_id] = preloaded_row_index
                preloaded_row_index += 1
        preloaded_embeddings.setflags(write=False)
        self._preloaded_embeddings = preloaded_embeddings
        self._preloaded_row_by_sample_id = preloaded_row_by_sample_id
        self.preloaded = True
        logger.info(
            "preloaded %d embedding(s) into a resident %d x %d fp32 array (%.1f MiB) from %s",
            embedding_count,
            embedding_count,
            self._embedding_dim,
            preloaded_embeddings.nbytes / (1024 * 1024),
            self._embeddings_dir,
        )

    def _load_cached_shard(self, shard_index: int) -> np.ndarray:
        """Load one shard, retaining it in the bounded least-recently-used cache."""
        cached_shard_embeddings = self._shard_cache.get(shard_index)
        if cached_shard_embeddings is not None:
            self._shard_cache.move_to_end(shard_index)
            return cached_shard_embeddings
        shard_embeddings = load_shard(shard_path(self._embeddings_dir, shard_index))
        self._shard_cache[shard_index] = shard_embeddings
        self._shard_cache.move_to_end(shard_index)
        while len(self._shard_cache) > self._shard_cache_size:
            self._shard_cache.popitem(last=False)
        return shard_embeddings

    def __call__(self, sample_id: str) -> np.ndarray:
        """Return the ``(dim,) float32`` embedding for *sample_id*.

        Raises
        ------
        KeyError
            If *sample_id* has no embedding in the index (message names it).
        """
        normalized_sample_id = str(sample_id)
        if self._preloaded_embeddings is not None:
            preloaded_row_index = self._preloaded_row_by_sample_id.get(normalized_sample_id)
            if preloaded_row_index is None:
                raise KeyError(
                    f"sample_id {sample_id!r} not found in embedding index at "
                    f"{self._embeddings_dir}"
                )
            # Zero-copy view into the resident store; the store is read-only
            # (see _preload_all) so this view is immutable and cannot corrupt it.
            embedding: np.ndarray = self._preloaded_embeddings[preloaded_row_index]
            return embedding
        location = self._location_by_sample_id.get(normalized_sample_id)
        if location is None:
            raise KeyError(
                f"sample_id {sample_id!r} not found in embedding index at {self._embeddings_dir}"
            )
        shard_embeddings = self._load_cached_shard(location.shard_index)
        return np.asarray(shard_embeddings[location.row_index], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._location_by_sample_id)
