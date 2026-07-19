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
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

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

#: Ordered columns of the parquet index.
INDEX_COLUMNS = ("sample_id", "shard", "row")


def shard_filename(shard_index: int) -> str:
    """Filename for shard *shard_index* (``embeddings-{i:05d}.safetensors``)."""
    return f"embeddings-{shard_index:05d}.safetensors"


def shard_path(embeddings_dir: str | Path, shard_index: int) -> Path:
    """Full path to shard *shard_index* inside *embeddings_dir*."""
    return Path(embeddings_dir) / shard_filename(shard_index)


def _atomic_replace(path: Path, write: Any) -> None:
    """Write via a same-directory temp file, then ``os.replace`` onto *path*.

    *write* is a callable taking the temp :class:`~pathlib.Path`.  The temp file
    is removed if *write* raises, so a failed write never leaves debris.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    ok = False
    try:
        write(tmp)
        os.replace(tmp, path)
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)


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
    _atomic_replace(out, lambda tmp: save_file({EMBEDDINGS_TENSOR_KEY: arr}, str(tmp)))
    return out


def load_shard(path: str | Path) -> np.ndarray:
    """Load the fp16 ``(N, dim)`` embedding matrix from a safetensors shard."""
    from safetensors.numpy import load_file

    return load_file(str(path))[EMBEDDINGS_TENSOR_KEY]


def write_index(embeddings_dir: str | Path, index: pd.DataFrame) -> Path:
    """Atomically write the parquet index into *embeddings_dir*."""
    out = Path(embeddings_dir) / INDEX_FILENAME
    _atomic_replace(out, lambda tmp: index.to_parquet(tmp, index=False))
    return out


def read_index(embeddings_dir: str | Path) -> pd.DataFrame | None:
    """Read the parquet index, or ``None`` when it does not exist yet."""
    import pandas as pd

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

    _atomic_replace(out, _write)
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
      default (see ``data.embeddings_preload``).  Keep the LRU path for ad-hoc
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
        self._dir = Path(embeddings_dir)
        index = read_index(self._dir)
        if index is None:
            raise FileNotFoundError(f"no embedding index ({INDEX_FILENAME}) found in {self._dir}")
        self._locations: dict[str, tuple[int, int]] = {
            str(sid): (int(shard), int(row))
            for sid, shard, row in zip(
                index["sample_id"], index["shard"], index["row"], strict=True
            )
        }
        self.meta = read_meta(self._dir)
        self._dim = int(self.meta["dim"])
        self._cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        #: True when all shards are resident in ``self._matrix`` (preload mode).
        self.preloaded = False
        self._matrix: np.ndarray | None = None
        self._row_of: dict[str, int] = {}
        if preload:
            self._preload_all()

    @property
    def dim(self) -> int:
        """Embedding dimension (from ``embeddings.meta.json``)."""
        return self._dim

    def sample_ids(self) -> list[str]:
        """All ``sample_id`` values the reader can serve (index order)."""
        return list(self._locations)

    def _preload_all(self) -> None:
        """Load every shard once into one contiguous ``(N, dim)`` fp32 array."""
        n = len(self._locations)
        matrix = np.empty((n, self._dim), dtype=np.float32)
        row_of: dict[str, int] = {}
        by_shard: dict[int, list[tuple[str, int]]] = {}
        for sid, (shard, row) in self._locations.items():
            by_shard.setdefault(shard, []).append((sid, row))
        cursor = 0
        for shard in sorted(by_shard):
            arr = load_shard(shard_path(self._dir, shard))
            for sid, row in by_shard[shard]:
                matrix[cursor] = arr[row]
                row_of[sid] = cursor
                cursor += 1
        self._matrix = matrix
        self._row_of = row_of
        self.preloaded = True
        logger.info(
            "preloaded %d embedding(s) into a resident %d x %d fp32 array (%.1f MiB) from %s",
            n,
            n,
            self._dim,
            matrix.nbytes / (1024 * 1024),
            self._dir,
        )

    def _shard(self, shard_index: int) -> np.ndarray:
        cached = self._cache.get(shard_index)
        if cached is not None:
            self._cache.move_to_end(shard_index)
            return cached
        arr = load_shard(shard_path(self._dir, shard_index))
        self._cache[shard_index] = arr
        self._cache.move_to_end(shard_index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def __call__(self, sample_id: str) -> np.ndarray:
        """Return the ``(dim,) float32`` embedding for *sample_id*.

        Raises
        ------
        KeyError
            If *sample_id* has no embedding in the index (message names it).
        """
        key = str(sample_id)
        if self._matrix is not None:
            row = self._row_of.get(key)
            if row is None:
                raise KeyError(
                    f"sample_id {sample_id!r} not found in embedding index at {self._dir}"
                )
            resident: np.ndarray = self._matrix[row]
            return resident
        location = self._locations.get(key)
        if location is None:
            raise KeyError(f"sample_id {sample_id!r} not found in embedding index at {self._dir}")
        shard_index, row = location
        return np.asarray(self._shard(shard_index)[row], dtype=np.float32)

    def __len__(self) -> int:
        return len(self._locations)
