"""Visual-embedding precompute pipeline (backbone, sharded storage, extraction).

This package turns sky JPEGs into precomputed visual embeddings that the
training stack consumes through :class:`allsky.data.datasets.MultimodalEmbeddingDataset`.

Three concerns, three modules:

- :mod:`~allsky.embeddings.backbone` — the :class:`VisualBackbone` protocol, the
  DINOv2 (``torch.hub``) backbone pinned to a fixed revision, and a torch-free
  :class:`FakeBackbone` used by every test.
- :mod:`~allsky.embeddings.storage` — safetensors shards + a parquet index +
  ``embeddings.meta.json``, plus :class:`SafetensorsEmbeddingReader` (satisfies
  the :class:`allsky.data.datasets.EmbeddingReader` protocol) and
  :func:`validate_embeddings`.
- :mod:`~allsky.embeddings.extract` — :func:`extract_embeddings`, a resumable,
  batched, atomically-written extraction loop.

All ``torch``/``torch.hub`` imports are lazy: importing this package never pulls
torch, mirroring the ``allsky.dataset`` / ``allsky.video`` contract.
"""

from __future__ import annotations

from allsky.embeddings.backbone import (
    AVAILABLE_BACKBONES,
    DINOV2_MODEL,
    DINOV2_REPO,
    DINOV2_REVISION,
    DinoV2Backbone,
    FakeBackbone,
    VisualBackbone,
    build_backbone,
)
from allsky.embeddings.extract import extract_embeddings
from allsky.embeddings.storage import (
    EMBEDDINGS_TENSOR_KEY,
    META_FILENAME,
    EmbeddingValidationReport,
    SafetensorsEmbeddingReader,
    load_shard,
    read_index,
    read_meta,
    save_shard,
    shard_path,
    validate_embeddings,
    write_index,
    write_meta,
)

__all__ = [
    "AVAILABLE_BACKBONES",
    "DINOV2_MODEL",
    "DINOV2_REPO",
    "DINOV2_REVISION",
    "EMBEDDINGS_TENSOR_KEY",
    "META_FILENAME",
    "DinoV2Backbone",
    "EmbeddingValidationReport",
    "FakeBackbone",
    "SafetensorsEmbeddingReader",
    "VisualBackbone",
    "build_backbone",
    "extract_embeddings",
    "load_shard",
    "read_index",
    "read_meta",
    "save_shard",
    "shard_path",
    "validate_embeddings",
    "write_index",
    "write_meta",
]
