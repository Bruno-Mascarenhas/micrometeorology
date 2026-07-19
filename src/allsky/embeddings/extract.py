"""Resumable, batched, atomically-written embedding extraction.

:func:`extract_embeddings` iterates a v2 manifest, encodes each frame's visual
embedding with a :class:`~allsky.embeddings.backbone.VisualBackbone` and writes
the result as safetensors shards plus a parquet index and a provenance meta
sidecar (see :mod:`allsky.embeddings.storage`).

Guarantees
----------
- **Resumable** — the index is the source of truth: on resume every
  ``sample_id`` already present in ``index.parquet`` is skipped, so a rerun does
  no duplicate work.  A shard is written *and* indexed together atomically, so a
  crash mid-run leaves a consistent (possibly shorter) index and no orphan rows.
- **Atomic** — shards, the index and the meta sidecar are each written to a
  temp file and ``os.replace``-d into place; a partial shard is never visible.
- **Single-process** — the backbone (and any model download) is created once by
  the caller; this loop never forks workers, so a hub model is fetched at most
  once.

``torch`` is only reached transitively through ``backbone.encode``; importing
this module never pulls it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from allsky.data.contracts import resolve
from allsky.embeddings.storage import (
    META_FILENAME,
    read_index,
    read_meta,
    save_shard,
    shard_path,
    write_index,
    write_meta,
)

if TYPE_CHECKING:
    import pandas as pd

    from allsky.embeddings.backbone import VisualBackbone

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]


def _load_uint8(path: Path) -> np.ndarray:
    """Load an image as a ``uint8`` HWC RGB array (grayscale -> 3-channel)."""
    import imageio.v3 as iio

    image = np.asarray(iio.imread(path))
    if image.ndim == 2:  # pragma: no cover - grayscale safety net
        image = np.stack([image] * 3, axis=-1)
    return image.astype(np.uint8, copy=False)


def _encode_batch(backbone: VisualBackbone, images: list[np.ndarray]) -> np.ndarray:
    """Transform + encode one batch of frames to an ``(B, dim)`` fp32 array."""
    batch = backbone.transform(images)
    encoded = backbone.encode(batch)
    if hasattr(encoded, "detach"):  # torch.Tensor
        encoded = encoded.detach().cpu().numpy()
    return np.asarray(encoded, dtype=np.float32)


def extract_embeddings(
    manifest_df: pd.DataFrame,
    backbone: VisualBackbone,
    out_dir: str | Path,
    *,
    data_root: str | Path,
    batch_size: int = 32,
    device: str | None = None,
    shard_size: int = 1024,
    resume: bool = True,
    dry_run: bool = False,
    config_sha256: str | None = None,
) -> dict[str, Any]:
    """Extract visual embeddings for every manifest sample into sharded storage.

    Parameters
    ----------
    manifest_df:
        v2 manifest with ``sample_id`` and ``image_path`` (relative POSIX)
        columns.
    backbone:
        A :class:`~allsky.embeddings.backbone.VisualBackbone`.
    out_dir:
        Output embeddings directory (shards + index + meta).
    data_root:
        Root the manifest's relative ``image_path`` values resolve against.
        (Required to load frames; not part of the manifest, which stores paths
        relative to this root.)
    batch_size:
        Frames encoded per backbone call (>= 1).
    device:
        Recorded in the summary/meta for provenance; the backbone owns actual
        device placement.
    shard_size:
        Rows per safetensors shard (>= 1); the final shard may be shorter.
    resume:
        When True, skip ``sample_id`` values already present in the index.
    dry_run:
        When True, compute and log the plan but write nothing (no directory,
        shards, index or meta are created).
    config_sha256:
        Optional content hash of the embeddings config, stored in the meta.

    Returns
    -------
    dict
        Summary: ``out_dir``, ``backbone``, ``revision``, ``pooling``, ``dim``,
        ``dtype``, ``device``, ``total``, ``skipped``, ``encoded``,
        ``shards_written``, ``resume`` and ``dry_run``.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    for column in ("sample_id", "image_path"):
        if column not in manifest_df.columns:
            raise ValueError(f"manifest is missing required column {column!r}")

    out = Path(out_dir)
    pooling = getattr(backbone, "pooling", "n/a")
    dtype = "fp16"  # storage dtype (safetensors shards are always fp16)

    # Resume must not silently mix incompatible embeddings into one store: if a
    # prior meta exists, the incoming backbone/config must match it exactly.
    if resume:
        _check_resume_compatible(out, backbone, pooling, config_sha256)

    # Resume bookkeeping: the index is the source of truth for done work.
    existing_index = read_index(out) if resume else None
    done_ids: set[str] = set()
    next_shard = 0
    prior_rows = 0
    if existing_index is not None and len(existing_index) > 0:
        done_ids = {str(s) for s in existing_index["sample_id"]}
        next_shard = int(existing_index["shard"].max()) + 1
        prior_rows = len(existing_index)

    samples = [
        (str(sid), str(path))
        for sid, path in zip(manifest_df["sample_id"], manifest_df["image_path"], strict=True)
        if str(sid) not in done_ids
    ]
    total = len(manifest_df)
    skipped = total - len(samples)

    summary: dict[str, Any] = {
        "out_dir": str(out),
        "backbone": backbone.name,
        "revision": backbone.revision,
        "pooling": pooling,
        "dim": int(backbone.dim),
        "dtype": dtype,
        "device": device,
        "total": total,
        "skipped": skipped,
        "encoded": 0,
        "shards_written": 0,
        "resume": resume,
        "dry_run": dry_run,
    }

    if dry_run:
        planned_shards = -(-len(samples) // shard_size)  # ceil division
        logger.info(
            "extract_embeddings[dry-run]: %d sample(s) total, %d already done, "
            "%d to encode -> ~%d new shard(s); writing nothing",
            total,
            skipped,
            len(samples),
            planned_shards,
        )
        return summary

    if not samples:
        logger.info("extract_embeddings: all %d sample(s) already embedded; nothing to do", total)
        # Refresh the meta so provenance reflects this (possibly new) backbone/config.
        if existing_index is not None:
            _write_meta(out, backbone, pooling, dtype, config_sha256, prior_rows)
        return summary

    out.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, Any]] = []
    if existing_index is not None:
        index_rows = [
            {"sample_id": str(rec["sample_id"]), "shard": int(rec["shard"]), "row": int(rec["row"])}
            for rec in existing_index.to_dict("records")
        ]

    buffer: np.ndarray | None = None
    buffer_ids: list[str] = []
    encoded = 0
    shards_written = 0

    def flush(*, final: bool) -> None:
        """Emit full shards from the buffer (or the trailing partial when final)."""
        nonlocal buffer, buffer_ids, next_shard, shards_written
        while buffer is not None and (len(buffer) >= shard_size or (final and len(buffer) > 0)):
            take = min(shard_size, len(buffer))
            shard_emb = buffer[:take]
            shard_ids = buffer_ids[:take]
            path = shard_path(out, next_shard)
            save_shard(path, shard_emb)
            for row, sid in enumerate(shard_ids):
                index_rows.append({"sample_id": sid, "shard": next_shard, "row": row})
            # Index only becomes truth after both the shard and index land.
            _write_index_atomic(out, index_rows)
            logger.info("extract_embeddings: wrote shard %s (%d embeddings)", path.name, take)
            shards_written += 1
            next_shard += 1
            remainder = buffer[take:]
            buffer = remainder if len(remainder) > 0 else None
            buffer_ids = buffer_ids[take:]

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        images = [_load_uint8(resolve(path, data_root)) for _, path in batch]
        vectors = _encode_batch(backbone, images)
        if vectors.shape[1] != backbone.dim:
            raise ValueError(
                f"backbone {backbone.name!r} produced dim {vectors.shape[1]}, "
                f"expected {backbone.dim}"
            )
        buffer = vectors if buffer is None else np.vstack([buffer, vectors])
        buffer_ids.extend(sid for sid, _ in batch)
        encoded += len(batch)
        flush(final=False)

    flush(final=True)
    _write_meta(out, backbone, pooling, dtype, config_sha256, prior_rows + encoded)

    summary["encoded"] = encoded
    summary["shards_written"] = shards_written
    logger.info(
        "extract_embeddings: done (%d encoded, %d skipped, %d shard(s) written) -> %s",
        encoded,
        skipped,
        shards_written,
        out,
    )
    return summary


def _check_resume_compatible(
    out: Path,
    backbone: VisualBackbone,
    pooling: str,
    config_sha256: str | None,
) -> None:
    """Refuse to resume into a store built with a different backbone/config.

    When ``resume=True`` and an ``embeddings.meta.json`` already exists in *out*,
    the incoming ``backbone`` (name/revision/pooling/dim) and ``config_sha256``
    must match the recorded provenance exactly.  Any mismatch would silently mix
    embeddings from two different encoders into one index, so this raises a clear
    :class:`RuntimeError` instead.

    Raises
    ------
    RuntimeError
        If any of ``backbone``/``revision``/``pooling``/``dim``/``config_sha256``
        in the existing meta differs from the incoming values.
    """
    if not (out / META_FILENAME).exists():
        return
    prior = read_meta(out)
    incoming = {
        "backbone": backbone.name,
        "revision": backbone.revision,
        "pooling": pooling,
        "dim": int(backbone.dim),
        "config_sha256": config_sha256,
    }
    mismatches = [
        f"{key}: existing={prior.get(key)!r} incoming={value!r}"
        for key, value in incoming.items()
        if prior.get(key) != value
    ]
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(
            f"cannot resume embedding extraction into {out}: the existing "
            f"embeddings.meta.json is incompatible with the requested backbone/config "
            f"({joined}). Rerun with --no-resume (resume=False) to overwrite, or point "
            f"at a fresh output directory."
        )


def _write_index_atomic(out: Path, index_rows: list[dict[str, Any]]) -> None:
    """Write the full index parquet atomically from accumulated rows."""
    import pandas as pd

    frame = pd.DataFrame(index_rows, columns=["sample_id", "shard", "row"])
    frame = frame.astype({"sample_id": "string", "shard": "int64", "row": "int64"})
    write_index(out, frame)


def _write_meta(
    out: Path,
    backbone: VisualBackbone,
    pooling: str,
    dtype: str,
    config_sha256: str | None,
    count: int,
) -> None:
    """Write the provenance meta sidecar for the embeddings directory."""
    meta = {
        "backbone": backbone.name,
        "revision": backbone.revision,
        "pooling": pooling,
        "dim": int(backbone.dim),
        "transform": getattr(backbone, "transform_description", ""),
        "config_sha256": config_sha256,
        "count": count,
        "dtype": dtype,
    }
    write_meta(out, meta)
