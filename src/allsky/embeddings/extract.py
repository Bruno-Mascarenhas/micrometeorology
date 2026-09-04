"""Resumable, batched, atomically-written embedding extraction.

:func:`extract_embeddings` iterates a v2 manifest, encodes each frame's visual
embedding with a :class:`~allsky.embeddings.backbone.VisualBackbone` and writes
the result as safetensors shards plus a parquet index and a provenance meta
sidecar (see :mod:`allsky.embeddings.storage`).

Guarantees
----------
- **Resumable** — the index is the source of truth: on resume every
  ``sample_id`` already present in the index (the consolidated ``index.parquet``
  **plus** any per-shard ``index.part-NNNNN.parquet`` files left by an
  interrupted run) is skipped, so a rerun does no duplicate work and re-extracts
  only the missing ids.
- **Incremental index** — each shard flush writes a small per-shard *part* file
  holding only that shard's rows (``O(shard_size)``), instead of rewriting the
  whole index every flush (which would be ``O(N^2 / shard_size)`` over a run); the
  parts are consolidated into a single ``index.parquet`` atomically at
  completion and then removed.  The final consolidated index equals the union of
  all parts (plus any prior consolidated rows).
- **Atomic + crash-consistent** — shards, index parts, the consolidated index
  and the meta sidecar are each written to a temp file and ``os.replace``-d into
  place.  A part is written only *after* its shard lands, so a crash never leaves
  a part referencing a missing shard; a crash before consolidation is recovered
  on the next resume by reading consolidated + parts.  The meta sidecar is
  written *before* the first shard (and a non-resume run drops the index it is
  about to invalidate before that), so every store on disk is self-describing
  and no crash state can be resumed into with a different backbone.
- **Single-process** — the backbone (and any model download) is created once by
  the caller; this loop never forks workers, so a hub model is fetched at most
  once.  Only the per-batch JPEG decode is threaded (``decode_workers``), which
  touches no model state.

``torch`` is only reached transitively through ``backbone.encode``; importing
this module never pulls it.
"""

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.data.contracts import resolve
from allsky.embeddings.backbone import VisualBackbone
from allsky.embeddings.storage import (
    INDEX_FILENAME,
    META_FILENAME,
    read_index,
    read_meta,
    save_shard,
    shard_path,
    write_index,
    write_meta,
)
from allsky.frame_pixels import decode_rgb
from labmim_core.atomic import atomic_write

logger = logging.getLogger(__name__)

__all__ = ["extract_embeddings"]

#: Glob matching the per-shard index parts written by :func:`_write_index_part`.
_INDEX_PART_GLOB = "index.part-*.parquet"


def _index_part_path(out: Path, shard_index: int) -> Path:
    """Path to the per-shard index part for *shard_index* inside *out*."""
    return out / f"index.part-{shard_index:05d}.parquet"


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
    pixel_config_sha256: str | None = None,
    decode_workers: int = 4,
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
    pixel_config_sha256:
        Optional digest of the config deciding which pixels are encoded (mask
        including its file's bytes, crop, resize, the video time fields), stored
        in the meta, so a later run can tell a store whose frames were shaped
        differently from one whose encoder merely moved.
    decode_workers:
        Threads used to decode each batch's JPEGs (>= 1, capped at the CPU
        count).  Decode order — and therefore every shard, row and index entry —
        is unchanged; only the wall clock of the decode step moves.

    Returns
    -------
    dict
        Summary: ``out_dir``, ``backbone``, ``revision``, ``pooling``, ``dim``,
        ``dtype``, ``device``, ``total``, ``skipped``, ``encoded``,
        ``shards_written``, ``resume`` and ``dry_run``.

    Raises
    ------
    ValueError
        If *batch_size*, *shard_size* or *decode_workers* is below 1, if the
        manifest lacks ``sample_id`` / ``image_path``, or if the backbone emits
        vectors of a width other than its declared ``dim``.
    RuntimeError
        On a resume into a store whose ``embeddings.meta.json`` describes a
        different backbone/config, or which has an index but no meta at all (see
        :func:`_check_resume_compatible`).
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    if decode_workers < 1:
        raise ValueError(f"decode_workers must be >= 1, got {decode_workers}")
    for column in ("sample_id", "image_path"):
        if column not in manifest_df.columns:
            raise ValueError(f"manifest is missing required column {column!r}")

    out = Path(out_dir)
    pooling = getattr(backbone, "pooling", "n/a")
    # Two different dtypes, recorded apart because they are read for different
    # things: the shards are always fp16 on disk, while the backbone may have
    # computed in fp32. Writing only one made the snapshot rebuild the encoder at
    # the STORAGE precision, which is not the computation the vectors came from.
    storage_dtype = "fp16"
    compute_dtype = str(getattr(backbone, "dtype", storage_dtype))
    dtype = storage_dtype
    decode_threads = min(decode_workers, os.cpu_count() or 1)

    # Resume must not silently mix incompatible embeddings into one store: if a
    # prior meta exists, the incoming backbone/config must match it exactly.
    if resume:
        _check_resume_compatible(out, backbone, pooling, config_sha256)

    # Resume bookkeeping: the index (consolidated + any un-consolidated parts from
    # an interrupted run) is the source of truth for done work.  A non-resume run
    # ignores it; its stale index/parts are dropped below, after the dry-run and
    # nothing-to-do early returns and before any shard is written.
    existing_index = _read_index_and_parts(out) if resume else None
    done_ids: set[str] = set()
    next_shard = 0
    prior_rows = 0
    # Rows carried forward for the final consolidation (seeded from existing work).
    index_rows: list[dict[str, Any]] = []
    if existing_index is not None and len(existing_index) > 0:
        done_ids = {str(s) for s in existing_index["sample_id"]}
        next_shard = int(existing_index["shard"].max()) + 1
        prior_rows = len(existing_index)
        index_rows = [
            {"sample_id": str(rec["sample_id"]), "shard": int(rec["shard"]), "row": int(rec["row"])}
            for rec in existing_index.to_dict("records")
        ]

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
        "storage_dtype": storage_dtype,
        "compute_dtype": compute_dtype,
        "device": device,
        "total": total,
        "skipped": skipped,
        "encoded": 0,
        "shards_written": 0,
        "resume": resume,
        "dry_run": dry_run,
    }

    if dry_run:
        planned_shards = math.ceil(len(samples) / shard_size)
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
        if existing_index is not None:
            # Consolidate any parts left by an interrupted prior run into
            # index.parquet, and refresh provenance for this backbone/config.
            _consolidate_index(out, index_rows)
            _write_meta(
                out,
                backbone,
                pooling,
                dtype,
                config_sha256,
                prior_rows,
                pixel_config_sha256=pixel_config_sha256,
            )
        return summary

    out.mkdir(parents=True, exist_ok=True)
    if not resume:
        # index.parquet describes the shard bytes this run is about to overwrite
        # from shard 0 onward, so it goes before the first flush: a crash then
        # leaves an index-less store that fails loudly, instead of an index that
        # silently maps sample_ids onto other samples' rows.
        _remove_index_parts(out)
        (out / INDEX_FILENAME).unlink(missing_ok=True)
    # Provenance before the first shard, not only at completion: any store with a
    # shard on disk is then self-describing, so a later resume cannot slip past
    # _check_resume_compatible (which short-circuits on a missing meta) and append
    # a different backbone's vectors onto these shards.  The final _write_meta
    # only corrects ``count``.
    _write_meta(
        out,
        backbone,
        pooling,
        dtype,
        config_sha256,
        prior_rows,
        pixel_config_sha256=pixel_config_sha256,
    )

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
            part_rows = [
                {"sample_id": sid, "shard": next_shard, "row": row}
                for row, sid in enumerate(shard_ids)
            ]
            index_rows.extend(part_rows)
            # Write only this shard's index part (O(shard_size)), atomically and
            # AFTER the shard lands, so the part never references a missing shard.
            _write_index_part(out, next_shard, part_rows)
            logger.info("extract_embeddings: wrote shard %s (%d embeddings)", path.name, take)
            shards_written += 1
            next_shard += 1
            remainder = buffer[take:]
            buffer = remainder if len(remainder) > 0 else None
            buffer_ids = buffer_ids[take:]

    # Pillow releases the GIL while decoding, so a small thread pool overlaps the
    # batch's JPEG decodes.  One pool for the whole run and one batch in flight:
    # ``Executor.map`` returns results in submission order, so shard/row assignment
    # (and therefore the index) is exactly as it is single-threaded.
    from PIL import Image

    Image.preinit()  # register the JPEG plugin here, not in N threads at once
    with ThreadPoolExecutor(max_workers=decode_threads) as decode_pool:
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            images = list(
                decode_pool.map(
                    lambda path: decode_rgb(resolve(path, data_root)),
                    [path for _, path in batch],
                )
            )
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
    # Consolidate all parts (+ prior rows) into a single index.parquet atomically,
    # then remove the parts. index.parquet is the source of truth for the reader.
    _consolidate_index(out, index_rows)
    _write_meta(
        out,
        backbone,
        pooling,
        dtype,
        config_sha256,
        prior_rows + encoded,
        pixel_config_sha256=pixel_config_sha256,
    )

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

    An indexed store with **no** provenance at all is refused too: it can only
    come from a version that wrote the meta at completion, so there is nothing to
    check the incoming backbone against.

    No mismatch is negotiable.  A store stamped by a superseded digest formula is
    refused like any other: what a resume has to establish is that the frames
    behind the stored vectors are the ones this config produces, and a store
    written before ``pixel_config_sha256`` was recorded holds nothing that could
    establish it — appending vectors of newly preprocessed frames to a store of
    the old ones, then restamping it, would leave no later run able to tell.

    Raises
    ------
    RuntimeError
        If any of ``backbone``/``revision``/``pooling``/``dim``/``config_sha256``
        in the existing meta differs from the incoming values, or the store has an
        index but no ``embeddings.meta.json``.
    """
    if not (out / META_FILENAME).exists():
        if read_index(out) is None and not _index_parts(out):
            return
        raise RuntimeError(
            f"cannot resume embedding extraction into {out}: it has an embedding "
            f"index but no {META_FILENAME} (an extraction interrupted by an older "
            f"version), so the backbone/pooling/dtype that produced those shards is "
            f"unknown. Drop in a matching {META_FILENAME}, or rerun with --no-resume "
            f"(resume=False) to re-extract from scratch."
        )
    prior = read_meta(out)
    incoming = {
        "backbone": backbone.name,
        "revision": backbone.revision,
        "pooling": pooling,
        "dim": int(backbone.dim),
        "config_sha256": config_sha256,
    }
    mismatched = [key for key, value in incoming.items() if prior.get(key) != value]
    if not mismatched:
        return
    joined = "; ".join(
        f"{key}: existing={prior.get(key)!r} incoming={incoming[key]!r}" for key in mismatched
    )
    raise RuntimeError(
        f"cannot resume embedding extraction into {out}: the existing "
        f"embeddings.meta.json is incompatible with the requested backbone/config "
        f"({joined}). Rerun with --no-resume (resume=False) to overwrite, or point "
        f"at a fresh output directory."
    )


def _index_frame(index_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the canonical-dtype index DataFrame from accumulated rows."""
    frame = pd.DataFrame(index_rows, columns=["sample_id", "shard", "row"])
    return frame.astype({"sample_id": "string", "shard": "int64", "row": "int64"})


def _write_index_part(out: Path, shard_index: int, part_rows: list[dict[str, Any]]) -> None:
    """Atomically write a per-shard index part (only *shard_index*'s rows)."""
    frame = _index_frame(part_rows)
    atomic_write(_index_part_path(out, shard_index), lambda tmp: frame.to_parquet(tmp, index=False))


def _index_parts(out: Path) -> list[Path]:
    """Sorted list of existing per-shard index part files in *out*."""
    return sorted(out.glob(_INDEX_PART_GLOB))


def _remove_index_parts(out: Path) -> None:
    """Delete every per-shard index part file in *out*."""
    for part in _index_parts(out):
        part.unlink(missing_ok=True)


def _read_index_and_parts(out: Path) -> pd.DataFrame | None:
    """Union the consolidated ``index.parquet`` with any un-consolidated parts.

    Returns the deduplicated (by ``sample_id``, consolidated rows winning) index,
    or ``None`` when neither a consolidated index nor any part exists.  This is the
    resume source of truth: a crash before consolidation still surfaces every id
    that has a written shard, so those ids are skipped and only the truly missing
    ones re-extract.
    """
    frames: list[pd.DataFrame] = []
    consolidated = read_index(out)
    if consolidated is not None:
        frames.append(consolidated)
    frames.extend(pd.read_parquet(part) for part in _index_parts(out))
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="sample_id", keep="first").reset_index(drop=True)
    return merged


def _consolidate_index(out: Path, index_rows: list[dict[str, Any]]) -> None:
    """Write the single consolidated ``index.parquet`` atomically, then drop parts.

    The consolidated index equals the union of all per-shard parts plus any prior
    consolidated rows (carried in *index_rows*).  Parts are removed only after the
    consolidated file lands, so an interrupted consolidation leaves the parts in
    place for the next resume.
    """
    write_index(out, _index_frame(index_rows))
    _remove_index_parts(out)


def _write_meta(
    out: Path,
    backbone: VisualBackbone,
    pooling: str,
    dtype: str,
    config_sha256: str | None,
    count: int,
    *,
    pixel_config_sha256: str | None,
) -> None:
    """Write the provenance meta sidecar for the embeddings directory."""
    meta = {
        "backbone": backbone.name,
        "revision": backbone.revision,
        "pooling": pooling,
        "dim": int(backbone.dim),
        "transform": getattr(backbone, "transform_description", ""),
        "config_sha256": config_sha256,
        "pixel_config_sha256": pixel_config_sha256,
        "count": count,
        # `dtype` is the STORAGE precision, kept under its original name for the
        # stores already on disk; `compute_dtype` is what the backbone ran at,
        # which a live prediction has to rebuild the encoder with.
        "dtype": dtype,
        "storage_dtype": dtype,
        "compute_dtype": str(getattr(backbone, "dtype", dtype)),
    }
    write_meta(out, meta)
