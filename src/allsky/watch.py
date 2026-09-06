"""Polls the camera's live frame and scores each datalogger block as it closes.

Timestamps here are **naive local time** on the camera's own clock, as in
:mod:`allsky.snapshot`: a frame is filed under the block its overlay stamp
rounds up to, and a block is closed on that same clock.

The state is the disk. ``<out_dir>/frames/`` holds every captured frame and
its sidecar, ``<out_dir>/blocks/`` one ``<YYYYMMDD-HHMM>.prediction.json`` per
scored block or a ``.skipped.json`` naming why it was not, so a restarted
watch rebuilds its index from the sidecars and never rewrites a block.
"""

import http.client
import json
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.archive import ArchiveError
from allsky.config import SiteConfig
from allsky.snapshot import (
    Snapshot,
    SolarElevationBelowFloorError,
    _site_now,
    block_checkpoint_window_minutes,
    block_end_of,
    predict_block,
    predict_snapshot,
)
from labmim_core.atomic import atomic_write_strict_json

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCKS_SUBDIR",
    "FRAMES_SUBDIR",
    "ensemble_prediction",
    "frames_on_disk",
    "run_watch",
]

FRAMES_SUBDIR = "frames"
BLOCKS_SUBDIR = "blocks"
BLOCK_STEM_FORMAT = "%Y%m%d-%H%M"
PREDICTION_SUFFIX = ".prediction.json"
SKIPPED_SUFFIX = ".skipped.json"
CAPTURE_ERRORS = (ArchiveError, ValueError, OSError, http.client.HTTPException)
PREDICTION_ERRORS = (ValueError, KeyError, RuntimeError, OSError)
STALE_CAPTURE_SOURCE = "local-clock"
UNFILED_CAPTURE_SOURCES = (STALE_CAPTURE_SOURCE, "server-last-modified")


def frames_on_disk(frames_dir: str | Path) -> list[Snapshot]:
    """Rebuild the frame index from the sidecars under *frames_dir*, in time order.

    A sidecar that cannot be read, names a missing image or was not stamped
    from the frame's own overlay is logged and left out rather than failing
    the watch: the directory is the watch's own history, and one bad entry
    must not stop it from resuming. Training filed every frame under the
    block its overlay stamp rounds up to, so a frame stamped from the server's
    ``Last-Modified`` (a cluster whose backends disagree by more than half a
    block) or from the host clock (no capture time at all) cannot be filed
    the same way.

    Parameters
    ----------
    frames_dir:
        Directory :func:`allsky.snapshot.capture_snapshot` wrote into.

    Returns
    -------
    list of Snapshot
        One per readable sidecar, sorted by naive local ``captured_at``.
    """
    directory = Path(frames_dir)
    records: list[Snapshot] = []
    for sidecar in sorted(directory.glob("*.json")):
        if sidecar.name.endswith(PREDICTION_SUFFIX):
            continue
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            image = directory / str(meta["image"])
            captured = pd.Timestamp(meta["captured_at"])
            source = str(meta["captured_at_source"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("skipping unreadable frame sidecar %s: %s", sidecar, exc)
            continue
        if source in UNFILED_CAPTURE_SOURCES:
            logger.warning("skipping %s: its capture time came from the %s", sidecar.name, source)
            continue
        if not image.is_file():
            logger.warning("skipping %s: its image %s is gone", sidecar.name, image.name)
            continue
        records.append(Snapshot(image_path=image, metadata_path=sidecar, captured_at=captured))
    return sorted(records, key=lambda record: record.captured_at)


def _capture_source(snapshot: Snapshot) -> str:
    meta = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    return str(meta["captured_at_source"])


def ensemble_prediction(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Average the block predictions of several checkpoints into one record.

    ``dhi``, ``kindex`` and ``cloud_fraction`` are the means over the members
    that carry the head; ``sky_probabilities`` is the per-class mean over the
    members that carry one and ``sky_class`` its argmax, first name on a tie.
    The ``block`` record is the first member's — every member was fed the
    same frames — and each member's own ``predictions`` and ``model`` are kept
    under ``members`` and ``models``.

    Parameters
    ----------
    results:
        The :func:`~allsky.snapshot.predict_block` records, one per checkpoint.

    Returns
    -------
    dict
        ``predictions``, ``block``, ``members`` and ``models``.

    Raises
    ------
    ValueError
        If *results* is empty.
    """
    if not results:
        raise ValueError("an ensemble needs at least one member prediction")
    predictions: dict[str, Any] = {}
    for name in ("dhi", "kindex", "cloud_fraction"):
        values = [r["predictions"][name] for r in results if name in r["predictions"]]
        if values:
            predictions[name] = float(np.mean(values))
    probability_maps = [
        r["predictions"]["sky_probabilities"]
        for r in results
        if "sky_probabilities" in r["predictions"]
    ]
    if probability_maps:
        names = list(probability_maps[0])
        mean = {name: float(np.mean([p[name] for p in probability_maps])) for name in names}
        predictions["sky_probabilities"] = mean
        predictions["sky_class"] = names[int(np.argmax([mean[name] for name in names]))]
    return {
        "predictions": predictions,
        "block": results[0]["block"],
        "members": [r["predictions"] for r in results],
        "models": [r["model"] for r in results],
    }


def _score_frame(
    snapshot: Snapshot,
    checkpoint: Path,
    *,
    site: SiteConfig | None,
    device: str,
    trust_checkpoint: bool,
) -> None:
    try:
        prediction = predict_snapshot(
            snapshot.image_path,
            checkpoint,
            timestamp=snapshot.captured_at,
            site=site,
            device=device,
            trust_checkpoint=trust_checkpoint,
        )
        path = atomic_write_strict_json(
            snapshot.image_path.with_suffix(PREDICTION_SUFFIX), prediction
        )
    except PREDICTION_ERRORS as exc:
        logger.error("frame prediction failed for %s: %s", snapshot.image_path.name, exc)
        return
    logger.info("frame prediction %s: %s", path.name, prediction["predictions"])


def _score_block(
    members: Sequence[Snapshot],
    end: pd.Timestamp,
    checkpoints: Sequence[Path],
    *,
    min_solar_elevation_deg: float,
    site: SiteConfig | None,
    device: str,
    trust_checkpoint: bool,
    image_backbone_builder: Callable[[], Any] | None,
) -> dict[str, Any]:
    frames = [(snapshot.image_path, snapshot.captured_at) for snapshot in members]
    results = [
        predict_block(
            frames,
            checkpoint,
            min_solar_elevation_deg=min_solar_elevation_deg,
            block_end=end,
            site=site,
            device=device,
            trust_checkpoint=trust_checkpoint,
            image_backbone_builder=image_backbone_builder,
        )
        for checkpoint in checkpoints
    ]
    return results[0] if len(results) == 1 else ensemble_prediction(results)


def _block_record_paths(blocks_dir: Path, end: pd.Timestamp) -> tuple[Path, Path]:
    stem = f"{end:{BLOCK_STEM_FORMAT}}"
    return blocks_dir / f"{stem}{PREDICTION_SUFFIX}", blocks_dir / f"{stem}{SKIPPED_SUFFIX}"


def _ready_block_ends(
    first: pd.Timestamp,
    *,
    latest: pd.Timestamp,
    now: pd.Timestamp,
    block_minutes: float,
    grace_seconds: float,
) -> Iterator[tuple[pd.Timestamp, str]]:
    """Every block end from *first* on that is ready, with what closed it.

    Whenever a later block is ready every earlier one is too, so the walk
    stops at the first block still open; a block no frame fell in is yielded
    like any other, which is what turns a capture gap into a record instead
    of a missing file.
    """
    width = pd.Timedelta(minutes=block_minutes)
    grace = pd.Timedelta(seconds=grace_seconds)
    end = first
    while True:
        if latest > end:
            yield end, "later_frame"
        elif now >= end + grace:
            yield end, "grace"
        else:
            return
        end = end + width


def _close_ready_blocks(
    known: dict[pd.Timestamp, Snapshot],
    blocks_dir: Path,
    checkpoints: Sequence[Path],
    *,
    now: pd.Timestamp,
    block_minutes: float,
    min_frames: int,
    grace_seconds: float,
    min_solar_elevation_deg: float,
    site: SiteConfig | None,
    device: str,
    trust_checkpoint: bool,
    image_backbone_builder: Callable[[], Any] | None,
    closed_through: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Close every ready block from the earliest frame's or after *closed_through*.

    The walk resumes past the last block closed on the previous poll, unless a
    frame older than that has arrived since — a backend serving a stale but
    still fresh frame — in which case it starts again at that frame's block.
    Returns the last block end walked, the cursor for the next poll.
    """
    if not known:
        return closed_through
    ordered = sorted(known.values(), key=lambda snapshot: snapshot.captured_at)
    by_block: dict[pd.Timestamp, list[Snapshot]] = {}
    for snapshot in ordered:
        by_block.setdefault(block_end_of(snapshot.captured_at, block_minutes), []).append(snapshot)
    earliest = block_end_of(ordered[0].captured_at, block_minutes)
    first = (
        min(closed_through + pd.Timedelta(minutes=block_minutes), earliest)
        if closed_through is not None
        else earliest
    )
    ready = _ready_block_ends(
        first,
        latest=ordered[-1].captured_at,
        now=now,
        block_minutes=block_minutes,
        grace_seconds=grace_seconds,
    )
    for end, closed_by in ready:
        closed_through = end
        members = by_block.get(end, [])
        prediction_path, skipped_path = _block_record_paths(blocks_dir, end)
        stem = prediction_path.name.removesuffix(PREDICTION_SUFFIX)
        if prediction_path.exists() or skipped_path.exists():
            continue
        record: dict[str, Any] = {
            "block_end": end.isoformat(),
            "closed_by": closed_by,
            "n_frames": len(members),
        }
        if len(members) < min_frames:
            atomic_write_strict_json(
                skipped_path,
                record | {"reason": "insufficient_frames", "min_frames": min_frames},
            )
            logger.info(
                "block %s skipped: %d frame(s), fewer than %d", stem, len(members), min_frames
            )
            continue
        try:
            payload = _score_block(
                members,
                end,
                checkpoints,
                min_solar_elevation_deg=min_solar_elevation_deg,
                site=site,
                device=device,
                trust_checkpoint=trust_checkpoint,
                image_backbone_builder=image_backbone_builder,
            )
            atomic_write_strict_json(prediction_path, payload | {"closed_by": closed_by})
        except SolarElevationBelowFloorError as exc:
            logger.info("block %s skipped: %s", stem, exc)
            atomic_write_strict_json(
                skipped_path,
                record
                | {
                    "reason": "below_elevation_floor",
                    "solar_elevation_deg": exc.elevation_deg,
                    "min_solar_elevation_deg": exc.floor_deg,
                },
            )
            continue
        except PREDICTION_ERRORS as exc:
            logger.error("block %s prediction failed, recording the failure: %s", stem, exc)
            atomic_write_strict_json(
                skipped_path, record | {"reason": "prediction_failed", "error": str(exc)}
            )
            continue
        logger.info(
            "block %s closed from %d frame(s): %s", stem, len(members), payload["predictions"]
        )
    return closed_through


def _inspect_block_checkpoints(
    checkpoints: Sequence[Path],
    *,
    block_minutes: float,
    min_solar_elevation_deg: float | None,
    device: str,
    trust_checkpoint: bool,
) -> float:
    if min_solar_elevation_deg is None:
        raise ValueError(
            "a block checkpoint needs min_solar_elevation_deg: the checkpoint does not record "
            "the night_filter.min_solar_elevation_deg its manifest was built with"
        )
    for checkpoint in checkpoints:
        window = block_checkpoint_window_minutes(
            checkpoint, device=device, trust_checkpoint=trust_checkpoint
        )
        if abs(window - block_minutes) > 1e-9:
            raise ValueError(
                f"{checkpoint} pools a {window:g} min window but the watch closes "
                f"{block_minutes:g} min blocks; a block must be the checkpoint's own window"
            )
    return min_solar_elevation_deg


def _discard_stale_capture(snapshot: Snapshot) -> None:
    logger.warning(
        "discarding %s: the camera has not advanced, so its capture time came from the host "
        "clock, not the frame",
        snapshot.image_path.name,
    )
    snapshot.image_path.unlink(missing_ok=True)
    snapshot.metadata_path.unlink(missing_ok=True)


def run_watch(
    capture: Callable[[], Snapshot],
    out_dir: str | Path,
    *,
    checkpoint_frame: Path | None = None,
    checkpoint_blocks: Sequence[Path] = (),
    poll_seconds: float = 20.0,
    block_minutes: float = 5.0,
    min_frames: int = 3,
    grace_seconds: float = 90.0,
    min_solar_elevation_deg: float | None = None,
    site: SiteConfig | None = None,
    device: str = "cpu",
    trust_checkpoint: bool = False,
    image_backbone_builder: Callable[[], Any] | None = None,
    clock: Callable[[], pd.Timestamp] = _site_now,
    sleep: Callable[[float], None] = time.sleep,
    max_polls: int | None = None,
) -> int:
    """Poll *capture*, index each new frame, and score every block once it closes.

    Each poll calls *capture*, which is expected to write the frame and its
    sidecar under ``<out_dir>/frames/`` (the CLI passes
    :func:`allsky.snapshot.capture_snapshot` bound to the camera client). A
    frame is new when its ``captured_at`` was not seen before, on this run or
    in the sidecars already on disk; a repeated stamp means the camera has not
    advanced, and the capture overwrote the same file with the same bytes. A
    frame named from the host clock is a stale frame the camera has served for
    longer than :data:`allsky.snapshot.LIVE_FRAME_MAX_AGE`: it is deleted
    again, since a poll every *poll_seconds* would otherwise fill the
    directory with copies of one image. A frame stamped from the server's
    ``Last-Modified`` stays on disk but is not filed under a block (see
    :func:`frames_on_disk`). With *checkpoint_frame* every new frame that is
    filed is scored by :func:`~allsky.snapshot.predict_snapshot` into
    ``<image>.prediction.json``.

    Block ``t`` (a multiple of *block_minutes* on the camera's clock, covering
    ``(t - block_minutes, t]``) is ready once a frame stamped after ``t`` has
    arrived or *clock* reads ``t + grace_seconds``; the record says which
    under ``closed_by``. Every block from the earliest frame's on is walked,
    so a capture gap leaves ``insufficient_frames`` records with ``n_frames``
    0 rather than no file. A ready block with no record under
    ``<out_dir>/blocks/`` is scored by every checkpoint in *checkpoint_blocks*
    through :func:`~allsky.snapshot.predict_block` — averaged by
    :func:`ensemble_prediction` when there is more than one — when it holds
    at least *min_frames* frames, and otherwise recorded as
    ``<YYYYMMDD-HHMM>.skipped.json`` so it is not revisited. A block whose
    representative frame has the sun below *min_solar_elevation_deg* is
    recorded as ``below_elevation_floor``; a prediction that raises is
    recorded as ``prediction_failed``, with the error, rather than retried on
    every poll. A block already recorded is never rewritten, so a frame that
    arrives for a block closed by the grace is indexed with a warning and not
    fed. With no block checkpoint the watch only archives frames.

    Every block checkpoint is inspected at start-up: one
    :func:`~allsky.snapshot.predict_block` would refuse, or whose window is
    not *block_minutes* wide, stops the watch before any record is written.

    Capture failures (network, TLS, an empty payload — everything the archive
    client raises through :func:`~allsky.snapshot.capture_snapshot`) are
    logged and the loop goes on to the next poll.

    Parameters
    ----------
    capture:
        Zero-argument capture returning the written :class:`Snapshot`.
    out_dir:
        Watch root; frames under ``frames/``, block records under ``blocks/``.
    checkpoint_frame:
        Single-frame checkpoint to score every new frame with; None scores
        no frames.
    checkpoint_blocks:
        Block checkpoints; empty archives frames only.
    poll_seconds:
        Handed to *sleep* between polls.
    block_minutes:
        The logger's averaging interval, the block width; must equal the
        block checkpoints' ``alignment.window_minutes``.
    min_frames:
        Fewest frames a block needs to be scored rather than skipped.
    grace_seconds:
        How long past its end a block waits for a later frame before closing.
    min_solar_elevation_deg:
        The ``night_filter.min_solar_elevation_deg`` the block checkpoints'
        manifest was built with; required with a block checkpoint, since the
        checkpoint does not record it.
    site:
        Observation site for the solar geometry; None is the module default.
    device:
        Torch device for both checkpoints.
    trust_checkpoint:
        Allow unpickling checkpoints that are not weights-only.
    image_backbone_builder:
        Injection hook for the block checkpoints' visual backbone, as
        :func:`~allsky.snapshot.predict_block` takes it.
    clock:
        Current naive local time on the camera's clock; injected for tests.
    sleep:
        Called with *poll_seconds* between polls; injected for tests.
    max_polls:
        Stop after this many polls; None polls until interrupted.

    Returns
    -------
    int
        Number of polls made.

    Raises
    ------
    ValueError
        At start-up, for a block checkpoint :func:`~allsky.snapshot.predict_block`
        refuses, one whose window is not *block_minutes* wide, or block
        checkpoints without *min_solar_elevation_deg*.
    """
    root = Path(out_dir)
    frames_dir = root / FRAMES_SUBDIR
    blocks_dir = root / BLOCKS_SUBDIR
    checkpoints = tuple(checkpoint_blocks)
    elevation_floor = (
        _inspect_block_checkpoints(
            checkpoints,
            block_minutes=block_minutes,
            min_solar_elevation_deg=min_solar_elevation_deg,
            device=device,
            trust_checkpoint=trust_checkpoint,
        )
        if checkpoints
        else None
    )
    known = {snapshot.captured_at: snapshot for snapshot in frames_on_disk(frames_dir)}
    logger.info("watching with %d frame(s) already under %s", len(known), frames_dir)
    closed_through: pd.Timestamp | None = None
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        try:
            snapshot = capture()
        except CAPTURE_ERRORS as exc:
            logger.warning("capture failed on poll %d: %s", polls, exc)
        else:
            _index_capture(
                snapshot,
                known,
                blocks_dir,
                checkpoint_frame=checkpoint_frame,
                filed_under_blocks=bool(checkpoints),
                block_minutes=block_minutes,
                site=site,
                device=device,
                trust_checkpoint=trust_checkpoint,
            )
        if checkpoints and elevation_floor is not None:
            closed_through = _close_ready_blocks(
                known,
                blocks_dir,
                checkpoints,
                now=clock(),
                block_minutes=block_minutes,
                min_frames=min_frames,
                grace_seconds=grace_seconds,
                min_solar_elevation_deg=elevation_floor,
                site=site,
                device=device,
                trust_checkpoint=trust_checkpoint,
                image_backbone_builder=image_backbone_builder,
                closed_through=closed_through,
            )
        if max_polls is None or polls < max_polls:
            sleep(poll_seconds)
    return polls


def _index_capture(
    snapshot: Snapshot,
    known: dict[pd.Timestamp, Snapshot],
    blocks_dir: Path,
    *,
    checkpoint_frame: Path | None,
    filed_under_blocks: bool,
    block_minutes: float,
    site: SiteConfig | None,
    device: str,
    trust_checkpoint: bool,
) -> None:
    if snapshot.captured_at in known:
        logger.info(
            "frame %s already indexed; the camera has not advanced", snapshot.image_path.name
        )
        return
    source = _capture_source(snapshot)
    if source == STALE_CAPTURE_SOURCE:
        _discard_stale_capture(snapshot)
        return
    if source in UNFILED_CAPTURE_SOURCES:
        logger.warning(
            "not filing %s: its capture time came from the %s, not the frame; it stays on disk",
            snapshot.image_path.name,
            source,
        )
        return
    known[snapshot.captured_at] = snapshot
    logger.info("new frame %s", snapshot.image_path.name)
    if filed_under_blocks:
        end = block_end_of(snapshot.captured_at, block_minutes)
        if any(path.exists() for path in _block_record_paths(blocks_dir, end)):
            logger.warning(
                "frame %s falls in block %s, which is already closed; it is indexed but not fed",
                snapshot.image_path.name,
                f"{end:{BLOCK_STEM_FORMAT}}",
            )
    if checkpoint_frame is not None:
        _score_frame(
            snapshot,
            checkpoint_frame,
            site=site,
            device=device,
            trust_checkpoint=trust_checkpoint,
        )
