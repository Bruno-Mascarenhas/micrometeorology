"""Polls the camera's live frame and scores each datalogger block as it closes.

Timestamps here are **naive local time** on the camera's own clock, as in
:mod:`allsky.snapshot`: a frame is filed under the block its overlay stamp
rounds up to, and a block is closed on that same clock.

The state is the disk. ``<out_dir>/frames/`` holds every captured frame, its
sidecar and — with a frame checkpoint — its ``.prediction.json``;
``<out_dir>/blocks/`` one ``<YYYYMMDD-HHMM>.prediction.json`` per scored block
or a ``.skipped.json`` naming why it was not, so a restarted watch rebuilds
its index from the sidecars and never rewrites a block.

Several checkpoints of one kind are served as an ensemble whose members play
a **role** read off the file stem — ``best.ckpt`` is ``best``, ``last.ckpt``
is ``last``, anything else ``other`` — and a role selector per head group
(``best``, ``last`` or ``all``) says whose sky heads and whose regression
heads are averaged. A block record carries the block checkpoints' prediction
when there are any, and otherwise the mean of the per-frame predictions
already written beside its frames; ``source`` says which.
"""

import http.client
import json
import logging
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.archive import ArchiveError
from allsky.config import SiteConfig
from allsky.serving import ALL_ROLES, HeadRoles, RoleSelector
from allsky.snapshot import (
    ServedInput,
    ServedModel,
    Snapshot,
    SolarElevationBelowFloorError,
    _site_now,
    block_end_of,
    load_served_model,
    solar_elevation_at,
)
from labmim_core.atomic import atomic_write_strict_json

logger = logging.getLogger(__name__)

__all__ = [
    "BLOCKS_SUBDIR",
    "CHECKPOINT_ROLES",
    "FRAMES_SUBDIR",
    "ROLE_SELECTORS",
    "checkpoint_role",
    "ensemble_prediction",
    "envelope_of",
    "frame_aggregate",
    "frames_newest_first",
    "frames_on_disk",
    "run_watch",
]

FRAMES_SUBDIR = "frames"
BLOCKS_SUBDIR = "blocks"
BLOCK_STEM_FORMAT = "%Y%m%d-%H%M"
PREDICTION_SUFFIX = ".prediction.json"
SKIPPED_SUFFIX = ".skipped.json"
CHECKPOINT_ROLES = ("best", "last", "other")
ROLE_SELECTORS = tuple(selector.value for selector in RoleSelector)
REGRESSION_HEADS = ("dhi", "kindex", "cloud_fraction")
SKY_HEAD = "sky"
SOURCE_BLOCK_MODEL = "block_model"
SOURCE_FRAME_AGGREGATE = "frame_aggregate"
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
    records = _snapshots_of(directory, sorted(_frame_sidecars(directory)))
    return sorted(records, key=lambda record: record.captured_at)


def frames_newest_first(frames_dir: str | Path) -> Iterator[Snapshot]:
    """The frames of :func:`frames_on_disk`, newest name first, each read only when reached.

    The capture files are named by their overlay stamp, so the name order is
    the capture order; a reader after the newest usable frame stops after one
    or two sidecars instead of parsing the whole day.
    """
    directory = Path(frames_dir)
    return _snapshots_of(directory, sorted(_frame_sidecars(directory), reverse=True))


def _frame_sidecars(directory: Path) -> Iterator[Path]:
    return (
        sidecar
        for sidecar in directory.glob("*.json")
        if not sidecar.name.endswith(PREDICTION_SUFFIX)
    )


def _snapshots_of(directory: Path, sidecars: Sequence[Path]) -> Iterator[Snapshot]:
    for sidecar in sidecars:
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
        yield Snapshot(image_path=image, metadata_path=sidecar, captured_at=captured)


def _capture_source(snapshot: Snapshot) -> str:
    meta = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    return str(meta["captured_at_source"])


def checkpoint_role(checkpoint: str | Path) -> str:
    """The role a checkpoint plays in an ensemble, read off its file stem.

    ``best.ckpt`` plays ``best``, ``last.ckpt`` plays ``last`` and any other
    stem plays ``other``, which only an ``all`` selector picks up.

    Parameters
    ----------
    checkpoint:
        Checkpoint path; only the stem is read.

    Returns
    -------
    str
        One of :data:`CHECKPOINT_ROLES`.
    """
    stem = Path(checkpoint).stem
    return stem if stem in ("best", "last") else "other"


def _selector(selector: str) -> RoleSelector:
    try:
        return RoleSelector(selector)
    except ValueError:
        raise ValueError(
            f"unknown role selector {selector!r}; expected one of {', '.join(ROLE_SELECTORS)}"
        ) from None


def _plays(role: str, selector: str) -> bool:
    chosen = _selector(selector)
    return chosen is RoleSelector.all or role == chosen.value


@dataclass(frozen=True, slots=True)
class _Ensemble:
    """The served checkpoints of one kind and which of them each head group is read from."""

    kind: ServedInput
    members: tuple[ServedModel, ...]
    roles: HeadRoles

    @property
    def checkpoints(self) -> tuple[Path, ...]:
        return tuple(member.checkpoint_path for member in self.members)


def _refuse_an_empty_role(
    kind: str, checkpoints: Sequence[Path], *, sky_roles: str, dhi_roles: str
) -> None:
    for head, selector in ((SKY_HEAD, sky_roles), ("dhi", dhi_roles)):
        if not any(_plays(checkpoint_role(path), selector) for path in checkpoints):
            stems = ", ".join(f"{path.stem} ({checkpoint_role(path)})" for path in checkpoints)
            raise ValueError(
                f"no {kind} checkpoint plays the {selector!r} role the {head} heads "
                f"are read from; given: {stems}"
            )


def _mean_predictions(
    regression_sources: Sequence[dict[str, Any]], sky_sources: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Mean of each head over the predictions that carry it; a class from the mean probabilities."""
    predictions: dict[str, Any] = {}
    for name in REGRESSION_HEADS:
        values = [p[name] for p in regression_sources if name in p]
        if values:
            predictions[name] = float(np.mean(values))
    probability_maps = [p["sky_probabilities"] for p in sky_sources if "sky_probabilities" in p]
    if probability_maps:
        names = list(probability_maps[0])
        mean = {name: float(np.mean([p[name] for p in probability_maps])) for name in names}
        predictions["sky_probabilities"] = mean
        predictions["sky_class"] = names[int(np.argmax([mean[name] for name in names]))]
    return predictions


def ensemble_prediction(
    results: Sequence[dict[str, Any]], *, sky_roles: str = "all", dhi_roles: str = "all"
) -> dict[str, Any]:
    """Average the predictions of several checkpoints into one record, by role.

    Each member's role is :func:`checkpoint_role` of the checkpoint its
    ``model`` names. ``dhi``, ``kindex`` and ``cloud_fraction`` are the means
    over the members playing *dhi_roles* that carry the head;
    ``sky_probabilities`` is the per-class mean over the members playing
    *sky_roles* that carry one and ``sky_class`` its argmax, first name on a
    tie. A head no selected member carries is absent, not an error. The
    ``block`` and ``image`` records, when the members have them, are the
    first member's — every member was fed the same input.

    Parameters
    ----------
    results:
        :func:`~allsky.snapshot.predict_block` or
        :func:`~allsky.snapshot.predict_snapshot` records, one per checkpoint.
    sky_roles, dhi_roles:
        One of :data:`ROLE_SELECTORS`, naming whose sky heads and whose
        regression heads are averaged.

    Returns
    -------
    dict
        ``predictions``; ``members`` listing per member its ``checkpoint``,
        ``role``, the ``heads`` it contributed and its own ``predictions``;
        ``models``; plus ``block`` and ``image`` when the members carry them.

    Raises
    ------
    ValueError
        If *results* is empty, a selector is unknown, or a selector picks no
        member.
    """
    if not results:
        raise ValueError("an ensemble needs at least one member prediction")
    members: list[dict[str, Any]] = []
    regression_sources: list[dict[str, Any]] = []
    sky_sources: list[dict[str, Any]] = []
    for result in results:
        checkpoint = str(result["model"]["checkpoint"])
        role = checkpoint_role(checkpoint)
        predictions = result["predictions"]
        heads: list[str] = []
        if _plays(role, dhi_roles):
            regression_sources.append(predictions)
            heads.extend(name for name in REGRESSION_HEADS if name in predictions)
        if _plays(role, sky_roles) and "sky_probabilities" in predictions:
            sky_sources.append(predictions)
            heads.append(SKY_HEAD)
        members.append(
            {"checkpoint": checkpoint, "role": role, "heads": heads, "predictions": predictions}
        )
    _refuse_an_empty_role(
        "member",
        [Path(m["checkpoint"]) for m in members],
        sky_roles=sky_roles,
        dhi_roles=dhi_roles,
    )
    record: dict[str, Any] = {
        "predictions": _mean_predictions(regression_sources, sky_sources),
        "members": members,
        "models": [r["model"] for r in results],
    }
    for shared in ("block", "image"):
        if shared in results[0]:
            record[shared] = results[0][shared]
    return record


def envelope_of(record: dict[str, Any]) -> dict[str, Any]:
    """*record* in the ensemble shape, whether the watch that wrote it had one member or several.

    The watch writes every record through :func:`ensemble_prediction` since
    the ensemble shape became the only one; a bare :meth:`ServedModel.predict_frame`
    record on disk predates that and is wrapped here, so every reader handles
    one shape.
    """
    if "members" in record:
        return record
    return ensemble_prediction([record])


def frame_aggregate(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean of the per-frame predictions of one block, as one prediction.

    The same averaging as :func:`ensemble_prediction` — every head over the
    frames that carry it, the class from the mean probabilities — applied
    across the frames of a block instead of across checkpoints.

    Parameters
    ----------
    predictions:
        The ``predictions`` of each scored frame's ``.prediction.json``.

    Returns
    -------
    dict
        Whichever of ``dhi``, ``kindex``, ``cloud_fraction``,
        ``sky_probabilities`` and ``sky_class`` the frames carry.

    Raises
    ------
    ValueError
        If *predictions* is empty.
    """
    if not predictions:
        raise ValueError("a frame aggregate needs at least one scored frame")
    return _mean_predictions(predictions, predictions)


def _frame_prediction_path(snapshot: Snapshot) -> Path:
    return snapshot.image_path.with_suffix(PREDICTION_SUFFIX)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _malformed_head(predictions: dict[str, Any]) -> str | None:
    for name in REGRESSION_HEADS:
        if name in predictions and not _is_finite_number(predictions[name]):
            return f"{name} is {predictions[name]!r}, not a finite number"
    if "sky_probabilities" not in predictions:
        return None
    probabilities = predictions["sky_probabilities"]
    if not isinstance(probabilities, dict) or not all(
        _is_finite_number(value) for value in probabilities.values()
    ):
        return f"sky_probabilities is {probabilities!r}, not a map of finite numbers"
    return None


def _validated_predictions(payload: Any, path: Path) -> dict[str, Any] | None:
    try:
        predictions = dict(payload["predictions"])
    except (KeyError, TypeError) as exc:
        logger.warning("ignoring unreadable frame prediction %s: %s", path.name, exc)
        return None
    fault = _malformed_head(predictions)
    if fault is not None:
        logger.warning("ignoring malformed frame prediction %s: %s", path.name, fault)
        return None
    return predictions


class _FrameIndex:
    """The frames the watch knows, bucketed by block, with the predictions it wrote for them.

    The disk is the state (:mod:`allsky.watch`); this is its index, built
    once from the sidecars at start-up and extended by each capture, so a
    poll never re-reads or re-sorts the day's history. A frame's prediction
    is kept from the write that produced it and read back from disk only for
    frames scored before this process started.
    """

    def __init__(self, snapshots: Sequence[Snapshot], block_minutes: float) -> None:
        self.block_minutes = block_minutes
        self.known: dict[pd.Timestamp, Snapshot] = {}
        self.by_block: dict[pd.Timestamp, list[Snapshot]] = {}
        self._predictions: dict[pd.Timestamp, dict[str, Any]] = {}
        self.earliest: pd.Timestamp | None = None
        self.latest: pd.Timestamp | None = None
        for snapshot in snapshots:
            self.add(snapshot)

    def __len__(self) -> int:
        return len(self.known)

    def __contains__(self, captured_at: pd.Timestamp) -> bool:
        return captured_at in self.known

    def add(self, snapshot: Snapshot) -> None:
        when = snapshot.captured_at
        self.known[when] = snapshot
        bucket = self.by_block.setdefault(block_end_of(when, self.block_minutes), [])
        bucket.append(snapshot)
        bucket.sort(key=lambda member: member.captured_at)
        self.earliest = when if self.earliest is None else min(self.earliest, when)
        self.latest = when if self.latest is None else max(self.latest, when)

    def remember(self, snapshot: Snapshot, predictions: dict[str, Any]) -> None:
        self._predictions[snapshot.captured_at] = predictions

    def predictions_of(self, snapshot: Snapshot) -> dict[str, Any] | None:
        remembered = self._predictions.get(snapshot.captured_at)
        if remembered is not None:
            return remembered
        path = _frame_prediction_path(snapshot)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("ignoring unreadable frame prediction %s: %s", path.name, exc)
            return None
        predictions = _validated_predictions(payload, path)
        if predictions is not None:
            self._predictions[snapshot.captured_at] = predictions
        return predictions


def _readable_frame_predictions(
    members: Sequence[Snapshot], index: _FrameIndex
) -> list[tuple[Snapshot, dict[str, Any]]]:
    return [
        (snapshot, predictions)
        for snapshot in members
        if (predictions := index.predictions_of(snapshot)) is not None
    ]


def _aggregate_of_scored_frames(
    members: Sequence[Snapshot], index: _FrameIndex
) -> dict[str, Any] | None:
    readable = _readable_frame_predictions(members, index)
    class_names = frozenset[str]().union(
        *(frozenset(p["sky_probabilities"]) for _, p in readable if "sky_probabilities" in p)
    )
    scored: list[tuple[Snapshot, dict[str, Any]]] = []
    for snapshot, predictions in readable:
        if "sky_probabilities" in predictions and (
            frozenset(predictions["sky_probabilities"]) != class_names
        ):
            logger.warning(
                "ignoring frame prediction %s: its sky classes %s lack some of the block's %s",
                _frame_prediction_path(snapshot).name,
                sorted(predictions["sky_probabilities"]),
                sorted(class_names),
            )
            continue
        scored.append((snapshot, predictions))
    if not scored:
        return None
    return {
        "predictions": frame_aggregate([predictions for _, predictions in scored]),
        "n_frames": len(scored),
        "frames": [
            {"path": str(snapshot.image_path), "captured_at": snapshot.captured_at.isoformat()}
            for snapshot, _ in scored
        ],
    }


def _averaged(results: Sequence[dict[str, Any]], ensemble: _Ensemble) -> dict[str, Any]:
    return ensemble_prediction(
        results, sky_roles=ensemble.roles.sky.value, dhi_roles=ensemble.roles.dhi.value
    )


def _score_frame(
    snapshot: Snapshot, ensemble: _Ensemble, index: _FrameIndex, *, site: SiteConfig
) -> None:
    try:
        results = [
            member.predict_frame(snapshot.image_path, timestamp=snapshot.captured_at, site=site)
            for member in ensemble.members
        ]
        payload = _averaged(results, ensemble)
        path = atomic_write_strict_json(_frame_prediction_path(snapshot), payload)
    except PREDICTION_ERRORS as exc:
        logger.error("frame prediction failed for %s: %s", snapshot.image_path.name, exc)
        return
    index.remember(snapshot, payload["predictions"])
    logger.info("frame prediction %s: %s", path.name, payload["predictions"])


def _score_block(
    members: Sequence[Snapshot],
    end: pd.Timestamp,
    ensemble: _Ensemble,
    *,
    min_solar_elevation_deg: float,
    site: SiteConfig,
) -> dict[str, Any]:
    frames = [(snapshot.image_path, snapshot.captured_at) for snapshot in members]
    results = [
        member.predict_block(
            frames, min_solar_elevation_deg=min_solar_elevation_deg, block_end=end, site=site
        )
        for member in ensemble.members
    ]
    return _averaged(results, ensemble)


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
    index: _FrameIndex,
    blocks_dir: Path,
    ensemble: _Ensemble | None,
    *,
    now: pd.Timestamp,
    block_minutes: float,
    min_frames: int,
    grace_seconds: float,
    min_solar_elevation_deg: float,
    site: SiteConfig,
    closed_through: pd.Timestamp | None,
) -> pd.Timestamp | None:
    """Close every ready block from the earliest frame's or after *closed_through*.

    The walk resumes past the last block closed on the previous poll, unless a
    frame older than that has arrived since — a backend serving a stale but
    still fresh frame — in which case it starts again at that frame's block.
    Returns the last block end walked, the cursor for the next poll. With no
    block *ensemble* a block is recorded from the predictions already written
    beside its frames, and skipped when none of them was scored.
    """
    if index.earliest is None or index.latest is None:
        return closed_through
    earliest = block_end_of(index.earliest, block_minutes)
    first = (
        min(closed_through + pd.Timedelta(minutes=block_minutes), earliest)
        if closed_through is not None
        else earliest
    )
    ready = _ready_block_ends(
        first,
        latest=index.latest,
        now=now,
        block_minutes=block_minutes,
        grace_seconds=grace_seconds,
    )
    for end, closed_by in ready:
        closed_through = end
        members = index.by_block.get(end, [])
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
        aggregate = _aggregate_of_scored_frames(members, index)
        if ensemble is None and aggregate is None:
            atomic_write_strict_json(skipped_path, record | {"reason": "no_frame_predictions"})
            logger.info("block %s skipped: none of its %d frame(s) was scored", stem, len(members))
            continue
        try:
            payload = (
                _block_model_payload(
                    record,
                    members,
                    end,
                    ensemble,
                    min_solar_elevation_deg=min_solar_elevation_deg,
                    site=site,
                )
                if ensemble is not None
                else record | {"source": SOURCE_FRAME_AGGREGATE}
            )
            if aggregate is not None:
                payload["frame_aggregate"] = aggregate
                payload.setdefault("predictions", aggregate["predictions"])
            atomic_write_strict_json(prediction_path, payload)
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
            "block %s closed from %d frame(s) by the %s: %s",
            stem,
            len(members),
            payload["source"],
            payload["predictions"],
        )
    return closed_through


def _block_model_payload(
    record: dict[str, Any],
    members: Sequence[Snapshot],
    end: pd.Timestamp,
    ensemble: _Ensemble,
    *,
    min_solar_elevation_deg: float,
    site: SiteConfig,
) -> dict[str, Any]:
    block_model = _score_block(
        members, end, ensemble, min_solar_elevation_deg=min_solar_elevation_deg, site=site
    )
    return record | {
        "source": SOURCE_BLOCK_MODEL,
        "predictions": block_model["predictions"],
        "block_model": block_model,
    }


def _elevation_floor(given: float | None, members: Sequence[ServedModel]) -> float:
    """The solar-elevation floor the watch scores nothing under.

    Every member that records the ``night_filter.min_solar_elevation_deg``
    its manifest was built with must agree with *given* when the operator
    supplied one, and with each other when not; a member recording none
    leaves the operator's value to stand and, absent that, is refused.
    """
    recorded = {
        member.checkpoint_path: member.min_solar_elevation_deg
        for member in members
        if member.min_solar_elevation_deg is not None
    }
    if given is not None:
        disagreeing = {path: floor for path, floor in recorded.items() if abs(floor - given) > 1e-9}
        if disagreeing:
            listed = ", ".join(f"{path} ({floor:g} deg)" for path, floor in disagreeing.items())
            raise ValueError(
                f"min_solar_elevation_deg {given:g} is not the floor these checkpoints' "
                f"manifests were built with: {listed}"
            )
        return given
    floors = set(recorded.values())
    if len(floors) == 1:
        return floors.pop()
    if not floors:
        raise ValueError(
            "a checkpoint needs min_solar_elevation_deg: none of these checkpoints records the "
            "night_filter.min_solar_elevation_deg its manifest was built with"
        )
    listed = ", ".join(f"{path} ({floor:g} deg)" for path, floor in recorded.items())
    raise ValueError(
        f"the checkpoints record different elevation floors, pass min_solar_elevation_deg "
        f"to choose: {listed}"
    )


def _refuse_another_window(ensemble: _Ensemble, block_minutes: float) -> None:
    for member in ensemble.members:
        if abs(member.window_minutes - block_minutes) > 1e-9:
            raise ValueError(
                f"{member.checkpoint_path} pools a {member.window_minutes:g} min window but the "
                f"watch closes {block_minutes:g} min blocks; a block must be the checkpoint's "
                "own window"
            )


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
    checkpoint_frames: Sequence[Path] = (),
    checkpoint_blocks: Sequence[Path] = (),
    frame_roles: HeadRoles = ALL_ROLES,
    block_roles: HeadRoles = ALL_ROLES,
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
    :func:`frames_on_disk`). Every checkpoint is loaded once, at start-up
    (:func:`~allsky.snapshot.load_served_model`), and stays resident. With
    *checkpoint_frames* every new frame that is filed and has the sun at or
    above the elevation floor is scored by each member's
    :meth:`~allsky.snapshot.ServedModel.predict_frame` into
    ``<image>.prediction.json``, always in the shape of
    :func:`ensemble_prediction` under *frame_roles*, one member or several;
    a frame below the floor is logged and left unscored.

    Block ``t`` (a multiple of *block_minutes* on the camera's clock, covering
    ``(t - block_minutes, t]``) is ready once a frame stamped after ``t`` has
    arrived or *clock* reads ``t + grace_seconds``; the record says which
    under ``closed_by``. Every block from the earliest frame's on is walked,
    so a capture gap leaves ``insufficient_frames`` records with ``n_frames``
    0 rather than no file. A ready block with no record under
    ``<out_dir>/blocks/`` and at least *min_frames* frames is scored by every
    member of *checkpoint_blocks* through
    :meth:`~allsky.snapshot.ServedModel.predict_block` — averaged by
    :func:`ensemble_prediction` under *block_roles* — and its record carries that prediction
    under ``block_model`` and, when its frames were scored, their mean under
    ``frame_aggregate``; with no block checkpoint the record is the frame
    aggregate alone, and a block none of whose frames was scored is recorded
    as ``no_frame_predictions`` — at night, when every frame prediction
    failed, or when its frames were captured before there was a frame
    checkpoint (an archive-only history, or a watch restarted between a
    frame's capture and its prediction): resuming re-indexes the frames on
    disk and never scores them after the fact. A frame ``.prediction.json``
    the aggregate cannot average (a head that is not a finite number, a
    probability map that is not, or one missing a class another frame of
    the block carries) is left out with a warning. ``source`` names which
    of the two the top-level ``predictions`` are. A block with fewer than *min_frames*
    frames is recorded as ``<YYYYMMDD-HHMM>.skipped.json`` so it is not
    revisited; one whose representative frame has the sun below
    *min_solar_elevation_deg* as ``below_elevation_floor``; a block prediction
    that raises as ``prediction_failed``, with the error, rather than retried
    on every poll. A block already recorded is never rewritten, so a frame
    that arrives for a block closed by the grace is indexed with a warning
    and not fed. With no checkpoint at all the watch only archives frames.

    Start-up refuses, before any capture, a role selector no checkpoint of
    its kind plays, an elevation floor the checkpoints do not settle (see
    *min_solar_elevation_deg*), a member that cannot be loaded or built, a
    windowed checkpoint under *checkpoint_frames*, a ``center_frame`` or
    embedding-mode one under *checkpoint_blocks*, and a block checkpoint
    whose window is not *block_minutes* wide.

    Capture failures (network, TLS, an empty payload — everything the archive
    client raises through :func:`~allsky.snapshot.capture_snapshot`) are
    logged and the loop goes on to the next poll.

    Parameters
    ----------
    capture:
        Zero-argument capture returning the written :class:`Snapshot`.
    out_dir:
        Watch root; frames under ``frames/``, block records under ``blocks/``.
    checkpoint_frames:
        Single-frame checkpoints every new frame is scored with; empty scores
        no frames.
    checkpoint_blocks:
        Block checkpoints; empty scores no block model.
    frame_roles, block_roles:
        Which members of the frame and block ensembles, by
        :func:`checkpoint_role`, the sky heads and the regression heads are
        read from.
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
        The ``night_filter.min_solar_elevation_deg`` the checkpoints'
        manifests were built with. A checkpoint written since that floor
        joined its provenance records it, and every recorded floor must
        agree with this value when it is given and with each other when it
        is not; with no checkpoint recording one, it is required.
    site:
        Observation site for the solar geometry; None is the module default.
    device:
        Torch device for every checkpoint.
    trust_checkpoint:
        Allow unpickling checkpoints that are not weights-only.
    image_backbone_builder:
        Injection hook for the block checkpoints' visual backbone, as
        :func:`~allsky.snapshot.load_served_model` takes it.
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
        At start-up, for any of the refusals listed above.
    """
    root = Path(out_dir)
    frames_dir = root / FRAMES_SUBDIR
    blocks_dir = root / BLOCKS_SUBDIR
    resolved_site = site or SiteConfig()
    frame_ensemble = _ensemble_or_none(
        "frame", checkpoint_frames, frame_roles, device=device, trust_checkpoint=trust_checkpoint
    )
    block_ensemble = _ensemble_or_none(
        "block",
        checkpoint_blocks,
        block_roles,
        device=device,
        trust_checkpoint=trust_checkpoint,
        image_backbone_builder=image_backbone_builder,
    )
    served = [
        *(frame_ensemble.members if frame_ensemble else ()),
        *(block_ensemble.members if block_ensemble else ()),
    ]
    elevation_floor = _elevation_floor(min_solar_elevation_deg, served) if served else None
    if block_ensemble is not None:
        _refuse_another_window(block_ensemble, block_minutes)
    index = _FrameIndex(frames_on_disk(frames_dir), block_minutes)
    logger.info("watching with %d frame(s) already under %s", len(index), frames_dir)
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
                index,
                blocks_dir,
                frame_ensemble=frame_ensemble,
                filed_under_blocks=elevation_floor is not None,
                min_solar_elevation_deg=elevation_floor,
                site=resolved_site,
            )
        if elevation_floor is not None:
            closed_through = _close_ready_blocks(
                index,
                blocks_dir,
                block_ensemble,
                now=clock(),
                block_minutes=block_minutes,
                min_frames=min_frames,
                grace_seconds=grace_seconds,
                min_solar_elevation_deg=elevation_floor,
                site=resolved_site,
                closed_through=closed_through,
            )
        if max_polls is None or polls < max_polls:
            sleep(poll_seconds)
    return polls


def _ensemble_or_none(
    kind: ServedInput,
    checkpoints: Sequence[Path],
    roles: HeadRoles,
    *,
    device: str,
    trust_checkpoint: bool,
    image_backbone_builder: Callable[[], Any] | None = None,
) -> _Ensemble | None:
    """Load the *kind* checkpoints once, refusing what the ensemble could not serve.

    A member that cannot be loaded or built stops the start: the watch logs a
    per-frame failure without stopping, which is right for a bad frame and
    wrong for a bad member, which would otherwise leave the service running
    all day, scoring nothing.
    """
    if not checkpoints:
        return None
    paths = tuple(Path(path) for path in checkpoints)
    _refuse_an_empty_role(kind, paths, sky_roles=roles.sky.value, dhi_roles=roles.dhi.value)
    members = tuple(
        load_served_model(
            path,
            device=device,
            trust_checkpoint=trust_checkpoint,
            image_backbone_builder=image_backbone_builder,
            expect=kind,
        )
        for path in paths
    )
    return _Ensemble(kind, members, roles)


def _index_capture(
    snapshot: Snapshot,
    index: _FrameIndex,
    blocks_dir: Path,
    *,
    frame_ensemble: _Ensemble | None,
    filed_under_blocks: bool,
    min_solar_elevation_deg: float | None,
    site: SiteConfig,
) -> None:
    if snapshot.captured_at in index:
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
    index.add(snapshot)
    logger.info("new frame %s", snapshot.image_path.name)
    if filed_under_blocks:
        end = block_end_of(snapshot.captured_at, index.block_minutes)
        if any(path.exists() for path in _block_record_paths(blocks_dir, end)):
            logger.warning(
                "frame %s falls in block %s, which is already closed; it is indexed but not fed",
                snapshot.image_path.name,
                f"{end:{BLOCK_STEM_FORMAT}}",
            )
    if frame_ensemble is None or min_solar_elevation_deg is None:
        return
    elevation_deg = solar_elevation_at(snapshot.captured_at, site)
    if elevation_deg < min_solar_elevation_deg:
        logger.info(
            "frame %s not scored: the sun is %.1f deg above the horizon, below the %g deg floor",
            snapshot.image_path.name,
            elevation_deg,
            min_solar_elevation_deg,
        )
        return
    _score_frame(snapshot, frame_ensemble, index, site=site)
