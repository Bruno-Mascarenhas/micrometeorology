"""Archive-mirroring CLI commands: ``sync-archive`` and ``snapshot``."""

import datetime as dt
import http.client
import logging
import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer

from allsky.archive import (
    ARCHIVE_BASE_URL,
    LEDGER_FILENAME,
    ArchiveClient,
    ArchiveEntry,
    ArchiveError,
    Ledger,
    build_ssl_context,
    ledger_lock,
)
from allsky.cli.runtime import configure_cli_logging
from allsky.drive import DriveTarget, RcloneError, RcloneUploader

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/all-sky")
VIDEOS_SUBDIR = "videos"
FRAMES_SUBDIR = "frames"
STATE_SUBDIR = ".state"
REMOTE_VIDEOS = "videos"
REMOTE_FRAMES = "frames"
REMOTE_SNAPSHOTS = "snapshots"
FRAME_PATTERN = "*.jpg"


class UploadChoice(StrEnum):
    """What ``sync-archive`` pushes to Drive: nothing, the mp4, the frames or both."""

    none = "none"
    videos = "videos"
    frames = "frames"
    both = "both"

    @property
    def wants_videos(self) -> bool:
        return self in (UploadChoice.videos, UploadChoice.both)

    @property
    def wants_frames(self) -> bool:
        return self in (UploadChoice.frames, UploadChoice.both)


class TimestampSource(StrEnum):
    """Where a frame's wall-clock time comes from.

    ``overlay`` reads the stamp the camera burns into the frame, which is what
    this camera's varying capture cadence requires; ``config`` hands the choice
    to the ``video`` section of the :class:`PrepareConfig` YAML, which names
    either that same overlay reader or the modelled cadence that places frame N
    at ``start_time + N x minutes_per_frame`` — a constant interval this camera
    does not follow (selecting it logs a warning; see
    ``docs/allsky-archive.md``).  The ledger records the clock the config
    resolved to, never this flag.
    """

    overlay = "overlay"
    config = "config"


@dataclass(frozen=True)
class DayPlan:
    """The transfers one archived day still needs, decided against the ledger.

    Attributes
    ----------
    entry:
        The published archive entry (filename, date and key) the plan is for.
    download:
        Fetch the day's mp4: it is absent from disk and either a later step needs
        its bytes or the ledger has never recorded it.
    extract:
        Run frame extraction for the day (no frames recorded for this
        ``step``/``resize``/timestamp-source combination yet).
    upload_video:
        Push the mp4 to its Drive destination, which the ledger has not recorded.
    upload_frames:
        Push the day's frame directory to its Drive destination.
    """

    entry: ArchiveEntry
    download: bool
    extract: bool
    upload_video: bool
    upload_frames: bool

    @property
    def has_work(self) -> bool:
        """True when at least one transfer or extraction is still pending."""
        return self.download or self.extract or self.upload_video or self.upload_frames

    def describe(self) -> str:
        """One-line ``filename: action, action`` summary for the plan listing."""
        wanted = [
            name
            for name, flag in (
                ("download", self.download),
                ("extract", self.extract),
                ("upload-video", self.upload_video),
                ("upload-frames", self.upload_frames),
            )
            if flag
        ]
        return f"{self.entry.filename}: {', '.join(wanted)}"


def _parse_date(raw: str | None, label: str) -> dt.date | None:
    if raw is None:
        return None
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        raise typer.BadParameter(f"{label} must be YYYY-MM-DD, got {raw!r}") from None


def _build_client(
    base_url: str,
    state_dir: Path,
    *,
    ca_file: Path | None,
    insecure: bool,
    timeout: float,
    retries: int,
    delay: float,
) -> ArchiveClient:
    if insecure:
        logger.warning(
            "TLS verification is OFF (--insecure): the downloaded frames are unauthenticated"
        )
    state_dir.mkdir(parents=True, exist_ok=True)
    context = build_ssl_context(
        state_dir=state_dir, ca_file=ca_file, insecure=insecure, timeout=timeout
    )
    return ArchiveClient(
        base_url,
        context=context,
        timeout=timeout,
        retries=retries,
        delay=delay,
        allow_plaintext=base_url.startswith("http://"),
    )


def _select(
    entries: tuple[ArchiveEntry, ...], *, since: dt.date | None, until: dt.date | None
) -> tuple[ArchiveEntry, ...]:
    return tuple(
        entry
        for entry in entries
        if (since is None or entry.date >= since) and (until is None or entry.date <= until)
    )


def _plan_day(
    entry: ArchiveEntry,
    *,
    ledger: Ledger,
    root: Path,
    target: DriveTarget | None,
    extract: bool,
    step: int,
    resize: int | None,
    timestamps: TimestampSource,
    upload: UploadChoice,
) -> DayPlan:
    """Decide what *entry* still needs by asking the ledger what it already holds.

    A day is downloaded when its mp4 is not on disk and either extraction or an
    upload needs the bytes or the ledger never recorded it — so a day already
    uploaded and pruned is not fetched again.  It is extracted when no frame set
    matches this *step*/*resize*/*timestamps* combination, and uploaded when the
    ledger holds no record of that exact Drive destination.  Frames are only
    queued for upload once they exist or are about to.

    A day being extracted again is always queued for upload, whatever the ledger
    records: the destination is keyed by the day alone, so the record of the
    superseded set names the same remote folder as the set replacing it, and
    reading it as "already there" would leave the remote holding frames that no
    longer exist locally and no later run would ever reconcile.

    A day the ledger records as unusable under these very parameters is not
    queued for extraction again: its video decodes the same way every time, and
    re-reading it nightly costs the whole decode to reach the same refusal.
    """
    frames_done = ledger.frames_match(entry.key, step=step, resize=resize, timestamps=timestamps)
    faulted = ledger.extraction_faulted(
        entry.key, step=step, resize=resize, timestamps=timestamps.value
    )
    wants_extract = extract and not frames_done and faulted is None
    wants_video_upload = upload.wants_videos and not ledger.uploaded(
        entry.key, _video_destination(target, entry)
    )
    wants_frame_upload = upload.wants_frames and (
        wants_extract
        or (frames_done and not ledger.uploaded(entry.key, _frames_destination(target, entry)))
    )
    video_on_disk = ledger.has_video(entry.key, local_root=root)
    needs_video_bytes = wants_extract or wants_video_upload
    return DayPlan(
        entry=entry,
        download=not video_on_disk and (needs_video_bytes or not ledger.has_video(entry.key)),
        extract=wants_extract,
        upload_video=wants_video_upload,
        upload_frames=wants_frame_upload,
    )


def _warn_about_ambiguously_clocked_days(
    ledger: Ledger,
    selected: tuple[ArchiveEntry, ...],
    *,
    clock: TimestampSource,
    step: int,
    resize: int | None,
    extract: bool,
) -> None:
    """Announce the days a ledger written before the clock was recorded honestly will redo.

    Earlier releases filed ``timestamps="config"`` for every ``--timestamps
    config`` run, whatever clock the resolved section actually named — so a day
    recorded that way may hold overlay-stamped frames or modelled ones, and
    nothing on disk distinguishes them.  Re-extracting is the only way to settle
    it, and because the Drive destination is keyed by the day alone, each one is
    re-uploaded too.

    That is a correct migration and an expensive one, so it is named up front
    rather than discovered as a cron that ran all night: the days are counted
    before any work starts, with the flag that skips it.
    """
    if not extract or clock is not TimestampSource.overlay:
        return
    ambiguous = [
        entry.key
        for entry in selected
        if (record := ledger.frames(entry.key)) is not None
        and record.get("timestamps") == TimestampSource.config.value
        and record.get("step") == step
        and record.get("resize") == resize
    ]
    if not ambiguous:
        return
    logger.warning(
        "%d day(s) were extracted by a release that recorded 'config' as the clock whichever "
        "clock it actually used, so they cannot be told from overlay-stamped days and will be "
        "re-extracted (and re-uploaded) once: %s%s. Pass --no-extract to defer this.",
        len(ambiguous),
        ", ".join(ambiguous[:5]),
        "..." if len(ambiguous) > 5 else "",
    )


def _video_destination(target: DriveTarget | None, entry: ArchiveEntry) -> str:
    return target.path(REMOTE_VIDEOS, entry.filename) if target else ""


def _frames_destination(target: DriveTarget | None, entry: ArchiveEntry) -> str:
    return target.path(REMOTE_FRAMES, entry.key) if target else ""


def sync_archive(
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Root for videos/, frames/ and the ledger.")
    ] = DEFAULT_DATA_DIR,
    base_url: Annotated[
        str, typer.Option(help="All-sky website root to mirror.")
    ] = ARCHIVE_BASE_URL,
    since: Annotated[
        str | None, typer.Option(help="Only days on/after this date (YYYY-MM-DD).")
    ] = None,
    until: Annotated[
        str | None, typer.Option(help="Only days on/before this date (YYYY-MM-DD).")
    ] = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Stop after N days (backfill in slices).")
    ] = None,
    list_only: Annotated[
        bool, typer.Option("--list", help="Report the plan only, transfer nothing.")
    ] = False,
    extract: Annotated[
        bool, typer.Option("--extract/--no-extract", help="Extract JPEG frames from new videos.")
    ] = False,
    step: Annotated[int, typer.Option(min=1, help="Keep every Nth video frame.")] = 1,
    resize: Annotated[
        int | None, typer.Option(min=1, help="Resize extracted frames to NxN pixels.")
    ] = None,
    timestamps: Annotated[
        TimestampSource,
        typer.Option(
            help=(
                "Frame timestamps: 'overlay' reads the stamp the camera burns into each "
                "frame; 'config' uses whichever clock the PrepareConfig video section names."
            )
        ),
    ] = TimestampSource.overlay,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="PrepareConfig YAML used when --timestamps config.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    upload: Annotated[
        UploadChoice, typer.Option(help="What to push to Drive.")
    ] = UploadChoice.none,
    drive_remote: Annotated[
        str | None, typer.Option(help="rclone remote name (from `rclone config`).")
    ] = None,
    drive_root: Annotated[
        str, typer.Option(help="Folder prefix inside the remote.")
    ] = "LabMiM/allsky",
    rclone_arg: Annotated[
        list[str] | None, typer.Option("--rclone-arg", help="Extra rclone flag (repeatable).")
    ] = None,
    prune_uploaded: Annotated[
        bool, typer.Option("--prune-uploaded", help="Delete the local mp4 once it is uploaded.")
    ] = False,
    ca_file: Annotated[
        Path | None,
        typer.Option(help="PEM bundle to verify against instead of the AIA repair.", exists=True),
    ] = None,
    insecure: Annotated[
        bool, typer.Option("--insecure", help="Disable TLS verification (last resort).")
    ] = False,
    timeout: Annotated[float, typer.Option(help="Per-request timeout in seconds.")] = 60.0,
    retries: Annotated[int, typer.Option(min=1, help="Attempts per request.")] = 3,
    delay: Annotated[float, typer.Option(help="Pause between downloads, in seconds.")] = 1.0,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report the plan; transfer nothing.")
    ] = False,
) -> None:
    """Mirror new days from the Salvador all-sky archive; optionally extract frames and upload.

    The ledger under ``<data-dir>/.state`` records every downloaded video,
    extracted frame set and completed upload, and is held under a lock for the
    whole run, so a rerun transfers only what is missing.  It is saved after each
    day, so a failure mid-run keeps the days already done.

    A day whose burned-in stamps cannot be read is skipped rather than fatal, on
    the same reasoning ``prepare-local`` states: the fault is a permanent
    property of that day's bytes, so a run that failed on it fails on it forever
    and an exit code that can never go green signals nothing.  The refusal is
    filed in the ledger against the extraction parameters and the video's digest,
    so the daily job stops re-decoding that video every night while a
    re-download or a different ``--step`` still gets a fresh attempt.

    Raises
    ------
    typer.BadParameter
        For a malformed ``--since``/``--until``, ``--upload`` without
        ``--drive-remote``, ``--prune-uploaded`` without ``--upload``, or
        ``--upload frames``/``both`` without ``--extract``.
    typer.Exit
        Code 1 when the archive listing or rclone is unreachable, or when any day
        failed to process for a reason a rerun could get past.
    """
    configure_cli_logging()
    root = data_dir
    videos_dir = root / VIDEOS_SUBDIR
    frames_root = root / FRAMES_SUBDIR
    state_dir = root / STATE_SUBDIR
    since_date = _parse_date(since, "--since")
    until_date = _parse_date(until, "--until")

    if upload is not UploadChoice.none and drive_remote is None:
        raise typer.BadParameter("--upload needs --drive-remote (the name from `rclone config`)")
    if upload is UploadChoice.none and prune_uploaded:
        raise typer.BadParameter("--prune-uploaded only makes sense together with --upload")
    if upload.wants_frames and not extract:
        raise typer.BadParameter("--upload frames/both needs --extract")

    target = DriveTarget(remote=drive_remote, root=drive_root) if drive_remote else None

    try:
        client = _build_client(
            base_url,
            state_dir,
            ca_file=ca_file,
            insecure=insecure,
            timeout=timeout,
            retries=retries,
            delay=delay,
        )
        published = client.list_videos()
    except ArchiveError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc

    from allsky.overlay import OverlayTimestampError

    selected = _select(published, since=since_date, until=until_date)
    video_config = _video_config(timestamps, config)
    clock = _clock(video_config)
    failures = 0
    with ledger_lock(state_dir / LEDGER_FILENAME):
        ledger = Ledger.load(state_dir / LEDGER_FILENAME)
        plans = [
            plan
            for plan in (
                _plan_day(
                    entry,
                    ledger=ledger,
                    root=root,
                    target=target,
                    extract=extract,
                    step=step,
                    resize=resize,
                    timestamps=clock,
                    upload=upload,
                )
                for entry in selected
            )
            if plan.has_work
        ]
        _warn_about_ambiguously_clocked_days(
            ledger, selected, clock=clock, step=step, resize=resize, extract=extract
        )
        typer.echo(
            f"published: {len(published)} day(s); selected: {len(selected)}; "
            f"needing work: {len(plans)}"
        )
        if not plans:
            typer.echo("nothing to do — the local mirror and the remote are up to date")
            return
        if limit is not None:
            plans = plans[:limit]
            typer.echo(f"limited to {len(plans)} day(s) this run")
        if list_only or dry_run:
            for plan in plans:
                typer.echo(f"  {plan.describe()}")
            return

        uploader = (
            RcloneUploader(target=target, extra_args=tuple(rclone_arg or ()))
            if target is not None and upload is not UploadChoice.none
            else None
        )
        if uploader is not None:
            try:
                uploader.check()
            except RcloneError as exc:
                logger.error("%s", exc)
                raise typer.Exit(code=1) from exc

        processed = 0
        unusable = 0
        for plan in plans:
            try:
                _process_day(
                    plan,
                    client=client,
                    ledger=ledger,
                    root=root,
                    videos_dir=videos_dir,
                    frames_root=frames_root,
                    video_config=video_config,
                    step=step,
                    resize=resize,
                    uploader=uploader,
                    prune_uploaded=prune_uploaded,
                )
                processed += 1
            except OverlayTimestampError as exc:
                unusable += 1
                ledger.record_extraction_fault(
                    plan.entry.key,
                    reason=str(exc),
                    step=step,
                    resize=resize,
                    timestamps=clock.value,
                )
                logger.warning("%s cannot be timestamped, skipping: %s", plan.entry.filename, exc)
            except (
                ArchiveError,
                RcloneError,
                OSError,
                ValueError,
                http.client.HTTPException,
            ) as exc:
                failures += 1
                logger.error("%s failed: %s", plan.entry.filename, exc)
            finally:
                ledger.save()

        unusable_note = f", {unusable} unusable" if unusable else ""
        typer.echo(f"done: {processed} day(s) processed{unusable_note}, {failures} failed")
        typer.echo(f"ledger: {ledger.path}")
    if failures:
        raise typer.Exit(code=1)


def _video_config(timestamps: TimestampSource, config: Path | None) -> Any:
    """Resolve the video section that drives frame timestamps into a ``VideoConfig``.

    ``--timestamps overlay`` is the built-in default section, so ``--config``
    carries nothing on that path; ``--timestamps config`` defers the whole clock
    decision to the YAML, including the case where the YAML names the overlay.
    Which clock the resolved section selects — and the warning the modelled
    cadence earns — is :func:`allsky.overlay.extract_frames_for`'s to state, so
    this returns a section rather than a verdict.
    """
    from allsky.config import PrepareConfig, VideoConfig, load_prepare_config

    if timestamps is TimestampSource.overlay:
        if config is not None:
            logger.warning("--config is ignored with --timestamps overlay")
        return VideoConfig()
    cfg = PrepareConfig() if config is None else load_prepare_config(config)
    return cfg.video


def _clock(video_config: Any) -> TimestampSource:
    """The timestamp source *video_config* selects, in the ledger's vocabulary.

    The ledger and the resume gate speak the flag's names, the config speaks
    ``overlay``/``modelled``; recording the clock the extraction actually used —
    rather than the flag that was typed — is what keeps a day extracted through a
    config naming the overlay from being extracted again on every run.
    """
    return (
        TimestampSource.overlay if video_config.timestamps == "overlay" else TimestampSource.config
    )


def _process_day(
    plan: DayPlan,
    *,
    client: ArchiveClient,
    ledger: Ledger,
    root: Path,
    videos_dir: Path,
    frames_root: Path,
    video_config: Any,
    step: int,
    resize: int | None,
    uploader: RcloneUploader | None,
    prune_uploaded: bool,
) -> None:
    """Execute one :class:`DayPlan`, recording each completed step in the ledger.

    The ledger is written to disk as each transfer lands, so an interruption —
    including one that never reaches the caller's ``finally`` — leaves the work
    already done visible to the next run.  With *prune_uploaded* the local mp4 is
    deleted only after the ledger confirms the upload.  A re-extraction produces
    its new frame set beside the day's previous one and only replaces it once the
    new set exists, so a failure keeps the previous frames and the record that
    describes them: some days carry a fault in the video bytes that no rerun can
    get past, and their frames are unrecoverable once deleted.
    """
    entry = plan.entry
    video_path = ledger.video_path(entry.key, root=root) or videos_dir / entry.filename
    if plan.download:
        result = client.download(entry, videos_dir)
        ledger.record_video(result, root=root)
        ledger.save()
        video_path = result.path

    frames_dir = frames_root / entry.key
    if plan.extract:
        manifest = _extract_replacing_previous_frames(
            video_path, frames_dir, video_config, step=step, resize=resize
        )
        ledger.record_frames(
            entry.key,
            directory=frames_dir.relative_to(root)
            if frames_dir.is_relative_to(root)
            else frames_dir,
            count=len(manifest),
            step=step,
            resize=resize,
            timestamps=_clock(video_config).value,
        )
        ledger.save()

    if uploader is None:
        return

    if plan.upload_video:
        destination = uploader.upload_file(video_path, REMOTE_VIDEOS, entry.filename)
        record = ledger.video(entry.key) or {}
        ledger.record_upload(
            entry.key,
            destination,
            kind="video",
            size=record.get("size"),
            sha256=record.get("sha256"),
        )
        ledger.save()

    if plan.upload_frames:
        recorded = ledger.frames(entry.key) or {}
        source = ledger.frames_dir(entry.key, root=root) or frames_dir
        destination = uploader.upload_dir(source, REMOTE_FRAMES, entry.key, pattern=FRAME_PATTERN)
        ledger.record_upload(
            entry.key, destination, kind="frames", count=recorded.get("count"), step=step
        )
        ledger.save()

    if prune_uploaded and ledger.uploaded(entry.key, _video_destination(uploader.target, entry)):
        video_path.unlink(missing_ok=True)
        ledger.mark_pruned(entry.key)
        ledger.save()
        logger.info("pruned local %s (kept on Drive)", video_path.name)


def _discard_previous_frames(frames_dir: Path) -> None:
    """Delete the JPEGs an earlier extraction of this day left in *frames_dir*.

    Both extraction paths name a frame after that frame's own capture time, so a
    re-extraction under a different clock, step or size writes new filenames
    beside the old ones instead of replacing them.  The two sets describe the
    same day with different times, and the ledger count and the Drive upload
    would carry both.  Only :data:`FRAME_PATTERN` — what the uploader mirrors —
    is removed, never the directory itself nor anything else it holds.
    """
    discarded = [path for path in sorted(frames_dir.glob(FRAME_PATTERN)) if path.is_file()]
    for path in discarded:
        path.unlink()
    if discarded:
        logger.info(
            "discarded %d frame(s) from an earlier extraction in %s", len(discarded), frames_dir
        )


def _extract_replacing_previous_frames(
    video_path: Path, frames_dir: Path, video_config: Any, *, step: int, resize: int | None
) -> Any:
    """Extract *video_path* into a staging sibling, then swap it over *frames_dir*.

    Extraction fails for whole days — a video whose burned-in stamps cannot be
    read is refused every time it is tried — so the frames a previous run left
    are only removed once a full replacement set exists beside them.  The staging
    directory is named after the day and cleared on entry rather than made unique
    per process: the ledger lock already serialises runs, and a unique name would
    turn every crash into debris nothing ever collects.

    A replacement set that holds no frame is refused before anything is
    discarded, and a swap that fails partway keeps the staging directory the
    frames it had not moved yet: both are cases where deleting first and finding
    out afterwards costs a day that no rerun can rebuild once the video is gone.

    Returns
    -------
    pandas.DataFrame
        The frame manifest, with ``frame_path`` pointing at the frames where they
        now live rather than at the staging directory they were written in.

    Raises
    ------
    allsky.overlay.OverlayTimestampError
        If the video decodes to no usable frame, in which case *frames_dir* is
        left exactly as it was.
    """
    from allsky.overlay import MANIFEST_FILENAME, OverlayTimestampError, extract_frames_for
    from labmim_core.atomic import atomic_write

    staging = frames_dir.with_name(f".{frames_dir.name}.incoming")
    shutil.rmtree(staging, ignore_errors=True)
    swapping = False
    try:
        manifest = extract_frames_for(video_path, staging, video_config, step=step, resize=resize)
        staged = sorted(staging.glob(FRAME_PATTERN))
        if not staged:
            raise OverlayTimestampError(
                f"{video_path.name} produced no frame — keeping the frames already in {frames_dir}"
            )
        swapping = True
        _discard_previous_frames(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame in staged:
            frame.replace(frames_dir / frame.name)
        manifest["frame_path"] = [
            str(frames_dir / Path(written).name) for written in manifest["frame_path"]
        ]
        atomic_write(
            frames_dir / MANIFEST_FILENAME, lambda tmp: manifest.to_parquet(tmp, index=False)
        )
        swapping = False
    finally:
        if not swapping:
            shutil.rmtree(staging, ignore_errors=True)
    return manifest


def snapshot(
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Directory for the image, sidecar and prediction.")
    ],
    base_url: Annotated[str, typer.Option(help="All-sky website root.")] = ARCHIVE_BASE_URL,
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            "-k",
            help="Trained checkpoint; supplying it runs a prediction on the frame.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    embeddings_dir: Annotated[
        Path | None,
        typer.Option(
            "--embeddings-dir",
            help="Embedding store to encode against, for a checkpoint moved off its "
            "training machine (embedding-mode checkpoints only).",
            exists=True,
            file_okay=False,
        ),
    ] = None,
    sensor_csv: Annotated[
        Path | None,
        typer.Option(
            help="Logger export supplying the meteorological features (else imputed).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    tolerance_minutes: Annotated[
        float | None,
        typer.Option(
            min=0,
            help="Max age of the sensor row used, in minutes; default: the checkpoint's own.",
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device for inference.")] = "cpu",
    trust_checkpoint: Annotated[
        bool,
        typer.Option(
            "--trust-checkpoint/--no-trust-checkpoint",
            help="Read the checkpoint with the unrestricted unpickler (own files only).",
        ),
    ] = False,
    drive_remote: Annotated[
        str | None, typer.Option(help="Upload the snapshot to this rclone remote.")
    ] = None,
    drive_root: Annotated[
        str, typer.Option(help="Folder prefix inside the remote.")
    ] = "LabMiM/allsky",
    rclone_arg: Annotated[
        list[str] | None, typer.Option("--rclone-arg", help="Extra rclone flag (repeatable).")
    ] = None,
    ca_file: Annotated[
        Path | None,
        typer.Option(help="PEM bundle to verify against instead of the AIA repair.", exists=True),
    ] = None,
    insecure: Annotated[
        bool, typer.Option("--insecure", help="Disable TLS verification (last resort).")
    ] = False,
    timeout: Annotated[float, typer.Option(help="Request timeout in seconds.")] = 60.0,
) -> None:
    """Capture the camera's current frame, optionally predict on it, and write both.

    Writes the JPEG plus a metadata sidecar into *out_dir*; with ``--checkpoint``
    it adds ``<image>.prediction.json``.  Meteorological features come from
    ``--sensor-csv`` when a row falls within ``--tolerance-minutes`` of the
    capture, and any feature without one is imputed at its training mean and
    named in the warning the command prints.  ``--embeddings-dir`` names the
    store the live frame is encoded against, for an embedding-mode checkpoint
    copied away from the machine whose absolute path it carries.

    Raises
    ------
    typer.Exit
        Code 1 when the capture, the prediction or the Drive upload fails.
    """
    configure_cli_logging()
    from allsky.snapshot import capture_snapshot

    try:
        client = _build_client(
            base_url,
            out_dir / STATE_SUBDIR,
            ca_file=ca_file,
            insecure=insecure,
            timeout=timeout,
            retries=3,
            delay=0.0,
        )
        captured = capture_snapshot(client, out_dir)
    except (ArchiveError, ValueError, OSError, http.client.HTTPException) as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"captured {captured.image_path} at {captured.captured_at}")
    typer.echo(f"  metadata: {captured.metadata_path}")

    prediction_path = (
        _predict_snapshot(
            captured,
            checkpoint,
            sensor_csv=sensor_csv,
            tolerance_minutes=tolerance_minutes,
            device=device,
            trust_checkpoint=trust_checkpoint,
            embeddings_dir=embeddings_dir,
        )
        if checkpoint is not None
        else None
    )

    if drive_remote is not None:
        uploader = RcloneUploader(
            target=DriveTarget(remote=drive_remote, root=drive_root),
            extra_args=tuple(rclone_arg or ()),
        )
        day = f"{captured.captured_at:%Y%m%d}"
        try:
            for path in _existing(captured.image_path, captured.metadata_path, prediction_path):
                uploader.upload_file(path, REMOTE_SNAPSHOTS, day, path.name)
        except RcloneError as exc:
            logger.error("%s", exc)
            raise typer.Exit(code=1) from exc
        typer.echo(f"  uploaded to {uploader.target.path(REMOTE_SNAPSHOTS, day)}")


def _predict_snapshot(
    captured: Any,
    checkpoint: Path,
    *,
    sensor_csv: Path | None,
    tolerance_minutes: float | None,
    device: str,
    trust_checkpoint: bool,
    embeddings_dir: Path | None,
) -> Path:
    """Predict on the captured frame and write ``<image>.prediction.json``.

    Any feature the sensor export could not supply within *tolerance_minutes* is
    imputed at its training mean, and those names are echoed as a warning so a
    prediction is never read as fully informed when it is not.  Left None the
    checkpoint's own pairing window applies, which is the one its manifest was
    built with; a free-standing default here paired live rows three times wider
    than training accepted.  The payload is
    written through the strict JSON writer, so a non-finite prediction fails the
    command instead of landing on disk as a bare ``NaN`` token that every strict
    reader of the file rejects wholesale.
    """
    import pandas as pd

    from allsky.snapshot import predict_snapshot
    from labmim_core.atomic import atomic_write_strict_json

    try:
        prediction = predict_snapshot(
            captured.image_path,
            checkpoint,
            timestamp=captured.captured_at,
            sensor_csv=sensor_csv,
            tolerance=(
                None if tolerance_minutes is None else pd.Timedelta(minutes=tolerance_minutes)
            ),
            device=device,
            trust_checkpoint=trust_checkpoint,
            embeddings_dir=embeddings_dir,
        )
    except (ValueError, KeyError, RuntimeError, OSError) as exc:
        logger.error("prediction failed: %s", exc)
        raise typer.Exit(code=1) from exc

    try:
        path = atomic_write_strict_json(
            captured.image_path.with_suffix(".prediction.json"), prediction
        )
    except ValueError as exc:
        logger.error("prediction is not publishable: %s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"  prediction: {path}")
    for name, value in prediction["predictions"].items():
        typer.echo(f"    {name}: {value}")
    imputed: list[str] = prediction["features"]["imputed"]
    if imputed:
        typer.echo(
            f"  WARNING: {len(imputed)} feature(s) imputed at the training mean "
            f"({', '.join(imputed)}) — pass --sensor-csv for a fully informed prediction"
        )
    return path


def _existing(*paths: Any) -> tuple[Path, ...]:
    return tuple(Path(path) for path in paths if path is not None and Path(path).is_file())


def register(app: typer.Typer) -> None:
    """Attach the ``sync-archive`` and ``snapshot`` commands onto *app*."""
    app.command("sync-archive")(sync_archive)
    app.command("snapshot")(snapshot)
