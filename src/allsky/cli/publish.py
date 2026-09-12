"""The ``publish-site`` command: write the sky page's model documents from the watch.

One oneshot run: verify the pin, load the served member and the two controls
once, build ``frame.json`` and its images, ``timeline.json`` and
``model.json``, write them in the order the page can tolerate (images, then
the two documents, ``frame.json`` last), prune old frames, optionally upload.
The contract and the reasons behind every step are in ``docs/allsky-site.md``.

Exit codes: 0 published; 1 the pin failed verification or a document could
not be written; 2 published, but the watch looks dead.
"""

import datetime as dt
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from allsky.archive import STATE_SUBDIR
from allsky.cli.runtime import configure_cli_logging
from allsky.serving import (
    PinVerificationError,
    ServingConfigError,
    load_serving_config,
    sha256_of_file,
    verify_pinned_checkpoints,
)
from labmim_core.atomic import JsonObjectError, atomic_write_strict_json, read_json_object

logger = logging.getLogger(__name__)

__all__ = ["EXIT_WATCH_STALE", "PUBLISHED_FILES", "register"]

EXIT_WATCH_STALE = 2
TIMELINE_FILENAME = "timeline.json"
MODEL_FILENAME = "model.json"
FRAME_FILENAME = "frame.json"
MODEL_STATE_FILENAME = "publish-model.json"
DEFAULT_DAYS = 3
DEFAULT_BLOCK_MINUTES = 5.0
DEFAULT_PRUNE_DAYS = 14
PRUNABLE_SUFFIXES = (".jpg", ".json")

#: Upload order: images before the documents that name them, ``frame.json`` last.
PUBLISHED_FILES = (
    "allsky.jpg",
    "input.jpg",
    "attribution.png",
    TIMELINE_FILENAME,
    MODEL_FILENAME,
    FRAME_FILENAME,
)


def _prune_frames(watch_dir: Path, *, older_than_days: int, now_host: dt.datetime) -> int:
    """Delete frame files (JPEG, sidecar, prediction record) older than *older_than_days*; never blocks."""
    from allsky.watch import FRAMES_SUBDIR

    frames_dir = watch_dir / FRAMES_SUBDIR
    if not frames_dir.is_dir():
        return 0
    cutoff = (now_host - dt.timedelta(days=older_than_days)).timestamp()
    removed = 0
    for path in frames_dir.iterdir():
        if path.suffix not in PRUNABLE_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    if removed:
        logger.info("pruned %d frame file(s) older than %d day(s)", removed, older_than_days)
    return removed


def _write_images(out_dir: Path, images: Mapping[str, bytes]) -> None:
    from labmim_core.atomic import atomic_write

    for name, payload in images.items():

        def write(tmp: Path, data: bytes = payload) -> int:
            return tmp.write_bytes(data)

        atomic_write(out_dir / name, write)
        logger.info("wrote %s (%d bytes)", out_dir / name, len(payload))


def _card_fingerprint(pin_path: Path, digests: list[str], inputs: list[Path]) -> list[Any]:
    """What the card is a pure function of: the pin's text, the weights, the reports, the code."""
    from allsky.provenance import code_version

    return [
        sha256_of_file(pin_path),
        list(digests),
        [[str(path), *_file_signature(path)] for path in inputs],
        code_version(),
    ]


def _file_signature(path: Path) -> list[int | None]:
    if not path.is_file():
        return [None, None]
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _cached_card(state_path: Path, fingerprint: list[Any]) -> dict[str, Any] | None:
    """The card the last publish wrote for exactly these inputs, or ``None``."""
    if not state_path.is_file():
        return None
    try:
        state = read_json_object(state_path)
    except JsonObjectError as exc:
        logger.warning("ignoring the model card cache: %s", exc)
        return None
    card = state.get("card")
    if state.get("fingerprint") != fingerprint or not isinstance(card, dict):
        return None
    return card


def _host_now() -> dt.datetime:
    """The host clock, aware UTC: file modification times are compared against it."""
    return dt.datetime.now(tz=dt.UTC)


def publish_site(
    serving: Annotated[
        Path,
        typer.Option(
            "--serving", help="Serving pin (configs/allsky/serving/<id>.yaml).", exists=True
        ),
    ],
    watch_dir: Annotated[
        Path,
        typer.Option(
            "--watch-dir",
            help="The watch root this pin's frames and blocks land in.",
            exists=True,
            file_okay=False,
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out", "-o", help="The site's Ceu/ directory the documents are written into."
        ),
    ],
    days: Annotated[
        int, typer.Option(min=1, help="Days of block predictions the timeline covers.")
    ] = DEFAULT_DAYS,
    block_minutes: Annotated[
        float, typer.Option(min=0.0, help="The logger's averaging interval, in minutes.")
    ] = DEFAULT_BLOCK_MINUTES,
    sensor_csv: Annotated[
        Path | None,
        typer.Option(
            help="Station export with a PSP_Wm2_Avg column; publishes the measured diffuse per block.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device for the probes.")] = "cpu",
    prune_frames_days: Annotated[
        int | None,
        typer.Option(min=1, help="Delete watch frames older than this many days (never blocks)."),
    ] = DEFAULT_PRUNE_DAYS,
    rclone_remote: Annotated[
        str | None,
        typer.Option(
            help="rclone NAME:path to copy exactly the published files into, after a successful publish."
        ),
    ] = None,
    trust_checkpoint: Annotated[
        bool,
        typer.Option(
            "--trust-checkpoint/--no-trust-checkpoint",
            help="Read the checkpoints with the unrestricted unpickler (own files only).",
        ),
    ] = False,
) -> None:
    """Write frame.json, its images, timeline.json and model.json for the sky page.

    A timeline or a card that cannot be built keeps the previous document in
    place and is logged; the frame and the images still publish from a
    digest-verified pin.

    Raises
    ------
    typer.Exit
        Code 1 when the pin is malformed or fails verification, the watch
        directory holds neither frames nor blocks, or a document cannot be
        written; code 2 after a complete publish whose frame status says the
        watch looks dead.
    """
    configure_cli_logging()
    import pandas as pd

    from allsky.config import SiteConfig
    from allsky.publish.dataset import train_max_solar_elevation_deg
    from allsky.publish.encoding import (
        MODEL_SCHEMA,
        document_header,
        publish_stamp,
        write_document,
    )
    from allsky.publish.frame import FramePublishError, LoadedControls, build_frame_artifacts
    from allsky.publish.model_card import (
        ModelCardError,
        build_model_card,
        card_inputs,
        checkpoint_metadata,
    )
    from allsky.publish.timeline import TimelineError, build_timeline
    from allsky.snapshot import _site_now, load_served_model, read_station_export
    from allsky.watch import BLOCKS_SUBDIR, FRAMES_SUBDIR

    if not (watch_dir / FRAMES_SUBDIR).is_dir() and not (watch_dir / BLOCKS_SUBDIR).is_dir():
        logger.error(
            "%s holds neither %s/ nor %s/: not a watch directory",
            watch_dir,
            FRAMES_SUBDIR,
            BLOCKS_SUBDIR,
        )
        raise typer.Exit(code=1)
    try:
        pin = load_serving_config(serving)
        digests = verify_pinned_checkpoints(pin, cache_dir=watch_dir / STATE_SUBDIR)
    except (ServingConfigError, PinVerificationError) as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    digest_of = {
        member.path.resolve(): digest
        for member, digest in zip(pin.verified_checkpoints, digests, strict=True)
    }
    stamp = publish_stamp()
    now_local = _site_now()
    now_host_utc = pd.Timestamp(_host_now()).floor("s")
    site = SiteConfig()
    train_max = train_max_solar_elevation_deg(pin.reports.dataset)
    if prune_frames_days is not None and prune_frames_days <= days:
        logger.error("--prune-frames-days must exceed --days (%d)", days)
        raise typer.Exit(code=1)
    try:
        station = read_station_export(sensor_csv) if sensor_csv is not None else None
    except (OSError, ValueError) as exc:
        logger.error("cannot read the station export %s: %s", sensor_csv, exc)
        raise typer.Exit(code=1) from exc

    served = load_served_model(
        pin.attribution_member.path, device=device, trust_checkpoint=trust_checkpoint
    )
    recorded_floor = served.min_solar_elevation_deg
    if recorded_floor is not None and abs(recorded_floor - pin.min_elevation_deg) > 1e-9:
        logger.error(
            "pin %s declares min_elevation_deg %g but %s records the floor %g its manifest "
            "was built with",
            pin.id,
            pin.min_elevation_deg,
            pin.attribution_member.path,
            recorded_floor,
        )
        raise typer.Exit(code=1)
    controls = LoadedControls(
        sensor_only=load_served_model(
            pin.controls.sensor_only.checkpoint.path,
            device=device,
            trust_checkpoint=trust_checkpoint,
        ),
        climatology=load_served_model(
            pin.controls.climatology.checkpoint.path,
            device=device,
            trust_checkpoint=trust_checkpoint,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        frame = build_frame_artifacts(
            watch_dir,
            pin=pin,
            digests=digest_of,
            served=served,
            controls=controls,
            stamp=stamp,
            now_local=now_local,
            now_host_utc=now_host_utc,
            site=site,
            out_dir=out_dir,
            block_minutes=block_minutes,
            train_max_elevation_deg=train_max,
            sensor_csv=station,
        )
    except FramePublishError as exc:
        logger.error("frame.json not published: %s", exc)
        raise typer.Exit(code=1) from exc

    timeline: dict[str, Any] | None
    try:
        timeline = build_timeline(
            watch_dir,
            pin=pin,
            stamp=stamp,
            now_local=now_local,
            days=days,
            site=site,
            block_minutes=block_minutes,
            sensor_csv=station,
            train_max_elevation_deg=train_max,
        )
    except TimelineError as exc:
        logger.error("timeline.json not published (the previous one is kept): %s", exc)
        timeline = None

    card: dict[str, Any] | None
    card_state = watch_dir / STATE_SUBDIR / MODEL_STATE_FILENAME
    fingerprint = _card_fingerprint(serving, digests, card_inputs(pin))
    cached = _cached_card(card_state, fingerprint)
    if cached is not None:
        card = {**cached, **document_header(MODEL_SCHEMA, stamp)}
        logger.info("model.json inputs unchanged; re-stamping the cached card")
    else:
        try:
            members = [
                checkpoint_metadata(
                    member.path,
                    digest_of[member.path.resolve()],
                    trust_checkpoint=trust_checkpoint,
                    payload=served.checkpoint if member is pin.attribution_member else None,
                )
                for member in pin.frame_checkpoints
            ]
            card = build_model_card(pin, members, stamp=stamp)
        except (ModelCardError, KeyError, ValueError, OSError) as exc:
            logger.error("model.json not published (the previous one is kept): %s", exc)
            card = None
        if card is not None:
            atomic_write_strict_json(card_state, {"fingerprint": fingerprint, "card": card})

    try:
        _write_images(out_dir, frame.images)
        if timeline is not None:
            write_document(out_dir / TIMELINE_FILENAME, timeline)
        if card is not None:
            write_document(out_dir / MODEL_FILENAME, card)
        write_document(out_dir / FRAME_FILENAME, frame.document)
    except (OSError, ValueError) as exc:
        logger.error("could not write the documents: %s", exc)
        raise typer.Exit(code=1) from exc

    if prune_frames_days is not None:
        _prune_frames(watch_dir, older_than_days=prune_frames_days, now_host=_host_now())

    if rclone_remote is not None:
        from allsky.drive import DriveTarget, RcloneError, RcloneUploader

        remote, _, root = rclone_remote.partition(":")
        uploader = RcloneUploader(DriveTarget(remote=remote, root=root))
        present = [name for name in PUBLISHED_FILES if (out_dir / name).is_file()]
        try:
            uploader.copy_files(out_dir, present)
        except RcloneError as exc:
            logger.error("upload failed: %s", exc)
            raise typer.Exit(code=1) from exc

    status = frame.document["status"]
    typer.echo(
        f"published {stamp.version} to {out_dir}: frame {status['reason']}, "
        f"timeline.json {'written' if timeline is not None else 'kept'}, "
        f"model.json {'written' if card is not None else 'kept'}"
    )
    if not status["watch_alive"]:
        logger.error("the watch looks dead: %s", status["reason_pt"])
        raise typer.Exit(code=EXIT_WATCH_STALE)


def register(app: typer.Typer) -> None:
    """Attach the ``publish-site`` command onto *app*."""
    app.command("publish-site")(publish_site)
