"""The ``watch`` command: poll the live frame and score each datalogger block."""

import http.client
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from allsky.archive import ARCHIVE_BASE_URL, ArchiveError
from allsky.cli.archive import STATE_SUBDIR, _build_client
from allsky.cli.runtime import configure_cli_logging

logger = logging.getLogger(__name__)


class RoleChoice(StrEnum):
    """Which members of an ensemble a head group is read from, by checkpoint stem.

    ``best`` reads the ``best.ckpt`` members, ``last`` the ``last.ckpt`` ones
    and ``all`` every member, whatever its stem.
    """

    best = "best"
    last = "last"
    all = "all"


def watch(
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Watch root: frames/ and blocks/ are written here.")
    ],
    base_url: Annotated[str, typer.Option(help="All-sky website root.")] = ARCHIVE_BASE_URL,
    checkpoint_frame: Annotated[
        list[Path] | None,
        typer.Option(
            "--checkpoint-frame",
            help="Single-frame checkpoint; scores every new frame (repeatable; more than one "
            "is averaged).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    checkpoint_block: Annotated[
        list[Path] | None,
        typer.Option(
            "--checkpoint-block",
            help="Block checkpoint (repeatable; more than one is averaged).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    frame_sky_role: Annotated[
        RoleChoice,
        typer.Option(help="Frame checkpoints whose sky heads are averaged, by file stem."),
    ] = RoleChoice.all,
    frame_dhi_role: Annotated[
        RoleChoice,
        typer.Option(
            help="Frame checkpoints whose dhi/kindex/cloud_fraction heads are averaged, by "
            "file stem."
        ),
    ] = RoleChoice.all,
    block_sky_role: Annotated[
        RoleChoice,
        typer.Option(help="Block checkpoints whose sky heads are averaged, by file stem."),
    ] = RoleChoice.all,
    block_dhi_role: Annotated[
        RoleChoice,
        typer.Option(
            help="Block checkpoints whose dhi/kindex/cloud_fraction heads are averaged, by "
            "file stem."
        ),
    ] = RoleChoice.all,
    poll_seconds: Annotated[float, typer.Option(min=0.0, help="Seconds between polls.")] = 20.0,
    block_minutes: Annotated[
        float, typer.Option(min=0.0, help="The logger's averaging interval, in minutes.")
    ] = 5.0,
    min_frames: Annotated[
        int, typer.Option(min=1, help="Fewest frames a block needs to be scored.")
    ] = 3,
    grace_seconds: Annotated[
        float,
        typer.Option(
            min=0.0, help="Seconds past a block's end before it closes without a later frame."
        ),
    ] = 90.0,
    min_elevation_deg: Annotated[
        float | None,
        typer.Option(
            help="night_filter.min_solar_elevation_deg the checkpoints' manifest was built "
            "with; a frame or block with the sun below it is not scored. Required with "
            "--checkpoint-frame and with --checkpoint-block."
        ),
    ] = None,
    device: Annotated[str, typer.Option(help="Torch device for inference.")] = "cpu",
    trust_checkpoint: Annotated[
        bool,
        typer.Option(
            "--trust-checkpoint/--no-trust-checkpoint",
            help="Read the checkpoints with the unrestricted unpickler (own files only).",
        ),
    ] = False,
    max_polls: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many polls (default: run until stopped)."),
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
    """Poll the camera's live frame and score each datalogger block as it closes.

    Frames land under ``<out>/frames/`` with their sidecars; every new frame
    with the sun at or above ``--min-elevation-deg`` is scored by the
    ``--checkpoint-frame`` checkpoints into ``<image>.prediction.json``. Each
    ``--block-minutes`` block is recorded under
    ``<out>/blocks/<YYYYMMDD-HHMM>.prediction.json`` once a later frame arrives
    or ``--grace-seconds`` pass: from the ``--checkpoint-block`` checkpoints
    when given (``source: block_model``), else from the mean of its frames'
    predictions (``source: frame_aggregate``). A block with fewer than
    ``--min-frames`` frames, with the sun below the floor, or with no scored
    frame and no block checkpoint is recorded as skipped. More than one
    checkpoint of a kind is averaged; the ``--*-role`` options read a head
    group from the ``best.ckpt`` members, the ``last.ckpt`` ones, or all.
    Restarting resumes from what is on disk. Ctrl-C stops cleanly.

    Raises
    ------
    typer.Exit
        Code 1 when the client cannot be built, a role selects no checkpoint,
        a checkpoint is given without ``--min-elevation-deg``, or a
        checkpoint of either kind is refused at start-up.
    """
    configure_cli_logging()
    from allsky.snapshot import capture_snapshot
    from allsky.watch import FRAMES_SUBDIR, run_watch

    frames_dir = out_dir / FRAMES_SUBDIR
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
        polls = run_watch(
            lambda: capture_snapshot(client, frames_dir),
            out_dir,
            checkpoint_frames=tuple(checkpoint_frame or ()),
            checkpoint_blocks=tuple(checkpoint_block or ()),
            frame_sky_role=frame_sky_role.value,
            frame_dhi_role=frame_dhi_role.value,
            block_sky_role=block_sky_role.value,
            block_dhi_role=block_dhi_role.value,
            poll_seconds=poll_seconds,
            block_minutes=block_minutes,
            min_frames=min_frames,
            grace_seconds=grace_seconds,
            min_solar_elevation_deg=min_elevation_deg,
            device=device,
            trust_checkpoint=trust_checkpoint,
            max_polls=max_polls,
        )
    except KeyboardInterrupt:
        typer.echo("watch stopped")
        return
    except (ArchiveError, ValueError, OSError, http.client.HTTPException) as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(f"watch finished after {polls} poll(s)")


def register(app: typer.Typer) -> None:
    """Attach the ``watch`` command onto *app*."""
    app.command("watch")(watch)
