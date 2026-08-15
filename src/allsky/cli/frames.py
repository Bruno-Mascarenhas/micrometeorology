"""``extract-frames`` CLI command.

Extracts timestamped JPEG frames from a single all-sky timelapse video.  The
``video.timestamps`` field of a :class:`allsky.config.PrepareConfig` (built-in
defaults when ``--config`` is omitted) selects the frame -> wall-clock time
mapping, the same way ``prepare-local`` and ``sync-archive`` do: the burned-in
overlay by default, the ``start_time``/``minutes_per_frame`` model on request.
This is the low-level, single-video entry point; the full local pipeline
(extract -> manifest -> splits) lives in ``allsky prepare-local``.

imageio-ffmpeg is imported lazily inside the command so ``allsky --help`` stays
light and torch-free.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from allsky.config import PrepareConfig, load_prepare_config

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="PrepareConfig YAML for the video time mapping (defaults to built-in defaults).",
        exists=True,
        dir_okay=False,
    ),
]


def extract_frames_cmd(
    video: Annotated[
        Path,
        typer.Argument(
            help="One-day timelapse mp4 (allsky-YYYYMMDD.mp4).", exists=True, dir_okay=False
        ),
    ],
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Directory for JPEG frames + manifest parquet.")
    ],
    step: Annotated[int, typer.Option(min=1, help="Keep every Nth frame.")] = 1,
    resize: Annotated[
        int | None, typer.Option(min=1, help="Resize frames to NxN pixels before writing.")
    ] = None,
    config: ConfigOption = None,
) -> None:
    """Extract timestamped JPEG frames from an all-sky video.

    ``video.timestamps`` picks the clock: ``overlay`` (the default) reads the
    time the camera burns into each frame, ``modelled`` places frame N at
    ``start_time + N x minutes_per_frame``.

    Raises
    ------
    typer.Exit
        Code 1 when the day's burned-in overlays cannot be read.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    cfg = PrepareConfig() if config is None else load_prepare_config(config)

    from allsky.overlay import OverlayTimestampError, extract_frames_for

    try:
        manifest = extract_frames_for(video, out_dir, cfg.video, step=step, resize=resize)
    except OverlayTimestampError as exc:
        typer.echo(f"ERROR: {video} cannot be timestamped: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Extracted {len(manifest)} frames from {video} into {out_dir}")


def register(app: typer.Typer) -> None:
    """Attach the ``extract-frames`` command onto *app*."""
    app.command("extract-frames")(extract_frames_cmd)
