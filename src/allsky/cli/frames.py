"""``extract-frames`` CLI command.

Extracts timestamped JPEG frames from a single all-sky timelapse video, using
the ``video`` section of a :class:`allsky.config.PrepareConfig` (built-in
defaults when ``--config`` is omitted) for the frame -> wall-clock time mapping.
This is the low-level, single-video entry point; the full local pipeline
(extract -> manifest -> splits) lives in ``allsky prepare-local``.

imageio-ffmpeg is imported lazily inside the command so ``allsky --help`` stays
light and torch-free.
"""

from __future__ import annotations

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
    """Extract timestamped JPEG frames from an all-sky video."""
    cfg = PrepareConfig() if config is None else load_prepare_config(config)
    from allsky.video import extract_frames  # lazy: needs imageio-ffmpeg

    manifest = extract_frames(video, out_dir, cfg.video, step=step, resize=resize)
    typer.echo(f"Extracted {len(manifest)} frames from {video} into {out_dir}")


def register(app: typer.Typer) -> None:
    """Attach the ``extract-frames`` command onto *app*."""
    app.command("extract-frames")(extract_frames_cmd)
