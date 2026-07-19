"""Legacy ``allsky`` CLI commands (info, extract-frames, build-index, train).

These four commands are the pre-refactor pipeline, moved here verbatim from the
old ``allsky/cli.py`` module during the Wave C1b package split. They are
attached to the package app by :func:`register`, so command names and
``--help`` output are byte-identical to the old single-module CLI.

Heavy dependencies (torch, imageio-ffmpeg) and the sibling pipeline modules are
imported lazily inside each command so ``allsky info`` / ``--help`` work in a
minimal environment.
"""

from __future__ import annotations

import glob
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from allsky.config import is_experiment_config, load_config, load_experiment_config

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Pipeline config YAML (defaults to built-in defaults).",
        exists=True,
        dir_okay=False,
    ),
]


class DeviceChoice(StrEnum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


def info(config: ConfigOption = None) -> None:
    """Print the video -> time mapping and the fully resolved configuration."""
    cfg = load_config(config)
    video = cfg.video
    typer.echo("All-sky pipeline")
    typer.echo(f"  config file:    {config if config is not None else '<built-in defaults>'}")
    typer.echo("Video -> time mapping")
    typer.echo(f"  frame 0 at:     {video.start_time} local time")
    typer.echo(f"  frame step:     {video.minutes_per_frame:g} minute(s) per frame")
    typer.echo(f"  date from name: {video.filename_date_format}")
    matches = sorted(glob.glob(video.pattern))
    typer.echo(f"  videos matched: {len(matches)} ({video.pattern})")
    for path in matches[:10]:
        typer.echo(f"    {path}")
    if cfg.sensor.diffuse_column is None:
        typer.echo(
            "Diffuse target: Erbs PSEUDO-target derived from "
            f"{cfg.sensor.ghi_column} (no shaded pyranometer yet)"
        )
    else:
        typer.echo(f"Diffuse target: measured column {cfg.sensor.diffuse_column}")
    typer.echo("Resolved config:")
    typer.echo(json.dumps(cfg.model_dump(), indent=2, default=str))


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
    cfg = load_config(config)
    from allsky.video import extract_frames  # lazy: needs imageio-ffmpeg

    manifest = extract_frames(video, out_dir, cfg.video, step=step, resize=resize)
    typer.echo(f"Extracted {len(manifest)} frames from {video} into {out_dir}")


def build_index_cmd(
    manifest: Annotated[
        Path,
        typer.Option(
            help="Frame manifest parquet from extract-frames.", exists=True, dir_okay=False
        ),
    ],
    out: Annotated[Path, typer.Option(help="Output index parquet.")] = Path(
        "output/allsky/index.parquet"
    ),
    config: ConfigOption = None,
) -> None:
    """Pair extracted frames with sensor records into a training index."""
    cfg = load_config(config)
    import pandas as pd

    from allsky.dataset import build_index  # lazy: sibling pipeline modules
    from allsky.sensors import derive_targets, load_sensor_frame

    manifest_df = pd.read_parquet(manifest)
    sensor_df = load_sensor_frame(cfg.sensor)
    sensor_df = derive_targets(sensor_df, cfg.site, cfg.sensor, cfg.labels)
    index_df = build_index(manifest_df, sensor_df, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_parquet(out)
    n_days = pd.to_datetime(index_df["timestamp"]).dt.normalize().nunique()
    typer.echo(f"Index: {len(index_df)} rows over {n_days} day(s) -> {out}")
    if "target_source" in index_df.columns:
        counts = index_df["target_source"].value_counts().to_dict()
        typer.echo(f"Target sources: {counts}")


def train(
    config: ConfigOption = None,
    index: Annotated[
        Path | None,
        typer.Option(
            help="Index parquet from build-index (default: <out-dir>/index.parquet).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option(
            help=(
                "Checkpoint to resume from. Legacy configs: path to <out-dir>/last.pt. "
                "Experiment configs: 'auto' (find last.ckpt in the run dir) or a path."
            ),
        ),
    ] = None,
    epochs: Annotated[int | None, typer.Option(min=1, help="Override train.epochs.")] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Override train.batch_size.")
    ] = None,
    device: Annotated[DeviceChoice | None, typer.Option(help="Override train.device.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Override the output/run directory.")] = None,
    data_root: Annotated[
        Path | None,
        typer.Option(help="Experiment configs: data root for the manifest/split/embeddings."),
    ] = None,
    amp: Annotated[
        bool | None,
        typer.Option("--amp/--no-amp", help="Experiment configs: override mixed precision."),
    ] = None,
    val_fraction: Annotated[
        float, typer.Option(min=0.01, max=0.99, help="Fraction of DAYS held out for validation.")
    ] = 0.2,
) -> None:
    """Train on a built index (legacy) or run a multimodal experiment.

    A config declaring ``experiment: true`` routes to the new multimodal engine
    (``--data-root`` / ``--amp`` apply, ``--resume`` accepts ``auto`` or a path);
    any other config keeps the byte-identical legacy SkyFusionNet behaviour.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    if config is not None and is_experiment_config(config):
        _train_experiment(config, epochs, batch_size, device, out_dir, data_root, amp, resume)
        return

    cfg = load_config(config)
    if epochs is not None:
        cfg.train.epochs = epochs
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if device is not None:
        cfg.train.device = str(device)
    if out_dir is not None:
        cfg.train.out_dir = str(out_dir)

    # Preserve the legacy --resume semantics (a path that must exist) now that the
    # option type is a plain string shared with the experiment dispatch.
    resume_path: Path | None = None
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_file():
            raise typer.BadParameter(
                f"resume checkpoint does not exist: {resume}", param_hint="--resume"
            )

    from allsky.training import train as run_training  # lazy: pulls torch at run time

    metrics = run_training(cfg, index_path=index, resume=resume_path, val_fraction=val_fraction)
    typer.echo(json.dumps(metrics, indent=2, default=str))


def _train_experiment(
    config: Path,
    epochs: int | None,
    batch_size: int | None,
    device: DeviceChoice | None,
    out_dir: Path | None,
    data_root: Path | None,
    amp: bool | None,
    resume: str | None,
) -> None:
    """Dispatch an ``experiment: true`` config to the multimodal training engine."""
    exp_cfg = load_experiment_config(config)
    if epochs is not None:
        exp_cfg.train.epochs = epochs
    if batch_size is not None:
        exp_cfg.train.batch_size = batch_size
    if device is not None:
        exp_cfg.train.device = str(device)

    resume_arg: str | None = None
    if resume is not None:
        if resume != "auto" and not Path(resume).exists():
            raise typer.BadParameter(
                f"resume checkpoint does not exist: {resume}", param_hint="--resume"
            )
        resume_arg = resume

    from allsky.training import run_experiment  # lazy: pulls torch at run time

    summary = run_experiment(
        exp_cfg,
        data_root=data_root,
        output_dir=out_dir,
        device=str(device) if device is not None else None,
        amp=amp,
        resume=resume_arg,
    )
    typer.echo(json.dumps(summary, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach the four legacy commands onto *app*.

    Command names are pinned explicitly so ``extract-frames`` / ``build-index``
    keep their hyphenated names (Typer would otherwise derive
    ``extract-frames-cmd`` from the function name); ``info`` and ``train`` use
    their function names unchanged. The result is identical to the historical
    ``@app.command()`` decorators on the old single-module CLI.
    """
    app.command()(info)
    app.command("extract-frames")(extract_frames_cmd)
    app.command("build-index")(build_index_cmd)
    app.command()(train)
