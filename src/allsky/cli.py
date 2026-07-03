"""``allsky`` command-line interface.

Examples
--------
Show the resolved configuration and the video -> wall-clock mapping:
    allsky info --config configs/allsky/default.yaml

Extract every 60th frame from a one-day timelapse:
    allsky extract-frames data/all-sky/allsky-20260625.mp4 --out scratch/frames --step 60

Pair frames with sensor records into a training index:
    allsky build-index --manifest scratch/frames/manifest.parquet --out output/allsky/index.parquet

Train (or resume) SkyFusionNet:
    allsky train --index output/allsky/index.parquet --epochs 2 --device cpu

Heavy dependencies (torch, imageio-ffmpeg) and the sibling pipeline modules
are imported lazily inside each command so ``allsky info``/``--help`` work in
a minimal environment.
"""

from __future__ import annotations

import glob
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from allsky.config import load_config

app = typer.Typer(
    name="allsky",
    no_args_is_help=True,
    help="All-sky camera + radiation-sensor fusion pipeline (LabMiM/UFBA).",
)

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


@app.command()
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


@app.command("extract-frames")
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


@app.command("build-index")
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


@app.command()
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
        Path | None,
        typer.Option(
            help="Checkpoint to resume from (e.g. <out-dir>/last.pt).", exists=True, dir_okay=False
        ),
    ] = None,
    epochs: Annotated[int | None, typer.Option(min=1, help="Override train.epochs.")] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Override train.batch_size.")
    ] = None,
    device: Annotated[DeviceChoice | None, typer.Option(help="Override train.device.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Override train.out_dir.")] = None,
    val_fraction: Annotated[
        float, typer.Option(min=0.01, max=0.99, help="Fraction of DAYS held out for validation.")
    ] = 0.2,
) -> None:
    """Train SkyFusionNet on a built index (day-based train/val split)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    cfg = load_config(config)
    if epochs is not None:
        cfg.train.epochs = epochs
    if batch_size is not None:
        cfg.train.batch_size = batch_size
    if device is not None:
        cfg.train.device = str(device)
    if out_dir is not None:
        cfg.train.out_dir = str(out_dir)

    from allsky.training import train as run_training  # lazy: pulls torch at run time

    metrics = run_training(cfg, index_path=index, resume=resume, val_fraction=val_fraction)
    typer.echo(json.dumps(metrics, indent=2, default=str))


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
