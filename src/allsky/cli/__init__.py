"""``allsky`` command-line interface (package).

Examples
--------
Mirror the public archive, then capture the camera's current frame:
    allsky sync-archive --data-dir data/all-sky --extract
    allsky snapshot --out output/allsky-mm/snapshots

Keep polling the live frame, score every frame with two checkpoints (the
sky class from best.ckpt, the diffuse from last.ckpt) and record each
5-minute datalogger block from the mean of its frames:
    allsky watch --out output/allsky-mm/watch \\
        --checkpoint-frame run/best.ckpt --checkpoint-frame run/last.ckpt \\
        --frame-sky-role best --frame-dhi-role last --min-elevation-deg 10

Score each block with a block checkpoint instead:
    allsky watch --out output/allsky-mm/watch --checkpoint-block run/best.ckpt --min-elevation-deg 10

Extract every 60th frame from a one-day timelapse:
    allsky extract-frames data/all-sky/allsky-20260625.mp4 --out scratch/frames --step 60

Prepare a local dataset (frames -> v2 manifest -> day splits):
    allsky prepare-local --config configs/allsky/data/local_prepare.yaml

Precompute DINOv2 embeddings for the prepared dataset:
    allsky precompute-embeddings --config configs/allsky/data/local_prepare.yaml

Attach the overlay exposure and relative radiance to a prepared dataset:
    allsky exposure-features --data-root output/allsky-mm/dataset-iso \\
        --videos data/all-sky/videos --out output/allsky-mm/dataset-iso-exp

Train a multimodal experiment:
    allsky train --config configs/allsky/experiments/v4_film.yaml \\
        --data-root output/allsky-mm/dataset

Evaluate a trained checkpoint:
    allsky evaluate --checkpoint output/allsky-mm/experiments/v4_film/run/best.ckpt \\
        --split test --data-root output/allsky-mm/dataset

The CLI is a package: each command group lives in its own module
(:mod:`allsky.cli.archive`, :mod:`allsky.cli.frames`, :mod:`allsky.cli.train`,
:mod:`allsky.cli.prepare`, :mod:`allsky.cli.embeddings`,
:mod:`allsky.cli.exposure`, :mod:`allsky.cli.evaluate`, :mod:`allsky.cli.watch`) and exposes a
``register(app)`` function called once here, so ``__init__`` never needs editing
to add a command. Heavy dependencies (torch, imageio-ffmpeg) are imported
lazily inside each command so ``allsky --help`` works in a minimal environment.
"""

import typer

from allsky.cli import archive, embeddings, evaluate, exposure, frames, prepare, train, watch

app = typer.Typer(
    name="allsky",
    rich_markup_mode="markdown",
    no_args_is_help=True,
    help="All-sky camera + radiation-sensor fusion pipeline (LabMiM/UFBA).",
)

archive.register(app)
frames.register(app)
prepare.register(app)
embeddings.register(app)
exposure.register(app)
train.register(app)
evaluate.register(app)
watch.register(app)


def main() -> None:
    """Console-script entry point (pyproject: ``allsky = allsky.cli:main``)."""
    app()


__all__ = ["app", "main"]
