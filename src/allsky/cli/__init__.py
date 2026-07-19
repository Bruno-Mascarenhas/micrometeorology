"""``allsky`` command-line interface (package).

Examples
--------
Extract every 60th frame from a one-day timelapse:
    allsky extract-frames data/all-sky/allsky-20260625.mp4 --out scratch/frames --step 60

Prepare a local dataset (frames -> v2 manifest -> day splits):
    allsky prepare-local --config configs/allsky/data/local_prepare.yaml

Precompute DINOv2 embeddings for the prepared dataset:
    allsky precompute-embeddings --config configs/allsky/data/local_prepare.yaml

Train a multimodal experiment:
    allsky train --config configs/allsky/experiments/v4_film.yaml \\
        --data-root output/allsky-mm/dataset

Evaluate a trained checkpoint:
    allsky evaluate --checkpoint output/allsky-mm/experiments/v4_film/run/best.ckpt \\
        --split test --data-root output/allsky-mm/dataset

The CLI is a package: each command group lives in its own module
(:mod:`allsky.cli.frames`, :mod:`allsky.cli.train`, :mod:`allsky.cli.prepare`,
:mod:`allsky.cli.embeddings`, :mod:`allsky.cli.evaluate`) and exposes a
``register(app)`` function called once here, so ``__init__`` never needs editing
to add a command. Heavy dependencies (torch, imageio-ffmpeg) are imported
lazily inside each command so ``allsky --help`` works in a minimal environment.
"""

from __future__ import annotations

import typer

from allsky.cli import embeddings, evaluate, frames, prepare, train

app = typer.Typer(
    name="allsky",
    no_args_is_help=True,
    help="All-sky camera + radiation-sensor fusion pipeline (LabMiM/UFBA).",
)

# Register command groups. Each register() attaches its commands inside its own
# module, so adding a command never touches this file.
frames.register(app)
prepare.register(app)
embeddings.register(app)
train.register(app)
evaluate.register(app)


def main() -> None:
    """Console-script entry point (pyproject: ``allsky = allsky.cli:main``)."""
    app()


__all__ = ["app", "main"]
