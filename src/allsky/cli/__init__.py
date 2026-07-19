"""``allsky`` command-line interface (package).

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

The CLI is a package: legacy commands live in :mod:`allsky.cli.legacy`, and the
new command groups added by later waves live in :mod:`allsky.cli.prepare`,
:mod:`allsky.cli.embeddings` and :mod:`allsky.cli.evaluate`. Each exposes a
``register(app)`` function called once here, so ``__init__`` never needs editing
again as those modules are filled in. Heavy dependencies (torch,
imageio-ffmpeg) are imported lazily inside each command so ``allsky info`` /
``--help`` work in a minimal environment.
"""

from __future__ import annotations

import typer

from allsky.cli import embeddings, evaluate, legacy, prepare

app = typer.Typer(
    name="allsky",
    no_args_is_help=True,
    help="All-sky camera + radiation-sensor fusion pipeline (LabMiM/UFBA).",
)

# Register command groups. The legacy pipeline ships today; the stub modules are
# populated by later waves (C2: prepare/embeddings, C4: evaluate). Importing and
# calling each register() here means later waves add commands inside their own
# module and never touch this file.
legacy.register(app)
prepare.register(app)
embeddings.register(app)
evaluate.register(app)


def main() -> None:
    """Console-script entry point (pyproject: ``allsky = allsky.cli:main``)."""
    app()


__all__ = ["app", "main"]
