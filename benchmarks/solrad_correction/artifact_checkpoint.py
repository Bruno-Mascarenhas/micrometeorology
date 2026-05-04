"""Synthetic benchmark for artifact manifest and checkpoint serialization.

Examples
--------
Run with default settings:
    python benchmarks/solrad_correction/artifact_checkpoint.py

Run with a larger model:
    python benchmarks/solrad_correction/artifact_checkpoint.py --hidden-size 128 --layers 6
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Annotated

import typer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    hidden_size: Annotated[int, typer.Option(help="Hidden layer size.")] = 32,
    layers: Annotated[int, typer.Option(help="Number of layers.")] = 2,
    output_dir: Annotated[Path, typer.Option(help="Output directory for artifacts.")] = ROOT
    / "scratch"
    / "benchmarks"
    / "artifacts",
) -> None:
    """Benchmark artifact manifest and checkpoint serialization."""
    import torch

    from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
    from solrad_correction.utils.io import save_json
    from solrad_correction.utils.serialization import save_torch_checkpoint

    layout = ArtifactLayout.from_experiment_dir(output_dir)
    layout.ensure_directories()
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        *[
            torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))
            for _ in range(max(0, layers - 1))
        ],
        torch.nn.Linear(hidden_size, 1),
    )

    started = time.perf_counter()
    save_torch_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=None,
        config={"hidden_size": hidden_size, "layers": layers},
        epoch=1,
        path=layout.checkpoints_dir / "synthetic.pt",
        metadata={"benchmark": "artifact_checkpoint"},
    )
    checkpoint_seconds = time.perf_counter() - started

    save_json({"RMSE": 0.0}, layout.metrics)
    started = time.perf_counter()
    write_manifest(layout, extra={"benchmark": "artifact_checkpoint"})
    manifest_seconds = time.perf_counter() - started

    typer.echo(
        {
            "benchmark": "artifact_checkpoint",
            "checkpoint_bytes": (layout.checkpoints_dir / "synthetic.pt").stat().st_size,
            "manifest_entries": len(
                [p for p in layout.root.rglob("*") if p.is_file() and p != layout.manifest]
            ),
            "checkpoint_seconds": round(checkpoint_seconds, 6),
            "manifest_seconds": round(manifest_seconds, 6),
        }
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
