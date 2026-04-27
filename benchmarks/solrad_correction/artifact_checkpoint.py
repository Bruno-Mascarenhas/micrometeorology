"""Synthetic benchmark for artifact manifest and checkpoint serialization."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    import torch

    from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
    from solrad_correction.utils.io import save_json
    from solrad_correction.utils.serialization import save_torch_checkpoint

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output-dir", default=str(ROOT / "scratch" / "benchmarks" / "artifacts"))
    args = parser.parse_args()

    layout = ArtifactLayout.from_experiment_dir(args.output_dir)
    layout.ensure_directories()
    model = torch.nn.Sequential(
        torch.nn.Linear(args.hidden_size, args.hidden_size),
        *[
            torch.nn.Sequential(
                torch.nn.ReLU(),
                torch.nn.Linear(args.hidden_size, args.hidden_size),
            )
            for _ in range(max(0, args.layers - 1))
        ],
        torch.nn.Linear(args.hidden_size, 1),
    )

    started = time.perf_counter()
    save_torch_checkpoint(
        model_state=model.state_dict(),
        optimizer_state=None,
        config={"hidden_size": args.hidden_size, "layers": args.layers},
        epoch=1,
        path=layout.checkpoints_dir / "synthetic.pt",
        metadata={"benchmark": "artifact_checkpoint"},
    )
    checkpoint_seconds = time.perf_counter() - started

    save_json({"RMSE": 0.0}, layout.metrics)
    started = time.perf_counter()
    write_manifest(layout, extra={"benchmark": "artifact_checkpoint"})
    manifest_seconds = time.perf_counter() - started

    print(
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


if __name__ == "__main__":
    main()
