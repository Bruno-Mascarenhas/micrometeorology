"""Synthetic benchmark for lazy sequence DataLoader throughput."""

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
    from torch.utils.data import DataLoader

    from solrad_correction.datasets.sequence import WindowedSequenceDataset

    np = __import__("numpy")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    rng = np.random.default_rng(44)
    features = rng.normal(size=(args.rows, args.features)).astype("float32")
    target = rng.normal(size=args.rows).astype("float32")
    dataset = WindowedSequenceDataset(features, target, args.sequence_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    started = time.perf_counter()
    batches = 0
    samples = 0
    with torch.inference_mode():
        for x_batch, _y_batch in loader:
            batches += 1
            samples += int(x_batch.shape[0])
            if batches >= args.max_batches:
                break
    elapsed = time.perf_counter() - started
    print(
        {
            "benchmark": "sequence_dataloader",
            "rows": args.rows,
            "windows": len(dataset),
            "samples": samples,
            "batches": batches,
            "seconds": round(elapsed, 6),
            "samples_per_second": round(samples / elapsed, 3) if elapsed else None,
        }
    )


if __name__ == "__main__":
    main()
