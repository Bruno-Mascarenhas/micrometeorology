"""Synthetic benchmark for lazy sequence DataLoader throughput.

Examples
--------
Run with default settings:
    python benchmarks/solrad_correction/sequence_dataloader.py

Run with custom parameters:
    python benchmarks/solrad_correction/sequence_dataloader.py --rows 100000 --batch-size 256
"""

import time
from typing import Annotated

import typer

app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run(
    rows: Annotated[int, typer.Option(help="Number of synthetic rows.")] = 50_000,
    features: Annotated[int, typer.Option(help="Number of feature columns.")] = 24,
    sequence_length: Annotated[int, typer.Option(help="Window length.")] = 24,
    batch_size: Annotated[int, typer.Option(help="DataLoader batch size.")] = 128,
    num_workers: Annotated[int, typer.Option(help="DataLoader workers.")] = 0,
    max_batches: Annotated[int, typer.Option(help="Batches to iterate.")] = 20,
) -> None:
    """Benchmark lazy sequence DataLoader throughput."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from solrad_correction.datasets.sequence import (
        WindowedSequenceDataset,
        collate_sequence_batch,
    )

    rng = np.random.default_rng(44)
    feat_data = rng.normal(size=(rows, features)).astype("float32")
    target = rng.normal(size=rows).astype("float32")
    dataset = WindowedSequenceDataset(feat_data, target, sequence_length)
    # The dataset batches itself in __getitems__, so the shared collate is
    # mandatory here exactly as it is in Trainer and TorchRegressorModel.predict.
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_sequence_batch,
    )

    started = time.perf_counter()
    batches = 0
    samples = 0
    with torch.inference_mode():
        for x_batch, _y_batch in loader:
            batches += 1
            samples += int(x_batch.shape[0])
            if batches >= max_batches:
                break
    elapsed = time.perf_counter() - started
    typer.echo(
        {
            "benchmark": "sequence_dataloader",
            "rows": rows,
            "windows": len(dataset),
            "samples": samples,
            "batches": batches,
            "seconds": round(elapsed, 6),
            "samples_per_second": round(samples / elapsed, 3) if elapsed else None,
        }
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
