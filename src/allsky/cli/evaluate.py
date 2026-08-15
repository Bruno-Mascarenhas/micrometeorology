"""Evaluation CLI: the ``allsky evaluate`` command.

Runs a trained checkpoint over a split and writes a report directory
(``metrics.json`` / ``stratified.csv`` / ``report.md`` and, unless
``--no-predictions``, ``predictions.parquet``).  :mod:`allsky.evaluation` pulls
torch, so it is imported inside the command body to keep importing
:mod:`allsky.cli` torch-free.
"""

import logging
from pathlib import Path
from typing import Annotated

import typer

from allsky.cli.runtime import configure_cli_logging
from allsky.cli.train import (
    DeviceChoice,  # typer resolves this annotation at runtime
)

logger = logging.getLogger(__name__)


def evaluate_cmd(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            "-k",
            help="Checkpoint written by training (last.ckpt / best.ckpt).",
            exists=True,
            dir_okay=False,
        ),
    ],
    split: Annotated[str, typer.Option(help="Split to evaluate: val, test or train.")] = "val",
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Experiment YAML whose data.data_root overrides the checkpoint's.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    data_root: Annotated[
        Path | None,
        typer.Option(help="Data root for the manifest/split/embeddings (wins over --config)."),
    ] = None,
    report_dir: Annotated[
        Path | None,
        typer.Option(help="Report output dir (default: <checkpoint dir>/eval-<split>)."),
    ] = None,
    device: Annotated[
        DeviceChoice | None, typer.Option(help="Inference device (default: cpu).")
    ] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Inference batch size (default: config batch size).")
    ] = None,
    predictions: Annotated[
        bool,
        typer.Option("--predictions/--no-predictions", help="Write predictions.parquet."),
    ] = True,
    strict: Annotated[
        bool,
        typer.Option("--strict/--no-strict", help="Error (not warn) on manifest/split mismatch."),
    ] = False,
    trust_checkpoint: Annotated[
        bool,
        typer.Option(
            "--trust-checkpoint/--no-trust-checkpoint",
            help=(
                "Read the checkpoint with the unrestricted unpickler. Only for a file "
                "you produced yourself: the default restricted reader refuses a payload "
                "carrying unexpected Python objects instead of executing it."
            ),
        ),
    ] = False,
) -> None:
    """Evaluate a trained checkpoint on a split and write a report directory.

    The data root is resolved as ``--data-root``, else the ``data.data_root`` of
    ``--config``, else the one baked into the checkpoint; the report lands in
    ``--report-dir`` or ``<checkpoint dir>/eval-<split>``.

    Raises
    ------
    typer.Exit
        Code 1 when evaluation or report writing fails; the reason is logged
        before the exit.
    """
    configure_cli_logging()

    resolved_root = _resolve_data_root(data_root, config)
    out_dir = report_dir if report_dir is not None else checkpoint.parent / f"eval-{split}"

    from allsky.evaluation import evaluate_checkpoint, write_evaluation_report

    try:
        result = evaluate_checkpoint(
            checkpoint,
            split=split,
            data_root=resolved_root,
            batch_size=batch_size,
            device=str(device) if device is not None else None,
            strict=strict,
            trust_checkpoint=trust_checkpoint,
        )
        written = write_evaluation_report(result, out_dir, predictions=predictions)
    except Exception as exc:
        logger.error("evaluation failed: %s", exc)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Evaluated '{split}': {result.n_samples} samples over {result.enabled_targets}")
    typer.echo(f"Report written to {out_dir}")
    for name, path in written.items():
        typer.echo(f"  {name}: {path}")


def _resolve_data_root(data_root: Path | None, config: Path | None) -> Path | None:
    """Pick the data root: ``--data-root`` wins, else ``--config``'s data_root."""
    if data_root is not None:
        return data_root
    if config is not None:
        from allsky.config import load_experiment_config

        return Path(load_experiment_config(config).data.data_root)
    return None


def register(app: typer.Typer) -> None:
    """Attach the ``evaluate`` command onto *app*."""
    app.command("evaluate")(evaluate_cmd)
