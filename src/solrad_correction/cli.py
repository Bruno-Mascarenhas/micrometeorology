"""CLI entry point for solrad_correction experiments.

Examples
--------
Run an experiment from a config file:
    solrad-run --config configs/tcc/experiments/svm_hourly.yaml

Run a quick test on CPU with only 100 rows:
    solrad-run --config configs/tcc/experiments/svm_hourly.yaml --limit-rows 100 --device cpu

Run a smoke test without needing a config:
    solrad-run --smoke-test --dry-run

Resume a neural-network experiment from a checkpoint:
    solrad-run --config configs/tcc/experiments/lstm_hourly.yaml --resume output/checkpoints/last.pt
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from solrad_correction.experiments.overrides import (
    ExperimentOverrides,
    load_config_with_overrides,
)

if TYPE_CHECKING:
    from pathlib import Path


class DeviceChoice(StrEnum):
    """Compute-device options for the ``--device`` flag.

    ``auto`` selects CUDA when available and falls back to CPU; ``cuda`` errors
    out if no GPU is present (see :func:`resolve_device`).
    """

    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"


app = typer.Typer(rich_markup_mode="markdown", no_args_is_help=True)


@app.command()
def run_experiment_cli(
    config: Annotated[
        Path | None, typer.Option("-c", "--config", help="Experiment config YAML.", exists=True)
    ] = None,
    name: Annotated[str | None, typer.Option("-n", help="Override experiment name.")] = None,
    output_dir: Annotated[
        Path | None, typer.Option("-o", "--output-dir", help="Override output directory.")
    ] = None,
    validate_config: Annotated[
        bool, typer.Option("--validate-config", help="Validate config and exit.")
    ] = False,
    print_config: Annotated[
        bool, typer.Option("--print-config", help="Print resolved config and exit.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate and exit without loading data.")
    ] = False,
    smoke_test: Annotated[
        bool, typer.Option("--smoke-test", help="Run a small synthetic CPU-safe smoke experiment.")
    ] = False,
    limit_rows: Annotated[
        int | None, typer.Option(help="Limit loaded rows for development.")
    ] = None,
    profile: Annotated[
        bool, typer.Option("--profile", help="Write profile.json with stage timings.")
    ] = False,
    device: Annotated[DeviceChoice | None, typer.Option(help="Device to use.")] = None,
    num_workers: Annotated[int | None, typer.Option(help="Number of data loader workers.")] = None,
    pin_memory: Annotated[bool | None, typer.Option("--pin-memory/--no-pin-memory")] = None,
    amp: Annotated[bool | None, typer.Option("--amp/--no-amp")] = None,
    torch_compile: Annotated[bool | None, typer.Option("--compile/--no-compile")] = None,
    resume: Annotated[
        Path | None, typer.Option(help="Resume from checkpoint.", exists=True)
    ] = None,
) -> None:
    """Run a solrad_correction experiment from a YAML config file."""
    if not smoke_test and not config:
        typer.echo("Error: --config is required unless --smoke-test is used", err=True)
        raise typer.Exit(code=1)

    overrides = ExperimentOverrides(
        name=name,
        output_dir=str(output_dir) if output_dir else None,
        dry_run=dry_run,
        smoke_test=smoke_test,
        limit_rows=limit_rows,
        profile=profile,
        device=str(device) if device else None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        amp=amp,
        torch_compile=torch_compile,
        resume=str(resume) if resume else None,
    )
    cfg = load_config_with_overrides(
        str(config) if config else None, smoke_test=smoke_test, overrides=overrides
    )

    try:
        cfg.validate()
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if print_config:
        typer.echo(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str))
        return

    if validate_config:
        typer.echo("Config is valid.")
        return

    if dry_run:
        typer.echo("Dry run: config is valid. No data was loaded and no training was run.")
        return

    typer.echo(f"Experiment: {cfg.name}")
    typer.echo(f"Model:      {cfg.model.model_type}")
    typer.echo(f"Eval policy:{cfg.model.evaluation_policy:>16}")
    typer.echo(f"Output:     {cfg.experiment_dir}")

    from solrad_correction.experiments.runner import run_experiment

    run_experiment(cfg)


def main() -> None:
    """Console-script entry point (pyproject: ``solrad-run``)."""
    app()


if __name__ == "__main__":
    main()
