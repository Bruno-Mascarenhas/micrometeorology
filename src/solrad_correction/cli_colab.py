"""Click command for Google Colab / remote GPU solrad training.

Examples
--------
Run an experiment on Google Colab:
    solrad-colab --config configs/tcc/experiments/lstm_hourly.yaml

Save output to Google Drive:
    solrad-colab --config configs/tcc/experiments/lstm_hourly.yaml -o /content/drive/MyDrive/outputs/

Resume an interrupted run:
    solrad-colab --config configs/tcc/experiments/lstm_hourly.yaml --resume /content/drive/MyDrive/outputs/checkpoints/last.pt
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from solrad_correction.experiments.overrides import (
    ExperimentOverrides,
    load_config_with_overrides,
)


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
def run_colab_cli(
    config: Annotated[
        Path, typer.Option("-c", "--config", help="Experiment YAML config.", exists=True)
    ],
    name: Annotated[str | None, typer.Option("-n", help="Override experiment name.")] = None,
    output_dir: Annotated[
        Path | None, typer.Option("-o", "--output-dir", help="Drive-backed output directory.")
    ] = None,
    validate_config: Annotated[
        bool, typer.Option("--validate-config", help="Validate config and exit.")
    ] = False,
    print_config: Annotated[
        bool, typer.Option("--print-config", help="Print resolved config and exit.")
    ] = False,
    limit_rows: Annotated[
        int | None, typer.Option(help="Limit loaded rows for development.")
    ] = None,
    profile: Annotated[
        bool, typer.Option("--profile", help="Write profile.json with stage timings.")
    ] = False,
    device: Annotated[DeviceChoice, typer.Option(help="Device to use.")] = DeviceChoice.cuda,
    num_workers: Annotated[int | None, typer.Option(help="Number of data loader workers.")] = None,
    pin_memory: Annotated[bool | None, typer.Option("--pin-memory/--no-pin-memory")] = None,
    amp: Annotated[bool | None, typer.Option("--amp/--no-amp")] = None,
    torch_compile: Annotated[bool | None, typer.Option("--compile/--no-compile")] = None,
    resume: Annotated[
        Path | None, typer.Option(help="Path to checkpoints/last.pt.", exists=True)
    ] = None,
) -> None:
    """Run a solrad neural-network experiment with Colab-friendly defaults."""
    cfg = load_colab_config(
        config=str(config),
        name=name,
        output_dir=str(output_dir) if output_dir else None,
        limit_rows=limit_rows,
        profile=profile,
        device=str(device),
        num_workers=num_workers,
        pin_memory=pin_memory,
        amp=amp,
        torch_compile=torch_compile,
        resume=str(resume) if resume else None,
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

    # Fail fast BEFORE any data is loaded: on Colab the default runtime has
    # no GPU, and discovering that only at model-build time wastes the whole
    # load/feature/split/preprocess pipeline.
    from solrad_correction.training.dataloaders import resolve_device

    try:
        resolve_device(cfg.runtime.device)
    except ValueError as exc:
        typer.echo(
            f"Error: {exc}. Enable a GPU runtime (Runtime > Change runtime type) "
            "or pass --device cpu to train without a GPU.",
            err=True,
        )
        raise typer.Exit(code=1) from exc

    typer.echo(f"Experiment: {cfg.name}")
    typer.echo(f"Model:      {cfg.model.model_type}")
    typer.echo(f"Device:     {cfg.runtime.device}")
    typer.echo(f"Output:     {cfg.experiment_dir}")

    from solrad_correction.experiments.runner import run_experiment

    run_experiment(cfg)


def load_colab_config(
    *,
    config: str,
    name: str | None = None,
    output_dir: str | None = None,
    limit_rows: int | None = None,
    profile: bool = False,
    device: str | None = "cuda",
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    amp: bool | None = None,
    torch_compile: bool | None = None,
    resume: str | None = None,
):
    """Load config with the same override path used by local CLI."""
    return load_config_with_overrides(
        config,
        overrides=ExperimentOverrides(
            name=name,
            output_dir=output_dir,
            limit_rows=limit_rows,
            profile=profile,
            device=device,
            num_workers=num_workers,
            pin_memory=pin_memory,
            amp=amp,
            torch_compile=torch_compile,
            resume=resume,
        ),
    )


def main() -> None:
    """Console-script entry point (pyproject: ``solrad-colab``)."""
    app()


if __name__ == "__main__":
    main()
