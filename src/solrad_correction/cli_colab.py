"""Click command for Google Colab / remote GPU solrad training."""

from __future__ import annotations

import json

import click

from solrad_correction.experiments.overrides import (
    ExperimentOverrides,
    load_config_with_overrides,
)


@click.command(name="solrad-colab")
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True), help="Experiment YAML config."
)
@click.option("--name", "-n", default=None, help="Override experiment name.")
@click.option("--output-dir", "-o", default=None, help="Drive-backed experiment output directory.")
@click.option("--validate-config", is_flag=True, help="Validate config and exit without training.")
@click.option(
    "--print-config", "print_config", is_flag=True, help="Print resolved config and exit."
)
@click.option("--limit-rows", type=int, default=None, help="Limit loaded rows for development.")
@click.option("--profile", is_flag=True, help="Write profile.json with stage timings.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default="cuda")
@click.option("--num-workers", type=int, default=None)
@click.option("--pin-memory/--no-pin-memory", default=None)
@click.option("--amp/--no-amp", default=None)
@click.option("--compile/--no-compile", "torch_compile", default=None)
@click.option(
    "--resume", type=click.Path(exists=True), default=None, help="Path to checkpoints/last.pt."
)
def run_colab_cli(
    config: str,
    name: str | None,
    output_dir: str | None,
    validate_config: bool,
    print_config: bool,
    limit_rows: int | None,
    profile: bool,
    device: str | None,
    num_workers: int | None,
    pin_memory: bool | None,
    amp: bool | None,
    torch_compile: bool | None,
    resume: str | None,
) -> None:
    """Run a solrad neural-network experiment with Colab-friendly defaults."""
    cfg = load_colab_config(
        config=config,
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
    )

    try:
        cfg.validate()
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if print_config:
        click.echo(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False, default=str))
        return

    if validate_config:
        click.echo("Config is valid.")
        return

    click.echo(f"Experiment: {cfg.name}")
    click.echo(f"Model:      {cfg.model.model_type}")
    click.echo(f"Device:     {cfg.runtime.device}")
    click.echo(f"Output:     {cfg.experiment_dir}")

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
    run_colab_cli()


if __name__ == "__main__":
    main()
