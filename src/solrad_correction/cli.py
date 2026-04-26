"""CLI entry point for solrad_correction experiments.

Examples
--------
Run the default config:
    solrad-run --config configs/tcc/experiments/svm_hourly.yaml

Fair SVM/LSTM/Transformer comparison:
    set model.evaluation_policy: common_sequence_horizon in the YAML.

Opt into PyTorch compilation for longer neural-network runs:
    set model.torch_compile: true in the YAML.

Resume from a checkpoint:
    set model.pretrained_path: output/experiments/lstm_v1/model.pt.
"""

from __future__ import annotations

import click

from solrad_correction.config import ExperimentConfig


@click.command()
@click.option(
    "--config", "-c", required=True, type=click.Path(exists=True), help="Experiment config YAML."
)
@click.option("--name", "-n", default=None, help="Override experiment name.")
@click.option("--output", "-o", default=None, help="Override output directory.")
def run_experiment_cli(config: str, name: str | None, output: str | None) -> None:
    """Run a solrad_correction experiment from a YAML config file."""
    from solrad_correction.experiments.runner import run_experiment

    cfg = ExperimentConfig.from_yaml(config)

    if name:
        cfg.name = name
    if output:
        cfg.output_dir = output

    click.echo(f"Experiment: {cfg.name}")
    click.echo(f"Model:      {cfg.model.model_type}")
    click.echo(f"Eval policy:{cfg.model.evaluation_policy:>16}")
    click.echo(f"Output:     {cfg.experiment_dir}")

    run_experiment(cfg)


if __name__ == "__main__":
    run_experiment_cli()
