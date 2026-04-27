"""CLI entry point for solrad_correction experiments.

Examples
--------
Run the default config:
    solrad-run --config configs/tcc/experiments/svm_hourly.yaml

Fair SVM/LSTM/Transformer comparison:
    set model.evaluation_policy: common_sequence_horizon in the YAML.

Opt into PyTorch compilation for longer neural-network runs:
    set runtime.torch_compile: true in the YAML.

Resume from a checkpoint:
    set runtime.resume: output/experiments/lstm_v1/checkpoints/last.pt.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from solrad_correction.config import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    PreprocessConfig,
    SplitConfig,
)


@click.command()
@click.option(
    "--config", "-c", required=False, type=click.Path(exists=True), help="Experiment config YAML."
)
@click.option("--name", "-n", default=None, help="Override experiment name.")
@click.option("--output-dir", "-o", default=None, help="Override output directory.")
@click.option("--validate-config", is_flag=True, help="Validate config and exit without training.")
@click.option(
    "--print-config", "print_config", is_flag=True, help="Print resolved config and exit."
)
@click.option(
    "--dry-run", is_flag=True, help="Validate resolved config and exit without loading data."
)
@click.option("--smoke-test", is_flag=True, help="Run a small synthetic CPU-safe smoke experiment.")
@click.option("--limit-rows", type=int, default=None, help="Limit loaded rows for development.")
@click.option("--profile", is_flag=True, help="Write profile.json with stage timings.")
@click.option("--device", type=click.Choice(["auto", "cpu", "cuda"]), default=None)
@click.option("--num-workers", type=int, default=None)
@click.option("--pin-memory/--no-pin-memory", default=None)
@click.option("--amp/--no-amp", default=None)
@click.option("--compile/--no-compile", "torch_compile", default=None)
@click.option("--resume", type=click.Path(exists=True), default=None)
def run_experiment_cli(
    config: str | None,
    name: str | None,
    output_dir: str | None,
    validate_config: bool,
    print_config: bool,
    dry_run: bool,
    smoke_test: bool,
    limit_rows: int | None,
    profile: bool,
    device: str | None,
    num_workers: int | None,
    pin_memory: bool | None,
    amp: bool | None,
    torch_compile: bool | None,
    resume: str | None,
) -> None:
    """Run a solrad_correction experiment from a YAML config file."""
    from solrad_correction.experiments.runner import run_experiment

    if smoke_test:
        cfg = _build_smoke_config()
    elif config:
        cfg = ExperimentConfig.from_yaml(config)
    else:
        raise click.ClickException("--config is required unless --smoke-test is used")

    if name:
        cfg.name = name
    if output_dir:
        cfg.output_dir = output_dir

    _apply_cli_overrides(
        cfg,
        dry_run=dry_run,
        smoke_test=smoke_test,
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

    if dry_run:
        click.echo("Dry run: config is valid. No data was loaded and no training was run.")
        return

    click.echo(f"Experiment: {cfg.name}")
    click.echo(f"Model:      {cfg.model.model_type}")
    click.echo(f"Eval policy:{cfg.model.evaluation_policy:>16}")
    click.echo(f"Output:     {cfg.experiment_dir}")

    run_experiment(cfg)


def _apply_cli_overrides(
    cfg: ExperimentConfig,
    *,
    dry_run: bool,
    smoke_test: bool,
    limit_rows: int | None,
    profile: bool,
    device: str | None,
    num_workers: int | None,
    pin_memory: bool | None,
    amp: bool | None,
    torch_compile: bool | None,
    resume: str | None,
) -> None:
    cfg.runtime.dry_run = dry_run
    cfg.runtime.smoke_test = smoke_test
    cfg.runtime.profile = profile or cfg.runtime.profile
    if limit_rows is not None:
        cfg.runtime.limit_rows = limit_rows
    if device is not None:
        cfg.runtime.device = device
    if num_workers is not None:
        cfg.runtime.num_workers = num_workers
    if pin_memory is not None:
        cfg.runtime.pin_memory = pin_memory
    if amp is not None:
        cfg.runtime.amp = amp
    if torch_compile is not None:
        cfg.runtime.torch_compile = torch_compile
    if resume is not None:
        cfg.runtime.resume = resume


def _build_smoke_config() -> ExperimentConfig:
    import numpy as np
    import pandas as pd

    scratch = Path("scratch") / "solrad_smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    data_path = scratch / "smoke_hourly.csv"
    if not data_path.exists():
        index = pd.date_range("2024-01-01", periods=80, freq="1h")
        rng = np.random.default_rng(42)
        f1 = rng.normal(size=80).astype("float32")
        f2 = rng.normal(size=80).astype("float32")
        target = (0.4 * f1 - 0.1 * f2 + rng.normal(scale=0.01, size=80)).astype("float32")
        pd.DataFrame({"f1": f1, "f2": f2, "target": target}, index=index).to_csv(data_path)

    return ExperimentConfig(
        name="solrad_smoke",
        data=DataConfig(
            hourly_data_path=str(data_path),
            target_column="target",
            feature_columns=["f1", "f2"],
        ),
        split=SplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
        preprocess=PreprocessConfig(scaler_type="standard", impute_strategy="drop"),
        features=FeatureConfig(add_temporal=False, cyclic_encoding=False),
        model=ModelConfig(
            model_type="lstm",
            lstm_hidden_size=4,
            lstm_num_layers=1,
            sequence_length=4,
            batch_size=8,
            max_epochs=1,
            patience=2,
        ),
        output_dir=str(scratch / "output"),
    )


if __name__ == "__main__":
    run_experiment_cli()
