"""``train`` CLI command — multimodal experiment training.

``allsky train`` runs a multimodal experiment declared by an
``experiment: true`` config (see ``configs/allsky/experiments/``); anything
else — a config that is not an experiment config, or no config at all — is
rejected with a pointer to those configs.

torch is imported lazily inside the run helper so ``allsky --help`` stays light
and torch-free; :func:`micrometeorology.common.optional.require` guards that
boundary so a core-only install gets the extra's install command instead of a
bare ``ModuleNotFoundError``.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from allsky.cli.runtime import configure_cli_logging
from allsky.config import is_experiment_config, load_experiment_config
from micrometeorology.common.optional import require

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Experiment config YAML (must declare experiment: true).",
        exists=True,
        dir_okay=False,
    ),
]


class DeviceChoice(StrEnum):
    """Torch device a run may be pinned to; ``auto`` resolves cuda -> mps -> cpu."""

    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"


def train(
    config: ConfigOption = None,
    resume: Annotated[
        str | None,
        typer.Option(
            help="Checkpoint to resume from: 'auto' (find last.ckpt in the run dir) or a path.",
        ),
    ] = None,
    epochs: Annotated[int | None, typer.Option(min=1, help="Override train.epochs.")] = None,
    batch_size: Annotated[
        int | None, typer.Option(min=1, help="Override train.batch_size.")
    ] = None,
    device: Annotated[DeviceChoice | None, typer.Option(help="Override train.device.")] = None,
    out_dir: Annotated[Path | None, typer.Option(help="Override the run directory.")] = None,
    data_root: Annotated[
        Path | None,
        typer.Option(help="Data root for the manifest/split/embeddings."),
    ] = None,
    amp: Annotated[
        bool | None,
        typer.Option("--amp/--no-amp", help="Override mixed precision."),
    ] = None,
    trust_checkpoint: Annotated[
        bool,
        typer.Option(
            "--trust-checkpoint/--no-trust-checkpoint",
            help=(
                "Read the --resume checkpoint with the unrestricted unpickler. Only for "
                "a file you produced yourself: the default restricted reader refuses a "
                "payload carrying unexpected Python objects instead of executing it."
            ),
        ),
    ] = False,
) -> None:
    """Run a multimodal training experiment from an experiment: true config.

    ``--resume`` accepts ``auto`` (find ``last.ckpt`` in the run dir) or a path;
    ``--data-root`` / ``--amp`` / the override flags apply to the resolved
    experiment config. A resumed checkpoint is read under torch's restricted
    unpickler unless ``--trust-checkpoint`` is passed.
    """
    configure_cli_logging()

    if config is None or not is_experiment_config(config):
        raise typer.BadParameter(
            "allsky train requires a multimodal experiment config (a YAML declaring "
            "'experiment: true'). The legacy SkyFusionNet pipeline has been retired; "
            "see configs/allsky/experiments/ (e.g. v4_film.yaml).",
            param_hint="--config",
        )

    exp_cfg = load_experiment_config(config)
    if epochs is not None:
        exp_cfg.train.epochs = epochs
    if batch_size is not None:
        exp_cfg.train.batch_size = batch_size
    if device is not None:
        exp_cfg.train.device = str(device)

    if resume is not None and resume != "auto" and not Path(resume).exists():
        raise typer.BadParameter(
            f"resume checkpoint does not exist: {resume}", param_hint="--resume"
        )

    require("torch", "allsky")
    from allsky.training import TrainingError, run_experiment  # lazy: pulls torch at run time

    try:
        summary = run_experiment(
            exp_cfg,
            data_root=data_root,
            output_dir=out_dir,
            device=str(device) if device is not None else None,
            amp=amp,
            resume=resume,
            trust_checkpoint=trust_checkpoint,
        )
    except TrainingError as exc:
        # Every TrainingError names the knob to change. A traceback through the
        # engine's internals buries that message under frames the operator has no
        # use for, and a run that diverged is the one time they most need to read
        # it.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(summary, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach the ``train`` command onto *app*."""
    app.command()(train)
