"""Shared config loading and runtime override helpers."""

from dataclasses import dataclass
from pathlib import Path

from solrad_correction.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class ExperimentOverrides:
    """Optional command-line overrides for an experiment config.

    Every field mirrors a CLI flag. For most of them ``None`` means "leave the
    config alone", so a flag the operator did not pass never silently replaces
    a value the versioned config declares; ``dry_run``, ``smoke_test`` and
    ``profile`` are the exceptions, under the rules in
    :func:`apply_overrides`.

    Attributes
    ----------
    name:
        Replacement experiment name, which also renames the output directory.
    output_dir:
        Root under which the experiment directory is created.
    dry_run:
        Validate and set up the run without training.
    smoke_test:
        Build a synthetic config instead of loading one from disk.
    limit_rows:
        Truncate the loaded table to its first ``n`` rows, for a fast pass over
        the real pipeline.
    profile:
        Record per-stage timings. Never turned off by an override: a config
        that asks for profiling keeps it.
    device:
        ``auto``, ``cpu`` or ``cuda``.
    num_workers, pin_memory:
        DataLoader settings; ``None`` leaves them to be resolved from the
        device.
    amp:
        Force automatic mixed precision on or off; ``None`` resolves it from
        the device.
    torch_compile:
        Attempt ``torch.compile`` on the module.
    resume:
        Path to a checkpoint to continue training from.
    allow_preprocessing_change:
        Permit a resume whose refitted preprocessing no longer matches the one
        the checkpoint's weights were trained under. Only ever accepted
        explicitly, since it re-scales the inputs of restored weights.
    """

    name: str | None = None
    output_dir: str | None = None
    dry_run: bool = False
    smoke_test: bool = False
    limit_rows: int | None = None
    profile: bool = False
    device: str | None = None
    num_workers: int | None = None
    pin_memory: bool | None = None
    amp: bool | None = None
    torch_compile: bool | None = None
    resume: str | None = None
    allow_preprocessing_change: bool = False


def load_config_with_overrides(
    config_path: str | Path | None,
    *,
    smoke_test: bool = False,
    overrides: ExperimentOverrides | None = None,
) -> ExperimentConfig:
    """Load a YAML or synthetic smoke config and apply shared overrides.

    Parameters
    ----------
    config_path:
        Path to the versioned experiment YAML. Optional only under
        ``smoke_test``.
    smoke_test:
        Build the synthetic smoke config instead of reading from disk.
    overrides:
        Command-line overrides to apply. When omitted, the only override
        applied is the ``smoke_test`` flag itself.

    Returns
    -------
    ExperimentConfig
        The loaded config, already overridden.

    Raises
    ------
    ValueError
        If neither a config path nor ``smoke_test`` is given.
    """
    if smoke_test:
        from solrad_correction.dev.synthetic import build_smoke_config

        cfg = build_smoke_config()
    elif config_path is not None:
        cfg = ExperimentConfig.from_yaml(config_path)
    else:
        raise ValueError("config_path is required unless smoke_test is enabled")

    apply_overrides(cfg, overrides or ExperimentOverrides(smoke_test=smoke_test))
    return cfg


def apply_overrides(cfg: ExperimentConfig, overrides: ExperimentOverrides) -> ExperimentConfig:
    """Apply command-line overrides in-place and return *cfg* for chaining.

    Most fields are only touched when the override carries a value, so the
    versioned config stays the source of truth for everything the operator did
    not ask about. Three deliberately break that pattern: ``dry_run`` and
    ``smoke_test`` are written unconditionally, because they describe how this
    invocation runs rather than what the experiment is, and ``profile`` is
    OR-ed, so a config that requests profiling cannot be silenced by its
    absence from the command line.
    """
    if overrides.name:
        cfg.name = overrides.name
    if overrides.output_dir:
        cfg.output_dir = overrides.output_dir

    cfg.runtime.dry_run = overrides.dry_run
    cfg.runtime.smoke_test = overrides.smoke_test
    cfg.runtime.profile = overrides.profile or cfg.runtime.profile

    if overrides.limit_rows is not None:
        cfg.runtime.limit_rows = overrides.limit_rows
    if overrides.device is not None:
        cfg.runtime.device = overrides.device
    if overrides.num_workers is not None:
        cfg.runtime.num_workers = overrides.num_workers
    if overrides.pin_memory is not None:
        cfg.runtime.pin_memory = overrides.pin_memory
    if overrides.amp is not None:
        cfg.runtime.amp = overrides.amp
    if overrides.torch_compile is not None:
        cfg.runtime.torch_compile = overrides.torch_compile
    if overrides.resume is not None:
        cfg.runtime.resume = overrides.resume
    if overrides.allow_preprocessing_change:
        cfg.runtime.allow_preprocessing_change = True

    return cfg
