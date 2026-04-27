"""Shared config loading and runtime override helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from solrad_correction.config import ExperimentConfig

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ExperimentOverrides:
    """Optional command-line overrides for an experiment config."""

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


def load_config_with_overrides(
    config_path: str | Path | None,
    *,
    smoke_test: bool = False,
    overrides: ExperimentOverrides | None = None,
) -> ExperimentConfig:
    """Load a YAML or synthetic smoke config and apply shared overrides."""
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
    """Apply command-line overrides in-place and return *cfg* for chaining."""
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

    return cfg
