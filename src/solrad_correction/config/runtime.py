"""Runtime and hardware configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    """Operational settings for local CPU and Colab/GPU execution."""

    device: str = "auto"
    num_workers: int | None = None
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    amp: bool | None = None
    torch_compile: bool = False
    gradient_clip: float | None = 1.0
    checkpoint_dir: str | None = None
    checkpoint_every: int | None = 1
    resume: str | None = None
    profile: bool = False
    dry_run: bool = False
    smoke_test: bool = False
    limit_rows: int | None = None
