"""Runtime and hardware configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeConfig:
    """Operational settings for local CPU and Colab/GPU execution.

    ``device`` is one of ``auto``, ``cpu`` or ``cuda``; the loader knobs
    (``num_workers``, ``pin_memory``, ``persistent_workers``, ``prefetch_factor``)
    and ``amp`` left at ``None`` are resolved from the selected device rather
    than from a fixed default.

    ``allow_preprocessing_change`` governs what a resume does when the scaler no
    longer matches: a resume refits the scaler from the data on disk now, and
    when the refitted transform differs from the one the checkpoint's weights
    were trained under the resume is refused. Setting this downgrades the
    refusal to a warning — deliberately re-scaling an existing model.

    ``ExperimentConfig.validate`` enforces the remaining bounds: non-negative
    ``num_workers``, a positive ``prefetch_factor`` that requires
    ``num_workers > 0``, a non-negative ``gradient_clip``, and positive
    ``checkpoint_every``/``limit_rows`` when set.
    """

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
    allow_preprocessing_change: bool = False
    profile: bool = False
    dry_run: bool = False
    smoke_test: bool = False
    limit_rows: int | None = None
