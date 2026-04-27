"""Feature engineering configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FeatureConfig:
    """Feature engineering settings."""

    lag_steps: list[int] = field(default_factory=list)
    rolling_windows: list[int] = field(default_factory=list)
    rolling_aggs: list[str] = field(default_factory=lambda: ["mean", "std"])
    add_temporal: bool = True
    cyclic_encoding: bool = True
    add_diffs: bool = False
