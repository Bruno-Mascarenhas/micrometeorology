"""Temporal split configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SplitConfig:
    """Train / validation / test split settings."""

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False
