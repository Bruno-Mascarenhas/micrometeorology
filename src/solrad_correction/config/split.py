"""Temporal split configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class SplitConfig:
    """Train / validation / test split settings.

    The three ratios are non-negative fractions of the available rows and must
    sum to 1.0 (``ExperimentConfig.validate`` checks this to within 1e-6).

    ``shuffle`` is off by default and should stay off: the split is chronological
    so that validation and test rows always follow the training rows in time.
    Shuffling an irradiance series leaks, because neighbouring timestamps carry
    nearly the same sky state.
    """

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    shuffle: bool = False
