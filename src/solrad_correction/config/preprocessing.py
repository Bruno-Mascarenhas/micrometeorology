"""Preprocessing configuration."""

from dataclasses import dataclass


@dataclass(slots=True)
class PreprocessConfig:
    """Preprocessing pipeline settings.

    ``scaler_type`` is one of ``standard``, ``minmax`` or ``none``, and
    ``impute_strategy`` one of ``drop``, ``ffill``, ``mean`` or ``interpolate``.
    Every statistic behind them is fitted on the training split alone.

    ``drop_na_threshold`` is a NaN fraction in ``[0, 1]``: a column whose share
    of missing values in the training split exceeds it is dropped from the
    experiment entirely, in both directions of the split.
    """

    scaler_type: str = "standard"
    impute_strategy: str = "drop"
    drop_na_threshold: float = 0.5
