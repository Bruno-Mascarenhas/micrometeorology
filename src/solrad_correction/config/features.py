"""Feature engineering configuration."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class FeatureConfig:
    """Feature engineering settings.

    ``lag_steps``, ``rolling_windows`` and ``add_diffs`` are applied to the base
    columns listed in ``DataConfig.feature_columns`` and are counted in rows of
    the loaded frequency, not in fixed clock time. Lag steps must be >= 1 (0
    copies a column onto itself and a negative step reads a future value) and
    rolling windows must be >= 1.

    ``add_temporal`` adds the calendar columns and ``cyclic_encoding`` their
    sine/cosine pairs, both derived from the DatetimeIndex rather than from
    ``feature_columns``, so they stay available when that list is empty.
    """

    lag_steps: list[int] = field(default_factory=list)
    rolling_windows: list[int] = field(default_factory=list)
    rolling_aggs: list[str] = field(default_factory=lambda: ["mean", "std"])
    add_temporal: bool = True
    cyclic_encoding: bool = True
    add_diffs: bool = False
