"""Data source configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class DataConfig:
    """Data loading and preparation settings.

    ``wrf_data_path``, ``station_lat`` and ``station_lon`` are accepted for
    backward compatibility with existing YAML files and are read by no pipeline
    stage: the site a run describes is selected by ``sensor_data_path`` /
    ``hourly_data_path``, never by coordinates.
    """

    sensor_data_path: str | None = None
    sensor_pattern: str = "*.dat"
    calibrations_path: str | None = None
    hourly_data_path: str | None = None
    wrf_data_path: str | None = None
    source_format: str = "auto"
    datetime_column: str | int | None = 0
    datetime_index: bool = True
    load_columns: list[str] = field(default_factory=list)
    dtype_map: dict[str, str] = field(default_factory=dict)
    cache_dir: str | None = None

    target_column: str = "SW_dif"
    feature_columns: list[str] = field(default_factory=list)

    use_raw: bool = False
    resample_freq: str | None = None
    sensor_min_samples: int = 6

    station_lat: float = -12.95
    station_lon: float = -38.51
