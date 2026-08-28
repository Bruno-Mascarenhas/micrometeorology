"""Data source configuration."""

from dataclasses import dataclass, field

from micrometeorology.common.site import STATION_SITE


@dataclass(slots=True)
class DataConfig:
    """Data loading and preparation settings.

    ``wrf_data_path``, ``station_lat`` and ``station_lon`` are accepted for
    backward compatibility with existing YAML files and are read by no pipeline
    stage: the site a run describes is selected by ``sensor_data_path`` /
    ``hourly_data_path``, never by coordinates.

    ``feature_columns`` is also the base-column list every ``features`` stage
    engineers from, so an empty list makes ``lag_steps``/``rolling_windows``/
    ``add_diffs`` unreachable, and listing ``target_column`` there is rejected
    once rolling or diff features are enabled (they read the current row).
    ``ExperimentConfig.validate`` enforces both.
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

    station_lat: float = STATION_SITE.latitude
    station_lon: float = STATION_SITE.longitude
