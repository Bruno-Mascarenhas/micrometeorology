"""Application configuration using pydantic-settings + YAML.

Usage::

    from micrometeorology.common.config import get_settings
    settings = get_settings()
    print(settings.data_dir)

The configuration is loaded from up to three layers (later overrides earlier):
1. ``configs/default.yaml``      — shipped defaults
2. ``configs/<env>.yaml``        — environment-specific (server / local)
3. Environment variables         — ``LABMIM_*`` prefix

Set ``LABMIM_CONFIG_PATH`` to point to a custom YAML configuration.
Set ``LABMIM_ENV`` to ``server`` or ``local`` to auto-load the matching file.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Walk upward from this file to find the project root (contains pyproject.toml).

    A wheel has no ``pyproject.toml`` above it, so the walk falls through to the
    package directory, where ``configs/micromet`` is force-included as
    ``micrometeorology/configs/micromet``. A fallback that misses that YAML leaves
    every sensor rule at its empty default, and the CLIs then export unfiltered,
    uncalibrated data with exit code 0.
    """
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return current.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file does not exist."""
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    return {}


def _load_named_yaml(path: Path, variable: str) -> dict[str, Any]:
    """Load a YAML file the operator named explicitly, failing when it is absent.

    A missing *shipped* default is optional, so :func:`_load_yaml` returning
    ``{}`` is right for layer 1. A path the operator typed is not: a wrong one
    would make the whole override layer vanish with no exception and exit code 0.
    """
    if not path.is_file():
        raise FileNotFoundError(f"{variable} points to {path}, which is not a file")
    return _load_yaml(path)


class Settings(BaseSettings):
    """Global application settings.

    Values come from the YAML layers and from environment variables carrying
    the ``LABMIM_`` prefix, the environment winning where both name a key (see
    :func:`get_settings`, which is what enforces that order).

    Notes
    -----
    There are deliberately no WRF settings here. ``get_settings()`` is imported
    only by ``labmim-archive`` and ``labmim-sensor-process``, and each WRF CLI
    owns a ``DEFAULT_VARS`` list different from its siblings' on purpose — the
    GeoJSON exporter's matches the front-end variable registry, the figure
    renderer's the variables that have a colormap — so no key here could be
    authoritative for all three.
    """

    model_config = SettingsConfigDict(
        env_prefix="LABMIM_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    figures_dir: Path = Field(default=Path("output/figures"), description="Figure output directory")
    shapes_dir: Path = Field(default=Path("shapes_BR_cities"), description="Shapefile directory")
    configs_dir: Path = Field(
        default=Path("configs/micromet"), description="Configuration directory"
    )

    sensor_min_samples_per_hour: int = Field(
        default=6,
        description="Minimum valid samples required per hour for aggregation",
    )
    sensor_sentinel_value: float | None = Field(
        default=-900.0,
        description=(
            "Values at or below this are read as missing. `null` disables the rule: "
            "the -900 default matches nothing in the LabMiM archive, whose sentinels "
            "are 1000, 999, -46.8, -273.1, -7999, -6673 and a date-scoped 0, so an "
            "operator needs a way to turn off a guard that only looks like one."
        ),
    )
    sensor_limits: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Quality-control ranges, one dict per column: {column, lower, upper}",
    )
    sensor_step_limits: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Excursion (spike/dip) tests, one dict per column: "
            "{column, threshold, return_tol, max_len}. These reject the improbable, "
            "which the ranges above structurally cannot: a barometer dropping 17 hPa "
            "for one sample and returning stays inside every bound."
        ),
    )
    sensor_persistence_limits: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Exact-repeat run tests, one dict per column: "
            "{column, min_run, exempt_at_or_below}. These catch a railed sensor, "
            "which reads inside every bound forever."
        ),
    )
    sensor_sum_columns: list[str] = Field(
        default_factory=list,
        description="Columns aggregated by sum instead of mean",
    )
    sensor_wind_dir_columns: list[str] = Field(
        default_factory=list,
        description="Columns aggregated as vector wind direction",
    )
    sensor_wind_speed_column_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Direction column -> the speed column that weights its vector mean. "
            "Without the pairing every sample carries unit weight, so a calm hour "
            "counts as much as a squall and the hourly bearing lands more than 5 deg "
            "off in about one hour in six."
        ),
    )

    log_level: str = Field(default="INFO", description="Logging level")

    def resolve_paths(self, root: Path | None = None) -> None:
        """Resolve relative paths against the project root."""
        base = root or _project_root()
        for field_name in ("data_dir", "output_dir", "figures_dir", "shapes_dir", "configs_dir"):
            p = getattr(self, field_name)
            if not p.is_absolute():
                object.__setattr__(self, field_name, (base / p).resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the application settings.

    Loading order, each layer overriding the ones above it:

    1. ``configs/micromet/default.yaml``, falling back to ``configs/default.yaml``
    2. ``configs/micromet/<LABMIM_ENV>.yaml`` (if ``LABMIM_ENV`` is set)
    3. ``LABMIM_CONFIG_PATH`` (if set, applied on top of step 2)
    4. Environment variables named ``LABMIM_<key>``

    Returns
    -------
    Settings
        The process-wide settings, cached, with every path field resolved to an
        absolute path against the project root.

    Raises
    ------
    FileNotFoundError
        If no shipped default file is found, or if a file the operator named
        through ``LABMIM_ENV`` / ``LABMIM_CONFIG_PATH`` is absent. Every sensor
        rule lives in those files; without them ``Settings`` builds from bare
        defaults and the CLIs export unfiltered data with exit code 0.
    """
    root = _project_root()
    configs_dir = root / "configs" / "micromet"

    merged: dict[str, Any] = _load_yaml(configs_dir / "default.yaml")
    if not merged:
        merged = _load_yaml(root / "configs" / "default.yaml")
    if not merged:
        raise FileNotFoundError(
            "no shipped configuration found; searched "
            f"{configs_dir / 'default.yaml'} and {root / 'configs' / 'default.yaml'}"
        )

    env_name = os.environ.get("LABMIM_ENV", "")
    if env_name:
        merged.update(_load_named_yaml(configs_dir / f"{env_name}.yaml", "LABMIM_ENV"))

    config_path = os.environ.get("LABMIM_CONFIG_PATH")
    if config_path:
        merged.update(_load_named_yaml(Path(config_path), "LABMIM_CONFIG_PATH"))

    # pydantic-settings ranks init kwargs above env vars, so YAML keys the
    # operator has already set in the environment are withheld from the
    # constructor to keep the documented order.
    environment_names = {name.upper() for name in os.environ}
    yaml_without_env_overrides = {
        key: value
        for key, value in merged.items()
        if f"LABMIM_{key.upper()}" not in environment_names
    }
    settings = Settings(**yaml_without_env_overrides)
    settings.resolve_paths(root)
    return settings
