"""Application configuration using pydantic-settings + YAML.

Usage::

    from labmim_micrometeorology.common.config import get_settings
    settings = get_settings()
    print(settings.data_dir)

The configuration is loaded from up to three layers (later overrides earlier):
1. ``configs/default.yaml``      — shipped defaults
2. ``configs/<env>.yaml``        — environment-specific (server / local)
3. Environment variables         — ``LABMIM_*`` prefix

Set ``LABMIM_CONFIG_PATH`` to point to a custom YAML configuration.
Set ``LABMIM_ENV`` to ``server`` or ``local`` to auto-load the matching file.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Walk upward from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: three levels up from common/config.py
    return current.parent.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file does not exist."""
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
            return data if isinstance(data, dict) else {}
    return {}


class Settings(BaseSettings):
    """Global application settings.

    Values are read from environment variables with the ``LABMIM_`` prefix
    and can be overridden by a YAML config file.
    """

    model_config = SettingsConfigDict(
        env_prefix="LABMIM_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Paths (all cross-platform via pathlib)
    data_dir: Path = Field(default=Path("data"), description="Root data directory")
    output_dir: Path = Field(default=Path("output"), description="Output directory")
    figures_dir: Path = Field(default=Path("output/figures"), description="Figure output directory")
    shapes_dir: Path = Field(default=Path("shapes_BR_cities"), description="Shapefile directory")
    configs_dir: Path = Field(
        default=Path("configs/micromet"), description="Configuration directory"
    )

    # WRF defaults
    wrf_default_variables: list[str] = Field(
        default=["temperature", "pressure", "vapor", "wind", "rain", "HFX", "LH", "SWDOWN"],
        description="Default WRF variables to process",
    )

    # Sensor defaults
    sensor_min_samples_per_hour: int = Field(
        default=6,
        description="Minimum valid samples required per hour for aggregation",
    )
    sensor_sentinel_value: float = Field(
        default=-900.0,
        description="Sentinel value in Campbell Scientific data indicating missing data",
    )

    # Logging
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

    Loading order:
    1. ``configs/default.yaml``
    2. ``configs/<LABMIM_ENV>.yaml`` (if ``LABMIM_ENV`` is set)
    3. ``LABMIM_CONFIG_PATH`` (if set, overrides step 2)
    4. Environment variables
    """
    root = _project_root()
    configs_dir = root / "configs" / "micromet"

    # Layer 1: defaults (check micromet/ subdir first, then root configs/)
    merged: dict[str, Any] = _load_yaml(configs_dir / "default.yaml")
    if not merged:
        merged = _load_yaml(root / "configs" / "default.yaml")

    # Layer 2: environment-specific
    env_name = os.environ.get("LABMIM_ENV", "")
    if env_name:
        env_data = _load_yaml(configs_dir / f"{env_name}.yaml")
        merged.update(env_data)

    # Layer 3: explicit config path
    config_path = os.environ.get("LABMIM_CONFIG_PATH")
    if config_path:
        merged.update(_load_yaml(Path(config_path)))

    # Build settings; env vars (Layer 4) are handled by pydantic-settings
    settings = Settings(**merged)
    settings.resolve_paths(root)
    return settings
