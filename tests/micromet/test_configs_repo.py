"""Contract tests for the shipped ``configs/micromet/`` YAML tree.

``ingest_sensor_data`` resolves the calibration table through
``Settings.configs_dir`` behind an ``is_file()`` guard.  When that directory
does not name the place the shipped configs actually live, the guard fails
silently and sensitivity calibrations are skipped with exit code 0.  These
tests pin the resolved directory so the field default and the YAML cannot drift
apart again.
"""

from pathlib import Path
from typing import Any

import yaml

from micrometeorology.common.config import Settings

_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = _ROOT / "configs" / "micromet"
_DEFAULT_YAML = _CONFIGS / "default.yaml"


def _shipped_defaults() -> dict[str, Any]:
    with open(_DEFAULT_YAML, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict)
    return data


def _settings_from_shipped_defaults() -> Settings:
    """Build ``Settings`` the way ``get_settings()`` does, minus any undeclared key.

    Every shipped key is a declared field today; the filter stays so that a
    schema-drift failure surfaces in ``test_config.py`` rather than as a
    confusing path assertion here.
    """
    declared = {k: v for k, v in _shipped_defaults().items() if k in Settings.model_fields}
    settings = Settings(**declared)
    settings.resolve_paths(_ROOT)
    return settings


def test_resolved_configs_dir_holds_the_shipped_configs() -> None:
    """The directory the CLI searches must be the one holding the shipped files."""
    configs_dir = _settings_from_shipped_defaults().configs_dir
    assert (configs_dir / "default.yaml").is_file(), configs_dir
    assert (configs_dir / "calibrations.yaml").is_file(), configs_dir


def test_default_yaml_does_not_redeclare_its_own_directory() -> None:
    """``configs_dir`` is self-referential: the field default is the single source."""
    assert "configs_dir" not in _shipped_defaults()
