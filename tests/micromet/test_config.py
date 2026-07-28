"""Tests for the YAML + environment settings layer (``get_settings``).

The shipped ``configs/micromet/default.yaml`` is part of the contract here:
every key it declares must be a real field on ``Settings``, otherwise
``extra="forbid"`` rejects the repository's own configuration and every
consumer of ``get_settings()`` dies at import-time of its first statement.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import ValidationError

from micrometeorology.common.config import Settings, get_settings


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ambient ``LABMIM_*`` overrides and the process-global settings cache."""
    monkeypatch.delenv("LABMIM_ENV", raising=False)
    monkeypatch.delenv("LABMIM_CONFIG_PATH", raising=False)
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"LABMIM_{field_name.upper()}", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_settings_accepts_the_shipped_default_yaml() -> None:
    settings = get_settings()

    assert settings.sensor_limits, "shipped QC limits must survive validation"
    assert settings.sensor_sum_columns == ["precip", "PL01_mm_Tot"]
    assert settings.sensor_wind_dir_columns == ["WD_WXT", "WindDir_WVT"]


def test_shipped_limits_carry_the_keys_apply_physical_limits_reads() -> None:
    for limit in get_settings().sensor_limits:
        assert {"column", "lower", "upper"} <= limit.keys()


def test_environment_variables_outrank_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """``default.yaml`` says INFO; the documented layer 4 must still win."""
    monkeypatch.setenv("LABMIM_LOG_LEVEL", "WARNING")
    get_settings.cache_clear()

    assert get_settings().log_level == "WARNING"


def test_yaml_keys_the_model_does_not_declare_are_still_rejected() -> None:
    """A misspelled setting must stay loud instead of silently falling back."""
    misspelled_yaml: dict[str, Any] = {"sensor_limitz": []}

    with pytest.raises(ValidationError):
        Settings(**misspelled_yaml)
