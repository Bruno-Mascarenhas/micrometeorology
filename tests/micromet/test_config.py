"""Tests for the YAML + environment settings layer (``get_settings``).

The shipped ``configs/micromet/default.yaml`` is part of the contract here:
every key it declares must be a real field on ``Settings``, otherwise
``extra="forbid"`` rejects the repository's own configuration and every
consumer of ``get_settings()`` fails on its first call.
"""

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
    # Membership rather than an exact list: the rules are keyed to the RAW TOA5
    # column names the archive carries across nine logger-program eras, and that
    # set grows whenever an instrument is added. The names below must match a
    # column of the merged frame, or the rule they name never fires.
    assert {"PL01_mm_Tot", "Rain_WXT_Tot", "precip"} <= set(settings.sensor_sum_columns)
    assert {"WD_WXT_Avg", "WindDir_D1_WVT", "WindDir1_GMX"} <= set(settings.sensor_wind_dir_columns)


def test_shipped_limits_declare_a_range_that_can_reject_something() -> None:
    """The model makes the three fields mandatory; their ORDER it cannot."""
    for limit in get_settings().sensor_limits:
        assert limit.column
        assert limit.lower < limit.upper, limit.column


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


def test_a_limit_key_the_gate_does_not_read_is_rejected_at_load() -> None:
    """``exempt_at_or_below`` was documented for years and read by nothing."""
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "sensor_persistence_limits": [
                    {"column": "Temp1_Avg", "min_run": 36, "exempt_at_or_below": 0.5}
                ]
            }
        )


def test_a_limit_missing_a_field_the_gate_reads_fails_at_load_not_mid_run() -> None:
    """A ``min_run`` left out used to surface as a KeyError deep inside the mask."""
    with pytest.raises(ValidationError):
        Settings.model_validate({"sensor_persistence_limits": [{"column": "Temp1_Avg"}]})
