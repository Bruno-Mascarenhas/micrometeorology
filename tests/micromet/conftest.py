"""Shared fixtures and helpers for the micrometeorology suite."""

import inspect
from collections.abc import Iterator

import pytest

from micrometeorology.common.config import Settings, get_settings
from micrometeorology.stats.distributions import FAMILIES


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


@pytest.fixture(autouse=True)
def neutral_product_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop an ambient ``LABMIM_TIMEZONE`` so the product default decides the hour.

    A test that inherits the developer's shell pins the wrong local hour into
    every title, manifest and daylight gate it compares against.
    """
    monkeypatch.delenv("LABMIM_TIMEZONE", raising=False)


def required_fit_options(family: str) -> set[str]:
    """Keyword-only parameters of a family's estimator that carry no default.

    Derived from the signature rather than listed by hand, so a new family or a
    new covariate lands in the check without anyone remembering to say so.
    """
    signature = inspect.signature(FAMILIES[family].fit)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
