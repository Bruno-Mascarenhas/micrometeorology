"""Contract for the logging level every entry point resolves through.

``--log-level`` used to resolve through ``getattr(logging, name, INFO)``, so a
typo ran the whole pipeline at a verbosity nobody asked for and said nothing.
"""

import logging
from collections.abc import Iterator

import pytest

from micrometeorology.common.logging import setup_logging


@pytest.fixture(autouse=True)
def restored_root_logger() -> Iterator[None]:
    """``setup_logging`` reconfigures the PROCESS root logger, not a local one.

    Without this the last case parametrized below leaves every later test
    running at the level it happened to ask for.
    """
    root = logging.getLogger()
    level, handlers = root.level, root.handlers[:]
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.parametrize("level", ["DEBUG", "info", "Warning", "ERROR", "critical"])
def test_a_level_name_is_accepted_in_any_case(level: str) -> None:
    setup_logging(level)

    assert logging.getLogger().level == logging.getLevelNamesMapping()[level.upper()]


def test_a_misspelt_level_is_refused_instead_of_falling_back_to_info() -> None:
    with pytest.raises(ValueError, match="unknown logging level 'INFOO'"):
        setup_logging("INFOO")


def test_an_attribute_of_the_logging_module_that_is_not_a_level_is_refused() -> None:
    """``getattr`` accepted anything the module happened to export."""
    with pytest.raises(ValueError, match="unknown logging level"):
        setup_logging("BASICCONFIG")
