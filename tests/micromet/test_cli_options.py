"""Contracts for the shared repeatable/comma-separated option parsers.

Every WRF console script and ``labmim-metrics`` accepts ``-x a,b`` and
``-x a -x b`` interchangeably; the parsing used to be four private copies, one
of which had drifted. These tests pin the one implementation they now share.
"""

import pytest
import typer

from micrometeorology.cli import (
    compute_metrics,
    export_operational_series,
    export_wrf_geojson,
    render_wrf_maps,
    run_wrf_pipeline,
)
from micrometeorology.common.cli_options import parse_csv, parse_int_csv


class TestParseCsv:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, ()),
            ("", ()),
            ([], ()),
            ("temperature", ("temperature",)),
            ("temperature,wind", ("temperature", "wind")),
            (["temperature", "wind"], ("temperature", "wind")),
            (["temperature,wind", "rain"], ("temperature", "wind", "rain")),
            ("  temperature , wind ", ("temperature", "wind")),
            ("temperature,,wind", ("temperature", "wind")),
            (" ", ()),
        ],
    )
    def test_repeated_and_comma_separated_spellings_agree(self, raw, expected) -> None:
        assert parse_csv(raw) == expected


class TestParseIntCsv:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, ()),
            ("1", (1,)),
            ("1,4", (1, 4)),
            (["1", "4"], (1, 4)),
            ([" 1 , 4 ", "5"], (1, 4, 5)),
            ("1,,4", (1, 4)),
        ],
    )
    def test_domain_spellings_agree(self, raw, expected) -> None:
        assert parse_int_csv(raw) == expected

    def test_a_non_integer_domain_is_a_usage_error_not_a_traceback(self) -> None:
        """``-D d03`` is a typo an operator should be told about, not a crash."""
        with pytest.raises(typer.BadParameter, match="expected a whole number, got 'd03'"):
            parse_int_csv("1,d03")


def test_every_wrf_cli_shares_one_parser_pair() -> None:
    """A fix to the parsing must not have to be applied once per console script."""
    for module in (
        render_wrf_maps,
        export_wrf_geojson,
        run_wrf_pipeline,
        export_operational_series,
    ):
        assert module.parse_csv is parse_csv
        assert module.parse_int_csv is parse_int_csv
    assert compute_metrics.parse_csv is parse_csv
