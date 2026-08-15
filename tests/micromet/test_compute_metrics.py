"""CLI contracts for ``labmim-metrics``.

The command's job is to print numbers an operator will quote, so the two ways
it can quietly print the wrong ones — comparing columns nobody asked for, and
aligning two files by row position instead of by time — are pinned here.
"""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from micrometeorology.cli import compute_metrics


def _hourly_csv(path: Path, t2_offset: float) -> Path:
    """Six hourly rows with a TIMESTAMP index; only T2 is shifted by *t2_offset*."""
    index = pd.date_range("2020-01-01", periods=6, freq="1h", name="TIMESTAMP")
    values = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
    pd.DataFrame({"T2": [v + t2_offset for v in values], "RH": values}, index=index).to_csv(path)
    return path


def test_a_columns_option_naming_nothing_is_a_usage_error(tmp_path: Path) -> None:
    """``-c " "`` must not silently widen to every common column."""
    dataset_a = _hourly_csv(tmp_path / "a.csv", 0.0)
    dataset_b = _hourly_csv(tmp_path / "b.csv", 1.0)

    result = CliRunner().invoke(
        compute_metrics.app, ["-a", str(dataset_a), "-b", str(dataset_b), "-c", " "]
    )

    assert result.exit_code == 2
    assert "names no column" in result.output


def test_blank_tokens_between_column_names_are_dropped(tmp_path: Path) -> None:
    """``-c T2,`` names one column, so nothing may be reported as missing."""
    dataset_a = _hourly_csv(tmp_path / "a.csv", 0.0)
    dataset_b = _hourly_csv(tmp_path / "b.csv", 1.0)

    result = CliRunner().invoke(
        compute_metrics.app, ["-a", str(dataset_a), "-b", str(dataset_b), "-c", "T2,"]
    )

    assert result.exit_code == 0, result.output
    assert "Comparing 1 columns: ['T2']" in result.output
    assert "Columns not found" not in result.output


def test_positional_alignment_is_announced_when_no_file_carries_a_time_index(
    tmp_path: Path,
) -> None:
    """A file with no parseable timestamp still joins on row order — say so."""
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("T2\n20.0\n21.0\n22.0\n", encoding="utf-8")

    result = CliRunner().invoke(
        compute_metrics.app, ["-a", str(tmp_path / "a.csv"), "-b", str(tmp_path / "b.csv")]
    )

    assert result.exit_code == 0, result.output
    assert "rows are aligned by position, not by time" in result.output


def test_the_metrics_table_is_written_under_an_output_directory_that_does_not_exist_yet(
    tmp_path: Path,
) -> None:
    """The whole computation is discarded if ``-o`` cannot create its own directory."""
    dataset_a = _hourly_csv(tmp_path / "a.csv", 0.0)
    dataset_b = _hourly_csv(tmp_path / "b.csv", 1.0)
    output = tmp_path / "out" / "metrics" / "table.csv"

    result = CliRunner().invoke(
        compute_metrics.app,
        ["-a", str(dataset_a), "-b", str(dataset_b), "-o", str(output)],
    )

    assert result.exit_code == 0, result.exception or result.output
    assert sorted(pd.read_csv(output, index_col=0).columns) == ["RH", "T2"]


def test_a_datetime_indexed_pair_is_not_warned_about(tmp_path: Path) -> None:
    dataset_a = _hourly_csv(tmp_path / "a.csv", 0.0)
    dataset_b = _hourly_csv(tmp_path / "b.csv", 1.0)

    result = CliRunner().invoke(compute_metrics.app, ["-a", str(dataset_a), "-b", str(dataset_b)])

    assert result.exit_code == 0, result.output
    assert "aligned by position" not in result.output
