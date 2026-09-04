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


def test_nearest_join_needs_a_time_index_in_both_files(tmp_path: Path) -> None:
    """The positional fallback of ``--join nearest`` could never run: a
    ``Timedelta`` tolerance on an integer key raised a ``MergeError`` traceback."""
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_text("T2\n20.0\n21.0\n22.0\n", encoding="utf-8")

    result = CliRunner().invoke(
        compute_metrics.app,
        ["-a", str(tmp_path / "a.csv"), "-b", str(tmp_path / "b.csv"), "--join", "nearest"],
    )

    assert result.exit_code == 2, result.output
    assert "needs a time index in both files" in result.output


def test_nearest_join_with_no_pair_inside_the_tolerance_is_an_error_not_a_table_of_nan(
    tmp_path: Path,
) -> None:
    """``merge_asof`` is a LEFT join, so ``aligned.empty`` never fired for two
    disjoint years and the command printed every metric as NaN with exit 0."""
    dataset_a = _hourly_csv(tmp_path / "a.csv", 0.0)
    index = pd.date_range("2021-01-01", periods=6, freq="1h", name="TIMESTAMP")
    dataset_b = tmp_path / "b.csv"
    pd.DataFrame({"T2": [1.0] * 6, "RH": [1.0] * 6}, index=index).to_csv(dataset_b)

    result = CliRunner().invoke(
        compute_metrics.app, ["-a", str(dataset_a), "-b", str(dataset_b), "--join", "nearest"]
    )

    assert result.exit_code == 1, result.output
    assert "No overlapping data after alignment" in result.output


def test_nearest_join_pairs_rows_offset_by_less_than_the_tolerance(tmp_path: Path) -> None:
    """The successful half of `--join nearest` — and `--tolerance` itself."""
    index_a = pd.date_range("2024-01-01 00:00", periods=6, freq="1h", name="TIMESTAMP")
    index_b = index_a + pd.Timedelta(minutes=10)
    dataset_a = tmp_path / "a.csv"
    dataset_b = tmp_path / "b.csv"
    pd.DataFrame({"T2": [20.0] * 6}, index=index_a).to_csv(dataset_a)
    pd.DataFrame({"T2": [21.0] * 6}, index=index_b).to_csv(dataset_b)
    output = tmp_path / "out" / "metrics.csv"

    result = CliRunner().invoke(
        compute_metrics.app,
        [
            "-a",
            str(dataset_a),
            "-b",
            str(dataset_b),
            "--join",
            "nearest",
            "--tolerance",
            "30min",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    table = pd.read_csv(output, index_col=0)
    assert list(table.loc["n"]) == [6]


def test_a_tolerance_tighter_than_the_offset_pairs_nothing(tmp_path: Path) -> None:
    """The same two files, refused: the tolerance is what decides, so it has to be
    read from the flag rather than from a default nobody passes."""
    index_a = pd.date_range("2024-01-01 00:00", periods=6, freq="1h", name="TIMESTAMP")
    index_b = index_a + pd.Timedelta(minutes=10)
    dataset_a = tmp_path / "a.csv"
    dataset_b = tmp_path / "b.csv"
    pd.DataFrame({"T2": [20.0] * 6}, index=index_a).to_csv(dataset_a)
    pd.DataFrame({"T2": [21.0] * 6}, index=index_b).to_csv(dataset_b)

    result = CliRunner().invoke(
        compute_metrics.app,
        [
            "-a",
            str(dataset_a),
            "-b",
            str(dataset_b),
            "--join",
            "nearest",
            "--tolerance",
            "5min",
        ],
    )

    assert result.exit_code == 1, result.output
    assert "No overlapping data after alignment" in result.output
