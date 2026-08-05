"""Tests for Campbell `.dat` ingestion and multi-file merging."""

from pathlib import Path

import pandas as pd
import pytest

from micrometeorology.sensors.ingestion import merge_dat_files, read_campbell_dat


def _write_toa5(path: Path, columns: list[str], rows: list[tuple[str, list[float | str]]]) -> str:
    """Write a synthetic TOA5 file with the real 4-line header structure.

    ``columns`` are the data columns (beyond TIMESTAMP/RECORD), letting each
    file declare a different sensor set.  A cell given as ``str`` is written
    verbatim, which is how the datalogger's literal ``NAN`` token is reproduced.
    """
    names = ",".join(f'"{c}"' for c in ["TIMESTAMP", "RECORD", *columns])
    units = ",".join(f'"{u}"' for u in ["TS", "RN", *["W/meter^2"] * len(columns)])
    aggs = ",".join(f'"{a}"' for a in ["", "", *["Avg"] * len(columns)])
    lines = [
        '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","CPU:PRG_LABMIM.CR5","49836","LBM_lenta"',
        names,
        units,
        aggs,
    ]
    for i, (ts, values) in enumerate(rows):
        cells = ",".join(v if isinstance(v, str) else f"{v:.1f}" for v in values)
        lines.append(f'"{ts}",{i},{cells}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


class TestReadCampbellDat:
    @pytest.mark.filterwarnings("error::pandas.errors.PandasChangeWarning")
    def test_literal_nan_token_becomes_a_float_nan(self, tmp_path: Path) -> None:
        """The datalogger writes ``NAN``, which is not in read_csv's NA list.

        Without the text-to-numeric coercion the column stays textual and the
        sentinel comparison below it raises ``TypeError``.
        """
        path = _write_toa5(
            tmp_path / "nan.dat",
            ["shared"],
            [
                ("2025-06-25 12:00:00", ["NAN"]),
                ("2025-06-25 12:05:00", [-999.0]),
                ("2025-06-25 12:10:00", [450.0]),
            ],
        )

        df = read_campbell_dat(path)

        assert df["shared"].dtype == "float64"
        assert pd.isna(df.loc["2025-06-25 12:00:00", "shared"])
        assert pd.isna(df.loc["2025-06-25 12:05:00", "shared"])
        assert df.loc["2025-06-25 12:10:00", "shared"] == pytest.approx(450.0)


class TestMergeDatFiles:
    def test_disjoint_new_columns_merge_without_data_loss(self, tmp_path: Path) -> None:
        """A column only present in a later file survives an overlapping timestamp.

        The docstring promises "first non-null value per column"; keeping the
        first *row* whole would NaN out any column absent from the earlier file.
        """
        early = _write_toa5(
            tmp_path / "early.dat",
            ["shared", "only_early"],
            [
                ("2025-06-25 12:00:00", [400.0, 11.0]),
                ("2025-06-25 12:10:00", [500.0, 12.0]),
            ],
        )
        late = _write_toa5(
            tmp_path / "late.dat",
            ["shared", "only_late"],
            [
                ("2025-06-25 12:00:00", [999.0, 77.0]),  # overlaps early at 12:00
                ("2025-06-25 12:05:00", [450.0, 78.0]),
            ],
        )

        merged = merge_dat_files([early, late])

        assert merged.index.is_monotonic_increasing
        assert len(merged) == 3  # 12:00, 12:05, 12:10 (12:00 collapsed)
        overlap = merged.loc["2025-06-25 12:00:00"]
        # The later file's exclusive column is preserved, not NaN'd out.
        assert overlap["only_late"] == pytest.approx(77.0)
        assert overlap["only_early"] == pytest.approx(11.0)
        # A row unique to the later file keeps its columns; the other file's
        # exclusive column is simply missing there.
        assert merged.loc["2025-06-25 12:05:00", "only_late"] == pytest.approx(78.0)
        assert pd.isna(merged.loc["2025-06-25 12:05:00", "only_early"])

    def test_conflicting_values_keep_the_earlier_files_value(self, tmp_path: Path) -> None:
        early = _write_toa5(
            tmp_path / "early.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [400.0])],
        )
        late = _write_toa5(
            tmp_path / "late.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [999.0])],
        )

        merged = merge_dat_files([early, late])

        assert len(merged) == 1
        # Chronological file order → the earlier file wins the conflict.
        assert merged.loc["2025-06-25 12:00:00", "shared"] == pytest.approx(400.0)

    def test_earlier_null_falls_through_to_later_non_null(self, tmp_path: Path) -> None:
        """A sentinel/NaN in the earlier file yields to the later file's value."""
        early = _write_toa5(
            tmp_path / "early.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [-999.0])],  # sentinel -> NaN
        )
        late = _write_toa5(
            tmp_path / "late.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [450.0])],
        )

        merged = merge_dat_files([early, late])

        assert merged.loc["2025-06-25 12:00:00", "shared"] == pytest.approx(450.0)

    def test_empty_paths_rejected(self) -> None:
        with pytest.raises(ValueError, match="No files to merge"):
            merge_dat_files([])

    def test_the_path_lists_real_callers_build_are_accepted(self, tmp_path: Path) -> None:
        """Both spellings must type-check, not just run.

        ``common.paths.find_files`` hands over a ``list[Path]`` and the CLIs a
        ``list[str]``; an invariant ``list[str | Path]`` parameter rejects both
        under mypy, so the parameter has to stay a covariant
        ``Sequence[str | Path]``. This call is the mypy assertion.
        """
        written = _write_toa5(
            tmp_path / "one.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [400.0])],
        )
        as_strings: list[str] = [written]
        as_paths: list[Path] = [Path(written)]

        assert merge_dat_files(as_strings).equals(merge_dat_files(as_paths))
