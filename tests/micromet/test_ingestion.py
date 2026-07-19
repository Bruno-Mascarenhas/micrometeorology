"""Tests for Campbell `.dat` ingestion and multi-file merging."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from micrometeorology.sensors.ingestion import merge_dat_files


def _write_toa5(path: Path, columns: list[str], rows: list[tuple[str, list[float]]]) -> str:
    """Write a synthetic TOA5 file with the real 4-line header structure.

    ``columns`` are the data columns (beyond TIMESTAMP/RECORD), letting each
    file declare a different sensor set.
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
        cells = ",".join(f"{v:.1f}" for v in values)
        lines.append(f'"{ts}",{i},{cells}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


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
