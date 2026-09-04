"""Tests for Campbell `.dat` ingestion and multi-file merging."""

from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from micrometeorology.common.config import SensorRangeLimit
from micrometeorology.sensors.ingestion import (
    apply_physical_limits,
    merge_dat_files,
    read_campbell_dat,
    values_outside_declared_limits,
)


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
                ("2025-06-25 12:00:00", [999.0, 77.0]),
                ("2025-06-25 12:05:00", [450.0, 78.0]),
            ],
        )

        merged = merge_dat_files([early, late])

        assert merged.index.is_monotonic_increasing
        assert len(merged) == 3, "12:00 is written by both files and collapses to one row"
        overlap = merged.loc["2025-06-25 12:00:00"]
        assert overlap["only_late"] == pytest.approx(77.0)
        assert overlap["only_early"] == pytest.approx(11.0)
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
        assert merged.loc["2025-06-25 12:00:00", "shared"] == pytest.approx(400.0)

    def test_earlier_null_falls_through_to_later_non_null(self, tmp_path: Path) -> None:
        """A sentinel/NaN in the earlier file yields to the later file's value."""
        early = _write_toa5(
            tmp_path / "early.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [-999.0])],
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
        under mypy, so the parameter stays a covariant ``Sequence[str | Path]``.
        This call is the mypy assertion.
        """
        written = _write_toa5(
            tmp_path / "one.dat",
            ["shared"],
            [("2025-06-25 12:00:00", [400.0])],
        )
        as_strings: list[str] = [written]
        as_paths: list[Path] = [Path(written)]

        assert merge_dat_files(as_strings).equals(merge_dat_files(as_paths))


class TestTheGateAlsoHoldsAfterCalibration:
    """A value AT the boundary crosses it once an instrument factor scales it.

    ``CM3Up_Wm2_Avg`` capped at exactly 1500 W/m2 by its own gate reaches the
    published artifact at 1508.65 -- 1500 x its post-2019 factor -- and the
    Eppley PSP factor pushes 578 more over. The gate runs on the raw signal,
    which is what protects the calibration from multiplying a non-physical
    value, so the number that actually gets written must be re-checked too.
    """

    LIMITS: ClassVar[list[SensorRangeLimit]] = [
        SensorRangeLimit(column="CM3Up_Wm2_Avg", lower=-20.0, upper=1500.0)
    ]

    def test_a_calibrated_boundary_value_is_reported(self):
        frame = pd.DataFrame({"CM3Up_Wm2_Avg": [1000.0, 1500.0 * 1.0058, 1490.0]})

        assert values_outside_declared_limits(frame, self.LIMITS) == {"CM3Up_Wm2_Avg": 1}

    def test_a_frame_inside_its_gates_reports_nothing(self):
        frame = pd.DataFrame({"CM3Up_Wm2_Avg": [0.0, 1000.0, 1500.0, -20.0]})

        assert values_outside_declared_limits(frame, self.LIMITS) == {}

    def test_reapplying_the_gate_clears_the_report(self):
        frame = pd.DataFrame({"CM3Up_Wm2_Avg": [1000.0, 1508.65]})

        gated = apply_physical_limits(frame.copy(), self.LIMITS)

        assert values_outside_declared_limits(gated, self.LIMITS) == {}
        assert gated["CM3Up_Wm2_Avg"].notna().sum() == 1

    def test_a_column_the_frame_lacks_is_not_a_violation(self):
        assert values_outside_declared_limits(pd.DataFrame({"other": [1.0]}), self.LIMITS) == {}


class TestTextColumnsSurviveTheNumericCoercion:
    """The hourly `*_ok_fraction` columns are computed from the logger's own text
    quality flag, so a reader that coerced it to NaN would publish a completeness
    the station never reported — and nothing pinned the preservation."""

    def test_a_declared_text_column_keeps_its_strings(self, tmp_path: Path) -> None:
        path = tmp_path / "station.dat"
        _write_toa5(
            path,
            ["MetSENS1_Status", "AirT1_C_Avg"],
            [
                ("2025-06-25 12:00:00", ["OK", 25.0]),
                ("2025-06-25 12:05:00", ["Unknown Fault", 25.1]),
            ],
        )

        frame = read_campbell_dat(path, text_columns=["MetSENS1_Status"])

        assert frame["MetSENS1_Status"].tolist() == ["OK", "Unknown Fault"]
        assert frame["AirT1_C_Avg"].iloc[0] == pytest.approx(25.0)

    def test_the_same_column_read_without_the_declaration_becomes_nan(self, tmp_path: Path) -> None:
        """Which is why the declaration exists: numeric coercion is the default."""
        path = tmp_path / "station.dat"
        _write_toa5(path, ["MetSENS1_Status"], [("2025-06-25 12:00:00", ["OK"])])

        frame = read_campbell_dat(path)

        assert frame["MetSENS1_Status"].isna().all()
