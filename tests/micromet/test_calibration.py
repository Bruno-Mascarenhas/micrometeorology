"""Tests for calibration application."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors.calibration import (
    apply_calibrations,
    uncalibrated_mapping_windows,
    unify_sensor_columns,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create synthetic sensor data spanning 2019."""
    idx = pd.date_range("2018-06-01", "2019-06-01", freq="1h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "CM3Up_Wm2_Avg": rng.uniform(100, 500, len(idx)),
            "PSP1_Wm2_Avg": rng.uniform(100, 500, len(idx)),
            "CMP21_Wm2_Avg": rng.uniform(100, 500, len(idx)),
        },
        index=idx,
    )


class TestApplyCalibrations:
    def test_multiplicative_factor(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "test",
            }
        ]
        original = sample_data["CM3Up_Wm2_Avg"].copy()
        apply_calibrations(sample_data, cals)

        # A date-only end_date is inclusive of the WHOLE boundary day: every
        # sub-daily sample of 2018-12-31 is calibrated, and the correction stops
        # cleanly at the day boundary (2019-01-01 00:00 onward is untouched).
        mask_before = sample_data.index <= pd.Timestamp("2018-12-31 23:59:59")
        mask_after = sample_data.index >= pd.Timestamp("2019-01-01")

        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_before, "CM3Up_Wm2_Avg"],
            original[mask_before] * 0.5,
        )
        # Data from the next day on should be unchanged.
        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_after, "CM3Up_Wm2_Avg"],
            original[mask_after],
        )

    def test_null_factor_sets_nan(self, sample_data):
        cals = [
            {
                "column": "CMP21_Wm2_Avg",
                "start_date": None,
                "end_date": "2019-01-01",
                "factor": None,
                "description": "sensor not installed",
            }
        ]
        apply_calibrations(sample_data, cals)
        # The whole boundary day (2019-01-01) is NaN'd, not just its midnight
        # sample; the day after is left intact.
        assert sample_data.loc["2019-01-01", "CMP21_Wm2_Avg"].isna().all()
        assert sample_data.loc["2019-01-02", "CMP21_Wm2_Avg"].notna().all()

    def test_missing_column_skipped(self, sample_data):
        cals = [
            {
                "column": "NONEXISTENT",
                "start_date": None,
                "end_date": None,
                "factor": 2.0,
                "description": "should skip",
            }
        ]
        # Should not raise
        apply_calibrations(sample_data, cals)


@pytest.fixture
def five_min_data() -> pd.DataFrame:
    """5-minute cadence spanning the 2018→2019 boundary day."""
    idx = pd.date_range("2018-12-30 00:00", "2019-01-02 23:55", freq="5min")
    return pd.DataFrame({"CM3Up_Wm2_Avg": np.full(len(idx), 100.0)}, index=idx)


class TestBoundaryDayCalibration:
    """A date-only end_date must cover every sample of the boundary day."""

    def test_factor_applies_to_all_samples_of_end_day(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": "2018-12-30",
                    "end_date": "2018-12-31",
                    "factor": 0.5,
                    "description": "boundary",
                }
            ],
        )
        end_day = five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"]
        assert len(end_day) == 288  # all 5-min samples of the day
        assert (end_day == 50.0).all()
        # Correction stops exactly at the day boundary.
        assert (five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"] == 100.0).all()

    def test_null_factor_nans_whole_end_day(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": None,
                    "end_date": "2018-12-31",
                    "factor": None,
                    "description": "invalid",
                }
            ],
        )
        assert five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"].isna().all()
        assert five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"].notna().all()

    def test_explicit_time_end_date_is_honored_exactly(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                {
                    "column": "CM3Up_Wm2_Avg",
                    "start_date": "2018-12-31 00:00",
                    "end_date": "2018-12-31 12:00",
                    "factor": 0.5,
                    "description": "explicit",
                }
            ],
        )
        # Inclusive up to exactly 12:00; the very next sample is untouched.
        assert five_min_data.loc["2018-12-31 12:00", "CM3Up_Wm2_Avg"] == 50.0
        assert five_min_data.loc["2018-12-31 12:05", "CM3Up_Wm2_Avg"] == 100.0
        assert (five_min_data.loc["2018-12-31 00:00":"2018-12-31 12:00"] == 50.0).all().all()

    def test_unify_has_no_boundary_day_hole(self, five_min_data):
        five_min_data["B"] = 20.0
        five_min_data = five_min_data.rename(columns={"CM3Up_Wm2_Avg": "A"})
        unify_sensor_columns(
            five_min_data,
            [
                {
                    "unified_name": "U",
                    "mappings": [
                        {"column": "A", "start_date": "2018-12-30", "end_date": "2018-12-31"},
                        {"column": "B", "start_date": "2019-01-01", "end_date": "2019-01-02"},
                    ],
                }
            ],
        )
        # The abutting mappings leave no unfilled hole on the boundary day.
        assert five_min_data.loc["2018-12-31", "U"].notna().all()
        assert (five_min_data.loc["2018-12-31", "U"] == 100.0).all()
        assert (five_min_data.loc["2019-01-01", "U"] == 20.0).all()


class TestUnifySensorColumns:
    def test_basic_switch(self):
        idx = pd.date_range("2018-01-01", "2019-06-01", freq="1D")
        df = pd.DataFrame(
            {
                "sensor_A": np.ones(len(idx)) * 10,
                "sensor_B": np.ones(len(idx)) * 20,
            },
            index=idx,
        )
        switches = [
            {
                "unified_name": "unified",
                "mappings": [
                    {"column": "sensor_A", "start_date": "2018-01-01", "end_date": "2018-12-31"},
                    {"column": "sensor_B", "start_date": "2019-01-01", "end_date": "2019-06-01"},
                ],
            }
        ]
        unify_sensor_columns(df, switches)
        assert "unified" in df.columns
        assert df.loc["2018-06-01", "unified"] == 10
        assert df.loc["2019-03-01", "unified"] == 20


class TestOverlapGuard:
    """Inclusive end dates make same-day-abutting records a config error."""

    def test_same_day_abutment_raises_clear_error(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "first",
            },
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-12-31",
                "end_date": "2019-06-01",
                "factor": 0.9,
                "description": "second",
            },
        ]
        with pytest.raises(ValueError, match=r"Overlapping calibrations.*2018-12-31"):
            apply_calibrations(sample_data, cals)

    def test_next_day_abutment_is_clean(self, sample_data):
        cals = [
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "factor": 0.5,
                "description": "first",
            },
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2019-01-01",
                "end_date": "2019-06-01",
                "factor": 0.9,
                "description": "second",
            },
        ]
        original = sample_data["CM3Up_Wm2_Avg"].copy()
        apply_calibrations(sample_data, cals)
        # Whole boundary day gets the FIRST factor; next day starts the second.
        end_day = sample_data.loc["2018-12-31", "CM3Up_Wm2_Avg"]
        np.testing.assert_allclose(end_day, original.loc["2018-12-31"] * 0.5)
        next_day = sample_data.loc["2019-01-01", "CM3Up_Wm2_Avg"]
        np.testing.assert_allclose(next_day, original.loc["2019-01-01"] * 0.9)

    def test_unify_same_day_abutment_raises(self, sample_data):
        df = sample_data.rename(columns={"CM3Up_Wm2_Avg": "sensor_A", "PSP1_Wm2_Avg": "sensor_B"})
        switches = [
            {
                "unified_name": "unified",
                "mappings": [
                    {"column": "sensor_A", "start_date": "2018-06-01", "end_date": "2018-12-31"},
                    {"column": "sensor_B", "start_date": "2018-12-31", "end_date": "2019-06-01"},
                ],
            }
        ]
        with pytest.raises(ValueError, match="Overlapping sensor-switch mappings"):
            unify_sensor_columns(df, switches)


class TestARecordThatClosesBeforeTheDataBegins:
    """An open-ended record inherits the DATASET's first timestamp.

    So a record that closes before the data starts resolves to
    ``[dataset-start, its own end]`` with the two inverted. It covers nothing and
    simply does not apply — but counting it as an interval made every later
    record "overlap" it, and the guard aborted the run.

    This is the ordinary case for a recent logger table, and it made
    ``labmim-sensor-process`` unable to read ``data/LBM_lenta_2025.dat`` at all:
    the shipped CMP21 record ends 2019-10-12 and that file begins 2025-05-14.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range("2025-05-14", periods=48, freq="h")
        return pd.DataFrame({"CMP21_Wm2_Avg": np.full(48, 100.0)}, index=index)

    def test_the_empty_record_does_not_raise_a_spurious_overlap(self) -> None:
        calibrations: list[dict[str, Any]] = [
            # Closes six years before this frame starts: empty here.
            {
                "column": "CMP21_Wm2_Avg",
                "end_date": "2019-10-12",
                "factor": None,
                "description": "not installed yet",
            },
            # The one that actually applies.
            {
                "column": "CMP21_Wm2_Avg",
                "start_date": "2019-10-13",
                "factor": 0.985,
                "description": "sensitivity correction",
            },
        ]

        result = apply_calibrations(self._frame().copy(), calibrations)

        assert result["CMP21_Wm2_Avg"].notna().all()
        assert result["CMP21_Wm2_Avg"].iloc[0] == pytest.approx(98.5)

    def test_a_genuine_overlap_is_still_refused(self) -> None:
        """The guard must not be weakened: two records that really do cover the
        same day are still a configuration error."""
        # Both inside the 48-hour frame, so both resolve to real intervals, and
        # they share 2025-05-14: a genuine configuration error.
        calibrations: list[dict[str, Any]] = [
            {
                "column": "CMP21_Wm2_Avg",
                "start_date": "2025-05-14",
                "end_date": "2025-05-14",
                "factor": 1.0,
                "description": "a",
            },
            {
                "column": "CMP21_Wm2_Avg",
                "start_date": "2025-05-14",
                "factor": 2.0,
                "description": "b",
            },
        ]

        with pytest.raises(ValueError, match="Overlapping"):
            apply_calibrations(self._frame().copy(), calibrations)

    def test_the_shipped_calibrations_load_against_a_recent_file(self) -> None:
        """The end-to-end shape of the bug: the real config against a 2025 frame."""
        from micrometeorology.common.config import get_settings
        from micrometeorology.sensors.calibration import load_calibrations

        settings = get_settings()
        path = settings.configs_dir / "calibrations.yaml"
        if not path.is_file():  # pragma: no cover - only in a stripped checkout
            pytest.skip("shipped calibrations not present")

        result = apply_calibrations(self._frame().copy(), load_calibrations(path))

        assert len(result) == 48


class TestACalibrationSurvivesAColumnRename:
    """A record keyed on a column name dies when the logger renames the channel.

    The Eppley PSP's post-2019 sensitivity correction sat in the config for seven
    years naming only the pre-v11 spelling ``PSP1_Wm2_Avg``. The logger renamed
    the channel to ``PSP_Wm2_Avg`` on 2019-03-15 — a rename the sensor_switches
    block in the same file documents — so the correction reached 2019-02-26 and
    stopped. The PSP is the diffuse sensor of the current operational era, so the
    published diffuse ran 8.5% low and nothing said a word: the record was
    declared, it simply matched nothing.
    """

    def test_both_spellings_of_the_pyranometer_are_calibrated(self) -> None:
        from micrometeorology.common.config import get_settings
        from micrometeorology.sensors.calibration import load_calibrations

        settings = get_settings()
        path = settings.configs_dir / "calibrations.yaml"
        if not path.is_file():  # pragma: no cover - only in a stripped checkout
            pytest.skip("shipped calibrations not present")
        records = load_calibrations(path)

        factors = {
            record["column"]: record["factor"]
            for record in records
            if record["column"] in {"PSP1_Wm2_Avg", "PSP_Wm2_Avg"}
            and record.get("start_date", "") >= "2019"
        }

        assert "PSP_Wm2_Avg" in factors, "the renamed channel carries no calibration"
        assert factors["PSP_Wm2_Avg"] == factors["PSP1_Wm2_Avg"], (
            "same instrument, same programmed sensitivity: the correction must match"
        )

    def test_a_record_that_matches_nothing_is_reported(self, caplog) -> None:
        """'Declared' and 'applied' must be distinguishable in the output."""
        index = pd.date_range("2025-01-01", periods=4, freq="h")
        frame = pd.DataFrame({"PSP_Wm2_Avg": [1.0, 2.0, 3.0, 4.0]}, index=index)
        records: list[dict[str, Any]] = [
            {
                "column": "PSP_Wm2_Avg",
                "start_date": "2019-01-01",
                "end_date": "2019-12-31",
                "factor": 2.0,
                "description": "window with no data here",
            }
        ]

        with caplog.at_level("WARNING"):
            apply_calibrations(frame.copy(), records)

        assert "matched no populated sample" in caplog.text


class TestUncalibratedMappingWindows:
    """A column calibrated over PART of the window it feeds puts a step in the data.

    Measured on the real archive: no ``PSP1_Wm2_Avg`` record covers
    2016-09-29..2017-12-31 while 2018-01-01 onward is scaled by 0.9383408072, so
    the published global shortwave drops 6.2% at a date on which nothing
    physical happened -- only a file handover. The raw instrument record is
    continuous across it (monthly p95 clearness 0.7600 -> 0.7612); the published
    series reads 0.7600 -> 0.7143.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range("2016-01-01", "2019-12-31", freq="D")
        return pd.DataFrame({"PSP1_Wm2_Avg": 1.0, "Temp1_Avg": 25.0}, index=index)

    @staticmethod
    def _switches() -> list[dict[str, Any]]:
        return [
            {
                "unified_name": "Sw_dw",
                "mappings": [{"column": "PSP1_Wm2_Avg", "start_date": None, "end_date": None}],
            },
            {
                "unified_name": "T",
                "mappings": [{"column": "Temp1_Avg", "start_date": None, "end_date": None}],
            },
        ]

    def test_the_half_covered_window_is_reported(self):
        calibrations = [
            {"column": "PSP1_Wm2_Avg", "start_date": "2018-01-01", "end_date": None, "factor": 0.94}
        ]

        gaps = uncalibrated_mapping_windows(self._frame(), calibrations, self._switches())

        assert len(gaps) == 1
        unified, column, start, end = gaps[0]
        assert (unified, column) == ("Sw_dw", "PSP1_Wm2_Avg")
        assert start == pd.Timestamp("2016-01-01")
        assert end.date() == pd.Timestamp("2017-12-31").date()

    def test_a_column_with_no_record_at_all_is_not_a_gap(self):
        """The logger's own multiplier is the calibration for most channels.

        Reporting those too buries the one real finding under 37 non-findings,
        which is how a warning stops being read.
        """
        calibrations = [
            {"column": "PSP1_Wm2_Avg", "start_date": None, "end_date": None, "factor": 0.94}
        ]

        assert uncalibrated_mapping_windows(self._frame(), calibrations, self._switches()) == []

    def test_a_hole_between_two_records_is_reported(self):
        calibrations = [
            {
                "column": "PSP1_Wm2_Avg",
                "start_date": None,
                "end_date": "2016-12-31",
                "factor": 0.94,
            },
            {
                "column": "PSP1_Wm2_Avg",
                "start_date": "2018-01-01",
                "end_date": None,
                "factor": 1.09,
            },
        ]

        gaps = uncalibrated_mapping_windows(self._frame(), calibrations, self._switches())

        assert len(gaps) == 1
        _unified, _column, start, end = gaps[0]
        assert start.date() == pd.Timestamp("2017-01-01").date()
        assert end.date() == pd.Timestamp("2017-12-31").date()

    def test_a_null_factor_record_counts_as_covered(self):
        """`factor: null` is a decision about the window, not an absence of one."""
        calibrations: list[dict[str, Any]] = [
            {
                "column": "PSP1_Wm2_Avg",
                "start_date": None,
                "end_date": "2017-12-31",
                "factor": None,
            },
            {
                "column": "PSP1_Wm2_Avg",
                "start_date": "2018-01-01",
                "end_date": None,
                "factor": 0.94,
            },
        ]

        assert uncalibrated_mapping_windows(self._frame(), calibrations, self._switches()) == []
