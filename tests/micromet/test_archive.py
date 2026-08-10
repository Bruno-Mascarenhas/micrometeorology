"""Tests for micrometeorology.sensors.archive.

Offline: the staging repairs and the sentinel rules are exercised against
synthetic TOA5 files written into ``tmp_path``, never against ``data/``.

The manifests themselves are checked for the properties that make them a
manifest rather than a list — no duplicates, known staging directives, and a
loud failure when an entry is missing. The audited row counts are verified by
``labmim-archive --strict`` against the real archive, which no offline test can
reproduce.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from micrometeorology.sensors import archive
from micrometeorology.sensors.archive import (
    ARCHIVE_END,
    ARCHIVE_START,
    LENTA_MANIFEST,
    RAIN_MANIFEST,
    mask_sentinels,
    stage_archive,
    verify_frame,
)
from micrometeorology.sensors.calibration import resolve_mapping_windows, unify_sensor_columns

TOA5_METADATA = '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","TEST","1","LBM_test"'


def write_toa5(path, columns, rows) -> None:
    """Write a minimal but structurally faithful TOA5 table."""
    lines = [
        TOA5_METADATA,
        ",".join(f'"{name}"' for name in columns),
        ",".join('""' for _ in columns),
        ",".join('""' for _ in columns),
    ]
    lines += [
        ",".join(f'"{row[0]}"' if index == 0 else str(row[index]) for index in range(len(row)))
        for row in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestManifests:
    @pytest.mark.parametrize(
        ("name", "manifest"), [("lenta", LENTA_MANIFEST), ("rain", RAIN_MANIFEST)]
    )
    def test_paths_are_unique(self, name, manifest):
        """The same table twice would double-count its rows in the merge."""
        paths = [entry.path for entry in manifest]
        assert len(paths) == len(set(paths)), name

    @pytest.mark.parametrize(
        ("name", "manifest"), [("lenta", LENTA_MANIFEST), ("rain", RAIN_MANIFEST)]
    )
    def test_every_entry_explains_itself(self, name, manifest):
        """A note is what stops a future reader deleting a file that is a sole source."""
        assert all(entry.note for entry in manifest), name

    @pytest.mark.parametrize(
        ("name", "manifest"), [("lenta", LENTA_MANIFEST), ("rain", RAIN_MANIFEST)]
    )
    def test_staging_directives_are_implemented(self, name, manifest):
        directives = {entry.staging for entry in manifest if entry.staging}
        assert directives <= set(archive._STAGERS), name

    def test_the_misnamed_rain_table_is_in_the_rain_manifest(self):
        """dados-labmim/LBM_lenta.dat is the RAIN table and the sole source of Feb 2019."""
        rain_paths = {entry.path for entry in RAIN_MANIFEST}
        lenta_paths = {entry.path for entry in LENTA_MANIFEST}
        assert "dados-labmim/LBM_lenta.dat" in rain_paths
        assert "dados-labmim/LBM_lenta.dat" not in lenta_paths

    def test_the_backup_rotation_is_not_filtered_out(self):
        """Three .backup tables are each the only source of an austral winter."""
        winters = {
            "dados-labmim/LBM_lenta_2020.dat.backup",
            "dados-labmim/LBM_lenta_2022.dat.backup",
            "dados-labmim/LBM_lenta_2024.dat.backup",
        }
        assert winters <= {entry.path for entry in LENTA_MANIFEST}

    def test_a_different_station_is_never_listed(self):
        """BTS is a second site; merging it would corrupt a decade of statistics."""
        for manifest in (LENTA_MANIFEST, RAIN_MANIFEST):
            assert not [entry for entry in manifest if "BTS" in entry.path]


class TestStaging:
    def test_missing_entry_is_fatal_not_skipped(self, tmp_path):
        """A silently shorter record is the failure the manifest exists to prevent."""
        manifest = (archive.ArchiveFile("nao-existe.dat", note="teste"),)
        with pytest.raises(FileNotFoundError, match="manifest entry missing"):
            stage_archive(manifest, tmp_path, tmp_path / "staged")

    def test_unstaged_entry_is_returned_as_found(self, tmp_path):
        source = tmp_path / "plain.dat"
        write_toa5(source, ["TIMESTAMP", "x"], [("2020-01-01 00:00:00", 1.0)])
        resolved = stage_archive((archive.ArchiveFile("plain.dat"),), tmp_path, tmp_path / "staged")
        assert resolved == [source]

    def test_clock_shift_repairs_the_headerless_table(self, tmp_path):
        """Two defects at once: no TOA5 header, and early rows are one hour behind."""
        source = tmp_path / "LBM_lenta_2020_03.dat"
        source.write_text(
            "TIMESTAMP,RECORD,x\n"
            "2020-02-28 11:45:00,1,10.0\n"
            "2020-02-28 11:50:00,2,11.0\n"
            "2020-02-28 12:55:00,3,12.0\n",
            encoding="utf-8",
        )
        (staged,) = stage_archive(
            (archive.ArchiveFile("LBM_lenta_2020_03.dat", staging=archive._CLOCK_PLUS_ONE_HOUR),),
            tmp_path,
            tmp_path / "staged",
        )
        frame = pd.read_csv(staged, skiprows=[0, 2, 3])
        stamps = pd.to_datetime(frame["TIMESTAMP"]).tolist()
        # The two mis-stamped rows move forward one hour; the late one does not.
        assert stamps == [
            pd.Timestamp("2020-02-28 12:45:00"),
            pd.Timestamp("2020-02-28 12:50:00"),
            pd.Timestamp("2020-02-28 12:55:00"),
        ]

    def test_clock_shift_output_survives_the_standard_reader(self, tmp_path):
        """The synthetic header is the point: without it skiprows eats two data rows."""
        source = tmp_path / "LBM_lenta_2020_03.dat"
        source.write_text(
            "TIMESTAMP,RECORD,x\n"
            + "".join(f"2020-03-0{day} 00:00:00,{day},{day}.0\n" for day in range(1, 6)),
            encoding="utf-8",
        )
        (staged,) = stage_archive(
            (archive.ArchiveFile("LBM_lenta_2020_03.dat", staging=archive._CLOCK_PLUS_ONE_HOUR),),
            tmp_path,
            tmp_path / "staged",
        )
        from micrometeorology.sensors.ingestion import read_campbell_dat

        assert len(read_campbell_dat(staged, sentinel_value=None)) == 5

    def test_late_tail_is_dropped(self, tmp_path):
        source = tmp_path / "LBM_lenta_2019.dat"
        write_toa5(
            source,
            ["TIMESTAMP", "x"],
            [
                ("2020-01-07 00:00:00", 1.0),
                ("2020-01-07 01:05:00", 2.0),
                ("2020-01-07 01:10:00", 3.0),
            ],
        )
        (staged,) = stage_archive(
            (archive.ArchiveFile("LBM_lenta_2019.dat", staging=archive._DROP_LATE_TAIL),),
            tmp_path,
            tmp_path / "staged",
        )
        frame = pd.read_csv(staged, skiprows=[0, 2, 3])
        # The correctly clocked 00:00 row stays; the mis-stamped tail goes.
        assert frame["TIMESTAMP"].tolist() == ["2020-01-07 00:00:00"]

    def test_only_the_2023_block_of_the_spare_logger_is_kept(self, tmp_path):
        source = tmp_path / "CR5000_LBM_rain_18-21082023.dat"
        write_toa5(
            source,
            ["TIMESTAMP", "PL01_mm_Tot"],
            [
                ("2014-05-01 00:00:00", 0.0),
                ("2023-08-19 12:00:00", 0.254),
                ("2019-02-01 00:00:00", 0.0),
            ],
        )
        (staged,) = stage_archive(
            (
                archive.ArchiveFile(
                    "CR5000_LBM_rain_18-21082023.dat", staging=archive._KEEP_2023_BLOCK
                ),
            ),
            tmp_path,
            tmp_path / "staged",
        )
        frame = pd.read_csv(staged, skiprows=[0, 2, 3])
        assert frame["TIMESTAMP"].tolist() == ["2023-08-19 12:00:00"]

    def test_source_files_are_never_modified(self, tmp_path):
        source = tmp_path / "LBM_lenta_2019.dat"
        write_toa5(source, ["TIMESTAMP", "x"], [("2020-01-07 01:05:00", 1.0)])
        before = source.read_bytes()
        stage_archive(
            (archive.ArchiveFile("LBM_lenta_2019.dat", staging=archive._DROP_LATE_TAIL),),
            tmp_path,
            tmp_path / "staged",
        )
        assert source.read_bytes() == before


def frame_at(stamps, **columns) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.DatetimeIndex([pd.Timestamp(s) for s in stamps]))


class TestVerifyFrame:
    def _full_span(self, rows):
        index = pd.DatetimeIndex(
            [ARCHIVE_START]
            + [ARCHIVE_START + pd.Timedelta(minutes=5 * i) for i in range(1, rows - 1)]
            + [ARCHIVE_END]
        )
        return pd.DataFrame({"x": range(rows)}, index=index)

    def test_a_short_merge_is_reported(self):
        report = verify_frame(self._full_span(10), "lenta")
        assert not report.ok
        assert any("rows" in problem for problem in report.problems)

    def test_the_span_is_checked_not_just_the_count(self):
        frame = pd.DataFrame({"x": [1, 2]}, index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))
        report = verify_frame(frame, "rain")
        assert any("starts" in problem for problem in report.problems)
        assert any("ends" in problem for problem in report.problems)

    def test_duplicates_and_disorder_are_reported(self):
        frame = pd.DataFrame(
            {"x": [1, 2, 3]},
            index=pd.DatetimeIndex([ARCHIVE_END, ARCHIVE_START, ARCHIVE_START]),
        )
        report = verify_frame(frame, "lenta")
        assert report.duplicated == 1
        assert not report.monotonic
        assert any("duplicated" in problem for problem in report.problems)
        assert any("monotonic" in problem for problem in report.problems)


class TestMaskSentinels:
    def test_impossible_values_are_removed_everywhere(self):
        frame = frame_at(
            ["2019-01-01 00:00", "2019-01-01 00:05"], AirT1_C_Avg=[26.0, -46.8], RH1=[80.0, 999.0]
        )
        masked, removed = mask_sentinels(frame)
        assert masked["AirT1_C_Avg"].isna().tolist() == [False, True]
        assert masked["RH1"].isna().tolist() == [False, True]
        assert removed == {"AirT1_C_Avg": 1, "RH1": 1}

    def test_kelvin_thermistors_are_range_gated(self):
        frame = frame_at(["2017-01-01 00:00", "2017-01-01 00:05"], T_C1_Avg=[300.0, 12.0])
        masked, _removed = mask_sentinels(frame)
        assert masked["T_C1_Avg"].isna().tolist() == [False, True]

    def test_zero_is_masked_only_inside_its_window(self):
        """Zero is a real wind speed: a global rule would delete every calm hour."""
        frame = frame_at(["2021-06-01 12:00", "2023-06-01 12:00"], WS_WXT_Avg=[0.0, 0.0])
        masked, _removed = mask_sentinels(frame)
        assert masked["WS_WXT_Avg"].isna().tolist() == [False, True]

    def test_invalid_windows_remove_every_value_not_just_a_sentinel(self):
        """An unshaded pyranometer reads plausible numbers — they are still wrong."""
        frame = frame_at(["2020-04-15 12:00", "2021-04-15 12:00"], CMP21_Wm2_Avg=[900.0, 120.0])
        masked, _removed = mask_sentinels(frame)
        assert masked["CMP21_Wm2_Avg"].isna().tolist() == [True, False]

    def test_masking_is_per_channel_never_per_row(self):
        """When the thermohygrometer railed, pressure on the same logger stayed good."""
        frame = frame_at(["2026-01-01 00:00"], AirT1_C_Avg=[-46.8], BP1_mbar_Avg=[1011.0])
        masked, _removed = mask_sentinels(frame)
        assert masked["AirT1_C_Avg"].isna().all()
        assert masked["BP1_mbar_Avg"].notna().all()

    def test_absent_columns_are_ignored(self):
        frame = frame_at(["2020-01-01 00:00"], only_this=[1.0])
        masked, removed = mask_sentinels(frame)
        assert removed == {}
        assert masked["only_this"].tolist() == [1.0]


class TestNightCorruptedDays:
    """Timestamp-shifted days, found by irradiance recorded in deep night.

    42 such days are measured in docs/arqueologia/qc/med-fault-detection.md, and
    their consequence was published: the nighttime net-radiation climatology
    carried values up to 1313 W/m2 with the sun 20 deg below the horizon, which
    more than doubled the summer standard deviation the page prints.
    """

    @staticmethod
    def _frame(values: dict[str, list[float]], stamps: list[str]) -> pd.DataFrame:
        return pd.DataFrame(values, index=pd.DatetimeIndex(stamps))

    def test_a_clean_day_is_not_flagged(self) -> None:
        # Local midday, so the sun is up and the flux is ordinary.
        stamps = [f"2024-06-15 12:{minute:02d}" for minute in (0, 5, 10, 15)]
        frame = self._frame({"Sw_dw": [800.0, 810.0, 790.0, 805.0]}, stamps)

        assert archive.night_corrupted_days(frame) == []

    def test_deep_night_irradiance_flags_the_day(self) -> None:
        stamps = [f"2024-06-15 02:{minute:02d}" for minute in (0, 5, 10, 15)]
        frame = self._frame({"Sw_dw": [600.0, 610.0, 590.0, 605.0]}, stamps)

        flagged = archive.night_corrupted_days(frame)

        assert [day for day, _count in flagged] == ["2024-06-15"]
        assert flagged[0][1] == 4

    def test_fewer_samples_than_the_floor_is_not_an_episode(self) -> None:
        """One stray sample is noise; a shifted clock lasts."""
        stamps = ["2024-06-15 02:00", "2024-06-15 02:05", "2024-06-15 12:00"]
        frame = self._frame({"Sw_dw": [600.0, 610.0, 800.0]}, stamps)

        assert archive.night_corrupted_days(frame) == []

    def test_a_day_only_the_par_sensor_witnesses_is_still_found(self) -> None:
        """Keying the detector on Sw_dw alone missed ten real days.

        The shortwave channels do not share an outage, so any one of them can be
        the only surviving witness of a shifted day — 2018-10-22 carries 118
        deep-night PAR samples and no global ones at all.
        """
        stamps = [f"2018-10-22 02:{minute:02d}" for minute in (0, 5, 10)]
        frame = self._frame({"Sw_dw": [float("nan")] * 3, "Sw_par": [300.0, 310.0, 290.0]}, stamps)

        assert [day for day, _count in archive.night_corrupted_days(frame)] == ["2018-10-22"]

    def test_longwave_at_night_is_not_corruption(self) -> None:
        """A pyrgeometer reads 300-400 W/m2 all night by design.

        Including longwave in the detector would flag the entire record.
        """
        stamps = [f"2024-06-15 02:{minute:02d}" for minute in (0, 5, 10, 15)]
        frame = self._frame({"Lw_dw": [380.0] * 4, "Lw_up": [420.0] * 4}, stamps)

        assert archive.night_corrupted_days(frame) == []

    def test_the_mask_takes_the_whole_day_and_the_derived_net_with_it(self) -> None:
        """The clock is what is wrong, so the plausible-looking half of the day is
        exactly as misplaced as the half that is not. ``Net_CNR1`` goes too: the
        logger computes it from the four components, so leaving it would keep the
        corrupted contribution on disk."""
        stamps = [
            "2024-06-15 02:00",
            "2024-06-15 02:05",
            "2024-06-15 02:10",
            "2024-06-15 12:00",  # looks ordinary, same broken clock
            "2024-06-16 12:00",  # the next day is untouched
        ]
        frame = self._frame(
            {
                "Sw_dw": [600.0, 610.0, 590.0, 800.0, 850.0],
                "Sw_par": [300.0, 305.0, 295.0, 400.0, 420.0],
                "Net_CNR1": [500.0, 510.0, 490.0, 700.0, 720.0],
                "T": [21.0, 21.1, 21.2, 28.0, 29.0],
            },
            stamps,
        )

        masked, removed = archive.mask_night_corrupted_days(
            frame.copy(), archive.night_corrupted_days(frame)
        )

        assert masked["Sw_dw"].iloc[:4].isna().all()
        assert masked["Sw_par"].iloc[:4].isna().all()
        assert masked["Net_CNR1"].iloc[:4].isna().all()
        assert masked["Sw_dw"].iloc[4] == 850.0, "the following day must survive"
        # Non-radiation channels keep their values: the reading is real, only its
        # timestamp is wrong, and dropping them would delete good measurements.
        assert masked["T"].notna().all()
        assert removed == {"Sw_dw": 4, "Sw_par": 4, "Net_CNR1": 4}


class TestTheMasksReachTheRawTwin:
    """``unify_sensor_columns`` COPIES, so masking the alias alone cleans half a frame.

    Measured on a full build before this was fixed: 1,049 hourly stamps carried
    ``Sw_dw`` NaN beside ``CM3Up_Wm2_Avg`` still holding the value that triggered
    the mask, up to 1442.19 W/m2 of "global irradiance" at 05:00 local, plus
    1,127 / 1,145 / 883 / 935 more for Sw_up, Sw_par, Net_CNR1 and Sw_dif. The
    published QC artifact disagreed with itself, and the site's static graphs
    read the raw names.
    """

    @staticmethod
    def _switches() -> list[dict[str, Any]]:
        return [
            {
                "unified_name": "Sw_dw",
                "mappings": [
                    {"column": "PSP1_Wm2_Avg", "start_date": None, "end_date": "2024-06-15"},
                    {"column": "CM3Up_Wm2_Avg", "start_date": "2024-06-16", "end_date": None},
                ],
            }
        ]

    @staticmethod
    def _frame() -> pd.DataFrame:
        stamps = [
            "2024-06-15 02:00",
            "2024-06-15 02:05",
            "2024-06-15 02:10",
            "2024-06-16 02:00",
            "2024-06-16 02:05",
            "2024-06-16 02:10",
        ]
        return pd.DataFrame(
            {
                "PSP1_Wm2_Avg": [600.0, 610.0, 590.0, np.nan, np.nan, np.nan],
                "CM3Up_Wm2_Avg": [np.nan, np.nan, np.nan, 700.0, 710.0, 690.0],
            },
            index=pd.DatetimeIndex(stamps),
        )

    def test_the_day_mask_blanks_the_source_the_alias_was_copied_from(self) -> None:
        frame = unify_sensor_columns(self._frame(), self._switches())
        sources = resolve_mapping_windows(frame, self._switches(), ("Sw_dw",))

        masked, removed = archive.mask_night_corrupted_days(
            frame.copy(), archive.night_corrupted_days(frame), sources
        )

        assert masked["Sw_dw"].isna().all()
        assert masked["PSP1_Wm2_Avg"].isna().all(), "the rejected value must not survive by name"
        assert masked["CM3Up_Wm2_Avg"].isna().all()
        assert removed == {"Sw_dw": 6, "PSP1_Wm2_Avg": 3, "CM3Up_Wm2_Avg": 3}

    def test_without_the_source_map_the_raw_twin_still_publishes_the_value(self) -> None:
        """What the fix removes -- pinned so a silent regression is a failing test."""
        frame = unify_sensor_columns(self._frame(), self._switches())

        masked, _removed = archive.mask_night_corrupted_days(
            frame.copy(), archive.night_corrupted_days(frame)
        )

        assert masked["Sw_dw"].isna().all()
        assert masked["PSP1_Wm2_Avg"].notna().sum() == 3

    def test_the_bsrn_mask_stays_inside_the_era_window(self) -> None:
        """Per-sample, so it may only touch the column that IS that measurement.

        Blanking an instrument outside its own mapping window would remove data
        on a check it never failed -- the day mask is the one that is allowed to
        take everything, because a slipped clock is a fault of the logger.
        """
        stamps = pd.DatetimeIndex(["2024-06-15 05:00", "2024-06-16 05:00"])
        frame = pd.DataFrame(
            {
                "PSP1_Wm2_Avg": [1400.0, 1400.0],
                "CM3Up_Wm2_Avg": [1400.0, 1400.0],
                "Sw_dw": [1400.0, 1400.0],
            },
            index=stamps,
        )
        sources = resolve_mapping_windows(frame, self._switches(), ("Sw_dw",))

        masked, removed = archive.mask_impossible_shortwave(frame.copy(), sources)

        assert masked["Sw_dw"].isna().all(), "both samples exceed the BSRN ceiling"
        # Each raw column loses only the sample inside its own era.
        assert masked["PSP1_Wm2_Avg"].to_list() == [
            pytest.approx(float("nan"), nan_ok=True),
            1400.0,
        ]
        assert masked["CM3Up_Wm2_Avg"].to_list() == [
            1400.0,
            pytest.approx(float("nan"), nan_ok=True),
        ]
        assert removed == {"Sw_dw": 2, "PSP1_Wm2_Avg": 1, "CM3Up_Wm2_Avg": 1}


class TestNetRadiationClosesWithItsComponents:
    """The monitoring chart tells the reader to add the four bars and land on Rn.

    The logger derives its net from the same four channels, so calibrating one
    component and not the logger's precomputed sum broke the identity by
    construction: a systematic +1,28 W/m2 residual reaching 9,28, past the
    8,95 W/m2 ceiling the module's own comment asserts for this quantity.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=3, freq="5min")
        return pd.DataFrame(
            {
                "Sw_dw": [800.0, 900.0, np.nan],
                "Sw_up": [100.0, 110.0, 120.0],
                "Lw_dw": [400.0, 405.0, 410.0],
                "Lw_up": [450.0, 455.0, 460.0],
                "Net_CNR1": [1.0, 2.0, 3.0],
            },
            index=index,
        )

    def test_the_net_becomes_the_sum_of_the_published_components(self):
        closed, _gained, _dropped = archive.close_net_radiation(self._frame())

        residual = closed["Sw_dw"] - closed["Sw_up"] + closed["Lw_dw"] - closed["Lw_up"]
        residual = residual - closed["Net_CNR1"]
        assert float(np.nanmax(np.abs(residual))) == 0.0
        assert closed["Net_CNR1"].iloc[0] == pytest.approx(650.0)

    def test_a_missing_component_leaves_no_net(self):
        """A net without all four terms is not a net, whatever the logger wrote."""
        closed, _gained, dropped = archive.close_net_radiation(self._frame())

        assert np.isnan(closed["Net_CNR1"].iloc[2])
        assert dropped == 1

    def test_components_without_a_logger_net_are_published(self):
        frame = self._frame()
        frame["Net_CNR1"] = [np.nan, np.nan, np.nan]

        closed, gained, _dropped = archive.close_net_radiation(frame)

        assert gained == 2
        assert closed["Net_CNR1"].iloc[1] == pytest.approx(740.0)

    def test_a_frame_without_the_components_is_untouched(self):
        frame = pd.DataFrame({"Net_CNR1": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))

        closed, gained, dropped = archive.close_net_radiation(frame)

        assert closed["Net_CNR1"].to_list() == [1.0, 2.0]
        assert (gained, dropped) == (0, 0)
