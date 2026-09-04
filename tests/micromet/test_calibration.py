"""Tests for calibration application."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from micrometeorology.sensors import calibration
from micrometeorology.sensors.calibration import (
    CalibrationRecord,
    DatedColumnRecord,
    SensorSwitch,
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


CALIBRATIONS_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "micromet" / "calibrations.yaml"
)


class TestApplyCalibrations:
    def test_multiplicative_factor(self, sample_data):
        cals = [
            CalibrationRecord(
                column="CM3Up_Wm2_Avg",
                start_date="2018-06-01",
                end_date="2018-12-31",
                factor=0.5,
                description="test",
            )
        ]
        original = sample_data["CM3Up_Wm2_Avg"].copy()
        apply_calibrations(sample_data, cals)

        # A date-only end_date is inclusive of the WHOLE boundary day: every
        # sub-daily sample of 2018-12-31 is calibrated, and 2019-01-01 00:00
        # onward is untouched.
        mask_before = sample_data.index <= pd.Timestamp("2018-12-31 23:59:59")
        mask_after = sample_data.index >= pd.Timestamp("2019-01-01")

        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_before, "CM3Up_Wm2_Avg"],
            original[mask_before] * 0.5,
        )
        np.testing.assert_array_almost_equal(
            sample_data.loc[mask_after, "CM3Up_Wm2_Avg"],
            original[mask_after],
        )


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
                CalibrationRecord(
                    column="CM3Up_Wm2_Avg",
                    start_date="2018-12-30",
                    end_date="2018-12-31",
                    factor=0.5,
                    description="boundary",
                )
            ],
        )
        end_day = five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"]
        assert len(end_day) == 288  # all 5-min samples of the day
        assert (end_day == 50.0).all()
        assert (five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"] == 100.0).all()

    def test_null_factor_nans_whole_end_day(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                CalibrationRecord(
                    column="CM3Up_Wm2_Avg",
                    start_date=None,
                    end_date="2018-12-31",
                    factor=None,
                    description="invalid",
                )
            ],
        )
        assert five_min_data.loc["2018-12-31", "CM3Up_Wm2_Avg"].isna().all()
        assert five_min_data.loc["2019-01-01", "CM3Up_Wm2_Avg"].notna().all()

    def test_explicit_time_end_date_is_honored_exactly(self, five_min_data):
        apply_calibrations(
            five_min_data,
            [
                CalibrationRecord(
                    column="CM3Up_Wm2_Avg",
                    start_date="2018-12-31 00:00",
                    end_date="2018-12-31 12:00",
                    factor=0.5,
                    description="explicit",
                )
            ],
        )
        assert five_min_data.loc["2018-12-31 12:00", "CM3Up_Wm2_Avg"] == 50.0
        assert five_min_data.loc["2018-12-31 12:05", "CM3Up_Wm2_Avg"] == 100.0
        assert (five_min_data.loc["2018-12-31 00:00":"2018-12-31 12:00"] == 50.0).all().all()

    def test_unify_has_no_boundary_day_hole(self, five_min_data):
        five_min_data["B"] = 20.0
        five_min_data = five_min_data.rename(columns={"CM3Up_Wm2_Avg": "A"})
        unify_sensor_columns(
            five_min_data,
            [
                SensorSwitch(
                    unified_name="U",
                    mappings=(
                        DatedColumnRecord(
                            column="A", start_date="2018-12-30", end_date="2018-12-31"
                        ),
                        DatedColumnRecord(
                            column="B", start_date="2019-01-01", end_date="2019-01-02"
                        ),
                    ),
                )
            ],
        )
        assert five_min_data.loc["2018-12-31", "U"].notna().all()
        assert (five_min_data.loc["2018-12-31", "U"] == 100.0).all()
        assert (five_min_data.loc["2019-01-01", "U"] == 20.0).all()


class TestOverlapGuard:
    """Inclusive end dates make same-day-abutting records a config error."""

    def test_same_day_abutment_raises_clear_error(self, sample_data):
        cals = [
            CalibrationRecord(
                column="CM3Up_Wm2_Avg",
                start_date="2018-06-01",
                end_date="2018-12-31",
                factor=0.5,
                description="first",
            ),
            CalibrationRecord(
                column="CM3Up_Wm2_Avg",
                start_date="2018-12-31",
                end_date="2019-06-01",
                factor=0.9,
                description="second",
            ),
        ]
        with pytest.raises(ValueError, match=r"Overlapping calibrations.*2018-12-31"):
            apply_calibrations(sample_data, cals)

    def test_next_day_abutment_is_clean(self, sample_data):
        cals = [
            CalibrationRecord(
                column="CM3Up_Wm2_Avg",
                start_date="2018-06-01",
                end_date="2018-12-31",
                factor=0.5,
                description="first",
            ),
            CalibrationRecord(
                column="CM3Up_Wm2_Avg",
                start_date="2019-01-01",
                end_date="2019-06-01",
                factor=0.9,
                description="second",
            ),
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
            SensorSwitch(
                unified_name="unified",
                mappings=(
                    DatedColumnRecord(
                        column="sensor_A", start_date="2018-06-01", end_date="2018-12-31"
                    ),
                    DatedColumnRecord(
                        column="sensor_B", start_date="2018-12-31", end_date="2019-06-01"
                    ),
                ),
            )
        ]
        with pytest.raises(ValueError, match="Overlapping sensor-switch mappings"):
            unify_sensor_columns(df, switches)


class TestARecordThatClosesBeforeTheDataBegins:
    """An open-ended record inherits the DATASET's first timestamp when APPLIED.

    A record that closes before the data starts therefore masks nothing, which
    is the ordinary case for a recent logger table — the shipped CMP21 record
    ends 2019-10-12 and ``data/LBM_lenta_2025.dat`` begins 2025-05-14. The
    overlap guard reads the DECLARED range instead, so how much data happens to
    be loaded cannot decide whether ``calibrations.yaml`` is validated.
    """

    @staticmethod
    def _frame() -> pd.DataFrame:
        index = pd.date_range("2025-05-14", periods=48, freq="h")
        return pd.DataFrame({"CMP21_Wm2_Avg": np.full(48, 100.0)}, index=index)

    def test_the_empty_record_does_not_raise_a_spurious_overlap(self) -> None:
        calibrations: list[CalibrationRecord] = [
            # Closes six years before this frame starts: empty here.
            CalibrationRecord(
                column="CMP21_Wm2_Avg",
                end_date="2019-10-12",
                factor=None,
                description="not installed yet",
            ),
            # The one that actually applies.
            CalibrationRecord(
                column="CMP21_Wm2_Avg",
                start_date="2019-10-13",
                factor=0.985,
                description="sensitivity correction",
            ),
        ]

        result = apply_calibrations(self._frame().copy(), calibrations)

        assert result["CMP21_Wm2_Avg"].notna().all()
        assert result["CMP21_Wm2_Avg"].iloc[0] == pytest.approx(98.5)

    def test_a_genuine_overlap_is_still_refused(self) -> None:
        """The guard must not be weakened: two records that really do cover the
        same day are still a configuration error."""
        # Both resolve to real intervals inside the 48-hour frame, sharing 2025-05-14.
        calibrations: list[CalibrationRecord] = [
            CalibrationRecord(
                column="CMP21_Wm2_Avg",
                start_date="2025-05-14",
                end_date="2025-05-14",
                factor=1.0,
                description="a",
            ),
            CalibrationRecord(
                column="CMP21_Wm2_Avg",
                start_date="2025-05-14",
                factor=2.0,
                description="b",
            ),
        ]

        with pytest.raises(ValueError, match="Overlapping"):
            apply_calibrations(self._frame().copy(), calibrations)

    def test_a_declared_overlap_is_refused_against_a_frame_narrower_than_it(self) -> None:
        """The guard resolved open ends against the frame, so on the rolling
        ``--source`` window the earlier record inverted, was dropped as empty,
        and the later record's factor applied uncontested — the exact
        record-order dependence the ValueError exists to prevent."""
        one_row = pd.DataFrame(
            {"CMP21_Wm2_Avg": [100.0]}, index=pd.DatetimeIndex(["2026-06-01 12:00"])
        )
        calibrations: list[CalibrationRecord] = [
            CalibrationRecord(
                column="CMP21_Wm2_Avg", end_date="2020-12-31", factor=1.0, description="a"
            ),
            CalibrationRecord(
                column="CMP21_Wm2_Avg", start_date="2020-01-01", factor=2.0, description="b"
            ),
        ]

        with pytest.raises(ValueError, match="Overlapping"):
            apply_calibrations(one_row.copy(), calibrations)

    def test_the_shipped_calibrations_load_against_a_recent_file(self) -> None:
        """The shipped config must apply end to end against a 2025 frame.

        Resolved from this file, not from ``configs_dir``: reading through the
        settings meant an ambient ``LABMIM_CONFIGS_DIR`` pointing anywhere else
        made the test SKIP, which is the exact misconfiguration it exists to
        catch. And the assertion is on the factor the record declares, not on a
        row count the calibration cannot change.
        """
        from micrometeorology.sensors.calibration import load_calibrations

        result = apply_calibrations(self._frame().copy(), load_calibrations(CALIBRATIONS_YAML))

        # 9.38 / 9.52, the CMP21 sensitivity revision the shipped record names.
        assert result["CMP21_Wm2_Avg"].iloc[0] == pytest.approx(100.0 * 9.38 / 9.52, abs=1e-4)


class TestACalibrationSurvivesAColumnRename:
    """A record keyed on a column name dies when the logger renames the channel.

    The logger renamed the Eppley PSP channel from the pre-v11 ``PSP1_Wm2_Avg``
    to ``PSP_Wm2_Avg`` on 2019-03-15, a rename the sensor_switches block of the
    same config documents. The PSP is the diffuse sensor of the current
    operational era, so a post-2019 sensitivity correction naming only the old
    spelling stops at 2019-02-26 and leaves the published diffuse 8.5% low while
    still reading as declared.
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
            record.column: record.factor
            for record in records
            if record.column in {"PSP1_Wm2_Avg", "PSP_Wm2_Avg"}
            and (record.start_date or "") >= "2019"
        }

        assert "PSP_Wm2_Avg" in factors, "the renamed channel carries no calibration"
        assert factors["PSP_Wm2_Avg"] == factors["PSP1_Wm2_Avg"], (
            "same instrument, same programmed sensitivity: the correction must match"
        )

    def test_a_record_that_matches_nothing_is_reported(self, caplog) -> None:
        """'Declared' and 'applied' must be distinguishable in the output."""
        index = pd.date_range("2025-01-01", periods=4, freq="h")
        frame = pd.DataFrame({"PSP_Wm2_Avg": [1.0, 2.0, 3.0, 4.0]}, index=index)
        records: list[CalibrationRecord] = [
            CalibrationRecord(
                column="PSP_Wm2_Avg",
                start_date="2019-01-01",
                end_date="2019-12-31",
                factor=2.0,
                description="window with no data here",
            )
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
    def _switches() -> list[SensorSwitch]:
        return [
            SensorSwitch(
                unified_name="Sw_dw",
                mappings=(
                    DatedColumnRecord(column="PSP1_Wm2_Avg", start_date=None, end_date=None),
                ),
            ),
            SensorSwitch(
                unified_name="T",
                mappings=(DatedColumnRecord(column="Temp1_Avg", start_date=None, end_date=None),),
            ),
        ]

    def test_the_half_covered_window_is_reported(self):
        calibrations = [
            CalibrationRecord(
                column="PSP1_Wm2_Avg", start_date="2018-01-01", end_date=None, factor=0.94
            )
        ]

        gaps = uncalibrated_mapping_windows(self._frame(), calibrations, self._switches())

        assert len(gaps) == 1
        unified, column, start, end = gaps[0]
        assert (unified, column) == ("Sw_dw", "PSP1_Wm2_Avg")
        assert start == pd.Timestamp("2016-01-01")
        assert end.date() == pd.Timestamp("2017-12-31").date()

    def test_a_column_with_no_record_at_all_is_not_a_gap(self):
        """The logger's own multiplier is the calibration for most channels.

        Reporting those too would bury the one real finding under 37
        non-findings.
        """
        calibrations = [
            CalibrationRecord(column="PSP1_Wm2_Avg", start_date=None, end_date=None, factor=0.94)
        ]

        assert uncalibrated_mapping_windows(self._frame(), calibrations, self._switches()) == []

    def test_a_hole_between_two_records_is_reported(self):
        calibrations = [
            CalibrationRecord(
                column="PSP1_Wm2_Avg",
                start_date=None,
                end_date="2016-12-31",
                factor=0.94,
            ),
            CalibrationRecord(
                column="PSP1_Wm2_Avg",
                start_date="2018-01-01",
                end_date=None,
                factor=1.09,
            ),
        ]

        gaps = uncalibrated_mapping_windows(self._frame(), calibrations, self._switches())

        assert len(gaps) == 1
        _unified, _column, start, end = gaps[0]
        assert start.date() == pd.Timestamp("2017-01-01").date()
        assert end.date() == pd.Timestamp("2017-12-31").date()

    def test_a_null_factor_record_counts_as_covered(self):
        """`factor: null` is a decision about the window, not an absence of one."""
        calibrations: list[CalibrationRecord] = [
            CalibrationRecord(
                column="PSP1_Wm2_Avg",
                start_date=None,
                end_date="2017-12-31",
                factor=None,
            ),
            CalibrationRecord(
                column="PSP1_Wm2_Avg",
                start_date="2018-01-01",
                end_date=None,
                factor=0.94,
            ),
        ]

        assert uncalibrated_mapping_windows(self._frame(), calibrations, self._switches()) == []


class TestShadeRingCorrection:
    """O anel de sombreamento oculta parte do domo, e a difusa medida sob ele subestima.

    Medido neste acervo antes da correção: a fração difusa satura em 0,81 sob céu
    encoberto (Kt < 0,10), quando a física exige que tenda a 1. Com o fator
    geométrico aplicado ela vai a 0,97.
    """

    @staticmethod
    def _fatores(stamps: pd.DatetimeIndex, valor: float = 1.2) -> pd.Series:
        return pd.Series([valor] * len(stamps), index=stamps)

    def test_a_difusa_dentro_da_janela_e_escalada(self) -> None:
        stamps = pd.DatetimeIndex(["2021-06-15 12:00"])
        frame = pd.DataFrame({"CMP21_Wm2_Avg": [100.0]}, index=stamps)
        janelas = [("CMP21_Wm2_Avg", pd.Timestamp("2020-06-01"), pd.Timestamp("2025-03-12"))]

        corrigido, contagem = calibration.apply_shade_ring_correction(
            frame, self._fatores(stamps), janelas
        )

        assert corrigido["CMP21_Wm2_Avg"].iloc[0] == pytest.approx(120.0)
        assert contagem == {"CMP21_Wm2_Avg": 1}

    def test_a_mesma_coluna_fora_da_janela_fica_intocada(self) -> None:
        """Fora da janela o instrumento mede o GLOBAL, e escalá-lo corromperia Sw_dw."""
        stamps = pd.DatetimeIndex(["2017-06-15 12:00"])
        frame = pd.DataFrame({"PSP_Wm2_Avg": [800.0]}, index=stamps)
        janelas = [("PSP_Wm2_Avg", pd.Timestamp("2025-05-14"), pd.Timestamp("2026-08-15"))]

        corrigido, contagem = calibration.apply_shade_ring_correction(
            frame, self._fatores(stamps), janelas
        )

        assert corrigido["PSP_Wm2_Avg"].iloc[0] == pytest.approx(800.0)
        assert contagem == {}

    def test_fator_ausente_levanta_em_vez_de_apagar_a_medida(self) -> None:
        """Um fator ausente que virasse NaN apagaria a difusa em silêncio."""
        stamps = pd.DatetimeIndex(["2021-06-15 12:00"])
        frame = pd.DataFrame({"CMP21_Wm2_Avg": [100.0]}, index=stamps)
        janelas = [("CMP21_Wm2_Avg", pd.Timestamp("2020-06-01"), pd.Timestamp("2025-03-12"))]
        vazios = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))

        with pytest.raises(calibration.MissingShadeRingFactorError):
            calibration.apply_shade_ring_correction(frame, vazios, janelas)

    def test_uma_hora_sem_difusa_nao_exige_fator(self) -> None:
        stamps = pd.DatetimeIndex(["2021-06-15 12:00"])
        frame = pd.DataFrame({"CMP21_Wm2_Avg": [float("nan")]}, index=stamps)
        janelas = [("CMP21_Wm2_Avg", pd.Timestamp("2020-06-01"), pd.Timestamp("2025-03-12"))]
        vazios = pd.Series(dtype="float64", index=pd.DatetimeIndex([]))

        _corrigido, contagem = calibration.apply_shade_ring_correction(frame, vazios, janelas)

        assert contagem == {}


class TestSolarGeometryParquet:
    """The lab's 203 MB CSV, rewritten in the shape the rest of the archive uses.

    Measured on the real table: 29 MB against 203, and 0.037 s against 1.42 s to
    answer for one column — which the hourly window job pays every run.
    """

    @staticmethod
    def _csv(path: Path) -> Path:
        path.write_text(
            "lon,lat,ano_i,mes_i,dia_i,hor_i,min_i,oc_topo,fc\n"
            "-38.5,-13.0,2026,8,15,12,0,1200.5,1.1802\n"
            "-38.5,-13.0,2026,8,15,12,5,1201.0,1.1803\n",
            encoding="utf-8",
        )
        return path

    def test_the_derived_table_carries_the_factor_unchanged(self, tmp_path) -> None:
        """``fc`` multiplies the measured diffuse, so single precision is not an
        option: its 5.4e-08 relative error would move every corrected value."""
        origem = self._csv(tmp_path / calibration.SHADE_RING_FACTOR_FILE)
        destino = tmp_path / calibration.SHADE_RING_FACTOR_PARQUET

        calibration.solar_geometry_to_parquet(origem, destino)
        derivado = pd.read_parquet(destino)

        assert derivado["fc"].tolist() == [1.1802, 1.1803]
        assert derivado.index.tolist() == [
            pd.Timestamp("2026-08-15 12:00"),
            pd.Timestamp("2026-08-15 12:05"),
        ]

    def test_the_loader_prefers_the_derived_table(self, tmp_path) -> None:
        origem = self._csv(tmp_path / calibration.SHADE_RING_FACTOR_FILE)
        calibration.solar_geometry_to_parquet(
            origem, tmp_path / calibration.SHADE_RING_FACTOR_PARQUET
        )
        origem.unlink()

        fatores = calibration.load_shade_ring_factors(origem)

        assert fatores.tolist() == [1.1802, 1.1803]

    def test_the_csv_remains_the_fallback(self, tmp_path) -> None:
        origem = self._csv(tmp_path / calibration.SHADE_RING_FACTOR_FILE)

        fatores = calibration.load_shade_ring_factors(origem)

        assert fatores.tolist() == [1.1802, 1.1803]


def test_a_record_without_a_factor_is_refused_instead_of_blanking_the_window() -> None:
    """``factor`` defaulted to ``None``, and ``None`` is the value that declares a
    window invalid and blanks it: an appended record that merely forgot the key
    erased its whole window with exit code 0."""
    with pytest.raises(ValidationError, match="factor"):
        CalibrationRecord.model_validate(
            {
                "column": "CM3Up_Wm2_Avg",
                "start_date": "2018-06-01",
                "end_date": "2018-12-31",
                "description": "sem factor",
            }
        )
