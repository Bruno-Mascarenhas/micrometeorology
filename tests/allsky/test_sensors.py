"""Tests for allsky.sensors: TOA5 ingestion and target derivation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allsky import sensors, solar
from allsky.config import LabelConfig, SensorConfig, SiteConfig
from allsky.erbs import pseudo_diffuse

# Mirrors the real data/LBM_lenta_2025.dat TOA5 header structure
# (station line / column names / units / aggregation), reduced to the
# columns the default SensorConfig consumes.
_COLUMNS = [
    "CM3Up_Wm2_Avg",
    "CG3Up_Wm2_Avg",
    "CM3Dn_Wm2_Avg",
    "Net_Wm2_Avg",
    "CUV5_Wm2_Avg",
    "PAR_Wm2_Avg",
]


def _write_toa5(path: Path, rows: list[tuple[str, list[float]]]) -> Path:
    """Write a synthetic TOA5 file with the real 4-line header structure."""
    names = ",".join(f'"{c}"' for c in ["TIMESTAMP", "RECORD", *_COLUMNS])
    units = ",".join(f'"{u}"' for u in ["TS", "RN"] + ["W/meter^2"] * len(_COLUMNS))
    aggs = ",".join(f'"{a}"' for a in ["", ""] + ["Avg"] * len(_COLUMNS))
    lines = [
        '"TOA5","CR5000","CR5000","2754","CR5000.Std.06",'
        '"CPU:PRG_LABMIM_UFBA_v22.CR5","49836","LBM_lenta"',
        names,
        units,
        aggs,
    ]
    for i, (ts, values) in enumerate(rows):
        cells = ",".join(f"{v:.1f}" for v in values)
        lines.append(f'"{ts}",{i},{cells}')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _day_row(ghi: float) -> list[float]:
    """Plausible companion sensor values for a given GHI."""
    return [ghi, -30.0, 0.2 * ghi, 0.6 * ghi, 0.05 * ghi, 0.45 * ghi]


@pytest.fixture
def site() -> SiteConfig:
    return SiteConfig()


@pytest.fixture
def sensor_day(tmp_path: Path, site: SiteConfig) -> tuple[Path, pd.DataFrame]:
    """One synthetic day (2025-06-25, Salvador) exercising every code path.

    Rows: night (00:00, 22:00), sun-up-but-low (06:30, elevation ~6.8 deg),
    one row per cloud class (kt 0.80 / 0.50 / 0.20), a sentinel GHI (13:00)
    and a duplicated timestamp.
    """
    day_times = pd.DatetimeIndex(
        ["2025-06-25 09:00:00", "2025-06-25 11:00:00", "2025-06-25 12:00:00"]
    )
    e0h = solar.extraterrestrial_ghi(day_times, site)
    fractions = np.array([0.80, 0.50, 0.20])  # clear, partial, overcast
    ghi_values = fractions * e0h

    rows = [
        ("2025-06-25 00:00:00", _day_row(0.0)),
        ("2025-06-25 06:30:00", _day_row(50.0)),
        ("2025-06-25 09:00:00", _day_row(float(ghi_values[0]))),
        ("2025-06-25 11:00:00", _day_row(float(ghi_values[1]))),
        ("2025-06-25 12:00:00", _day_row(float(ghi_values[2]))),
        ("2025-06-25 13:00:00", [-999.0, *_day_row(600.0)[1:]]),  # sentinel GHI
        ("2025-06-25 13:00:00", _day_row(700.0)),  # duplicate timestamp
        ("2025-06-25 22:00:00", _day_row(0.0)),
    ]
    path = _write_toa5(tmp_path / "LBM_synth_20250625.dat", rows)
    expected = pd.DataFrame({"fraction": fractions, "cloud_class": [0, 1, 2]}, index=day_times)
    return path, expected


def _config_for(path: Path) -> SensorConfig:
    # The synthetic table has no CMP21 column: exercise the Erbs pseudo path.
    return SensorConfig(paths=[str(path)], diffuse_column=None)


# ---------------------------------------------------------------------------
# load_sensor_frame
# ---------------------------------------------------------------------------


class TestLoadSensorFrame:
    def test_columns_index_and_sentinel(self, sensor_day):
        path, _ = sensor_day
        df = sensors.load_sensor_frame(_config_for(path))

        assert list(df.columns) == _COLUMNS  # ghi first, then features (deduped)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing
        assert not df.index.duplicated().any()
        assert len(df) == 7  # 8 rows written, 1 duplicate timestamp dropped
        # Sentinel (-999 <= -900) became NaN.
        assert pd.isna(df.loc["2025-06-25 13:00:00", "CM3Up_Wm2_Avg"])

    def test_diffuse_column_included_when_configured(self, sensor_day):
        path, _ = sensor_day
        cfg = SensorConfig(paths=[str(path)], diffuse_column="CM3Dn_Wm2_Avg")
        df = sensors.load_sensor_frame(cfg)
        assert list(df.columns)[:2] == ["CM3Up_Wm2_Avg", "CM3Dn_Wm2_Avg"]
        assert list(df.columns).count("CM3Dn_Wm2_Avg") == 1  # not duplicated

    def test_missing_column_raises(self, sensor_day):
        path, _ = sensor_day
        cfg = SensorConfig(paths=[str(path)], feature_columns=["DOES_NOT_EXIST"])
        with pytest.raises(KeyError, match="DOES_NOT_EXIST"):
            sensors.load_sensor_frame(cfg)

    def test_multi_file_concat_sort_dedupe(self, tmp_path: Path):
        # File A out of order; file B overlaps A at 12:00 with another value.
        path_a = _write_toa5(
            tmp_path / "a.dat",
            [
                ("2025-06-25 12:10:00", _day_row(500.0)),
                ("2025-06-25 12:00:00", _day_row(400.0)),
            ],
        )
        path_b = _write_toa5(
            tmp_path / "b.dat",
            [
                ("2025-06-25 12:00:00", _day_row(999.0)),
                ("2025-06-25 12:05:00", _day_row(450.0)),
            ],
        )
        df = sensors.load_sensor_frame(
            SensorConfig(paths=[str(path_a), str(path_b)], diffuse_column=None)
        )
        assert len(df) == 3
        assert df.index.is_monotonic_increasing
        # First occurrence (file A) wins on the duplicated timestamp.
        assert df.loc["2025-06-25 12:00:00", "CM3Up_Wm2_Avg"] == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# derive_targets
# ---------------------------------------------------------------------------


class TestDeriveTargets:
    def test_targets_and_night_filtering(self, sensor_day, site: SiteConfig):
        path, expected = sensor_day
        sensor_cfg = _config_for(path)
        label_cfg = LabelConfig()
        df = sensors.load_sensor_frame(sensor_cfg)

        # Preconditions of the fixture geometry.
        elev_low = solar.solar_elevation(pd.DatetimeIndex(["2025-06-25 06:30:00"]), site)
        assert 0.0 < elev_low[0] < label_cfg.min_solar_elevation_deg

        out = sensors.derive_targets(df, site, sensor_cfg, label_cfg)

        # Night (00:00, 22:00), low sun (06:30) and NaN-GHI (13:00) dropped.
        assert list(out.index) == list(expected.index)
        for col in ("kt", "diffuse", "cloud_class", "target_source"):
            assert col in out.columns

        np.testing.assert_allclose(out["kt"], expected["fraction"], atol=0.005)
        assert list(out["cloud_class"]) == list(expected["cloud_class"])
        assert (out["target_source"] == "erbs_pseudo").all()

        # Pseudo-target: 0 <= diffuse <= GHI and equals the Erbs formula.
        ghi = out["CM3Up_Wm2_Avg"].to_numpy()
        assert (out["diffuse"].to_numpy() >= 0.0).all()
        assert (out["diffuse"].to_numpy() <= ghi).all()
        np.testing.assert_allclose(out["diffuse"], pseudo_diffuse(ghi, out["kt"].to_numpy()))

    def test_measured_diffuse_column(self, sensor_day, site: SiteConfig):
        path, expected = sensor_day
        sensor_cfg = SensorConfig(paths=[str(path)], diffuse_column="CM3Dn_Wm2_Avg")
        df = sensors.load_sensor_frame(sensor_cfg)

        out = sensors.derive_targets(df, site, sensor_cfg, LabelConfig())

        assert (out["target_source"] == "measured").all()
        np.testing.assert_allclose(out["diffuse"], out["CM3Dn_Wm2_Avg"])
        assert list(out.index) == list(expected.index)

    def test_elevation_threshold_configurable(self, sensor_day, site: SiteConfig):
        path, _ = sensor_day
        sensor_cfg = _config_for(path)
        df = sensors.load_sensor_frame(sensor_cfg)
        # Lowering the threshold below the 06:30 elevation keeps that row.
        out = sensors.derive_targets(df, site, sensor_cfg, LabelConfig(min_solar_elevation_deg=5.0))
        assert pd.Timestamp("2025-06-25 06:30:00") in out.index

    def test_requires_datetime_index(self, site: SiteConfig):
        df = pd.DataFrame({"CM3Up_Wm2_Avg": [500.0]}, index=[0])
        with pytest.raises(TypeError, match="DatetimeIndex"):
            sensors.derive_targets(df, site, SensorConfig(diffuse_column=None), LabelConfig())


# ---------------------------------------------------------------------------
# classify_cloud_condition
# ---------------------------------------------------------------------------


class TestClassifyCloudCondition:
    def test_bins_boundaries_and_nan(self):
        cfg = LabelConfig()  # kt_clear=0.65, kt_overcast=0.35
        kt = np.array([0.9, 0.65, 0.649, 0.35, 0.349, 0.0, np.nan])
        labels = sensors.classify_cloud_condition(kt, cfg)
        assert labels.tolist() == [
            sensors.CLASS_CLEAR,
            sensors.CLASS_CLEAR,  # kt >= kt_clear is clear (inclusive)
            sensors.CLASS_PARTIAL,
            sensors.CLASS_PARTIAL,  # kt >= kt_overcast is partial (inclusive)
            sensors.CLASS_OVERCAST,
            sensors.CLASS_OVERCAST,
            -1,  # NaN is unlabelable
        ]

    def test_custom_thresholds(self):
        cfg = LabelConfig(kt_clear=0.7, kt_overcast=0.3)
        labels = sensors.classify_cloud_condition(np.array([0.68, 0.32]), cfg)
        assert labels.tolist() == [sensors.CLASS_PARTIAL, sensors.CLASS_PARTIAL]

    def test_class_names_mapping(self):
        assert sensors.CLASS_NAMES[sensors.CLASS_CLEAR] == "clear"
        assert sensors.CLASS_NAMES[sensors.CLASS_PARTIAL] == "partial"
        assert sensors.CLASS_NAMES[sensors.CLASS_OVERCAST] == "overcast"


class TestDeadDiffuseChannelGuard:
    def test_all_zero_measured_diffuse_raises(self, sensor_day, site):
        path, _frame = sensor_day
        df = sensors.load_sensor_frame(SensorConfig(paths=[str(path)], diffuse_column=None))
        df["DEAD_Wm2_Avg"] = 0.0
        cfg = SensorConfig(diffuse_column="DEAD_Wm2_Avg")
        with pytest.raises(ValueError, match="effectively all zeros"):
            sensors.derive_targets(df, site, cfg, LabelConfig())
