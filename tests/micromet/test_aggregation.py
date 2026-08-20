"""Tests for temporal aggregation."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import yaml

from micrometeorology.common.config import Settings, get_settings
from micrometeorology.sensors.aggregation import aggregate_to_hourly


@pytest.fixture
def sample_5min_data() -> pd.DataFrame:
    """Create synthetic 5-minute data spanning 2 hours."""
    idx = pd.date_range("2024-01-01 00:00", "2024-01-01 01:55", freq="5min")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "Temp": 25.0 + rng.normal(0, 1, len(idx)),
            "precip": rng.exponential(0.1, len(idx)),
            "WD_WXT": rng.uniform(0, 360, len(idx)),
            "WS_WXT": rng.uniform(0, 5, len(idx)),
        },
        index=idx,
    )


class TestAggregateToHourly:
    def test_output_length(self, sample_5min_data):
        result = aggregate_to_hourly(sample_5min_data)
        assert len(result) == 2

    def test_sum_columns(self, sample_5min_data):
        result = aggregate_to_hourly(
            sample_5min_data,
            sum_columns=["precip"],
        )
        assert "precip" in result.columns
        raw_sum = sample_5min_data["precip"][:12].sum()
        assert result["precip"].iloc[0] == pytest.approx(raw_sum, abs=0.001)

    def test_min_samples_filter(self):
        """If fewer than min_samples valid values exist, result should be NaN."""
        idx = pd.date_range("2024-01-01 00:00", "2024-01-01 00:55", freq="5min")
        data = pd.DataFrame(
            {
                "Temp": [
                    25.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
            },
            index=idx,
        )
        result = aggregate_to_hourly(data, min_samples=6)
        assert np.isnan(result["Temp"].iloc[0])

    def test_wind_direction_vector_mean(self, sample_5min_data):
        result = aggregate_to_hourly(
            sample_5min_data,
            wind_dir_columns=["WD_WXT"],
        )
        assert "WD_WXT" in result.columns
        for val in result["WD_WXT"].dropna():
            assert 0 <= val < 360

        # The vector mean is taken per hour, not once over the whole frame.
        dir1 = result["WD_WXT"].iloc[0]
        dir2 = result["WD_WXT"].iloc[1]
        assert abs(dir1 - dir2) > 1.0, f"Hours 1 and 2 identical: {dir1} == {dir2}"

        arithmetic = sample_5min_data["WD_WXT"][:12].mean()
        assert dir1 != pytest.approx(arithmetic, abs=1.0)

    def test_wind_direction_wraps_around_north(self):
        """350° and 10° in the same hour average to 0°, not to the arithmetic 180°."""
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [350.0, 10.0] * 6, "WS_WXT": [3.0] * 12}, index=idx)
        result = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "WS_WXT"},
        )
        d = float(result["WD_WXT"].iloc[0])
        assert d < 1.0 or d > 359.0, d

    def test_speed_weighted_mean_differs_from_unit_speed(self):
        """The mapped speed column weights the components; without it speeds are unit."""
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [90.0, 180.0] * 6, "WS_WXT": [10.0, 1.0] * 6}, index=idx)
        weighted = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "WS_WXT"},
        )
        unit = aggregate_to_hourly(df, min_samples=6, wind_dir_columns=["WD_WXT"])
        # The 10 m/s easterly dominates the 1 m/s southerly
        assert weighted["WD_WXT"].iloc[0] == pytest.approx(95.71, abs=0.5)
        # Unit speeds give the plain bisector of 090 and 180
        assert unit["WD_WXT"].iloc[0] == pytest.approx(135.0, abs=0.5)

    def test_unit_speed_fallback_when_mapped_speed_column_is_absent(self):
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame({"WD_WXT": [90.0, 180.0] * 6, "WS_WXT": [10.0, 1.0] * 6}, index=idx)
        result = aggregate_to_hourly(
            df,
            min_samples=6,
            wind_dir_columns=["WD_WXT"],
            wind_speed_column_map={"WD_WXT": "MISSING"},
        )
        assert result["WD_WXT"].iloc[0] == pytest.approx(135.0, abs=0.5)


@pytest.fixture
def settings_without_ambient_overrides(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    """``get_settings()`` free of ambient ``LABMIM_*`` and of the process-wide cache."""
    monkeypatch.delenv("LABMIM_ENV", raising=False)
    monkeypatch.delenv("LABMIM_CONFIG_PATH", raising=False)
    for field_name in Settings.model_fields:
        monkeypatch.delenv(f"LABMIM_{field_name.upper()}", raising=False)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


class TestShippedConfigDrivesSpecialAggregation:
    """The sum / wind-direction rules only apply if the shipped YAML is findable.

    ``configs_dir`` is the directory the sensor pipeline resolves its
    configuration from, and every reader of it is guarded by an
    ``if ...exists()``. When it names a directory without ``default.yaml`` those
    guards fall through silently and both rule lists come back empty, so every
    column is arithmetically meaned: an hour of rain tips is reported as its
    per-sample average, and a bearing straddling north as the opposite bearing.
    """

    def _rules_from_the_resolved_configs_dir(
        self, settings: Settings
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Resolve the sum / wind-direction / speed-pairing rules through ``configs_dir``.

        Missing file means "no special aggregation", which is exactly how the
        pipeline treats it, so a wrong ``configs_dir`` surfaces here as wrong
        physics rather than as an ``OSError``.
        """
        config_file = settings.configs_dir / "default.yaml"
        if not config_file.exists():
            return [], [], {}
        with open(config_file, encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
        return (
            config.get("sensor_sum_columns", []),
            config.get("sensor_wind_dir_columns", []),
            config.get("sensor_wind_speed_column_map", {}),
        )

    def test_hourly_rain_is_the_total_and_wind_direction_is_the_vector_mean(
        self, settings_without_ambient_overrides: Settings
    ) -> None:
        """Twelve 0.5 mm tips are 6 mm of rain, and 350/010 averages to north."""
        sum_columns, wind_dir_columns, speed_map = self._rules_from_the_resolved_configs_dir(
            settings_without_ambient_overrides
        )
        idx = pd.date_range("2024-01-01 00:00", periods=12, freq="5min")
        df = pd.DataFrame(
            {
                "precip": [0.5] * 12,
                "WD_WXT_Avg": [350.0, 10.0] * 6,
                "WS_WXT_Avg": [3.0] * 12,
            },
            index=idx,
        )

        result = aggregate_to_hourly(
            df,
            min_samples=6,
            sum_columns=sum_columns,
            wind_dir_columns=wind_dir_columns,
            wind_speed_column_map=speed_map,
        )

        assert result["precip"].iloc[0] == pytest.approx(6.0), "rain was meaned, not summed"
        direction = float(result["WD_WXT_Avg"].iloc[0])
        assert direction < 1.0 or direction > 359.0, f"scalar-meaned bearing {direction}"

    def test_the_resolved_configs_dir_agrees_with_the_settings_fields(
        self, settings_without_ambient_overrides: Settings
    ) -> None:
        """Both routes to the rules must name the same columns, forever."""
        settings = settings_without_ambient_overrides
        sum_columns, wind_dir_columns, speed_map = self._rules_from_the_resolved_configs_dir(
            settings
        )

        assert sum_columns == settings.sensor_sum_columns
        assert wind_dir_columns == settings.sensor_wind_dir_columns
        assert speed_map == settings.sensor_wind_speed_column_map

    def test_every_shipped_direction_column_declares_the_speed_that_weights_it(
        self, settings_without_ambient_overrides: Settings
    ) -> None:
        """A direction with no pairing is silently vector-averaged with unit
        weight, so a calm sample counts as much as a squall in the published
        bearing."""
        settings = settings_without_ambient_overrides
        unpaired = set(settings.sensor_wind_dir_columns) - set(
            settings.sensor_wind_speed_column_map
        )

        assert not unpaired, f"direction columns with no speed pairing: {sorted(unpaired)}"


def test_a_complete_direction_hour_survives_a_dead_anemometer() -> None:
    """Completeness is judged on the direction channel, not on the u component.

    u is NaN whenever the speed is NaN, so gating the vector mean on it threw
    away hours whose directional record was complete. Measured on the archive,
    2,619 such hours were published as NaN, 1,701 of them in 2018.
    """
    stamps = pd.date_range("2024-06-15 00:00", periods=12, freq="5min")
    frame = pd.DataFrame(
        {"WindDir": [90.0] * 12, "WS_ms": [float("nan")] * 12},
        index=stamps,
    )

    hourly = aggregate_to_hourly(
        frame,
        min_samples=6,
        sum_columns=[],
        wind_dir_columns=["WindDir"],
        wind_speed_column_map={"WindDir": "WS_ms"},
    )

    assert hourly["WindDir"].iloc[0] == pytest.approx(90.0)


def test_a_sparse_direction_hour_is_still_rejected() -> None:
    """The gate still fires — on the quantity it is about."""
    stamps = pd.date_range("2024-06-15 00:00", periods=12, freq="5min")
    directions = [90.0] * 3 + [float("nan")] * 9
    frame = pd.DataFrame({"WindDir": directions, "WS_ms": [2.0] * 12}, index=stamps)

    hourly = aggregate_to_hourly(
        frame,
        min_samples=6,
        sum_columns=[],
        wind_dir_columns=["WindDir"],
        wind_speed_column_map={"WindDir": "WS_ms"},
    )

    assert np.isnan(hourly["WindDir"].iloc[0])


def test_speed_weighting_still_applies_when_the_anemometer_is_present() -> None:
    """Falling back to unit weight must not cost the weighting where it exists.

    Six samples due north at 10 m/s and six due east at 1 m/s: the speed-weighted
    mean leans north, the unweighted one would sit near the bisector at 45 deg.
    """
    stamps = pd.date_range("2024-06-15 00:00", periods=12, freq="5min")
    frame = pd.DataFrame(
        {"WindDir": [0.0] * 6 + [90.0] * 6, "WS_ms": [10.0] * 6 + [1.0] * 6},
        index=stamps,
    )

    hourly = aggregate_to_hourly(
        frame,
        min_samples=6,
        sum_columns=[],
        wind_dir_columns=["WindDir"],
        wind_speed_column_map={"WindDir": "WS_ms"},
    )

    assert hourly["WindDir"].iloc[0] == pytest.approx(5.71, abs=0.1)
