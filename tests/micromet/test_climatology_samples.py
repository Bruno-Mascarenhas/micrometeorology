"""Conditioning parity between the observed and the model subsets.

The climatology page exists to let the two histograms be compared, and a
comparison only means something when both are conditional on the same event.
They were not: the model's relative humidity kept its saturation mass while the
observed one removed it, the model's wind direction got no calm removal at all,
and the model's wind-speed atom was published as a hard-coded 0,0 % / 0 h while
98 model hours sat at or below the calm threshold -- inside the bars and inside
the Weibull fit that the page prints as a conditional density.
"""

import numpy as np
import pandas as pd
import pytest

from micrometeorology.cli.export_climatology import (
    CALM_THRESHOLD_MS,
    OBSERVED_COLUMN,
    SATURATION_RH,
    WRF_COLUMN,
    _observed_sample,
    _strip_atoms,
    _wrf_sample,
)


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=len(next(iter(columns.values()))), freq="h")
    return pd.DataFrame(columns, index=index)


class TestTheCalmCutIsTheStallValueNotAThreshold:
    def test_a_sample_exactly_on_the_threshold_is_a_calm(self):
        """`<=`, not `<`: 0,281 m/s is what the stalled cup REPORTS."""
        speed = pd.Series([0.10, CALM_THRESHOLD_MS, CALM_THRESHOLD_MS + 0.001, 3.0])

        values, atoms = _strip_atoms("wind_speed", speed, speed)

        assert atoms[0].count == 2
        assert sorted(values) == pytest.approx([CALM_THRESHOLD_MS + 0.001, 3.0])

    def test_direction_is_cut_by_the_paired_speed_not_by_itself(self):
        direction = pd.Series([10.0, 20.0, 30.0, 40.0])
        speed = pd.Series([0.10, 0.10, 5.0, 5.0])

        values, atoms = _strip_atoms("wind_direction", direction, speed)

        assert atoms[0].count == 2
        assert sorted(values) == [30.0, 40.0]


class TestBothSourcesAreConditionedIdentically:
    @staticmethod
    def _paired(values: list[float], speeds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One frame in observed spelling, one in WRF spelling, same numbers."""
        observed = _frame(
            {
                OBSERVED_COLUMN["relative_humidity"]: values,
                OBSERVED_COLUMN["wind_speed"]: speeds,
                OBSERVED_COLUMN["wind_direction"]: [45.0] * len(values),
            }
        )
        model = _frame(
            {
                WRF_COLUMN["relative_humidity"]: values,
                WRF_COLUMN["wind_speed"]: speeds,
                WRF_COLUMN["wind_direction"]: [45.0] * len(values),
            }
        )
        return observed, model

    def test_saturation_leaves_the_model_sample_too(self):
        values = [50.0, 80.0, SATURATION_RH, SATURATION_RH + 0.4]
        observed, model = self._paired(values, [3.0] * 4)

        obs_values, obs_atoms = _observed_sample("relative_humidity", observed)
        wrf_values, wrf_atoms = _wrf_sample("relative_humidity", model)

        assert [a.id for a in obs_atoms] == [a.id for a in wrf_atoms] == ["saturation"]
        assert obs_atoms[0].count == wrf_atoms[0].count == 2
        assert sorted(obs_values) == sorted(wrf_values) == [50.0, 80.0]

    def test_the_model_calm_atom_is_measured_not_declared_empty(self):
        """It was published as a literal 0,0 % / 0 h whatever the model did."""
        speeds = [0.10, CALM_THRESHOLD_MS, 4.0, 5.0]
        _observed, model = self._paired([70.0] * 4, speeds)

        values, atoms = _wrf_sample("wind_speed", model)

        assert atoms[0].count == 2
        assert atoms[0].fraction == pytest.approx(0.5)
        assert sorted(values) == [4.0, 5.0]

    def test_the_model_direction_gets_a_calm_atom(self):
        """It had none, so its rose kept the hours the observed rose removed."""
        speeds = [0.10, 4.0, 5.0, 6.0]
        _observed, model = self._paired([70.0] * 4, speeds)

        values, atoms = _wrf_sample("wind_direction", model)

        assert [a.id for a in atoms] == ["calm"]
        assert atoms[0].count == 1
        assert len(values) == 3

    def test_both_sources_publish_the_same_atom_ids(self):
        """Structural parity across every variable the model carries."""
        observed, model = self._paired([70.0, 80.0, 90.0, 99.9], [0.1, 2.0, 3.0, 4.0])

        for spec_id in ("relative_humidity", "wind_speed", "wind_direction"):
            _obs, obs_atoms = _observed_sample(spec_id, observed)
            _wrf, wrf_atoms = _wrf_sample(spec_id, model)
            assert [a.id for a in obs_atoms] == [a.id for a in wrf_atoms], spec_id


class TestAVariableWithNoPointMassGetsNoAtom:
    def test_temperature_passes_through_untouched(self):
        series = pd.Series([21.0, 22.0, 23.0])

        values, atoms = _strip_atoms("air_temperature", series, None)

        assert atoms == []
        np.testing.assert_array_equal(values, series.to_numpy())
