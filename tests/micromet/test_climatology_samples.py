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
    INVALID_DIRECTION,
    OBSERVED_COLUMN,
    SATURATION_RH,
    WRF_COLUMN,
    _available_hours,
    _observed_sample,
    _strip_atoms,
    _wrf_sample,
)


def _frame(columns: dict[str, list[float]]) -> pd.DataFrame:
    index = pd.date_range("2024-03-01", periods=len(next(iter(columns.values()))), freq="h")
    return pd.DataFrame(columns, index=index)


def _hours(values: list[float]) -> pd.Series:
    """A sample on consecutive hours, the shape every published sample has."""
    return pd.Series(values, index=pd.date_range("2024-03-01", periods=len(values), freq="h"))


class TestTheCalmCutIsTheStallValueNotAThreshold:
    def test_a_sample_exactly_on_the_threshold_is_a_calm(self):
        """`<=`, not `<`: 0,281 m/s is what the stalled cup REPORTS."""
        speed = _hours([0.10, CALM_THRESHOLD_MS, CALM_THRESHOLD_MS + 0.001, 3.0])

        values, atoms = _strip_atoms("wind_speed", speed, speed)

        assert atoms[0].count == 2
        assert sorted(values) == pytest.approx([CALM_THRESHOLD_MS + 0.001, 3.0])

    def test_direction_is_cut_by_the_paired_speed_not_by_itself(self):
        direction = _hours([10.0, 20.0, 30.0, 40.0])
        speed = _hours([0.10, 0.10, 5.0, 5.0])

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

    def test_both_sources_publish_the_same_value_based_atoms(self):
        """Parity is over the masses a VALUE creates, not over instrument history.

        ``arithmetic_mean_era`` is deliberately one-sided: it removes the window
        in which the logger averaged an angle arithmetically, which is a fault of
        this station's recording and not of anything the model did. Requiring it
        on both sides would delete good model hours to mirror a broken sensor.
        """
        observed, model = self._paired([70.0, 80.0, 90.0, 99.9], [0.1, 2.0, 3.0, 4.0])
        instrument_only = {"arithmetic_mean_era"}

        for spec_id in ("relative_humidity", "wind_speed", "wind_direction"):
            _obs, obs_atoms = _observed_sample(spec_id, observed)
            _wrf, wrf_atoms = _wrf_sample(spec_id, model)
            assert [a.id for a in obs_atoms if a.id not in instrument_only] == [
                a.id for a in wrf_atoms if a.id not in instrument_only
            ], spec_id


class TestAVariableWithNoPointMassGetsNoAtom:
    def test_temperature_passes_through_untouched(self):
        series = _hours([21.0, 22.0, 23.0])

        values, atoms = _strip_atoms("air_temperature", series, None)

        assert atoms == []
        np.testing.assert_array_equal(values, series.to_numpy())


class TestCoverageMeansSensorAvailability:
    """The panel is labelled "horas válidas" and read as uptime.

    Counting after the point masses left made the rain gauge report 6.980 valid
    hours where 80.518 exist -- 91,3% of them dry, which is weather, not an
    outage. A reader comparing that strip with the temperature strip (75.622)
    would conclude the gauge was out of service for 92% of the record.
    """

    def test_dry_hours_are_available_hours(self):
        frame = _frame({OBSERVED_COLUMN["precipitation"]: [0.0, 0.0, 0.0, 0.4, 1.2]})

        sample, atoms = _observed_sample("precipitation", frame)

        assert len(sample) == 2, "only the wet hours are fitted"
        assert atoms[0].count == 3
        assert _available_hours("precipitation", frame) == 5

    def test_the_published_identity_is_n_plus_the_atoms(self):
        """coverage == n + sum(atom counts), checkable from the bytes alone."""
        frame = _frame(
            {
                OBSERVED_COLUMN["wind_speed"]: [0.10, CALM_THRESHOLD_MS, 4.0, 5.0, 6.0],
                OBSERVED_COLUMN["relative_humidity"]: [50.0, 99.9, 70.0, 80.0, 90.0],
            }
        )

        for spec_id in ("wind_speed", "relative_humidity"):
            sample, atoms = _observed_sample(spec_id, frame)
            # The break markers are not hours: every published count is over the
            # finite entries, which is what `histogram` and `fit_distribution` see.
            measured = int(np.isfinite(sample).sum())
            assert _available_hours(spec_id, frame) == measured + sum(a.count for a in atoms)

    def test_the_daylight_gate_marks_where_it_breaks_the_hour_chain(self):
        """``effective_sample_size`` correlates consecutive ENTRIES, and the
        daylight gate splices one day's last daylight hour onto the next day's
        first with nothing marking the jump — so the lag-1 autocorrelation was
        taken between temporally unrelated neighbours and ``n_effective`` came
        out roughly twice too high."""
        from micrometeorology.stats.distributions import effective_sample_size

        index = pd.date_range("2024-03-01", periods=72, freq="h")
        shape = np.clip(np.sin((index.hour.to_numpy() - 6) / 12.0 * np.pi), 0.0, None)
        frame = pd.DataFrame({OBSERVED_COLUMN["shortwave_down"]: 900.0 * shape}, index=index)

        sample, _atoms = _observed_sample("shortwave_down", frame)

        # Three days of daylight hours: two breaks, one per night crossed.
        assert int((~np.isfinite(sample)).sum()) == 2
        marked_n_eff, marked_r1 = effective_sample_size(sample)
        spliced_n_eff, spliced_r1 = effective_sample_size(sample[np.isfinite(sample)])
        # The splice pairs an evening hour with the next morning's, which drags
        # r1 down and n_effective up: the published number was the optimistic one.
        assert marked_r1 > spliced_r1
        assert marked_n_eff < spliced_n_eff

    def test_a_missing_sensor_still_reports_zero(self):
        assert _available_hours("precipitation", _frame({"unrelated": [1.0, 2.0]})) == 0


class TestTheDirectionEraCutIsPublished:
    """46,5% of the record leaves, and it used to leave in silence.

    Every number on the rose is then a share of what is left, which is why its
    calm fraction (7,4%) disagreed with the wind-speed panel's (5,2%) for the
    same anemometer over the same declared period.
    """

    @staticmethod
    def _frame_spanning_the_cut() -> pd.DataFrame:
        first, last = INVALID_DIRECTION
        index = pd.DatetimeIndex(
            [
                first - pd.Timedelta(hours=1),
                first + pd.Timedelta(days=1),
                first + pd.Timedelta(days=2),
                last + pd.Timedelta(hours=1),
            ]
        )
        return pd.DataFrame(
            {
                OBSERVED_COLUMN["wind_direction"]: [10.0, 20.0, 30.0, 40.0],
                OBSERVED_COLUMN["wind_speed"]: [4.0, 4.0, 4.0, 4.0],
            },
            index=index,
        )

    def test_the_excluded_era_is_an_atom_not_a_silent_drop(self):
        frame = self._frame_spanning_the_cut()

        sample, atoms = _observed_sample("wind_direction", frame)

        assert [atom.id for atom in atoms] == ["arithmetic_mean_era", "calm"]
        assert atoms[0].count == 2
        assert atoms[0].fraction == pytest.approx(0.5)
        assert sorted(sample[np.isfinite(sample)]) == [10.0, 40.0]

    def test_the_excluded_hours_come_back_in_the_coverage_panel(self):
        """The logger did write them; what is wrong is how it averaged them."""
        frame = self._frame_spanning_the_cut()

        assert _available_hours("wind_direction", frame) == 4
