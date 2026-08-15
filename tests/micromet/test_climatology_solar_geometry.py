"""Solar geometry for a database of hourly MEANS stamped at the window's left edge.

Every row of ``station_hourly.parquet`` is an average over ``[T, T+1h)`` --
``sensors.aggregation.aggregate_to_hourly`` resamples on pandas' defaults, which
label a window by its start. Dividing such a mean by an extraterrestrial flux
evaluated instantaneously at ``T`` inflates the clearness index all morning and
deflates it all afternoon. The WRF point extraction writes instantaneous values at
the stamped hour, so the model branch must keep reading the geometry at the label
for the values it computes.

The daylight SELECTION is the one thing the two sides share: an hour the sun
crosses the floor inside belongs to neither sample, since a gate read at each
source's own instant admits different hours around sunrise and sunset and the
paired histograms would then be conditional on different events.
"""

import pandas as pd
import pytest

from allsky.solar import extraterrestrial_ghi
from micrometeorology.cli.export_climatology import (
    OBSERVED_COLUMN,
    SITE,
    UTC_OFFSET_HOURS,
    WRF_COLUMN,
    _observed_sample,
    _wrf_sample,
)

AVERAGING_WINDOW_MIDPOINT = pd.Timedelta(minutes=30)

# Two hours of 2024-03-20 the 10 deg floor falls inside: at 06:00 the sun is at
# 4,43 deg and at 11,73 deg half an hour later, at 17:00 at 10,18 deg and 2,87 deg.
TERMINATOR_HOURS = ("2024-03-20 06:00", "2024-03-20 17:00")


def _hourly(column: str, values: list[float], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: values}, index=pd.DatetimeIndex([pd.Timestamp(x) for x in labels]))


@pytest.mark.parametrize(
    ("label", "ghi_w_m2"),
    [("2024-03-20 07:00", 430.0), ("2024-03-20 15:00", 520.0)],
)
def test_clearness_divides_the_hourly_mean_by_the_flux_at_the_averaging_midpoint(
    label: str, ghi_w_m2: float
) -> None:
    frame = _hourly(OBSERVED_COLUMN["clearness_index"], [ghi_w_m2], [label])

    sample, _atoms = _observed_sample("clearness_index", frame)

    midpoint = pd.DatetimeIndex([pd.Timestamp(label) + AVERAGING_WINDOW_MIDPOINT])
    assert sample == pytest.approx(
        ghi_w_m2 / extraterrestrial_ghi(midpoint, SITE, UTC_OFFSET_HOURS)
    )


def test_an_hour_whose_midpoint_sits_below_the_elevation_floor_is_not_daytime() -> None:
    frame = _hourly(
        OBSERVED_COLUMN["shortwave_down"],
        [820.0, 65.0],
        ["2024-03-20 12:00", "2024-03-20 17:00"],
    )

    sample, _atoms = _observed_sample("shortwave_down", frame)

    assert sample == pytest.approx([820.0])


def test_the_model_series_reads_the_geometry_at_the_stamp_it_carries() -> None:
    label = "2024-03-20 07:00"
    frame = _hourly(WRF_COLUMN["clearness_index"], [430.0], [label])

    sample, _atoms = _wrf_sample("clearness_index", frame)

    stamp = pd.DatetimeIndex([pd.Timestamp(label)])
    assert sample == pytest.approx(430.0 / extraterrestrial_ghi(stamp, SITE, UTC_OFFSET_HOURS))


@pytest.mark.parametrize("label", TERMINATOR_HOURS)
def test_an_hour_the_sun_crosses_the_floor_inside_is_daytime_for_neither_source(
    label: str,
) -> None:
    observed = _hourly(OBSERVED_COLUMN["shortwave_down"], [120.0], [label])
    model = _hourly(WRF_COLUMN["shortwave_down"], [120.0], [label])

    observed_sample, _observed_atoms = _observed_sample("shortwave_down", observed)
    model_sample, _model_atoms = _wrf_sample("shortwave_down", model)

    assert observed_sample.size == 0
    assert model_sample.size == 0


@pytest.mark.parametrize("label", TERMINATOR_HOURS)
def test_the_clearness_gate_admits_the_terminator_hour_for_neither_source(label: str) -> None:
    observed = _hourly(OBSERVED_COLUMN["clearness_index"], [120.0], [label])
    model = _hourly(WRF_COLUMN["clearness_index"], [120.0], [label])

    observed_sample, _observed_atoms = _observed_sample("clearness_index", observed)
    model_sample, _model_atoms = _wrf_sample("clearness_index", model)

    assert observed_sample.size == 0
    assert model_sample.size == 0


def test_an_hour_that_begins_with_the_sun_up_is_not_a_night_sample() -> None:
    """At 18:00 the sun sits at +0,07 deg and sets inside the hour it labels."""
    frame = _hourly(
        OBSERVED_COLUMN["net_radiation_night"],
        [-30.0],
        ["2024-01-03 18:00"],
    )

    sample, _atoms = _observed_sample("net_radiation_night", frame)

    assert sample.size == 0


def test_an_hour_the_sun_rises_in_after_the_midpoint_is_not_a_night_sample() -> None:
    """At 05:00 the sun is at -6,92 deg and still -0,02 deg at the midpoint.

    It clears the horizon at roughly +31 min and reaches +6,95 deg by the end of
    the hour, so a bracket spanning only the two per-source geometry offsets reads
    the whole hour as night and averages some 29 min of daylight into the
    nocturnal distribution.
    """
    frame = _hourly(
        OBSERVED_COLUMN["net_radiation_night"],
        [-30.0],
        ["2024-01-31 05:00"],
    )

    sample, _atoms = _observed_sample("net_radiation_night", frame)

    assert sample.size == 0


def test_a_wholly_dark_hour_is_still_a_night_sample() -> None:
    """02:00 on the same day: the sun stays below the horizon for the whole hour."""
    frame = _hourly(
        OBSERVED_COLUMN["net_radiation_night"],
        [-30.0],
        ["2024-01-31 02:00"],
    )

    sample, _atoms = _observed_sample("net_radiation_night", frame)

    assert sample.size == 1
