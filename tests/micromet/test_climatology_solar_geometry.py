"""Solar geometry for a database of hourly MEANS stamped at the window's left edge.

Every row of ``station_hourly.parquet`` is an average over ``[T, T+1h)`` --
``sensors.aggregation.aggregate_to_hourly`` resamples on pandas' defaults, which
label a window by its start. Dividing such a mean by an extraterrestrial flux
evaluated instantaneously at ``T`` inflates the clearness index all morning and
deflates it all afternoon, and lets an hour that is mostly twilight through the
daylight gate. The WRF point extraction writes instantaneous values at the
stamped hour, so the model branch must keep reading the geometry at the label.
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
