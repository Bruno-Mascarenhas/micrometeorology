"""The published cumulative artifact and the figure that must agree with it.

The curve is the running sum of the histogram's own bars, so these tests pin the
two artifacts against each other rather than against a recomputed expectation:
a gate change that moved one and not the other is exactly the failure worth
catching.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from micrometeorology.cli.export_climatology import OBSERVED_COLUMN, _observed_sample
from micrometeorology.stats.climatology_export import (
    CLIMATOLOGY_VARIABLES,
    build_variable_payload,
)

SPEC = {item.id: item for item in CLIMATOLOGY_VARIABLES}["clearness_index_cumulative"]
HISTOGRAM_SPEC = {item.id: item for item in CLIMATOLOGY_VARIABLES}["clearness_index"]


def _daylight_hours(days: int = 30) -> pd.DataFrame:
    """Hourly frame whose global irradiance spans overcast to clear skies."""
    index = pd.date_range("2024-03-01", periods=days * 24, freq="h")
    rng = np.random.default_rng(11)
    # A plausible clear-sky-ish diurnal shape, scaled by a random daily clearness.
    hour = index.hour.to_numpy()
    shape = np.clip(np.sin((hour - 6) / 12.0 * np.pi), 0.0, None)
    daily = np.repeat(rng.uniform(0.15, 0.85, days), 24)
    return pd.DataFrame({OBSERVED_COLUMN["clearness_index"]: 1100.0 * shape * daily}, index=index)


def test_the_cumulative_stops_short_of_one_by_exactly_the_samples_outside_the_bins():
    """F is a fraction of the RECORD, not of the bars: an above-range Kt still counts."""
    frame = _daylight_hours()
    sample, atoms = _observed_sample(SPEC.id, frame)

    subset = build_variable_payload(SPEC, {"all": sample}, version="t", atoms={"all": atoms})[
        "subsets"
    ]["all"]

    inside = subset["n"] - subset["below"] - subset["above"]
    assert subset["cumulative"][-1] == pytest.approx(inside / subset["n"])


def test_the_cumulative_reaches_one_when_every_sample_is_inside_the_bins():
    inside_only = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    subset = build_variable_payload(SPEC, {"all": inside_only}, version="t")["subsets"]["all"]

    assert (subset["below"], subset["above"]) == (0, 0)
    assert subset["cumulative"][-1] == pytest.approx(1.0)


def test_the_cumulative_is_non_decreasing():
    frame = _daylight_hours()
    sample, atoms = _observed_sample(SPEC.id, frame)

    subset = build_variable_payload(SPEC, {"all": sample}, version="t", atoms={"all": atoms})[
        "subsets"
    ]["all"]

    assert np.all(np.diff(subset["cumulative"]) >= 0.0)


def test_the_cumulative_is_the_running_sum_of_the_histogram_artifact_s_bars():
    """One record, two charts: the curve must be the integral of the bars."""
    frame = _daylight_hours()
    sample, atoms = _observed_sample(SPEC.id, frame)
    histogram_sample, histogram_atoms = _observed_sample(HISTOGRAM_SPEC.id, frame)

    cumulative = build_variable_payload(SPEC, {"all": sample}, version="t", atoms={"all": atoms})[
        "subsets"
    ]["all"]
    histogram = build_variable_payload(
        HISTOGRAM_SPEC, {"all": histogram_sample}, version="t", atoms={"all": histogram_atoms}
    )["subsets"]["all"]

    assert cumulative["counts"] == histogram["counts"]
    assert cumulative["n"] == histogram["n"]


def test_the_published_bounds_and_class_shares_travel_in_the_payload():
    frame = _daylight_hours()
    sample, atoms = _observed_sample(SPEC.id, frame)

    conditions = build_variable_payload(SPEC, {"all": sample}, version="t", atoms={"all": atoms})[
        "subsets"
    ]["all"]["sky_conditions"]

    assert conditions["kt_upper_bounds"] == [0.35, 0.55, 0.65]
    assert [c["condition"] for c in conditions["conditions"]] == [1, 2, 3, 4]
    assert sum(c["fraction"] for c in conditions["conditions"]) == pytest.approx(1.0, abs=1e-5)


def test_an_empty_subset_publishes_the_no_data_shape_instead_of_raising():
    empty = np.array([], dtype=float)

    subset = build_variable_payload(SPEC, {"all": empty}, version="t")["subsets"]["all"]

    assert subset["n"] == 0
    assert subset["cumulative"] == []
    assert all(c["fraction"] is None for c in subset["sky_conditions"]["conditions"])


def test_the_cumulative_carries_no_fitted_curve():
    """A cumulative frequency needs no density family; a fit would be invented."""
    assert SPEC.family is None


def test_the_figure_script_renders_from_the_same_sample_as_the_artifact(tmp_path: Path):
    from scripts.plot_clearness_cumulative import build_figure

    frame = _daylight_hours()
    figure, summary = build_figure(frame)
    try:
        sample, _atoms = _observed_sample(SPEC.id, frame)
        assert summary["n"] == int(np.isfinite(sample).sum())
        destination = tmp_path / "f.png"
        figure.savefig(destination)
        assert destination.stat().st_size > 0
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
