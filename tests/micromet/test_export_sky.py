"""The two artifacts the sky page reads, and the gates that decide who is in them.

The cumulative and the Kt-Kd plane are published side by side but describe
DIFFERENT populations on purpose: F(Kt) covers every hour the global pyranometer
measured, while the Kt-Kd plane needs a diffuse measurement too. Conflating them
would publish a clearness record silently conditioned on the shaded sensor's
availability, so the difference is pinned here.
"""

import json

import numpy as np
import pandas as pd
import pytest

from labmim_core.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.cli.export_sky import SUBSET_LABELS, _seasonal_samples, build_payloads
from micrometeorology.stats import ktkd as ktkd_stats
from micrometeorology.stats.sky_condition import KT_CUMULATIVE_EDGES


def _hourly(days: int = 366, *, diffuse_days: int | None = None) -> pd.DataFrame:
    """A full year of hours, so every seasonal recorte has data.

    *diffuse_days* limits how many days carry a diffuse reading, which is how the
    two artifacts' populations are made to differ on purpose.
    """
    index = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    hour = index.hour.to_numpy()
    shape = np.clip(np.sin((hour - 6) / 12.0 * np.pi), 0.0, None)
    clearness = np.repeat(np.random.default_rng(5).uniform(0.2, 0.8, days), 24)
    global_flux = 1050.0 * shape * clearness
    diffuse = global_flux * np.clip(1.05 - clearness, 0.08, 0.95)
    frame = pd.DataFrame({"Sw_dw": global_flux, "Sw_dif": diffuse}, index=index)
    if diffuse_days is not None:
        frame.loc[frame.index[diffuse_days * 24 :], "Sw_dif"] = np.nan
    return frame


def test_the_cumulative_covers_hours_the_kt_kd_plane_cannot():
    """F(Kt) must not be conditioned on the diffuse sensor being up."""
    frame = _hourly(days=366, diffuse_days=60)

    clearness = ktkd_stats.prepare_clearness(
        frame, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    paired = ktkd_stats.prepare_ktkd(
        frame, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    ).kt

    assert len(clearness) > len(paired)


def test_both_artifacts_carry_the_same_run_stamp():
    """A ``Ceu/`` meio atualizado so e detectavel pelo navegador se os dois
    documentos concordarem sobre a versao em que foram produzidos."""
    payloads = build_payloads(_hourly(), version="20260101T000000Z")

    assert payloads["kt_cumulative.json"]["version"] == "20260101T000000Z"
    assert payloads["ktkd.json"]["version"] == "20260101T000000Z"


def test_the_command_writes_both_artifacts_into_the_output_directory(tmp_path):
    """Every other test calls ``build_payloads``; the writing wrapper had none."""
    from typer.testing import CliRunner

    from micrometeorology.cli.export_sky import app

    source = tmp_path / "station_hourly.parquet"
    _hourly().to_parquet(source)
    output_dir = tmp_path / "Ceu"

    result = CliRunner().invoke(app, ["-i", str(source), "-o", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in output_dir.glob("*.json")) == [
        "kt_cumulative.json",
        "ktkd.json",
    ]


def test_both_artifacts_are_published_under_their_declared_schemas():
    payloads = build_payloads(_hourly(), version="probe")

    assert payloads["kt_cumulative.json"]["format"] == "labmim-kt-cumulative-v1"
    assert payloads["ktkd.json"]["schema"] == "labmim-ktkd-v1"


def test_every_cumulative_subset_carries_its_own_caption():
    """`Ceu/` has no manifest beside it, so the label travels in the subset."""
    subsets = build_payloads(_hourly(), version="probe")["kt_cumulative.json"]["subsets"]

    assert set(subsets) == set(SUBSET_LABELS)
    for subset_id, subset in subsets.items():
        assert subset["label"] == SUBSET_LABELS[subset_id]


def test_the_cumulative_falls_short_of_one_by_exactly_the_out_of_range_samples():
    """`n` counts the whole subset, so F reaches 1 only when nothing overflowed."""
    subsets = build_payloads(_hourly(), version="probe")["kt_cumulative.json"]["subsets"]

    for subset in subsets.values():
        inside = subset["n"] - subset["below"] - subset["above"]
        assert subset["cumulative"][-1] == pytest.approx(inside / subset["n"], abs=1e-6)


def test_the_cumulative_publishes_one_value_per_bin_not_per_edge():
    payload = build_payloads(_hourly(), version="probe")["kt_cumulative.json"]

    assert len(payload["edges"]) == len(KT_CUMULATIVE_EDGES)
    for subset in payload["subsets"].values():
        assert len(subset["cumulative"]) == len(payload["edges"]) - 1
        assert len(subset["counts"]) == len(payload["edges"]) - 1


def test_the_class_shares_are_published_already_computed_with_their_bounds():
    subset = build_payloads(_hourly(), version="probe")["kt_cumulative.json"]["subsets"][
        "observed_all"
    ]

    conditions = subset["sky_conditions"]
    assert conditions["kt_upper_bounds"] == [0.35, 0.55, 0.65]
    assert sum(c["fraction"] for c in conditions["conditions"]) == pytest.approx(1.0, abs=1e-6)


def test_the_cumulative_selects_exactly_what_the_climatology_histogram_selects():
    """The published caveat says one is the integral of the other. It must be true.

    The two artifacts live on different pages, so nothing but this test stops
    their gates from drifting apart — and a drift shows up only as two row counts
    no reader can reconcile.
    """
    from micrometeorology.cli.export_climatology import _observed_sample

    frame = _hourly()
    histogram, _atoms = _observed_sample("clearness_index", frame)
    cumulative = ktkd_stats.prepare_clearness(
        frame, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )

    # The histogram sample carries a NaN wherever the daylight gate breaks the
    # hour chain, so `n_effective` is not computed across a splice; the
    # populations themselves are what must agree.
    measured = histogram[np.isfinite(histogram)]
    assert len(cumulative) == len(measured)
    np.testing.assert_allclose(np.sort(cumulative.to_numpy()), np.sort(measured))


def test_the_daylight_gate_brackets_the_whole_hour_not_only_its_midpoint():
    """An hour the sun rises inside is not daytime, however bright its midpoint."""
    from micrometeorology.stats.daylight import elevation_bounds

    # 2024-01-31 05:00 local: -6.92 deg at the start, -0.02 at the midpoint,
    # +6.95 by the end. A midpoint-only gate would call this hour daytime-adjacent.
    times = pd.DatetimeIndex(["2024-01-31 05:00"])
    lowest, highest = elevation_bounds(times, STATION_SITE, STATION_UTC_OFFSET_HOURS)

    assert lowest[0] < highest[0]
    assert lowest[0] == pytest.approx(-6.92, abs=0.05)
    assert highest[0] == pytest.approx(6.95, abs=0.05)


def test_every_model_publishes_a_short_label_beside_its_full_citation():
    """The page prints the producer's name, not one parsed out of the citation."""
    models = build_payloads(_hourly(), version="probe")["ktkd.json"]["models"]

    for model in models:
        assert model["label"]
        assert "," not in model["label"].split("(")[0].rstrip()
        assert model["reference"] != model["label"]


def test_the_density_matrix_rows_match_the_kd_edges_the_renderer_checks():
    density = build_payloads(_hourly(), version="probe")["ktkd.json"]["density"]

    assert len(density["counts"]) == len(density["kd_edges"]) - 1
    assert len(density["counts"][0]) == len(density["kt_edges"]) - 1


def test_each_seasonal_recorte_carries_only_its_own_months():
    """The recorte ids are a published contract, so a swapped tuple must fail here.

    The synthetic year is season-symmetric on purpose everywhere else in this
    file, which leaves the DJF/JJA split with nothing to hold it.
    """
    index = pd.date_range("2024-01-01", periods=366, freq="D")
    month = index.month.to_numpy()
    clearness = pd.Series(
        np.select([np.isin(month, (12, 1, 2)), np.isin(month, (6, 7, 8))], [0.75, 0.25], 0.5),
        index=index,
    )

    subsets = _seasonal_samples(clearness)

    np.testing.assert_allclose(subsets["observed_djf"], 0.75)
    np.testing.assert_allclose(subsets["observed_jja"], 0.25)
    assert len(subsets["observed_all"]) == len(index)


def test_neither_artifact_carries_a_non_finite_number():
    """Both are fetched by a browser that fails the whole document on a NaN token."""
    for payload in build_payloads(_hourly(), version="probe").values():
        json.dumps(payload, allow_nan=False)


def test_a_record_with_no_usable_hour_refuses_to_publish_empty_artifacts():
    """The global channel decides: with it absent neither document has a
    population, so the run refuses rather than publishing empty subsets."""
    blind = pd.DataFrame(
        {"Sw_dw": np.full(48, np.nan), "Sw_dif": np.full(48, np.nan)},
        index=pd.date_range("2024-01-01", periods=48, freq="h"),
    )

    with pytest.raises(ValueError, match="no hour survived"):
        build_payloads(blind, version="probe")


def test_a_dead_diffuse_sensor_still_publishes_the_cumulative():
    """``prepare_ktkd`` needs the diffuse channel and ``kt_cumulative.json`` does
    not, but the ``kt.empty`` refusal ran first and took both documents down on
    every PSP outage."""
    blind = _hourly(days=366)
    blind["Sw_dif"] = np.nan

    payloads = build_payloads(blind, version="probe")

    assert set(payloads) == {"kt_cumulative.json"}
    assert payloads["kt_cumulative.json"]["format"] == "labmim-kt-cumulative-v1"
