"""The Kt-Kd plane, its three published models and the artifact they are served in.

The models are transcribed coefficients, so these tests pin the published forms
against values computed by hand rather than against the implementation's own
output: a transposed sign inside the logistic would otherwise be invisible.
"""

import numpy as np
import pandas as pd
import pytest

from labmim_core.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from micrometeorology.stats import ktkd


def _daylight_frame(days: int = 12) -> pd.DataFrame:
    index = pd.date_range("2024-06-01", periods=days * 24, freq="h")
    hour = index.hour.to_numpy()
    shape = np.clip(np.sin((hour - 6) / 12.0 * np.pi), 0.0, None)
    rng = np.random.default_rng(3)
    clearness = np.repeat(rng.uniform(0.25, 0.8, days), 24)
    global_flux = 1050.0 * shape * clearness
    # A physical diffuse fraction: high under an overcast day, low under a clear one.
    diffuse = global_flux * np.clip(1.05 - clearness, 0.08, 0.95)
    return pd.DataFrame({"Sw_dw": global_flux, "Sw_dif": diffuse}, index=index)


def test_marques_filho_matches_the_published_logistic_by_hand():
    """Kd = 0.13 + 0.86 / (1 + exp(-6.29 + 12.26 Kt)) — Marques Filho et al. (2016).

    The three expected values are the logistic evaluated away from this package,
    not its coded expression retyped: a coefficient mistranscribed in both places
    would otherwise agree with itself.
    """
    kt = np.array([0.2, 0.5, 0.75])

    np.testing.assert_allclose(
        ktkd.marques_filho_2016(kt), [0.9718690, 0.5943268, 0.1746400], rtol=1e-6
    )


def test_the_two_brl_models_differ_only_in_their_coefficients():
    """Same functional form, different fits: a shared bug would move both together."""
    args = (np.array([0.5]), np.array([12.0]), np.array([60.0]), np.array([0.5]), np.array([0.5]))

    assert ktkd.lemos_2017(*args)[0] != pytest.approx(ktkd.ridley_brl_2010(*args)[0])


#: Predictors chosen so no term of the logistic vanishes: with AST or elevation at
#: zero a transposed sign on their coefficient would leave the exponent unchanged.
LEMOS_PROBE = (
    np.array([0.62]),
    np.array([9.25]),
    np.array([37.5]),
    np.array([0.54]),
    np.array([0.41]),
)


def test_lemos_matches_the_published_logistic_by_hand():
    """Kd = 1 / (1 + exp(-4.41 + 7.87 Kt - 0.088 AST - 0.0049 alpha + 1.47 Ktd + 1.10 psi)).

    Lemos et al. (2017), the BRL form refitted to Brazilian stations. Written out
    term by term here so a transposed sign is a failure, not a silent refit.
    """
    kt, ast, elevation, daily_kt, psi = (value[0] for value in LEMOS_PROBE)
    exponent = -4.41 + 7.87 * kt - 0.088 * ast - 0.0049 * elevation + 1.47 * daily_kt + 1.10 * psi

    expected = 1.0 / (1.0 + np.exp(exponent))

    np.testing.assert_allclose(ktkd.lemos_2017(*LEMOS_PROBE), [expected], rtol=1e-12)


def test_every_model_returns_a_fraction_between_zero_and_one():
    kt = np.linspace(0.0, 1.0, 51)
    zeros, noon, elevation = np.zeros_like(kt), np.full_like(kt, 12.0), np.full_like(kt, 45.0)

    for predicted in (
        ktkd.marques_filho_2016(kt),
        ktkd.lemos_2017(kt, noon, elevation, kt, kt),
        ktkd.ridley_brl_2010(kt, noon, elevation, kt, zeros),
    ):
        assert np.all((predicted >= 0.0) & (predicted <= 1.0))


def test_the_diffuse_fraction_falls_as_the_sky_clears():
    """The one qualitative claim every diffuse-fraction model makes."""
    kt = np.array([0.15, 0.45, 0.75])

    assert np.all(np.diff(ktkd.marques_filho_2016(kt)) < 0)


def test_apparent_solar_time_is_twelve_at_solar_noon():
    """The hour angle is zero at solar noon, so AST is 12 by construction."""
    index = pd.date_range("2024-03-20 00:00", periods=24 * 4, freq="15min")

    ast = ktkd.apparent_solar_time_hours(index, STATION_SITE.longitude, STATION_UTC_OFFSET_HOURS)
    from labmim_core.solar import hour_angle_deg

    noon = np.argmin(
        np.abs(
            hour_angle_deg(index, STATION_SITE.longitude, utc_offset_hours=STATION_UTC_OFFSET_HOURS)
        )
    )

    assert ast[noon] == pytest.approx(12.0, abs=0.2)
    assert np.all(np.diff(ast) > 0), "apparent solar time runs with the clock"
    assert ast[index.get_loc(pd.Timestamp("2024-03-20 09:00"))] < 12.0


def test_persistence_is_the_mean_of_the_two_neighbouring_hours():
    kt = pd.Series([0.2, 0.6, 0.4], index=pd.date_range("2024-01-01 09:00", periods=3, freq="h"))

    psi = ktkd.persistence_index(kt)

    assert psi.iloc[1] == pytest.approx((0.2 + 0.4) / 2.0)


def test_an_isolated_hour_keeps_its_own_clearness_rather_than_becoming_missing():
    """A gap is not bridged: persistence never invents a continuity."""
    kt = pd.Series([0.7], index=pd.DatetimeIndex(["2024-01-01 12:00"]))

    assert ktkd.persistence_index(kt).iloc[0] == pytest.approx(0.7)


def test_the_density_matrix_is_rows_kd_by_columns_kt():
    """The renderer refuses a transposed matrix; a wrong orientation must fail here."""
    kt = np.array([0.1, 0.1, 0.9])
    kd = np.array([0.9, 0.9, 0.1])
    edges = np.array([0.0, 0.5, 1.0])

    density = ktkd.density_2d(kt, kd, edges, edges)

    assert len(density["counts"]) == len(density["kd_edges"]) - 1
    assert len(density["counts"][0]) == len(density["kt_edges"]) - 1
    # Two samples at low Kt / high Kd land in the TOP-row, first-column cell.
    assert density["counts"][1][0] == 2
    assert density["counts"][0][1] == 1
    assert density["max_count"] == 2


def test_a_thin_band_publishes_null_quantiles_but_still_reports_its_real_count():
    """Consumers must decide on `median is None`, never on a zero count."""
    kt = np.array([0.11, 0.12, 0.61, 0.62, 0.63, 0.64, 0.65])
    predicted = np.linspace(0.2, 0.8, kt.size)
    edges = np.array([0.0, 0.5, 1.0])

    band = ktkd.model_band(kt, predicted, edges, min_samples_per_bin=3)

    assert band["median"][0] is None
    assert band["n_per_bin"][0] == 2
    assert band["median"][1] is not None


def test_regression_scores_are_none_rather_than_nan_when_nothing_pairs():
    scores = ktkd.regression_scores(np.array([np.nan]), np.array([0.5]))

    assert scores == {"rmse": None, "mbe": None, "mae": None, "n": 0}


def test_the_published_scores_are_signed_from_the_model_towards_the_measurement():
    """MBE is predicted - observed, so a model that runs high publishes a positive
    bias; the pairwise-finite mask must drop the two half-pairs before averaging,
    leaving residuals +0.2 and -0.1 over n = 2.
    """
    observed = np.array([0.5, 0.6, np.nan, 0.4])
    predicted = np.array([0.7, 0.5, 0.3, np.nan])

    scores = ktkd.regression_scores(observed, predicted)

    assert scores["n"] == 2
    assert scores["mbe"] == pytest.approx(0.05)
    assert scores["mae"] == pytest.approx(0.15)
    assert scores["rmse"] == pytest.approx(np.sqrt((0.2**2 + 0.1**2) / 2))


def test_the_gates_keep_only_daylight_hours_with_a_physical_ratio():
    prepared = ktkd.prepare_ktkd(
        _daylight_frame(), site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )

    assert len(prepared.kt) > 0
    assert prepared.kt.between(0.0, ktkd.MAX_RATIO).all()
    assert prepared.kd.between(0.0, ktkd.MAX_RATIO).all()
    assert len(prepared.ast) == len(prepared.kt)
    assert len(prepared.psi) == len(prepared.kt)


def test_a_dim_hour_and_an_hour_without_diffuse_are_both_dropped():
    """The ratio window alone admits the dim hour: 40 W/m2 against 20 W/m2 of
    diffuse gives a Kt and a Kd both inside [0, 1.2], so only the 50 W/m2 floor
    removes it. The hour with no diffuse reading is removed by its own gate.
    """
    hourly = _daylight_frame()
    index = pd.DatetimeIndex(hourly.index)
    noon = index[index.hour == 12]
    dim, without_diffuse = noon[0], noon[1]
    hourly.loc[dim, ["Sw_dw", "Sw_dif"]] = [40.0, 20.0]
    hourly.loc[without_diffuse, "Sw_dif"] = np.nan

    prepared = ktkd.prepare_ktkd(
        hourly, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )

    assert dim not in prepared.kt.index
    assert without_diffuse not in prepared.kt.index
    assert noon[2] in prepared.kt.index


def test_an_hour_without_diffuse_is_no_neighbour_of_the_persistence_index():
    """Dropping the pair is not enough: psi averages the hour before and after,
    so an hour the diffuse channel never covered must leave the neighbourhood
    too. The oracle is the same frame with the global channel also missing,
    which every gate excludes for a reason nobody disputes.
    """
    hourly = _daylight_frame()
    index = pd.DatetimeIndex(hourly.index)
    without_diffuse = index[index.hour == 12][1]
    neighbours = [without_diffuse - pd.Timedelta(hours=1), without_diffuse + pd.Timedelta(hours=1)]

    blind = hourly.copy()
    blind.loc[without_diffuse, "Sw_dif"] = np.nan
    both_blind = hourly.copy()
    both_blind.loc[without_diffuse, ["Sw_dw", "Sw_dif"]] = np.nan

    prepared = [
        ktkd.prepare_ktkd(frame, site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS)
        for frame in (blind, both_blind)
    ]
    psi = [pd.Series(one.psi, index=pd.DatetimeIndex(one.kt.index)) for one in prepared]

    np.testing.assert_allclose(psi[0][neighbours], psi[1][neighbours], rtol=1e-12)


def test_no_surviving_hour_dips_below_the_elevation_floor_at_either_end():
    """The floor is on the WHOLE hour, not on its midpoint: a terminator hour
    whose midpoint clears 10 degrees still starts below it, and the model's
    predictors would then describe a sky the sensor never saw.
    """
    prepared = ktkd.prepare_ktkd(
        _daylight_frame(), site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    kept = pd.DatetimeIndex(prepared.kt.index)

    lowest, _highest = ktkd.elevation_bounds(kept, STATION_SITE, STATION_UTC_OFFSET_HOURS)

    assert (lowest > 10.0).all()
    assert prepared.elevation.min() > 10.0


def test_the_payload_declares_the_published_schema_and_both_model_kinds():
    from micrometeorology.stats.sky_condition import sky_condition_summary

    prepared = ktkd.prepare_ktkd(
        _daylight_frame(), site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    kt, kd = prepared.kt, prepared.kd
    predictors = (prepared.ast, prepared.elevation, prepared.daily_kt, prepared.psi)
    edges = np.round(np.arange(0.0, 1.02, 0.02), 2)

    payload = ktkd.build_ktkd_payload(
        kt,
        kd,
        models={
            "marques_filho_2016": ktkd.marques_filho_2016(kt.to_numpy()),
            "lemos_2017": ktkd.lemos_2017(kt.to_numpy(), *predictors),
            "ridley_brl_2010": ktkd.ridley_brl_2010(kt.to_numpy(), *predictors),
        },
        sky_conditions=sky_condition_summary(kt.to_numpy()),
        kt_edges=edges,
        kd_edges=edges,
        station={"name": "LabMiM"},
        sources=["station_hourly.parquet"],
        filters=prepared.filters,
        caveats=["teste"],
        version="t",
    )

    assert payload["schema"] == "labmim-ktkd-v1"
    kinds = {model["id"]: model["kind"] for model in payload["models"]}
    assert kinds["marques_filho_2016"] == "curve"
    assert kinds["lemos_2017"] == "band"
    assert kinds["ridley_brl_2010"] == "band"
    assert payload["sky_conditions"]["kt_upper_bounds"] == [0.35, 0.55, 0.65]
    # The display name is published, so the page never derives one by splitting
    # the citation: a bibliography edit would otherwise relabel the chart.
    labels = {model["id"]: model["label"] for model in payload["models"]}
    assert labels["marques_filho_2016"] == "Marques Filho et al. (2016)"
    assert all(model["reference"] for model in payload["models"])
    assert len(payload["points"]) == len(kt)
    assert payload["points_format"] == ["kt", "kd", "t"]
    first = payload["points"][0]
    assert isinstance(first, list)
    assert len(first) == 3
    assert first[0] == pytest.approx(float(kt.iloc[0]), abs=5e-4)


def test_the_published_payload_carries_no_non_finite_number():
    """The site parses strictly: a bare NaN token loses the whole document."""
    import json

    from micrometeorology.stats.sky_condition import sky_condition_summary

    prepared = ktkd.prepare_ktkd(
        _daylight_frame(), site=STATION_SITE, utc_offset_hours=STATION_UTC_OFFSET_HOURS
    )
    kt = prepared.kt
    edges = np.round(np.arange(0.0, 1.02, 0.02), 2)
    payload = ktkd.build_ktkd_payload(
        kt,
        prepared.kd,
        models={"marques_filho_2016": ktkd.marques_filho_2016(kt.to_numpy())},
        sky_conditions=sky_condition_summary(kt.to_numpy()),
        kt_edges=edges,
        kd_edges=edges,
        station={},
        sources=[],
        filters=[],
        caveats=[],
        version="t",
    )

    json.dumps(payload, allow_nan=False)


def test_a_gap_in_the_global_record_leaves_the_daily_clearness_untouched():
    """``groupby().sum()`` counts a missing global hour as zero while the
    extraterrestrial hour still enters the denominator: a day with four missing
    noon hours published half its real Kt."""
    index = pd.date_range("2024-06-01 06:00", periods=12, freq="h")
    extraterrestrial = pd.Series(np.linspace(100.0, 1000.0, 12), index=index)
    global_flux = 0.6 * extraterrestrial
    global_flux.iloc[4:8] = np.nan

    kt = ktkd.daily_clearness_index(global_flux, extraterrestrial)

    assert kt.iloc[0] == pytest.approx(0.6)


def test_ridley_brl_matches_the_published_logistic_by_hand():
    """Ridley, Boland & Lauret (2010): d = 1 / (1 + exp(-5.38 + 6.63 kt + 0.006 AST
    - 0.007 alpha + 1.75 Kt + 1.31 psi)). The AST coefficient had been transcribed
    with its sign flipped."""
    kt, ast, alpha, daily_kt, psi = 0.5, 15.0, 40.0, 0.55, 0.5
    expected = 1.0 / (
        1.0 + np.exp(-5.38 + 6.63 * kt + 0.006 * ast - 0.007 * alpha + 1.75 * daily_kt + 1.31 * psi)
    )

    modelled = ktkd.ridley_brl_2010(
        np.array([kt]), np.array([ast]), np.array([alpha]), np.array([daily_kt]), np.array([psi])
    )

    assert modelled[0] == pytest.approx(expected, rel=1e-12)
