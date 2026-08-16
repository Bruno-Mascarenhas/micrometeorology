"""Tests for micrometeorology.stats.distributions.

Offline and fast: synthetic samples from a seeded generator, no I/O.

Where a reference exists, the estimators are checked against SciPy's own
maximum-likelihood fit rather than against a recorded constant: what is pinned
is that the published parameters are right, not that they are unchanged.
"""

import inspect

import numpy as np
import pytest
from scipy import stats

from micrometeorology.stats.distributions import (
    FAMILIES,
    cdf,
    circular_summary,
    effective_sample_size,
    fit_distribution,
    fit_von_mises_mixture,
    goodness_of_fit,
    histogram,
    pdf,
    ppf,
    sector_frequencies,
    von_mises_mixture_pdf,
)


def _flatten(values):
    """Family parameters, with the list-valued ones spread out."""
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            yield from (float(item) for item in value)
        else:
            yield float(value)


def _required_options(family):
    """Keyword-only parameters of a family's estimator that carry no default."""
    signature = inspect.signature(FAMILIES[family].fit)
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }


# Families estimable from a sample alone, and families that need a covariate the
# sample does not carry. Derived from the signatures rather than listed by hand,
# so a new family lands in the right group without anyone remembering to say so.
SAMPLE_ONLY_FAMILIES = sorted(family for family in FAMILIES if not _required_options(family))
COVARIATE_FAMILIES = sorted(family for family in FAMILIES if _required_options(family))


# Large enough that a maximum-likelihood estimate lands within a few parts in
# 1e3 of the generating parameters, small enough that the whole module runs in
# a couple of seconds under pytest-xdist.
SAMPLE_SIZE = 100_000


@pytest.fixture
def wind_speed() -> np.ndarray:
    """Weibull-distributed speeds, shape 1.9 and scale 2.4 m/s."""
    sample: np.ndarray = stats.weibull_min.rvs(
        1.9, loc=0.0, scale=2.4, size=SAMPLE_SIZE, random_state=11
    )
    return sample


@pytest.fixture
def temperature() -> np.ndarray:
    """Gaussian air temperature, 26.0 +/- 1.9 degC."""
    sample: np.ndarray = stats.norm.rvs(26.0, 1.9, size=SAMPLE_SIZE, random_state=14)
    return sample


class TestHistogram:
    def test_density_integrates_to_one(self, temperature):
        binned = histogram(temperature, np.arange(14.0, 40.01, 0.5))
        integral = float((binned.density * np.diff(binned.edges)).sum())
        assert integral == pytest.approx(1.0, abs=1e-12)

    def test_out_of_range_samples_are_tallied_not_dropped(self):
        binned = histogram(np.array([-5.0, 0.5, 1.5, 99.0]), [0.0, 1.0, 2.0])
        assert (binned.below, binned.n, binned.above) == (1, 2, 1)

    def test_non_finite_samples_are_dropped(self):
        binned = histogram(np.array([0.5, np.nan, np.inf, -np.inf, 1.5]), [0.0, 1.0, 2.0])
        assert binned.n == 2

    def test_centers_are_bin_midpoints(self):
        binned = histogram(np.array([0.5]), [0.0, 1.0, 3.0])
        np.testing.assert_allclose(binned.centers, [0.5, 2.0])

    def test_empty_sample_gives_zero_density_not_nan(self):
        binned = histogram(np.array([]), [0.0, 1.0, 2.0])
        assert binned.n == 0
        assert np.all(binned.density == 0.0)

    def test_too_few_edges_raises(self):
        with pytest.raises(ValueError, match="at least two edges"):
            histogram(np.array([1.0]), [0.0])

    def test_non_monotonic_edges_raise(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            histogram(np.array([1.0]), [0.0, 2.0, 1.0])


class TestWeibull:
    def test_matches_scipy_maximum_likelihood(self, wind_speed):
        fitted = fit_distribution("weibull", wind_speed)
        reference_shape, _loc, reference_scale = stats.weibull_min.fit(wind_speed, floc=0)
        assert fitted.params["shape"] == pytest.approx(reference_shape, rel=1e-4)
        assert fitted.params["scale"] == pytest.approx(reference_scale, rel=1e-4)

    def test_recovers_the_generating_parameters(self, wind_speed):
        fitted = fit_distribution("weibull", wind_speed)
        assert fitted.params["shape"] == pytest.approx(1.9, rel=0.02)
        assert fitted.params["scale"] == pytest.approx(2.4, rel=0.02)

    def test_calms_are_excluded_from_the_fit(self, wind_speed):
        """A zero atom must not reach the estimator; support is the open (0, inf)."""
        with_calms = np.concatenate([wind_speed, np.zeros(10_000)])
        fitted = fit_distribution("weibull", with_calms)
        assert fitted.n == wind_speed.size

    def test_fitting_over_zeros_would_bias_the_shape_down(self, wind_speed):
        """Guards the reason support exists, not the implementation."""
        clean = fit_distribution("weibull", wind_speed)
        naive = FAMILIES["weibull"].fit(np.concatenate([wind_speed, np.full(50_000, 1e-6)]))
        assert naive["shape"] < clean.params["shape"]

    def test_pdf_integrates_to_one(self, wind_speed):
        fitted = fit_distribution("weibull", wind_speed)
        grid = np.linspace(1e-9, 30.0, 300_001)
        assert float(np.trapezoid(pdf(fitted, grid), grid)) == pytest.approx(1.0, abs=1e-6)

    def test_pdf_and_cdf_are_zero_at_and_below_zero(self, wind_speed):
        fitted = fit_distribution("weibull", wind_speed)
        np.testing.assert_array_equal(pdf(fitted, np.array([-1.0, 0.0])), [0.0, 0.0])
        np.testing.assert_array_equal(cdf(fitted, np.array([-1.0, 0.0])), [0.0, 0.0])

    def test_large_speeds_do_not_overflow_the_shape_solver(self):
        """exp(k ln x) is used instead of x**k precisely so this stays finite."""
        gusty = stats.weibull_min.rvs(9.0, loc=0.0, scale=40.0, size=20_000, random_state=3)
        fitted = fit_distribution("weibull", gusty)
        assert np.isfinite(fitted.params["shape"])
        assert np.isfinite(fitted.params["scale"])


class TestGamma:
    def test_matches_scipy_maximum_likelihood(self):
        sample = stats.gamma.rvs(0.7, loc=0.0, scale=3.1, size=SAMPLE_SIZE, random_state=12)
        fitted = fit_distribution("gamma", sample)
        reference_shape, _loc, reference_scale = stats.gamma.fit(sample, floc=0)
        assert fitted.params["shape"] == pytest.approx(reference_shape, rel=1e-5)
        assert fitted.params["scale"] == pytest.approx(reference_scale, rel=1e-5)

    def test_handles_the_small_shape_of_hourly_rainfall(self):
        """Hourly intensity has shape well below 1 — the Thom start must survive it."""
        sample = stats.gamma.rvs(0.35, loc=0.0, scale=1.8, size=SAMPLE_SIZE, random_state=21)
        fitted = fit_distribution("gamma", sample)
        assert fitted.params["shape"] == pytest.approx(0.35, rel=0.03)

    def test_dry_hours_are_excluded_from_the_fit(self):
        sample = stats.gamma.rvs(0.7, loc=0.0, scale=3.1, size=1_000, random_state=12)
        fitted = fit_distribution("gamma", np.concatenate([sample, np.zeros(50_000)]))
        assert fitted.n == sample.size

    def test_cdf_matches_scipy(self):
        sample = stats.gamma.rvs(0.7, loc=0.0, scale=3.1, size=10_000, random_state=12)
        fitted = fit_distribution("gamma", sample)
        grid = np.linspace(0.0, 20.0, 501)
        reference = stats.gamma.cdf(grid, fitted.params["shape"], scale=fitted.params["scale"])
        np.testing.assert_allclose(cdf(fitted, grid), reference, atol=1e-12)


class TestBeta:
    def test_matches_scipy_maximum_likelihood(self):
        sample = stats.beta.rvs(4.0, 1.6, size=SAMPLE_SIZE, random_state=13)
        fitted = fit_distribution("beta", sample)
        reference_alpha, reference_beta, _loc, _scale = stats.beta.fit(sample, floc=0, fscale=1)
        assert fitted.params["alpha"] == pytest.approx(reference_alpha, rel=1e-5)
        assert fitted.params["beta"] == pytest.approx(reference_beta, rel=1e-5)

    def test_saturation_pileup_is_excluded_from_the_fit(self):
        """RH clipped at 100 % is an atom, not part of the continuous density."""
        sample = stats.beta.rvs(4.0, 1.6, size=10_000, random_state=13)
        fitted = fit_distribution("beta", np.concatenate([sample, np.ones(2_000)]))
        assert fitted.n == sample.size

    def test_pdf_integrates_to_one(self):
        sample = stats.beta.rvs(4.0, 1.6, size=20_000, random_state=13)
        fitted = fit_distribution("beta", sample)
        grid = np.linspace(1e-9, 1.0 - 1e-9, 200_001)
        assert float(np.trapezoid(pdf(fitted, grid), grid)) == pytest.approx(1.0, abs=1e-5)

    def test_pdf_is_zero_outside_the_unit_interval(self):
        sample = stats.beta.rvs(4.0, 1.6, size=1_000, random_state=13)
        fitted = fit_distribution("beta", sample)
        np.testing.assert_array_equal(pdf(fitted, np.array([-0.1, 0.0, 1.0, 1.1])), np.zeros(4))


class TestNormal:
    def test_parameters_are_the_sample_moments(self, temperature):
        fitted = fit_distribution("normal", temperature)
        assert fitted.params["mu"] == pytest.approx(float(temperature.mean()))
        assert fitted.params["sigma"] == pytest.approx(float(temperature.std(ddof=1)))

    def test_cdf_matches_scipy(self, temperature):
        fitted = fit_distribution("normal", temperature)
        grid = np.linspace(14.0, 40.0, 401)
        reference = stats.norm.cdf(grid, fitted.params["mu"], fitted.params["sigma"])
        np.testing.assert_allclose(cdf(fitted, grid), reference, atol=1e-12)


def test_a_stuck_channel_does_not_publish_a_normal_fit():
    """A constant sample is the Dirac limit, which no normal density represents.

    Reporting sigma = 0 would pass the caller's finite-parameter gate and ship a
    fit whose density is NaN at every abscissa, whose KS distance and density
    R-squared are NaN, and whose quantile gap is a perfect 0.0 — the best
    possible goodness-of-fit score, printed for a dead sensor.
    """
    fitted = fit_distribution("normal", np.full(500, 1010.0))

    assert np.isnan(fitted.params["mu"])
    assert np.isnan(fitted.params["sigma"])


class TestHollandsHuget:
    @pytest.fixture
    def clearness(self) -> np.ndarray:
        sample: np.ndarray = np.clip(
            stats.beta.rvs(2.2, 1.7, size=50_000, random_state=15) * 0.8, 0.0, None
        )
        return sample

    def test_pdf_integrates_to_one(self, clearness):
        fitted = fit_distribution("hollands_huget", clearness)
        grid = np.linspace(0.0, fitted.params["kt_max"], 200_001)
        assert float(np.trapezoid(pdf(fitted, grid), grid)) == pytest.approx(1.0, abs=1e-6)

    def test_fitted_mean_matches_the_sample_mean(self, clearness):
        """The single free parameter is fixed by matching the mean — verify it did."""
        fitted = fit_distribution("hollands_huget", clearness)
        grid = np.linspace(0.0, fitted.params["kt_max"], 200_001)
        modelled = float(np.trapezoid(grid * pdf(fitted, grid), grid))
        inside = clearness[clearness < fitted.params["kt_max"]]
        assert modelled == pytest.approx(float(inside.mean()), rel=1e-4)

    def test_cdf_and_ppf_round_trip(self, clearness):
        fitted = fit_distribution("hollands_huget", clearness)
        probabilities = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        np.testing.assert_allclose(
            cdf(fitted, ppf(fitted, probabilities)), probabilities, atol=1e-9
        )

    def test_kt_max_defaults_to_the_local_clear_sky_ceiling(self, clearness):
        fitted = fit_distribution("hollands_huget", clearness)
        assert fitted.params["kt_max"] == pytest.approx(float(np.percentile(clearness, 99.5)))

    def test_kt_max_can_be_pinned(self, clearness):
        fitted = fit_distribution("hollands_huget", clearness, kt_max=0.86)
        assert fitted.params["kt_max"] == 0.86

    def test_near_zero_lambda_falls_back_to_the_triangular_limit(self):
        """The closed forms are 0/0 at lambda = 0 and lose every digit before it."""
        ceiling = 0.8
        grid = np.linspace(0.0, ceiling, 100_001)
        triangular = np.trapezoid(
            pdf(
                fit_distribution("hollands_huget", np.array([ceiling / 3.0] * 5), kt_max=ceiling),
                grid,
            ),
            grid,
        )
        assert float(triangular) == pytest.approx(1.0, abs=1e-6)

    def test_rejects_an_option_the_family_does_not_declare(self, clearness):
        with pytest.raises(ValueError, match="does not accept option"):
            fit_distribution("hollands_huget", clearness, kt_min=0.1)


class TestFitDistribution:
    def test_unknown_family_raises(self):
        with pytest.raises(KeyError, match="unknown distribution family"):
            fit_distribution("lognormal", np.array([1.0, 2.0]))

    def test_option_on_a_family_without_options_raises(self, temperature):
        with pytest.raises(ValueError, match="does not accept option"):
            fit_distribution("normal", temperature, kt_max=0.8)

    @pytest.mark.parametrize("family", SAMPLE_ONLY_FAMILIES)
    def test_empty_sample_yields_nan_parameters_not_an_exception(self, family):
        fitted = fit_distribution(family, np.array([]))
        assert fitted.n == 0
        assert all(np.isnan(value) for value in _flatten(fitted.params.values()))

    @pytest.mark.parametrize("family", SAMPLE_ONLY_FAMILIES)
    def test_n_counts_only_the_samples_actually_used(self, family):
        sample = np.concatenate([np.full(100, 0.5), np.full(7, np.nan)])
        assert fit_distribution(family, sample).n == 100

    @pytest.mark.parametrize("family", COVARIATE_FAMILIES)
    def test_a_family_that_needs_a_covariate_refuses_to_guess(self, family):
        """The induced radiation densities cannot be estimated from the sample.

        Their parameters are inherited from another fit and their mixture comes
        from a covariate, so a default would silently publish a curve fitted on
        something other than what the caption claims. Failing loudly is the
        contract; `VariableSpec.fit_options` makes the caller supply them.
        """
        with pytest.raises(TypeError, match="required keyword"):
            fit_distribution(family, np.array([1.0, 2.0]))


class TestGoodnessOfFit:
    def test_ks_distance_matches_scipy(self, temperature):
        fitted = fit_distribution("normal", temperature)
        quality = goodness_of_fit(fitted, temperature)
        reference = stats.ks_1samp(
            temperature, stats.norm(fitted.params["mu"], fitted.params["sigma"]).cdf
        ).statistic
        assert quality.ks_distance == pytest.approx(float(reference), rel=1e-9)

    def test_a_correct_model_scores_near_zero_distance(self, wind_speed):
        fitted = fit_distribution("weibull", wind_speed)
        quality = goodness_of_fit(fitted, wind_speed)
        assert quality.ks_distance < 0.01
        assert quality.quantile_gap < 0.02

    def test_a_wrong_family_scores_worse_than_the_right_one(self, wind_speed):
        right = goodness_of_fit(fit_distribution("weibull", wind_speed), wind_speed)
        wrong = goodness_of_fit(fit_distribution("normal", wind_speed), wind_speed)
        assert wrong.ks_distance > right.ks_distance

    def test_quantile_gap_is_in_the_variables_own_units(self, temperature):
        """Shifting the sample by a constant leaves the gap unchanged."""
        base = goodness_of_fit(fit_distribution("normal", temperature), temperature)
        shifted_sample = temperature + 100.0
        shifted = goodness_of_fit(fit_distribution("normal", shifted_sample), shifted_sample)
        assert shifted.quantile_gap == pytest.approx(base.quantile_gap, rel=1e-6)

    def test_density_r_squared_is_reported_only_with_a_histogram(self, temperature):
        fitted = fit_distribution("normal", temperature)
        without = goodness_of_fit(fitted, temperature)
        binned = histogram(temperature, np.arange(14.0, 40.01, 0.5))
        with_bins = goodness_of_fit(fitted, temperature, binned=binned)
        assert np.isnan(without.density_r_squared)
        assert with_bins.density_r_squared > 0.99

    def test_too_few_samples_yield_nan_distances(self):
        fitted = fit_distribution("normal", np.array([1.0, 2.0, 3.0]))
        quality = goodness_of_fit(fitted, np.array([1.0]))
        assert np.isnan(quality.ks_distance)
        assert np.isnan(quality.quantile_gap)


class TestEffectiveSampleSize:
    def test_independent_samples_keep_their_size(self):
        generator = np.random.default_rng(5)
        sample = generator.normal(size=50_000)
        n_effective, lag1 = effective_sample_size(sample)
        assert abs(lag1) < 0.02
        assert n_effective == pytest.approx(50_000, rel=0.05)

    def test_autocorrelated_samples_lose_most_of_their_size(self):
        """An hourly meteorological series is the case this correction exists for."""
        generator = np.random.default_rng(6)
        noise = generator.normal(size=50_000)
        series = np.zeros_like(noise)
        for index in range(1, series.size):
            series[index] = 0.95 * series[index - 1] + noise[index]
        n_effective, lag1 = effective_sample_size(series)
        assert lag1 > 0.9
        assert n_effective < 0.1 * series.size

    def test_short_or_degenerate_input_is_not_a_crash(self):
        assert effective_sample_size(np.array([1.0]))[0] == 1.0
        assert np.isnan(effective_sample_size(np.ones(10))[1])


class TestCircularStatistics:
    def test_mean_direction_survives_the_zero_wrap(self):
        """The arithmetic mean of 355 and 5 is 180 — the opposite direction."""
        summary = circular_summary(np.array([355.0, 5.0]))
        assert summary["mean_direction"] == pytest.approx(0.0, abs=1e-9)

    def test_uniform_directions_have_no_resultant(self):
        summary = circular_summary(np.arange(0.0, 360.0, 1.0))
        assert summary["resultant_length"] == pytest.approx(0.0, abs=1e-12)
        assert summary["circular_variance"] == pytest.approx(1.0, abs=1e-12)

    def test_identical_directions_are_fully_concentrated(self):
        summary = circular_summary(np.full(100, 137.0))
        assert summary["mean_direction"] == pytest.approx(137.0)
        assert summary["resultant_length"] == pytest.approx(1.0)

    def test_empty_sample_is_all_nan(self):
        summary = circular_summary(np.array([]))
        assert summary["n"] == 0.0
        assert np.isnan(summary["mean_direction"])


class TestSectorFrequencies:
    def test_frequencies_sum_to_one(self):
        generator = np.random.default_rng(8)
        _centers, frequencies = sector_frequencies(generator.uniform(0.0, 360.0, 10_000))
        assert float(frequencies.sum()) == pytest.approx(1.0)

    def test_sectors_are_centred_on_the_compass_points(self):
        """348.75-11.25 deg is north; an off-by-half-sector rose splits the peak."""
        _centers, frequencies = sector_frequencies(np.array([355.0, 0.0, 5.0]), sectors=16)
        assert frequencies[0] == pytest.approx(1.0)

    def test_dominant_sector_matches_the_injected_direction(self):
        generator = np.random.default_rng(9)
        centers, frequencies = sector_frequencies(generator.normal(120.0, 3.0, 5_000), sectors=16)
        assert centers[int(np.argmax(frequencies))] == 112.5

    def test_angles_outside_zero_to_360_are_wrapped(self):
        _centers, frequencies = sector_frequencies(np.array([-5.0, 365.0]), sectors=16)
        assert frequencies[0] == pytest.approx(1.0)

    def test_empty_sample_gives_zero_frequencies(self):
        _centers, frequencies = sector_frequencies(np.array([]), sectors=16)
        assert frequencies.shape == (16,)
        assert not frequencies.any()

    def test_too_few_sectors_raises(self):
        with pytest.raises(ValueError, match="at least two sectors"):
            sector_frequencies(np.array([1.0]), sectors=1)


class TestVonMisesMixture:
    @pytest.fixture
    def bimodal_directions(self) -> np.ndarray:
        """Two modes, as a trade flow plus a sea-breeze veer would produce."""
        trade = stats.vonmises.rvs(kappa=6.0, loc=np.deg2rad(120.0), size=60_000, random_state=16)
        breeze = stats.vonmises.rvs(kappa=3.0, loc=np.deg2rad(200.0), size=40_000, random_state=17)
        return np.rad2deg(np.concatenate([trade, breeze])) % 360.0

    def test_recovers_both_components(self, bimodal_directions):
        mixture = fit_von_mises_mixture(bimodal_directions, components=2)
        assert mixture.weights[0] == pytest.approx(0.6, abs=0.02)
        assert mixture.mu_degrees[0] == pytest.approx(120.0, abs=1.0)
        assert mixture.mu_degrees[1] == pytest.approx(200.0, abs=2.0)
        assert mixture.kappa[0] == pytest.approx(6.0, rel=0.05)

    def test_components_are_sorted_by_descending_weight(self, bimodal_directions):
        mixture = fit_von_mises_mixture(bimodal_directions, components=3)
        assert list(mixture.weights) == sorted(mixture.weights, reverse=True)

    def test_is_reproducible_for_a_fixed_seed(self, bimodal_directions):
        first = fit_von_mises_mixture(bimodal_directions, components=2, seed=42)
        second = fit_von_mises_mixture(bimodal_directions, components=2, seed=42)
        assert first == second

    def test_single_component_matches_the_circular_summary(self):
        sample = np.rad2deg(
            stats.vonmises.rvs(kappa=4.0, loc=np.deg2rad(80.0), size=40_000, random_state=18)
        )
        mixture = fit_von_mises_mixture(sample % 360.0, components=1)
        summary = circular_summary(sample % 360.0)
        assert mixture.weights == (1.0,)
        assert mixture.mu_degrees[0] == pytest.approx(summary["mean_direction"], abs=0.5)

    def test_density_integrates_to_one_over_the_circle(self, bimodal_directions):
        mixture = fit_von_mises_mixture(bimodal_directions, components=2)
        grid = np.arange(0.0, 360.0, 0.05)
        integral = float(np.trapezoid(von_mises_mixture_pdf(mixture, grid), grid))
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_density_approximates_the_sector_frequencies(self, bimodal_directions):
        """pdf * sector width is what makes the curve overlay the rose."""
        mixture = fit_von_mises_mixture(bimodal_directions, components=2)
        centers, frequencies = sector_frequencies(bimodal_directions, sectors=36)
        modelled = von_mises_mixture_pdf(mixture, centers) * (360.0 / 36)
        assert float(np.max(np.abs(modelled - frequencies))) < 0.01

    def test_very_high_concentration_does_not_overflow(self):
        """i0e keeps the density finite where exp(kappa) would not."""
        mixture = fit_von_mises_mixture(np.full(1_000, 90.0) + 1e-6, components=1)
        assert np.isfinite(von_mises_mixture_pdf(mixture, np.array([90.0]))).all()

    def test_empty_sample_is_all_nan(self):
        mixture = fit_von_mises_mixture(np.array([]), components=2)
        assert mixture.n == 0
        assert all(np.isnan(weight) for weight in mixture.weights)

    def test_zero_components_raises(self):
        with pytest.raises(ValueError, match="at least one component"):
            fit_von_mises_mixture(np.array([1.0]), components=0)


class TestPublicShapes:
    def test_every_family_round_trips_cdf_and_ppf(self):
        """A quantile fed back through the CDF must return the probability it came from."""
        probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        samples = {
            "weibull": stats.weibull_min.rvs(1.9, scale=2.4, size=5_000, random_state=31),
            "gamma": stats.gamma.rvs(0.9, scale=2.0, size=5_000, random_state=32),
            "beta": stats.beta.rvs(3.0, 2.0, size=5_000, random_state=33),
            "normal": stats.norm.rvs(20.0, 3.0, size=5_000, random_state=34),
            "hollands_huget": stats.beta.rvs(2.0, 2.0, size=5_000, random_state=35) * 0.8,
        }
        for family, sample in samples.items():
            fitted = fit_distribution(family, sample)
            recovered = cdf(fitted, ppf(fitted, probabilities))
            np.testing.assert_allclose(recovered, probabilities, atol=1e-8, err_msg=family)
