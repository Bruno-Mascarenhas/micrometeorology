"""Tests for the induced radiation densities and the published bibliography.

Offline and synthetic. Three properties are checked for every new family, and
they are the ones that catch a wrong change of variable: the density integrates
to one, the derivative of the CDF equals the PDF, and the quantile function
inverts the CDF. A missing Jacobian breaks the first two and nothing else, which
is exactly why it is such an easy mistake to ship.
"""

import warnings

import numpy as np
import pytest
from scipy import integrate

from micrometeorology.stats import climatology_export
from micrometeorology.stats import distributions as dist
from micrometeorology.stats.climatology_export import (
    _REFERENCE_MARKER,
    CLIMATOLOGY_VARIABLES,
    REFERENCES,
    Reference,
)
from tests.micromet.conftest import required_fit_options

# A realistic extraterrestrial-irradiance mixture: sixty equal-weight bins over
# the range a tropical site actually sees at solar elevations above 10 degrees.
SCALES = np.linspace(200.0, 1390.0, 60)
WEIGHTS = np.full(60, 1.0 / 60.0)
PARENT = {"lam": 4.16238, "kt_max": 0.92717}


def _compound(**extra):
    return dist.fit_distribution(
        "compound_hollands_huget",
        np.array([300.0, 500.0]),
        scales=list(SCALES),
        weights=list(WEIGHTS),
        **PARENT,
        **extra,
    )


def _integrates_to_one(fit, low, high, tolerance=1e-5, *, kinks=()):
    """Integrate the density over its support and demand unit mass.

    The compound densities are piecewise — one kink per component, at
    ``kt_max * scale`` — and ``quad`` cannot find them on its own: left to
    itself it reports 'roundoff error is detected, which prevents the requested
    tolerance from being achieved' and its answer is no longer an oracle at the
    precision asserted here. The kinks are handed over in *points*, and the
    warning is promoted to an error so a future family cannot slip back into a
    result the integrator has disowned.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", integrate.IntegrationWarning)
        mass, _error = integrate.quad(
            lambda v: dist.pdf(fit, np.array([v]))[0],
            low,
            high,
            limit=500,
            points=sorted(point for point in kinks if low < point < high) or None,
        )
    assert mass == pytest.approx(1.0, abs=tolerance)


def _derivative_matches_density(fit, low, high, step, tolerance=1e-5):
    grid = np.linspace(low, high, 2000)
    numeric = (dist.cdf(fit, grid + step) - dist.cdf(fit, grid - step)) / (2.0 * step)
    assert np.abs(numeric - dist.pdf(fit, grid)).max() < tolerance


def _quantiles_invert(fit, tolerance=1e-6):
    probabilities = np.linspace(0.001, 0.999, 200)
    assert np.abs(dist.cdf(fit, dist.ppf(fit, probabilities)) - probabilities).max() < tolerance


class TestCompoundHollandsHuget:
    def test_is_a_proper_density(self):
        fit = _compound()
        ceiling = PARENT["kt_max"] * SCALES.max()
        _integrates_to_one(fit, 0.0, ceiling, kinks=PARENT["kt_max"] * SCALES)
        _derivative_matches_density(fit, 1.0, ceiling * 0.999, 1e-4 * ceiling)
        _quantiles_invert(fit)

    def test_inherits_rather_than_estimates(self):
        """The sample must not move the parameters: that is the family's claim."""
        one = _compound()
        other = dist.fit_distribution(
            "compound_hollands_huget",
            np.array([10.0, 20.0, 30.0, 900.0]),
            scales=list(SCALES),
            weights=list(WEIGHTS),
            **PARENT,
        )
        assert one.params["lambda"] == other.params["lambda"]
        assert one.params["kt_max"] == other.params["kt_max"]

    def test_gain_rescales_the_support_and_nothing_else(self):
        """Albedo and PAR fraction enter as one scalar; the shape is untouched."""
        plain = _compound()
        scaled = _compound(gain=0.25)
        ceiling = PARENT["kt_max"] * SCALES.max()
        probabilities = np.linspace(0.05, 0.95, 40)
        np.testing.assert_allclose(
            dist.ppf(scaled, probabilities), 0.25 * dist.ppf(plain, probabilities), rtol=1e-6
        )
        _integrates_to_one(scaled, 0.0, 0.25 * ceiling, kinks=0.25 * PARENT["kt_max"] * SCALES)

    def test_binning_the_covariate_is_not_a_material_approximation(self):
        """Sixty bins against one component per observed hour."""
        generator = np.random.default_rng(7)
        observed = generator.uniform(200.0, 1390.0, 5000)
        exact = dist.fit_distribution(
            "compound_hollands_huget",
            np.array([1.0]),
            scales=list(observed),
            weights=list(np.full(observed.size, 1.0 / observed.size)),
            **PARENT,
        )
        order = np.sort(observed)
        groups = np.array_split(order, 60)
        binned = dist.fit_distribution(
            "compound_hollands_huget",
            np.array([1.0]),
            scales=[float(group.mean()) for group in groups],
            weights=[group.size / observed.size for group in groups],
            **PARENT,
        )
        grid = np.linspace(0.0, PARENT["kt_max"] * observed.max(), 400)
        assert np.abs(dist.cdf(exact, grid) - dist.cdf(binned, grid)).max() < 1e-3


class TestPowerNormalMixture:
    @pytest.fixture
    def sample(self):
        generator = np.random.default_rng(11)
        temperature = np.concatenate(
            [generator.normal(298.06, 2.10, 6000), generator.normal(305.17, 4.30, 3800)]
        )
        return dist.STEFAN_BOLTZMANN * temperature**4

    def test_is_a_proper_density(self, sample):
        fit = dist.fit_distribution("power_normal_mixture", sample, components=2)
        _integrates_to_one(fit, 300.0, 900.0)
        _derivative_matches_density(fit, 330.0, 700.0, 1e-3)
        _quantiles_invert(fit)

    def test_recovers_the_two_planted_regimes(self, sample):
        """Blind EM must find the components, not merely fit something."""
        fit = dist.fit_distribution("power_normal_mixture", sample, components=2)
        assert fit.params["mu"][0] == pytest.approx(298.06, abs=0.4)
        assert fit.params["mu"][1] == pytest.approx(305.17, abs=0.6)

    def test_the_cdf_carries_no_jacobian(self, sample):
        """The map is monotone, so P(L < l) is exactly P(T < T(l))."""
        fit = dist.fit_distribution("power_normal_mixture", sample, components=2)
        flux = np.array([380.0, 450.0, 520.0])
        empirical = np.array([(sample < value).mean() for value in flux])
        assert np.abs(dist.cdf(fit, flux) - empirical).max() < 0.03


class TestCompoundHollandsGaussian:
    def _fit(self):
        return dist.fit_distribution(
            "compound_hollands_gaussian",
            np.array([100.0]),
            scales=list(SCALES),
            weights=list(WEIGHTS),
            slope=0.7752,
            intercept=-24.389,
            residual_sd=36.245,
            **PARENT,
        )

    def test_is_a_proper_density(self):
        fit = self._fit()
        _integrates_to_one(fit, -400.0, 1500.0)
        _derivative_matches_density(fit, -200.0, 1200.0, 1e-2)
        _quantiles_invert(fit, tolerance=1e-4)

    def test_supports_negative_values(self):
        """Net radiation is negative near the daylight gate's edges."""
        fit = self._fit()
        assert dist.pdf(fit, np.array([-60.0]))[0] > 0.0

    def test_quadrature_is_stable(self):
        """The published number must not be an artefact of the node count."""
        fit = self._fit()
        grid = np.linspace(-100.0, 1000.0, 50)
        assert np.all(np.diff(dist.cdf(fit, grid)) >= -1e-12)


class TestPublishedBibliography:
    def test_a_record_that_cites_itself_is_refused(self, monkeypatch):
        """The failure mode a global rename of the prose citations once caused.

        The check runs at import, so the shipped records are stood aside and a
        self-citing one is put in their place.
        """
        monkeypatch.setattr(climatology_export, "CLIMATOLOGY_VARIABLES", ())
        monkeypatch.setattr(
            climatology_export,
            "REFERENCES",
            {
                "selfcite": Reference(
                    key="selfcite",
                    short="[[selfcite]]",
                    citation="A record whose own prose carries its marker.",
                    url="https://doi.org/10.1000/selfcite",
                )
            },
        )

        with pytest.raises(ValueError, match=r"contain a \[\[marker\]\] of their own"):
            climatology_export._assert_references()

    def test_every_declared_reference_is_actually_cited(self):
        """A record nobody cites is a record nobody maintains."""
        cited: set[str] = set()
        for spec in CLIMATOLOGY_VARIABLES:
            for text in (spec.family_label, *spec.caveats):
                cited |= set(_REFERENCE_MARKER.findall(text))
        assert set(REFERENCES) == cited

    def test_every_record_is_complete_and_resolvable(self):
        for key, reference in REFERENCES.items():
            assert reference.short.strip(), key
            # Long enough to actually be a record rather than another short form.
            assert len(reference.citation) > 60, key
            assert reference.url.startswith("https://"), key

    def test_every_variable_declares_a_family_and_a_reference(self):
        """The requirement this work exists to satisfy: nothing merely descriptive."""
        for spec in CLIMATOLOGY_VARIABLES:
            assert spec.family is not None, spec.id
            assert _REFERENCE_MARKER.search(spec.family_label), spec.id

    def test_induced_specs_declare_the_options_their_family_needs(self):
        """A spec that forgets one would fit a curve on the wrong covariate, and
        the omission surfaces as a bare ``TypeError`` out of the estimator rather
        than as a named failure. Skipping the spec with no options at all would
        skip exactly that case, so the check runs on every spec whose family
        needs a covariate.
        """
        for spec in CLIMATOLOGY_VARIABLES:
            # The rose is fitted by `fit_von_mises_mixture` directly and never
            # enters the FAMILIES registry, so it declares no options by design.
            if spec.family is None or spec.family not in dist.FAMILIES:
                continue
            required = required_fit_options(spec.family)
            declared = set(dist.FAMILIES[spec.family].options)
            assert required <= set(spec.fit_options), spec.id
            assert set(spec.fit_options) <= declared, spec.id
