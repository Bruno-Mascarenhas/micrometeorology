r"""Empirical histograms and the theoretical distributions the literature pairs them with.

This module answers one question: *does an observed variable follow the law the
literature assigns to it?* It fits the canonical family by maximum likelihood,
evaluates its density on a **frozen** bin edge set, and reports how far the fit
sits from the data — as distances, never as p-values.

Families and where they come from
---------------------------------
- ``weibull``  — wind speed. Justus et al. (1978), *J. Appl. Meteorol.* 17, 350.
- ``gamma``    — wet-hour precipitation intensity. Thom (1958), *Mon. Weather
  Rev.* 86, 117; Wilks (2019), *Statistical Methods in the Atmospheric
  Sciences*, 4th ed., ch. 4.
- ``beta``     — relative humidity scaled to ``[0, 1]``. Raschke (2011),
  *Stoch. Environ. Res. Risk Assess.* 25, 79.
- ``normal``   — air temperature and station pressure. The reference model in
  Wilks (2019) ch. 4; expect visible departure, because pooled hourly values are
  a mixture over hour-of-day and season.
- ``hollands_huget`` — clearness index ``Kt`` on bounded support. Hollands &
  Huget (1983), *Solar Energy* 30, 195, reproducing the Liu & Jordan (1960)
  frequency curves.
- von Mises mixture — wind direction. Carta, Bueno & Ramirez (2008), *Energy
  Convers. Manag.* 49, 897. Kept out of :data:`FAMILIES` because direction is
  circular: it has no quantile function on the line and belongs on a rose, not
  a histogram.

Why no p-values
---------------
:func:`goodness_of_fit` deliberately returns distances only. At the sample sizes
a decade of hourly observations produces, Kolmogorov-Smirnov, Anderson-Darling
and chi-square are all *consistent* tests, so they reject any model that is not
literally true; hourly series are also strongly autocorrelated, which invalidates
their i.i.d. null anyway, and the parameters were estimated from the same sample
(the Lilliefors problem). The honest quantities are how far the fit is — the KS
distance, the mean quantile gap in the variable's own units, and the density
R-squared — reported next to ``n`` and the autocorrelation-corrected
``n_effective`` so a reader can judge them.

Relationship to the neighbouring modules (do **not** merge these)
-----------------------------------------------------------------
- :mod:`micrometeorology.stats.climatology` groups a record *by time* (diurnal,
  monthly, seasonal). This module ignores time entirely and describes the
  marginal distribution of the values it is handed; the two compose, with
  ``seasonal_groups`` selecting the subset that is fitted here.
- :mod:`micrometeorology.stats.metrics` scores *paired* model-versus-observation
  arrays. Nothing here is paired: a fit compares one sample against a
  parametric family.
- :mod:`micrometeorology.stats.radiation` produces the ``Kt`` and ``Kd`` series
  that :func:`fit_distribution` then fits with ``hollands_huget``.

No plotting lives here, mirroring the ``comparison`` / ``comparison_plots``
split: this module is pure NumPy/SciPy and is safe to import from a CLI that
must not pay the matplotlib import.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import special

logger = logging.getLogger(__name__)

__all__ = [
    "FAMILIES",
    "DistributionFit",
    "FitQuality",
    "Histogram",
    "VonMisesMixture",
    "cdf",
    "circular_summary",
    "effective_sample_size",
    "fit_distribution",
    "fit_von_mises_mixture",
    "goodness_of_fit",
    "histogram",
    "pdf",
    "ppf",
    "sector_frequencies",
    "von_mises_mixture_pdf",
]

# Newton/bisection budget for the three implicit maximum-likelihood equations
# (Weibull shape, gamma shape, beta pair). Every one of them converges in well
# under ten iterations from its method-of-moments start; the cap only exists so
# a pathological sample cannot spin.
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-10

# Shape parameters are clamped to this range while iterating. A sample of
# identical values drives the Weibull shape to infinity, which then overflows
# the power terms and returns NaN for an input that is merely degenerate.
_MIN_SHAPE = 1e-3
_MAX_SHAPE = 1e4

# Bracket for the Hollands-Huget lambda. The density is monotone in lambda and
# the physically meaningful range is roughly [-20, 20]; the wider bracket costs
# a few bisection steps and never binds on real data.
_LAMBDA_BRACKET = (-60.0, 60.0)


@dataclass(frozen=True)
class Histogram:
    """Counts and density of a sample on a frozen bin edge set.

    The edges are an input, never derived from the sample: the whole point of a
    climatology page is comparing subsets (full record, summer, winter, model)
    on one axis, and a per-subset Freedman-Diaconis width makes those bars
    incomparable. Anything outside the edges is counted in ``below``/``above``
    rather than silently dropped, so a reader can see that the axis is clipped.

    Attributes
    ----------
    edges:
        Bin boundaries, length ``len(counts) + 1``, strictly increasing.
    counts:
        Samples in each bin. The last bin is closed on both sides, matching
        :func:`numpy.histogram`.
    density:
        ``counts / (n * width)`` — integrates to 1 over the covered range when
        nothing overflowed, so a fitted PDF can be drawn on the same axis.
    n:
        Finite samples that landed inside the edges (the density denominator).
    below, above:
        Finite samples that fell under the first edge or over the last.
    """

    edges: NDArray
    counts: NDArray
    density: NDArray
    n: int
    below: int
    above: int

    @property
    def centers(self) -> NDArray:
        """Mid-point of each bin, the abscissa a fitted PDF is evaluated on."""
        midpoints: NDArray = 0.5 * (self.edges[:-1] + self.edges[1:])
        return midpoints


@dataclass(frozen=True)
class DistributionFit:
    """A fitted family plus the sample it was estimated from.

    ``params`` carries the natural parameters of the family and is what gets
    serialised: ``{"shape", "scale"}`` for Weibull and gamma, ``{"alpha",
    "beta"}`` for beta, ``{"mu", "sigma"}`` for normal, ``{"lambda", "kt_max"}``
    for Hollands-Huget.
    """

    family: str
    params: dict[str, float]
    n: int


@dataclass(frozen=True)
class FitQuality:
    """How far a fit sits from its sample. Distances only — see the module docstring.

    Attributes
    ----------
    ks_distance:
        ``sup |F_n(x) - F(x)|``, the largest cumulative discrepancy. Reads
        directly as "at worst the model misplaces this share of the record".
    quantile_gap:
        Mean absolute distance between the ordered sample and the fitted
        quantiles, **in the variable's own units** (m/s, degC, hPa). The most
        physically legible number on the page.
    density_r_squared:
        Coefficient of determination between the binned empirical density and
        the fitted PDF at the bin centres. Depends on the binning, which is why
        it is reported alongside the two binning-free distances and never alone.
        ``NaN`` when no histogram was supplied.
    n:
        Samples the fit and the distances were computed from.
    n_effective:
        ``n`` corrected for serial correlation (see
        :func:`effective_sample_size`). Printing it next to ``n`` is the single
        most honest element of a goodness-of-fit report on hourly data.
    lag1_autocorrelation:
        The correlation the correction used.
    """

    ks_distance: float
    quantile_gap: float
    density_r_squared: float
    n: int
    n_effective: float
    lag1_autocorrelation: float


@dataclass(frozen=True)
class VonMisesMixture:
    """A finite mixture of von Mises components on the circle.

    Attributes
    ----------
    weights:
        Mixture weights, summing to 1.
    mu_degrees:
        Component mean directions in ``[0, 360)``, meteorological convention
        (the direction the wind blows *from*).
    kappa:
        Component concentrations. Larger is tighter; ``kappa = 0`` is uniform.
    n:
        Angles the mixture was estimated from.
    """

    weights: tuple[float, ...]
    mu_degrees: tuple[float, ...]
    kappa: tuple[float, ...]
    n: int


@dataclass(frozen=True)
class Family:
    """One entry of :data:`FAMILIES`: how to fit a law and how to evaluate it.

    All three evaluators take ``(x, params)`` so the registry can be driven from
    a family name read out of a configuration file, the same way
    :data:`micrometeorology.stats.metrics.ALL_METRICS` is.
    """

    name: str
    # `...` rather than `[NDArray]`: hollands_huget takes an extra keyword-only
    # option, and a precise signature here would reject it at the registry.
    fit: Callable[..., dict[str, float]]
    pdf: Callable[[NDArray, Mapping[str, float]], NDArray]
    cdf: Callable[[NDArray, Mapping[str, float]], NDArray]
    ppf: Callable[[NDArray, Mapping[str, float]], NDArray]
    # Samples outside this open interval cannot be represented by the family and
    # are dropped before fitting (wind calms for Weibull, dry hours for gamma,
    # the saturation pile-up for beta). The caller reports the removed mass as a
    # separate atom; silently fitting over it is what pulls a Weibull shape down.
    support: tuple[float, float] = (-np.inf, np.inf)
    # Extra keyword arguments the estimator accepts, for families whose support
    # is data-dependent (Hollands-Huget needs kt_max).
    options: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Sample preparation
# ---------------------------------------------------------------------------


def _clean(values: NDArray) -> NDArray:
    """Flatten to 1-D float and drop every non-finite entry.

    ``+/-inf`` has to go with the NaNs for the same reason it does in
    :mod:`~micrometeorology.stats.metrics`: it survives an ``isnan`` filter and
    then makes every moment infinite, so one overflowed sample would destroy the
    fit of an otherwise usable record.
    """
    sample = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(sample)
    if finite.all():
        return sample
    logger.debug("dropped %d non-finite samples of %d", int((~finite).sum()), finite.size)
    return sample[finite]


def _wrap_degrees(degrees: float) -> float:
    """Fold an angle into ``[0, 360)``, never onto the excluded 360 endpoint.

    A bare ``% 360.0`` is not enough. ``arctan2`` returns a tiny negative angle
    for a direction that is exactly north, and ``-5.7e-15 % 360.0`` rounds to
    exactly ``360.0`` in float64 — so a due-north mean direction would be
    published as "360 deg", outside the range this module documents and outside
    what a compass rose can place.
    """
    wrapped = float(degrees) % 360.0
    return 0.0 if wrapped >= 360.0 else wrapped


def _within_support(sample: NDArray, support: tuple[float, float]) -> NDArray:
    """Keep the samples a family can actually represent (open interval)."""
    low, high = support
    inside = (sample > low) & (sample < high)
    if inside.all():
        return sample
    logger.debug("dropped %d samples outside %s", int((~inside).sum()), support)
    return sample[inside]


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


def histogram(values: NDArray, edges: Sequence[float] | NDArray) -> Histogram:
    """Bin a sample on a frozen edge set and normalise to density.

    Parameters
    ----------
    values:
        Sample; non-finite entries are dropped.
    edges:
        Strictly increasing bin boundaries, at least two of them.

    Returns
    -------
    Histogram
        Counts, density and the under/overflow tallies.

    Raises
    ------
    ValueError
        If fewer than two edges are given or they are not strictly increasing —
        both are authoring mistakes in a frozen edge set, and letting them
        through would produce a plausible-looking but meaningless density.
    """
    bounds = np.asarray(edges, dtype=float)
    if bounds.size < 2:
        raise ValueError(f"histogram needs at least two edges, got {bounds.size}")
    if not np.all(np.diff(bounds) > 0):
        raise ValueError("histogram edges must be strictly increasing")

    sample = _clean(values)
    below = int(np.count_nonzero(sample < bounds[0]))
    above = int(np.count_nonzero(sample > bounds[-1]))
    counts, _ = np.histogram(sample, bins=bounds)
    inside = int(counts.sum())

    widths = np.diff(bounds)
    density = np.zeros_like(widths)
    if inside > 0:
        density = counts / (inside * widths)
    return Histogram(
        edges=bounds,
        counts=counts.astype(np.int64),
        density=density,
        n=inside,
        below=below,
        above=above,
    )


# ---------------------------------------------------------------------------
# Weibull (wind speed)
# ---------------------------------------------------------------------------


def _weibull_shape(log_sample: NDArray, start: float) -> float:
    r"""Solve the Weibull shape likelihood equation by damped Newton.

    Formula
    -------
    The log-likelihood derivative vanishes at

    .. math::
        g(k) = \frac{\sum x_i^k \ln x_i}{\sum x_i^k} - \frac{1}{k}
               - \overline{\ln x} = 0

    which is monotone decreasing in ``k``. Everything is evaluated as
    ``exp(k ln x - max)`` rather than ``x**k``: at k around 10 a 40 m/s gust
    overflows float64, and the ratio the equation needs is unchanged by the
    shift.
    """
    shape = float(np.clip(start, _MIN_SHAPE, _MAX_SHAPE))
    mean_log = float(log_sample.mean())
    for _iteration in range(_MAX_ITERATIONS):
        scaled = shape * log_sample
        weights = np.exp(scaled - scaled.max())
        total = float(weights.sum())
        first = float((weights * log_sample).sum()) / total
        second = float((weights * log_sample * log_sample).sum()) / total
        residual = first - 1.0 / shape - mean_log
        derivative = second - first * first + 1.0 / (shape * shape)
        if derivative <= 0.0:
            break
        step = residual / derivative
        # Halving keeps the iterate positive: an undamped Newton step from a
        # near-degenerate sample can jump straight through zero into k < 0,
        # where the likelihood is undefined.
        while shape - step <= _MIN_SHAPE:
            step *= 0.5
        shape -= step
        if abs(step) < _TOLERANCE * max(1.0, shape):
            break
    return float(np.clip(shape, _MIN_SHAPE, _MAX_SHAPE))


def _fit_weibull(values: NDArray) -> dict[str, float]:
    """Two-parameter Weibull by maximum likelihood, location fixed at zero.

    The Justus moment estimator ``k = (s / v_bar) ** -1.086`` seeds the Newton
    iteration; the scale then follows in closed form from the fitted shape.
    Calms must already be removed — the family has no atom at zero, and fitting
    over one biases the shape low, which is the classic error in a published
    wind distribution.
    """
    sample = _clean(values)
    sample = sample[sample > 0.0]
    if sample.size < 2:
        return {"shape": float("nan"), "scale": float("nan")}

    mean = float(sample.mean())
    std = float(sample.std(ddof=1))
    start = (std / mean) ** -1.086 if mean > 0.0 and std > 0.0 else 2.0
    shape = _weibull_shape(np.log(sample), start)
    scale = float(np.mean(sample**shape) ** (1.0 / shape))
    return {"shape": shape, "scale": scale}


def _weibull_pdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    values = np.asarray(x, dtype=float)
    reduced = np.where(values > 0.0, values / scale, np.nan)
    density: NDArray = (shape / scale) * reduced ** (shape - 1.0) * np.exp(-(reduced**shape))
    return np.where(values > 0.0, density, 0.0)


def _weibull_cdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    values = np.asarray(x, dtype=float)
    reduced = np.where(values > 0.0, values / scale, 0.0)
    cumulative: NDArray = -np.expm1(-(reduced**shape))
    return np.where(values > 0.0, cumulative, 0.0)


def _weibull_ppf(q: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    quantiles = np.asarray(q, dtype=float)
    inverse: NDArray = scale * (-np.log1p(-quantiles)) ** (1.0 / shape)
    return inverse


# ---------------------------------------------------------------------------
# Gamma (wet-hour precipitation intensity)
# ---------------------------------------------------------------------------


def _fit_gamma(values: NDArray) -> dict[str, float]:
    r"""Two-parameter gamma by maximum likelihood, seeded by the Thom estimator.

    Formula
    -------
    With :math:`A = \ln\bar{x} - \overline{\ln x}`, Thom (1958) gives
    :math:`k_0 = (1 + \sqrt{1 + 4A/3}) / (4A)`, refined here by Newton on
    :math:`\ln k - \psi(k) = A` (derivative :math:`1/k - \psi'(k)`), and the
    scale follows as :math:`\theta = \bar{x} / k`.

    Only strictly positive values may be passed: the dry hours are an atom the
    caller reports separately, and a zero makes ``ln x`` diverge.
    """
    sample = _clean(values)
    sample = sample[sample > 0.0]
    if sample.size < 2:
        return {"shape": float("nan"), "scale": float("nan")}

    mean = float(sample.mean())
    disparity = float(np.log(mean) - np.mean(np.log(sample)))
    if disparity <= 0.0:
        # Jensen's inequality makes A >= 0 exactly, with equality only when every
        # value is identical; a floating-point zero here means a degenerate
        # sample, not a shape of infinity.
        return {"shape": float("nan"), "scale": float("nan")}

    shape = (1.0 + np.sqrt(1.0 + 4.0 * disparity / 3.0)) / (4.0 * disparity)
    for _iteration in range(_MAX_ITERATIONS):
        residual = float(np.log(shape) - special.digamma(shape) - disparity)
        derivative = float(1.0 / shape - special.polygamma(1, shape))
        if derivative == 0.0:
            break
        step = residual / derivative
        while shape - step <= _MIN_SHAPE:
            step *= 0.5
        shape -= step
        if abs(step) < _TOLERANCE * max(1.0, shape):
            break

    shape = float(np.clip(shape, _MIN_SHAPE, _MAX_SHAPE))
    return {"shape": shape, "scale": mean / shape}


def _gamma_pdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    values = np.asarray(x, dtype=float)
    reduced = np.where(values > 0.0, values / scale, np.nan)
    log_density = (shape - 1.0) * np.log(reduced) - reduced - special.gammaln(shape) - np.log(scale)
    return np.where(values > 0.0, np.exp(log_density), 0.0)


def _gamma_cdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    values = np.asarray(x, dtype=float)
    cumulative: NDArray = special.gammainc(shape, np.maximum(values, 0.0) / scale)
    return cumulative


def _gamma_ppf(q: NDArray, params: Mapping[str, float]) -> NDArray:
    shape, scale = params["shape"], params["scale"]
    inverse: NDArray = scale * special.gammaincinv(shape, np.asarray(q, dtype=float))
    return inverse


# ---------------------------------------------------------------------------
# Beta (relative humidity on [0, 1])
# ---------------------------------------------------------------------------


def _fit_beta(values: NDArray) -> dict[str, float]:
    r"""Beta by maximum likelihood on the open interval, via two-dimensional Newton.

    Formula
    -------
    The likelihood equations are
    :math:`\psi(\alpha) - \psi(\alpha+\beta) = \overline{\ln x}` and
    :math:`\psi(\beta) - \psi(\alpha+\beta) = \overline{\ln(1-x)}`, solved from
    the method-of-moments start
    :math:`\alpha_0 = \bar{x}\,(\bar{x}(1-\bar{x})/s^2 - 1)`.

    Samples at exactly 0 or 1 are dropped: the density diverges there, and on a
    humid coast the 100 % saturation pile-up is a sensor clip that belongs in a
    separately reported point mass, not in the continuous fit.
    """
    sample = _clean(values)
    sample = sample[(sample > 0.0) & (sample < 1.0)]
    if sample.size < 2:
        return {"alpha": float("nan"), "beta": float("nan")}

    mean = float(sample.mean())
    variance = float(sample.var(ddof=1))
    if variance <= 0.0:
        return {"alpha": float("nan"), "beta": float("nan")}
    common = mean * (1.0 - mean) / variance - 1.0
    if common <= 0.0:
        # An over-dispersed sample (variance above that of the uniform) has no
        # unimodal beta start; the moment estimator would hand Newton negative
        # parameters, so fall back to the uniform.
        common = 1.0
    alpha = max(mean * common, _MIN_SHAPE)
    beta = max((1.0 - mean) * common, _MIN_SHAPE)

    mean_log = float(np.mean(np.log(sample)))
    mean_log_complement = float(np.mean(np.log1p(-sample)))
    for _iteration in range(_MAX_ITERATIONS):
        digamma_total = float(special.digamma(alpha + beta))
        residual = np.array(
            [
                float(special.digamma(alpha)) - digamma_total - mean_log,
                float(special.digamma(beta)) - digamma_total - mean_log_complement,
            ]
        )
        trigamma_total = float(special.polygamma(1, alpha + beta))
        jacobian = np.array(
            [
                [float(special.polygamma(1, alpha)) - trigamma_total, -trigamma_total],
                [-trigamma_total, float(special.polygamma(1, beta)) - trigamma_total],
            ]
        )
        determinant = float(np.linalg.det(jacobian))
        if determinant == 0.0:
            break
        step = np.linalg.solve(jacobian, residual)
        # One shared damping factor for both parameters: scaling them
        # independently would walk off the Newton direction and stall.
        damping = 1.0
        while (alpha - damping * step[0] <= _MIN_SHAPE) or (beta - damping * step[1] <= _MIN_SHAPE):
            damping *= 0.5
            if damping < _TOLERANCE:
                break
        alpha -= damping * step[0]
        beta -= damping * step[1]
        if float(np.max(np.abs(damping * step))) < _TOLERANCE * max(1.0, alpha, beta):
            break

    return {"alpha": float(alpha), "beta": float(beta)}


def _beta_pdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    alpha, beta = params["alpha"], params["beta"]
    values = np.asarray(x, dtype=float)
    inside = (values > 0.0) & (values < 1.0)
    safe = np.where(inside, values, 0.5)
    log_density = (
        (alpha - 1.0) * np.log(safe)
        + (beta - 1.0) * np.log1p(-safe)
        - float(special.betaln(alpha, beta))
    )
    return np.where(inside, np.exp(log_density), 0.0)


def _beta_cdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    alpha, beta = params["alpha"], params["beta"]
    clipped = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    cumulative: NDArray = special.betainc(alpha, beta, clipped)
    return cumulative


def _beta_ppf(q: NDArray, params: Mapping[str, float]) -> NDArray:
    alpha, beta = params["alpha"], params["beta"]
    inverse: NDArray = special.betaincinv(alpha, beta, np.asarray(q, dtype=float))
    return inverse


# ---------------------------------------------------------------------------
# Normal (air temperature, station pressure)
# ---------------------------------------------------------------------------


def _fit_normal(values: NDArray) -> dict[str, float]:
    """Gaussian by maximum likelihood, reported with the unbiased spread.

    ``ddof=1`` is the display convention: at the sample sizes this module sees
    it is indistinguishable from the ``1/n`` maximum-likelihood variance, and it
    is the number the descriptive statistics beside it already quote.
    """
    sample = _clean(values)
    if sample.size < 2:
        return {"mu": float("nan"), "sigma": float("nan")}
    return {"mu": float(sample.mean()), "sigma": float(sample.std(ddof=1))}


def _normal_pdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    mu, sigma = params["mu"], params["sigma"]
    reduced = (np.asarray(x, dtype=float) - mu) / sigma
    density: NDArray = np.exp(-0.5 * reduced * reduced) / (sigma * np.sqrt(2.0 * np.pi))
    return density


def _normal_cdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    mu, sigma = params["mu"], params["sigma"]
    cumulative: NDArray = special.ndtr((np.asarray(x, dtype=float) - mu) / sigma)
    return cumulative


def _normal_ppf(q: NDArray, params: Mapping[str, float]) -> NDArray:
    mu, sigma = params["mu"], params["sigma"]
    inverse: NDArray = mu + sigma * special.ndtri(np.asarray(q, dtype=float))
    return inverse


# ---------------------------------------------------------------------------
# Hollands-Huget (clearness index)
# ---------------------------------------------------------------------------


def _hollands_moments(lam: float, kt_max: float) -> tuple[float, float]:
    r"""Return ``(normalising constant C, distribution mean)`` for one lambda.

    Formula
    -------
    With :math:`f(k_t) = C\,(1 - k_t/M)\,e^{\lambda k_t}` on :math:`[0, M]`,
    integrating gives :math:`C = M\lambda^2 / (e^{\lambda M} - 1 - \lambda M)`
    and the mean follows from :math:`I_n = \int_0^M k_t^n e^{\lambda k_t}
    \mathrm{d}k_t` as :math:`C\,(I_1 - I_2/M)`.

    The :math:`|\lambda M| < 10^{-4}` branch is not cosmetic: both expressions
    are :math:`0/0` at :math:`\lambda = 0`, and in float64 the cancellation in
    ``e^{x} - 1 - x`` has already destroyed every significant digit well before
    the exact singularity. The limit is the triangular density, mean ``M/3``.
    """
    scaled = lam * kt_max
    if abs(scaled) < 1e-4:
        return 2.0 / kt_max, kt_max / 3.0

    exponential = np.exp(scaled)
    constant = kt_max * lam * lam / (exponential - 1.0 - scaled)
    first_moment = (kt_max * exponential) / lam - (exponential - 1.0) / (lam * lam)
    second_moment = (kt_max * kt_max * exponential) / lam - 2.0 * first_moment / lam
    return float(constant), float(constant * (first_moment - second_moment / kt_max))


def _fit_hollands_huget(values: NDArray, *, kt_max: float | None = None) -> dict[str, float]:
    """Hollands-Huget clearness-index density, matched on the mean by bisection.

    The upper bound ``kt_max`` is a site property, not a universal constant. When
    it is not supplied it is taken as the 99.5th percentile of the sample, which
    tracks the local clear-sky ceiling instead of imposing a textbook 0.86 that
    a humid tropical coast never reaches.

    The single free parameter is fixed by matching the distribution mean to the
    observed mean; the mean is strictly increasing in ``lambda``, so bisection
    on the bracket is unconditionally stable where a Newton step is not.
    """
    sample = _clean(values)
    sample = sample[(sample >= 0.0) & np.isfinite(sample)]
    if sample.size < 2:
        return {"lambda": float("nan"), "kt_max": float("nan")}

    ceiling = float(np.percentile(sample, 99.5)) if kt_max is None else float(kt_max)
    if not np.isfinite(ceiling) or ceiling <= 0.0:
        return {"lambda": float("nan"), "kt_max": float("nan")}
    sample = sample[sample < ceiling]
    if sample.size < 2:
        return {"lambda": float("nan"), "kt_max": ceiling}

    target = float(sample.mean())
    low, high = _LAMBDA_BRACKET
    for _iteration in range(_MAX_ITERATIONS):
        middle = 0.5 * (low + high)
        _constant, mean = _hollands_moments(middle, ceiling)
        if mean < target:
            low = middle
        else:
            high = middle
        if high - low < _TOLERANCE:
            break
    return {"lambda": 0.5 * (low + high), "kt_max": ceiling}


def _hollands_pdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    lam, kt_max = params["lambda"], params["kt_max"]
    values = np.asarray(x, dtype=float)
    constant, _mean = _hollands_moments(lam, kt_max)
    inside = (values >= 0.0) & (values <= kt_max)
    safe = np.where(inside, values, 0.0)
    density = constant * (1.0 - safe / kt_max) * np.exp(lam * safe)
    return np.where(inside, density, 0.0)


def _hollands_cdf(x: NDArray, params: Mapping[str, float]) -> NDArray:
    r"""Closed-form CDF of the Hollands-Huget density.

    Formula
    -------
    :math:`F(k) = C\,[\,(e^{\lambda k}-1)/\lambda - ((k/\lambda -
    1/\lambda^2)e^{\lambda k} + 1/\lambda^2)/M\,]`, with the same
    triangular limit as :func:`_hollands_moments` when ``lambda`` vanishes.
    """
    lam, kt_max = params["lambda"], params["kt_max"]
    values = np.clip(np.asarray(x, dtype=float), 0.0, kt_max)
    if abs(lam * kt_max) < 1e-4:
        cumulative: NDArray = values / kt_max * (2.0 - values / kt_max)
        return cumulative
    constant, _mean = _hollands_moments(lam, kt_max)
    exponential = np.exp(lam * values)
    first = (exponential - 1.0) / lam
    second = (values / lam - 1.0 / (lam * lam)) * exponential + 1.0 / (lam * lam)
    bounded: NDArray = np.clip(constant * (first - second / kt_max), 0.0, 1.0)
    return bounded


def _hollands_ppf(q: NDArray, params: Mapping[str, float]) -> NDArray:
    """Quantiles by bisection on the closed-form CDF (no analytic inverse exists)."""
    kt_max = params["kt_max"]
    quantiles = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
    low = np.zeros_like(quantiles)
    high = np.full_like(quantiles, kt_max)
    for _iteration in range(64):
        middle = 0.5 * (low + high)
        overshoot = _hollands_cdf(middle, params) > quantiles
        high = np.where(overshoot, middle, high)
        low = np.where(overshoot, low, middle)
    midpoint: NDArray = 0.5 * (low + high)
    return midpoint


FAMILIES: dict[str, Family] = {
    "weibull": Family(
        name="weibull",
        fit=_fit_weibull,
        pdf=_weibull_pdf,
        cdf=_weibull_cdf,
        ppf=_weibull_ppf,
        support=(0.0, np.inf),
    ),
    "gamma": Family(
        name="gamma",
        fit=_fit_gamma,
        pdf=_gamma_pdf,
        cdf=_gamma_cdf,
        ppf=_gamma_ppf,
        support=(0.0, np.inf),
    ),
    "beta": Family(
        name="beta",
        fit=_fit_beta,
        pdf=_beta_pdf,
        cdf=_beta_cdf,
        ppf=_beta_ppf,
        support=(0.0, 1.0),
    ),
    "normal": Family(
        name="normal",
        fit=_fit_normal,
        pdf=_normal_pdf,
        cdf=_normal_cdf,
        ppf=_normal_ppf,
    ),
    "hollands_huget": Family(
        name="hollands_huget",
        fit=_fit_hollands_huget,
        pdf=_hollands_pdf,
        cdf=_hollands_cdf,
        ppf=_hollands_ppf,
        support=(0.0, np.inf),
        options=("kt_max",),
    ),
}


def fit_distribution(family: str, values: NDArray, **options: float) -> DistributionFit:
    """Fit one of :data:`FAMILIES` to a sample.

    Parameters
    ----------
    family:
        Key of :data:`FAMILIES`.
    values:
        Sample. Non-finite entries and entries outside the family's support are
        dropped; ``DistributionFit.n`` reports how many were actually used, so
        the caller can turn the difference into the point mass the page prints
        beside the curve.
    **options:
        Family-specific knobs declared in ``Family.options`` (today only
        ``kt_max`` for ``hollands_huget``).

    Returns
    -------
    DistributionFit
        Parameters, or NaN parameters when fewer than two usable samples remain.

    Raises
    ------
    KeyError
        If ``family`` is not a known family — a typo in a configuration file
        must fail loudly rather than silently skip a variable.
    ValueError
        If an option the family does not declare is passed.
    """
    if family not in FAMILIES:
        raise KeyError(f"unknown distribution family {family!r}; known: {sorted(FAMILIES)}")
    specification = FAMILIES[family]
    unexpected = set(options) - set(specification.options)
    if unexpected:
        raise ValueError(f"{family} does not accept option(s) {sorted(unexpected)}")

    sample = _within_support(_clean(values), specification.support)
    params = specification.fit(sample, **options) if options else specification.fit(sample)
    return DistributionFit(family=family, params=params, n=int(sample.size))


def pdf(distribution: DistributionFit, x: NDArray) -> NDArray:
    """Probability density of a fit at ``x`` (zero outside its support)."""
    return FAMILIES[distribution.family].pdf(np.asarray(x, dtype=float), distribution.params)


def cdf(distribution: DistributionFit, x: NDArray) -> NDArray:
    """Cumulative distribution of a fit at ``x``."""
    return FAMILIES[distribution.family].cdf(np.asarray(x, dtype=float), distribution.params)


def ppf(distribution: DistributionFit, q: NDArray) -> NDArray:
    """Quantiles of a fit at probabilities ``q``."""
    return FAMILIES[distribution.family].ppf(np.asarray(q, dtype=float), distribution.params)


# ---------------------------------------------------------------------------
# Goodness of fit — distances only
# ---------------------------------------------------------------------------


def effective_sample_size(values: NDArray) -> tuple[float, float]:
    r"""Return ``(n_effective, lag-1 autocorrelation)`` for a serially correlated sample.

    Formula
    -------
    .. math:: n_{\text{eff}} = n \,\frac{1 - r_1}{1 + r_1}

    the standard first-order-autoregressive correction. An hourly temperature
    record has :math:`r_1` above 0.95, so ``n_effective`` lands one to two orders
    of magnitude below ``n`` — which is precisely why a nominal p-value computed
    from ``n`` would be meaningless.

    The input must be in **chronological order**; the correlation is taken
    between consecutive entries as given. A negative or non-finite estimate
    falls back to ``n``.
    """
    sample = np.asarray(values, dtype=float).ravel()
    if sample.size < 3:
        return float(sample.size), float("nan")
    current, following = sample[:-1], sample[1:]
    paired = np.isfinite(current) & np.isfinite(following)
    if int(paired.sum()) < 3:
        return float(np.isfinite(sample).sum()), float("nan")
    current, following = current[paired], following[paired]
    if np.isclose(current.std(), 0.0) or np.isclose(following.std(), 0.0):
        return float(np.isfinite(sample).sum()), float("nan")

    lag1 = float(np.corrcoef(current, following)[0, 1])
    n = float(np.isfinite(sample).sum())
    if not np.isfinite(lag1) or lag1 <= 0.0:
        return n, lag1
    return n * (1.0 - lag1) / (1.0 + lag1), lag1


def goodness_of_fit(
    distribution: DistributionFit,
    values: NDArray,
    *,
    binned: Histogram | None = None,
) -> FitQuality:
    """Measure how far a fit sits from its sample, as distances.

    Parameters
    ----------
    distribution:
        The fit to score.
    values:
        The sample, in **chronological order** — the order is what makes the
        lag-1 autocorrelation, and therefore ``n_effective``, meaningful.
    binned:
        Optional histogram on the frozen edges the page will draw. Supplying it
        adds ``density_r_squared``; without it that field is ``NaN``.

    Returns
    -------
    FitQuality
        All-NaN when fewer than two samples survive cleaning, so a variable with
        no data produces an explicit gap rather than a fabricated score.
    """
    ordered_sample = _clean(values)
    n_effective, lag1 = effective_sample_size(ordered_sample)

    specification = FAMILIES[distribution.family]
    sample = np.sort(_within_support(ordered_sample, specification.support))
    n = int(sample.size)
    if n < 2:
        return FitQuality(
            ks_distance=float("nan"),
            quantile_gap=float("nan"),
            density_r_squared=float("nan"),
            n=n,
            n_effective=n_effective,
            lag1_autocorrelation=lag1,
        )

    theoretical = cdf(distribution, sample)
    upper = np.arange(1, n + 1) / n
    lower = np.arange(0, n) / n
    ks = float(np.max(np.maximum(np.abs(upper - theoretical), np.abs(theoretical - lower))))

    # Mid-rank plotting positions rather than i/n: the largest sample would
    # otherwise be compared against the infinite quantile of an unbounded family.
    positions = (np.arange(1, n + 1) - 0.5) / n
    gap = float(np.mean(np.abs(sample - ppf(distribution, positions))))

    r_squared = float("nan")
    if binned is not None and binned.n > 0:
        modelled = pdf(distribution, binned.centers)
        residual = float(np.sum(np.square(binned.density - modelled)))
        total = float(np.sum(np.square(binned.density - binned.density.mean())))
        if total > 0.0:
            r_squared = 1.0 - residual / total

    return FitQuality(
        ks_distance=ks,
        quantile_gap=gap,
        density_r_squared=r_squared,
        n=n,
        n_effective=n_effective,
        lag1_autocorrelation=lag1,
    )


# ---------------------------------------------------------------------------
# Circular statistics (wind direction)
# ---------------------------------------------------------------------------


def circular_summary(degrees: NDArray) -> dict[str, float]:
    r"""Mean direction and circular variance of an angular sample.

    Formula
    -------
    With :math:`C = \overline{\cos\theta}` and :math:`S = \overline{\sin\theta}`,
    the mean direction is :math:`\arctan2(S, C)` and the resultant length is
    :math:`\bar{R} = \sqrt{C^2 + S^2}`; the circular variance is
    :math:`1 - \bar{R}`.

    Limitation
    ----------
    The arithmetic mean and standard deviation of degrees are meaningless across
    the 0/360 wrap — a record split between 350 deg and 10 deg averages to 180,
    the exact opposite direction. Use these instead (Mardia & Jupp, 2000,
    *Directional Statistics*).

    Returns
    -------
    dict
        ``mean_direction`` in ``[0, 360)``, ``resultant_length`` in ``[0, 1]``,
        ``circular_variance``, and ``n``. All NaN for an empty sample.
    """
    sample = _clean(degrees)
    if sample.size == 0:
        return {
            "mean_direction": float("nan"),
            "resultant_length": float("nan"),
            "circular_variance": float("nan"),
            "n": 0.0,
        }
    radians = np.deg2rad(sample)
    cosine = float(np.mean(np.cos(radians)))
    sine = float(np.mean(np.sin(radians)))
    resultant = float(np.hypot(cosine, sine))
    return {
        "mean_direction": _wrap_degrees(float(np.rad2deg(np.arctan2(sine, cosine)))),
        "resultant_length": resultant,
        "circular_variance": 1.0 - resultant,
        "n": float(sample.size),
    }


def sector_frequencies(
    degrees: NDArray,
    *,
    sectors: int = 16,
) -> tuple[NDArray, NDArray]:
    """Bin directions into compass sectors **centred** on the compass points.

    With the default 16 sectors, north spans 348.75-11.25 deg rather than
    0-22.5 deg. Centring matters: an off-by-half-sector rose puts the dominant
    flow on a boundary and splits it across two petals.

    Parameters
    ----------
    degrees:
        Directions the wind blows *from*, any real values; wrapped into
        ``[0, 360)``. Calms must already be removed by the caller — direction is
        undefined at zero speed, and a logger that writes 0.0 for a stalled
        anemometer would otherwise pile a spurious north peak into the rose.
    sectors:
        Number of equal sectors; 16 (22.5 deg) is the wind-atlas convention, 36
        (10 deg) gives a smoother rose.

    Returns
    -------
    tuple
        ``(centers_degrees, frequencies)`` — sector mid-points and the fraction
        of the sample in each, summing to 1 (all zeros for an empty sample).

    Raises
    ------
    ValueError
        If ``sectors`` is below 2.
    """
    if sectors < 2:
        raise ValueError(f"a wind rose needs at least two sectors, got {sectors}")
    sample = _clean(degrees)
    width = 360.0 / sectors
    centers = np.arange(sectors, dtype=float) * width
    if sample.size == 0:
        return centers, np.zeros(sectors)
    # The half-sector shift is what centres the first bin on 0 deg.
    index = np.floor(((sample % 360.0) + width / 2.0) % 360.0 / width).astype(int)
    counts = np.bincount(index, minlength=sectors)[:sectors]
    return centers, counts / float(sample.size)


def _kappa_from_resultant(resultant: float) -> float:
    """Invert ``A(kappa) = I1/I0 = R`` with the Mardia & Jupp (2000) approximations.

    The three branches are the published piecewise estimator; the Newton
    refinement that follows uses ``A'(kappa) = 1 - A^2 - A/kappa`` and turns the
    approximation into an estimate accurate to machine precision.
    """
    if resultant <= 0.0:
        return 0.0
    if resultant >= 1.0:
        return _MAX_SHAPE
    if resultant < 0.53:
        kappa = 2.0 * resultant + resultant**3 + 5.0 * resultant**5 / 6.0
    elif resultant < 0.85:
        kappa = -0.4 + 1.39 * resultant + 0.43 / (1.0 - resultant)
    else:
        kappa = 1.0 / (resultant**3 - 4.0 * resultant**2 + 3.0 * resultant)

    for _iteration in range(_MAX_ITERATIONS):
        if kappa <= 0.0:
            return 0.0
        # i1e/i0e is I1/I0 with the shared exp(kappa) factor cancelled, which is
        # the only form that survives kappa above ~700 in float64.
        ratio = float(special.i1e(kappa) / special.i0e(kappa))
        derivative = 1.0 - ratio * ratio - ratio / kappa
        if derivative <= 0.0:
            break
        step = (ratio - resultant) / derivative
        kappa -= step
        if abs(step) < _TOLERANCE * max(1.0, kappa):
            break
    return float(np.clip(kappa, 0.0, _MAX_SHAPE))


def fit_von_mises_mixture(
    degrees: NDArray,
    *,
    components: int = 2,
    seed: int = 0,
) -> VonMisesMixture:
    """Fit a finite von Mises mixture to wind directions by expectation-maximisation.

    A single von Mises visibly fails at a coastal tropical site: the trade flow
    and the sea-breeze veer are two distinct modes, and one concentration cannot
    describe both. Carta, Bueno & Ramirez (2008) is the reference for the
    mixture treatment.

    Parameters
    ----------
    degrees:
        Directions the wind blows from; calms must already be removed.
    components:
        Number of mixture components. Fix it and state it on any published
        figure rather than choosing it per subset, otherwise summer and winter
        roses stop being comparable.
    seed:
        Seeds the initial mean directions, which are drawn as evenly spaced
        angles jittered by a deterministic generator. Fixing it keeps a
        published fit reproducible.

    Returns
    -------
    VonMisesMixture
        Components sorted by descending weight. All-NaN for an empty sample.

    Raises
    ------
    ValueError
        If ``components`` is below 1.
    """
    if components < 1:
        raise ValueError(f"a mixture needs at least one component, got {components}")
    sample = _clean(degrees)
    if sample.size < components:
        nan = (float("nan"),) * components
        return VonMisesMixture(weights=nan, mu_degrees=nan, kappa=nan, n=int(sample.size))

    radians = np.deg2rad(sample % 360.0)
    generator = np.random.default_rng(seed)
    mu = np.linspace(0.0, 2.0 * np.pi, components, endpoint=False)
    mu = mu + generator.normal(scale=0.1, size=components)
    kappa = np.ones(components)
    weights = np.full(components, 1.0 / components)

    for _iteration in range(_MAX_ITERATIONS):
        # exp(kappa (cos d - 1)) / (2 pi i0e(kappa)) is the von Mises density
        # with the exp(kappa) factor cancelled against i0e's scaling; the direct
        # form overflows for kappa above ~700.
        deviation = radians[:, None] - mu[None, :]
        log_component = kappa[None, :] * (np.cos(deviation) - 1.0)
        densities = np.exp(log_component) / (2.0 * np.pi * special.i0e(kappa)[None, :])
        weighted = weights[None, :] * densities
        total = weighted.sum(axis=1, keepdims=True)
        total[total <= 0.0] = 1.0
        responsibility = weighted / total

        mass = responsibility.sum(axis=0)
        mass[mass <= 0.0] = _TOLERANCE
        new_weights = mass / float(radians.size)
        cosine = (responsibility * np.cos(radians)[:, None]).sum(axis=0) / mass
        sine = (responsibility * np.sin(radians)[:, None]).sum(axis=0) / mass
        new_mu = np.arctan2(sine, cosine)
        new_kappa = np.array([_kappa_from_resultant(float(r)) for r in np.hypot(cosine, sine)])

        shift = float(np.max(np.abs(new_weights - weights)))
        weights, mu, kappa = new_weights, new_mu, new_kappa
        if shift < _TOLERANCE:
            break

    order = np.argsort(-weights)
    return VonMisesMixture(
        weights=tuple(float(w) for w in weights[order]),
        mu_degrees=tuple(_wrap_degrees(float(np.rad2deg(m))) for m in mu[order]),
        kappa=tuple(float(k) for k in kappa[order]),
        n=int(sample.size),
    )


def von_mises_mixture_pdf(mixture: VonMisesMixture, degrees: NDArray) -> NDArray:
    """Mixture density per **degree**, so it overlays a sector-frequency rose directly.

    The von Mises density is defined per radian; dividing by ``180/pi`` converts
    it to the per-degree scale a sector frequency lives on, which is what makes
    ``frequency ~= pdf * sector_width`` hold.
    """
    angles = np.deg2rad(np.asarray(degrees, dtype=float) % 360.0)
    density = np.zeros_like(angles)
    for weight, mu, kappa in zip(mixture.weights, mixture.mu_degrees, mixture.kappa, strict=True):
        if not np.isfinite(weight) or not np.isfinite(kappa):
            return np.full_like(angles, np.nan)
        deviation = angles - np.deg2rad(mu)
        density += (
            weight
            * np.exp(kappa * (np.cos(deviation) - 1.0))
            / (2.0 * np.pi * float(special.i0e(kappa)))
        )
    per_degree: NDArray = density * np.pi / 180.0
    return per_degree
