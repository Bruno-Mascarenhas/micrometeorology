"""Reduction of WRF surface pressure to mean sea level.

Surface pressure over terrain is not a meteorological field.  On the operational
domains it correlates with terrain height at ``r = -0.998``, and between two
steps two days apart its spatial pattern correlates with itself at ``0.9996``:
contouring it draws the relief, frozen, and not the circulation.

The algorithm is RIP4's ``seaprs``, the routine wrf-python's ``slp`` and NCL's
``wrf_slp`` both inherit, reimplemented here on numpy so the laboratory does not
take a Fortran dependency for one field.  Its reference temperature is read
100 hPa ABOVE the lowest model level rather than at 2 m, which keeps the reduced
field from inheriting the surface diurnal cycle: measured on the operational
d02, the diurnal swing of the high-terrain minus low-ground pressure difference
falls from 5.3 hPa, extrapolating from 2 m, to 2.0 hPa.

This implementation deliberately differs from its reference in one place; see
the interpolation weight below.  The published field therefore does not match
wrf-python bit for bit — the two differ by up to 0.36 hPa, correlated with
terrain — and a comparison against another group's ``slp`` output should expect
that much.

What no formulation fixes is that the reduction stays a fiction below ground.
Two consequences are measured rather than assumed, and both are limits of the
method, not of the code:

- a terrain-locked residual remains after the reference level is raised.  On d02
  the high-minus-low difference still swings 2.0 hPa between pre-dawn and
  afternoon, which at the spacings this domain publishes is about two contour
  intervals: a feature over the Chapada that pulses through the animation and is
  an artifact, though the daily MEAN difference is real.
- above roughly 1500 m no reduction is trustworthy at all.  The operational
  domains top out at 1375 m and never reach it, so
  :data:`MAX_TRUSTWORTHY_TERRAIN_M` guards a future nested domain rather than
  today's — the artifact above is already material well below that edge.
"""

import numpy as np
from numpy.typing import NDArray

from micrometeorology.common.physics import PASCAL_PER_HECTOPASCAL
from micrometeorology.wrf.reader import WRFDataset

# RIP4 seaprs constants, kept at RIP's own values rather than the repository's
# general-purpose ones: the Ferrel correction below is a fit against THESE, and
# mixing a more precise R with that empirical quadratic is not more accurate.
_DRY_AIR_GAS_CONSTANT = 287.04
_GRAVITY = 9.81
_STANDARD_LAPSE_RATE_K_M = 0.0065

#: Reference level for the extrapolation temperature, as a pressure drop below
#: the lowest model level. RIP4 uses 100 hPa: high enough to clear the daytime
#: mixed layer, low enough to still represent the column being reduced.
_REFERENCE_PRESSURE_DROP_PA = 10000.0

#: Threshold of the MM5/RIP "ridiculous" correction, which caps the sea-level
#: temperature extrapolated from a hot surface and bends it back quadratically,
#: stopping a summer afternoon over high terrain from producing a fictitious low.
_HOT_SURFACE_THRESHOLD_K = 273.16 + 17.5
_HOT_SURFACE_RELAXATION = 0.005

# WRF stores potential temperature as a perturbation from 300 K, and the Poisson
# exponent for dry air is R/cp = 2/7.
_POTENTIAL_TEMPERATURE_BASE_K = 300.0
_REFERENCE_PRESSURE_PA = 1.0e5
_POISSON_EXPONENT = 2.0 / 7.0

# Virtual-temperature coefficient, (1 - epsilon)/epsilon for water vapour in air.
_VIRTUAL_TEMPERATURE_COEFFICIENT = 0.608

#: Terrain height above which no sea-level reduction is trustworthy, because the
#: fictitious column below ground becomes deeper than the layer whose temperature
#: is being extrapolated. Crossing it is reported, not raised: the field is still
#: the best available and the rest of the domain stays valid.
MAX_TRUSTWORTHY_TERRAIN_M = 1500.0


def terrain_reduction_caveat(terrain_m: NDArray) -> str | None:
    """The operator-facing caveat when terrain leaves the reduction's valid range.

    Returned rather than logged so it travels in the work unit's warnings, which
    is the channel the exporter prints; a log line alone is invisible to an
    operator running at ``--log-level ERROR``.

    Parameters
    ----------
    terrain_m:
        Terrain height, any shape, metres above sea level.

    Returns
    -------
    str | None
        The caveat, or ``None`` when the domain stays inside the range where the
        reduction is defensible.
    """
    highest = float(np.max(terrain_m))
    if highest <= MAX_TRUSTWORTHY_TERRAIN_M:
        return None
    return (
        f"terrain reaches {highest:.0f} m, above the {MAX_TRUSTWORTHY_TERRAIN_M:.0f} m "
        "beyond which no sea-level reduction is trustworthy; isobars over the "
        "highest cells are indicative"
    )


def sea_level_pressure_hpa(
    pressure_pa: NDArray,
    height_m: NDArray,
    temperature_k: NDArray,
    vapor_mixing_ratio: NDArray,
) -> NDArray:
    """Reduce one time step's pressure field to mean sea level.

    Parameters
    ----------
    pressure_pa:
        Full pressure on mass levels, ``(K, Y, X)``, Pa, bottom level first.
    height_m:
        Geopotential height of the same mass levels, ``(K, Y, X)``, metres above
        sea level.
    temperature_k:
        Absolute temperature on the same levels, ``(K, Y, X)``, K.
    vapor_mixing_ratio:
        Water-vapour mixing ratio on the same levels, ``(K, Y, X)``, kg/kg.

    Returns
    -------
    numpy.ndarray
        Mean sea level pressure, ``(Y, X)``, hPa, ``float64``.

    Raises
    ------
    ValueError
        When the inputs disagree on shape, when fewer than two mass levels are
        given, or when some column never reaches the reference level — a model
        top below the surface pressure minus 100 hPa, which no usable run has.
    """
    shapes = {array.shape for array in (pressure_pa, height_m, temperature_k, vapor_mixing_ratio)}
    if len(shapes) != 1:
        raise ValueError(f"sea-level reduction needs one common (K, Y, X) shape, got {shapes}")
    if pressure_pa.ndim != 3:
        raise ValueError(f"sea-level reduction needs (K, Y, X) fields, got {pressure_pa.shape}")
    if pressure_pa.shape[0] < 2:
        raise ValueError("sea-level reduction needs at least two mass levels to interpolate")

    virtual_temperature = temperature_k * (
        1.0 + _VIRTUAL_TEMPERATURE_COEFFICIENT * vapor_mixing_ratio
    )
    lowest_pressure, lowest_height = pressure_pa[0], height_m[0]
    reference_pressure = lowest_pressure - _REFERENCE_PRESSURE_DROP_PA

    above_reference = pressure_pa < reference_pressure[np.newaxis, :, :]
    if not above_reference.any(axis=0).all():
        raise ValueError(
            "no model level reaches 100 hPa above the surface; the reference "
            "temperature of the reduction is undefined for this file"
        )
    # The bottom level never satisfies the test, so this index is always >= 1 and
    # brackets the reference level together with the one below it.
    upper = above_reference.argmax(axis=0)
    lower = upper - 1
    rows, columns = np.indices(upper.shape)

    pressure_below = pressure_pa[lower, rows, columns]
    pressure_above = pressure_pa[upper, rows, columns]
    if not (np.isfinite(pressure_below).all() and np.isfinite(pressure_above).all()):
        raise ValueError(
            "a bracketing model level is not finite; the reduction would return a "
            "NaN column that only surfaces as a missing contour much later"
        )
    # DELIBERATE DEVIATION from RIP4 seaprs, which wrf-python transliterates:
    # both read `LOG(p*/phi) * LOG(plo/phi)`, the two logarithms MULTIPLIED, which
    # is not an interpolation weight — it evaluates to ~2.5e-4 here and silently
    # collapses the result onto the bracketing level above. The quotient below is
    # the log-linear interpolation the surrounding algorithm is written for, exact
    # for an isothermal layer. Measured cost over the operational domains: up to
    # 0.36 hPa, correlated with terrain at r ~ -0.8.
    weight = np.log(reference_pressure / pressure_above) / np.log(pressure_below / pressure_above)

    def _interpolate(field: NDArray) -> NDArray:
        below, above = field[lower, rows, columns], field[upper, rows, columns]
        return np.asarray(above - (above - below) * weight, dtype=np.float64)

    temperature_at_reference = _interpolate(virtual_temperature)
    height_at_reference = _interpolate(height_m)

    surface_temperature = temperature_at_reference * (lowest_pressure / reference_pressure) ** (
        _STANDARD_LAPSE_RATE_K_M * _DRY_AIR_GAS_CONSTANT / _GRAVITY
    )
    sea_level_temperature = (
        temperature_at_reference + _STANDARD_LAPSE_RATE_K_M * height_at_reference
    )

    # RIP4's three-branch "ridiculous MM5" correction, applied in its own order:
    # a hot SURFACE bends the sea-level temperature back quadratically, and a
    # merely hot extrapolation under a cool surface is capped flat.
    hot_surface = surface_temperature > _HOT_SURFACE_THRESHOLD_K
    sea_level_temperature = np.where(
        hot_surface,
        _HOT_SURFACE_THRESHOLD_K
        - _HOT_SURFACE_RELAXATION * (surface_temperature - _HOT_SURFACE_THRESHOLD_K) ** 2,
        np.where(
            sea_level_temperature >= _HOT_SURFACE_THRESHOLD_K,
            _HOT_SURFACE_THRESHOLD_K,
            sea_level_temperature,
        ),
    )

    reduced_pa = lowest_pressure * np.exp(
        2.0
        * _GRAVITY
        * lowest_height
        / (_DRY_AIR_GAS_CONSTANT * (sea_level_temperature + surface_temperature))
    )
    return np.asarray(reduced_pa / PASCAL_PER_HECTOPASCAL, dtype=np.float64)


def read_sea_level_pressure_hpa(dataset: WRFDataset, step: int) -> NDArray:
    """Read one time step's 3-D fields and reduce them to mean sea level.

    Read a step at a time rather than through the whole-variable extractors in
    :mod:`micrometeorology.wrf.variables`: the six 3-D fields this needs are two
    orders of magnitude larger than the 2-D surface fields those return, and a
    whole-run read of them would not fit the per-worker budget.

    Parameters
    ----------
    dataset:
        An open wrfout carrying ``P``, ``PB``, ``PH``, ``PHB``, ``T`` and
        ``QVAPOR``.
    step:
        Time index to reduce.

    Returns
    -------
    numpy.ndarray
        Mean sea level pressure, ``(Y, X)``, hPa, ``float64``.
    """

    def _block(name: str) -> NDArray:
        return np.asarray(dataset.get_variable_block(name, step, step + 1)[0], dtype=np.float64)

    pressure_pa = _block("P") + _block("PB")
    # Geopotential is stored on staggered levels; mass levels are their midpoints.
    staggered_height = (_block("PH") + _block("PHB")) / _GRAVITY
    height_m = 0.5 * (staggered_height[:-1] + staggered_height[1:])
    potential_temperature = _block("T") + _POTENTIAL_TEMPERATURE_BASE_K
    temperature_k = (
        potential_temperature * (pressure_pa / _REFERENCE_PRESSURE_PA) ** _POISSON_EXPONENT
    )

    return sea_level_pressure_hpa(pressure_pa, height_m, temperature_k, _block("QVAPOR"))
