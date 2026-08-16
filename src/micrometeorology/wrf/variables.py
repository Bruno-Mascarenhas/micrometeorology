"""WRF variable extraction, unit conversion and derived surface diagnostics.

One extractor per published variable, each returning the field together with
the ``(vmin, vmax)`` colour-scale bounds the map and JSON writers render from.

Derived surface radiation budget
--------------------------------
wrfout carries the DOWNWELLING fluxes (SWDOWN, GLW) always, the upwelling ones
only with RRTMG's bottom-of-atmosphere outputs on: LWUPB/SWUPB are in the 2013
archive runs and absent from the 2026 operational runs. Every budget term here
is therefore rebuilt from fields every wrfout generation carries — EMISS, TSK,
GLW, SWDOWN, ALBEDO, T2, COSZEN — so a term publishes identically across runs.
Validated where WRF's own fluxes exist (d02 2013, 9801 cells x 9 steps): LWUP
reproduces LWUPB to MAE 0.82 W/m2 (0.11% of a ~420 W/m2 signal), SWUP
reproduces SWUPB exactly. Per-term formulas, their sources in the WRF source
tree and their limitations live on the ``compute_*`` functions below.
"""

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from allsky.solar import SOLAR_CONSTANT_WM2, eccentricity_correction
from micrometeorology.common.physics import STEFAN_BOLTZMANN
from micrometeorology.wrf.reader import WRFDataset
from micrometeorology.wrf.safety import assert_reasonable_array_size

logger = logging.getLogger(__name__)


def _drop_spinup_step(value: NDArray) -> NDArray:
    """Drop the spin-up first time step.

    A single-timestep input is returned whole, so reductions never see an empty tail.
    """
    if value.shape[0] <= 1:
        return value
    return value[1:, :]


def squeeze_array(value: NDArray) -> NDArray:
    """Drop every singleton axis, turning a one-step ``(1, ny, nx)`` slice into a frame.

    Returns
    -------
    NDArray
        A view of *value* with all length-1 axes removed, in its dtype and unit.
    """
    return np.squeeze(value)


def materialize_2d(value: NDArray) -> NDArray:
    """Validate and return a 2-D ``(ny, nx)`` worker payload in the field's unit.

    Raises
    ------
    ValueError
        When the squeezed input is not 2-D, which would reach the writers as a
        values array whose length no longer matches the published grid.
    MemoryError
        When the frame exceeds the array-size ceiling.
    """
    squeezed = np.squeeze(value)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2-D worker payload, got shape {squeezed.shape!r}")
    assert_reasonable_array_size(squeezed.shape, squeezed.dtype, context="materialize_2d")
    return np.asarray(squeezed)


def blank_uninitialised_radiation(values: NDArray, glw: NDArray) -> NDArray:
    """NaN every step at which WRF had not yet called the radiation scheme.

    *values* is a derived ``(T, ny, nx)`` field and *glw* the ``(T, ny, nx)``
    downwelling longwave (W/m2) it was derived from; they share the time axis.
    The result is *values* itself when no step is uninitialised, and otherwise a
    float64 copy with the affected steps set to NaN.

    On a cold start the first output step precedes the first radiation call, so
    ``GLW`` is identically zero domain-wide — an unambiguous marker, not a
    threshold, because downwelling longwave is never zero over land. The
    GLW-derived fields at that step are impossible: net radiation ~-416 W/m2
    domain-mean, sky emissivity exactly 0.00, upwelling longwave ~18 W/m2 low
    with the reflected ``(1 - eps) * GLW`` term missing.

    Blanking here rather than at the writer keeps the colour-scale sample and the
    published frame one population: :func:`percentile_scale_bounds` already skips
    step 0, and the publish gate drops a frame only when every cell is non-finite
    — so the blanked step is dropped instead of shipping under a legend computed
    without it. Continuation runs keep their radiation state and are untouched;
    shortwave is not covered, because a zero ``SWDOWN`` is also what night looks
    like, and ``DAYLIGHT_ONLY_VARIABLES`` gates those instead.
    """
    steps = np.asarray(glw).reshape(np.shape(glw)[0], -1)
    uninitialised = np.all(steps == 0.0, axis=1)
    if not uninitialised.any():
        return values
    blanked = np.array(values, dtype=np.float64, copy=True)
    blanked[uninitialised] = np.nan
    return blanked


def percentile_scale_bounds(variable: NDArray) -> tuple[float, float]:
    """Return color-scale bounds ``(low, high)`` for a 3-D variable, skipping the first step.

    The lower bound is the true minimum, but the **upper bound is the
    98th-percentile saturation cap, not the maximum**, so a handful of extreme
    cells cannot blow out the map colour scale. The pair is a rendering
    ``(vmin, vmax)``, not the data's actual range.
    """
    flat = _drop_spinup_step(variable).ravel()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: NDArray, v: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays, skipping the first step."""
    speed = np.hypot(_drop_spinup_step(u).ravel(), _drop_spinup_step(v).ravel())
    return float(np.nanmin(speed)), float(np.nanmax(speed))


def get_low_high_rain(variable: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` of incremental precipitation from *cumulative* rain."""
    arr = np.asarray(variable)
    if arr.ndim < 3:
        flat = arr.ravel()
        return float(np.nanmin(flat)), float(np.nanmax(flat))
    diffs = np.diff(arr, axis=0)
    if diffs.size == 0:
        return 0.0, 0.0
    flat = diffs.ravel()
    return float(np.nanmin(flat)), float(np.nanmax(flat))


def extract_temperature(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract 2-m temperature (°C).

    Returns the raw ``(T, ny, nx)`` Kelvin array — converted per step by
    :func:`extract_temperature_step` — with bounds already in °C. Callers needing
    surface pressure read PSFC themselves, so this never loads an unused variable.
    """
    t2 = ds.get_variable("T2")

    t_min, t_max = percentile_scale_bounds(t2)
    t_min -= 273.15
    t_max -= 273.15

    return t2, t_min, t_max


def extract_temperature_step(t2_step: NDArray) -> NDArray:
    """Convert a single time-step of T2 from Kelvin to Celsius."""
    return squeeze_array(t2_step) - 273.15


def extract_skin_temperature(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract WRF surface skin temperature, ``TSK_C = TSK_K - 273.15``.

    Limitation
        Model skin temperature: not a 2-m air temperature and not an observed
        land-surface temperature product.
    """
    tsk = ds.get_variable("TSK")
    t_min, t_max = percentile_scale_bounds(tsk)
    return tsk, t_min - 273.15, t_max - 273.15


def extract_pressure(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract surface pressure (hPa)."""
    psfc = ds.get_variable("PSFC")
    p_min, p_max = percentile_scale_bounds(psfc)
    return psfc / 100.0, p_min / 100.0, p_max / 100.0


def extract_vapor(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract 2-m water-vapor mixing ratio: ``Q2`` (``QV at 2 M``, kg/kg) scaled to g/kg.

    The exported site variable keeps the legacy ``VAPOR`` id.
    """
    q2 = ds.get_variable("Q2")
    q_min, q_max = percentile_scale_bounds(q2)
    return q2 * 1000.0, q_min * 1000.0, q_max * 1000.0


def compute_relative_humidity(q2: NDArray, t2: NDArray, psfc: NDArray) -> NDArray:
    """Compute 2-m relative humidity from ``Q2`` (kg/kg), ``T2`` (K) and ``PSFC`` (Pa).

    Formula
        ``100 * e / es`` with vapor pressure ``e = q * p / (epsilon + q)``,
        ``epsilon = 0.622`` — that denominator assumes Q2 is a mixing ratio,
        matching WRF's QV convention — over the Bolton/Tetens saturation
        ``es = 611.2 * exp(17.67 * Tc / (Tc + 243.5))``.
    Output
        Percent, clipped to the physical display range 0-100%.
    """
    epsilon = 0.622
    temp_c = t2 - 273.15
    vapor_pressure = q2 * psfc / (epsilon + q2)
    saturation_pressure = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = 100.0 * (vapor_pressure / saturation_pressure)
    clipped: NDArray = np.clip(rh, 0.0, 100.0)
    return clipped


def extract_relative_humidity(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived 2-m relative humidity (%)."""
    q2 = ds.get_variable("Q2")
    t2 = ds.get_variable("T2")
    psfc = ds.get_variable("PSFC")
    rh = compute_relative_humidity(q2, t2, psfc)
    rh_min, rh_max = percentile_scale_bounds(rh)
    return rh, rh_min, rh_max


def rotate_to_earth_relative(
    ds: WRFDataset, u: NDArray, v: NDArray, t0: int | None = None, t1: int | None = None
) -> tuple[NDArray, NDArray]:
    """Rotate grid-relative wind components onto true north.

    ``U10``/``V10`` are grid-relative: on a Lambert or polar-stereographic domain
    the grid's y axis is not north, so every bearing taken from them is off by the
    local rotation angle — tens of degrees at the edges of a wide domain.
    ``COSALPHA``/``SINALPHA`` carry that angle per cell, making the rotation exact
    rather than estimated. The operational domains are Mercator, where
    ``SINALPHA`` is 0 everywhere and this is the identity; a file without the
    fields is passed through unrotated.

    ``t0``/``t1`` rotate one time block, reading the angle for the same block —
    without them a 64-step block would meet the file's full 76-step field. A
    ``u``/``v`` carrying a vertical axis gets one broadcast into the angle, which
    has none.
    """
    if not (ds.has_variable("COSALPHA") and ds.has_variable("SINALPHA")):
        return u, v
    if t0 is None or t1 is None:
        cos_alpha = ds.get_variable("COSALPHA")
        sin_alpha = ds.get_variable("SINALPHA")
    else:
        cos_alpha = ds.get_variable_block("COSALPHA", t0, t1)
        sin_alpha = ds.get_variable_block("SINALPHA", t0, t1)
    if u.ndim == cos_alpha.ndim + 1:
        cos_alpha = cos_alpha[:, np.newaxis, :, :]
        sin_alpha = sin_alpha[:, np.newaxis, :, :]
    return u * cos_alpha + v * sin_alpha, v * cos_alpha - u * sin_alpha


def extract_wind(ds: WRFDataset) -> tuple[NDArray, NDArray, float, float]:
    """Extract 10-m U/V wind components and compute speed bounds.

    Returns
    -------
    tuple[NDArray, NDArray, float, float]
        ``(u10, v10, speed_min, speed_max)``: two ``(T, ny, nx)`` arrays in m/s,
        earth-relative (rotated onto true north by
        :func:`rotate_to_earth_relative`), and the true min/max of the resulting
        speed in m/s — no percentile cap here, unlike the scalar fields.
    """
    u10, v10 = rotate_to_earth_relative(ds, ds.get_variable("U10"), ds.get_variable("V10"))
    ws_min, ws_max = get_low_high_wind(u10, v10)
    return u10, v10, ws_min, ws_max


def extract_rain(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract total precipitation, ``RAINC + RAINNC``, in mm.

    The field is CUMULATIVE since the run start, as WRF writes it; the published
    per-step increment comes from :func:`extract_rain_step`. The bounds are
    already those of the increment, so the map legend matches what is drawn.
    """
    rainc = ds.get_variable("RAINC")
    rainnc = ds.get_variable("RAINNC")
    total = rainc + rainnc
    r_min, r_max = get_low_high_rain(total)
    return total, r_min, r_max


def extract_rain_step(total: NDArray, i: int) -> NDArray:
    """Compute incremental rain for step *i* from cumulative totals.

    *total* is the ``(T, ny, nx)`` cumulative field in mm; the result is the
    ``(ny, nx)`` accumulation over the step ending at *i*, also in mm.

    Step 0 has no increment to compute at a file or restart boundary, so it is
    no-value rather than zero: a field of zeros is the statement that it rained
    nowhere, which the model never made. The publish gate drops an all-NaN frame,
    so the step leaves the manifest's ``availability`` like the radiation fields
    the cold start has not initialised.
    """
    if i == 0:
        return np.full_like(np.squeeze(total[i : i + 1, :, :]), np.nan)
    current: NDArray = np.squeeze(total[i : i + 1, :, :])
    previous: NDArray = np.squeeze(total[i - 1 : i, :, :])
    increment: NDArray = current - previous
    return increment


def extract_scalar(ds: WRFDataset, var_name: str) -> tuple[NDArray, float, float]:
    """Generic extractor for scalar fields published in their wrfout unit.

    Used for the fields that need no conversion — HFX, LH and SWDOWN, all W/m2 —
    returning the ``(T, ny, nx)`` array and its percentile scale bounds unchanged.
    """
    var = ds.get_variable(var_name)
    v_min, v_max = percentile_scale_bounds(var)
    return var, v_min, v_max


#: Sun-elevation floor for the clearness index: below it the extraterrestrial
#: denominator is small enough that ``kt`` tracks the horizon singularity rather
#: than sky condition, so those cells publish "no value".
#:
#: cos(z) = 0.1736 is a solar zenith angle of 80 degrees (elevation 10), the
#: daytime threshold of the ARM best-estimate radiation VAP (Shi & Long,
#: DOE/SC-ARM/TR-008); BSRN's component tests tighten to SZA < 75. Those bound
#: pyranometer cosine-response error, which model output does not have, but the
#: same grazing geometry is where WRF's plane-parallel radiative transfer is
#: least trustworthy. On a 76-step operational d04 run this keeps 30 of the 39
#: daylight frames and ~77% of their cells.
MIN_COSZEN_FOR_CLEARNESS = 0.1736


def _finite_scale_bounds(values: NDArray, fallback: tuple[float, float]) -> tuple[float, float]:
    """Percentile scale bounds, falling back when the field is entirely no-value.

    Only the clearness index reaches this: an all-night file leaves every cell
    NaN and the values writer rejects non-finite bounds outright. Those steps are
    gated out and never published, but the bounds are computed before the gate.
    """
    with warnings.catch_warnings():
        # nanmin/nanpercentile warn on an all-NaN input; that NaN is the signal
        # acted on below, so the warning is noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = percentile_scale_bounds(values)
    if not (np.isfinite(low) and np.isfinite(high)):
        return fallback
    return low, high


def compute_upwelling_longwave(emiss: NDArray, tsk: NDArray, glw: NDArray) -> NDArray:
    """Upwelling longwave radiation at the surface, W/m2 positive upward.

    Formula
        ``LWup = eps * sigma * TSK^4 + (1 - eps) * GLW`` from ``EMISS`` (0-1),
        ``TSK`` (K) and ``GLW`` (W/m2): graybody emission plus the sky radiance a
        non-black surface reflects. Dropping the reflected term is the common
        mistake and costs ~15 W/m2, nearly 20x the error of the full form.

        This is what WRF does, not an approximation of it. RRTMG's ``rtrnmc``
        forms the upward radiance per band as ``radlu = rad0 + reflect * radld``
        with ``rad0 = semiss * B(Ts)`` and ``reflect = 1 - semiss``
        (``phys/module_ra_rrtmg_lw.F``, under "Add in specular reflection of
        surface downward radiance"), and Noah's urban branch writes it longhand:
        ``rl_up_rural = -emiss*sigma*tsk**4 - (1.-emiss)*glw``
        (``phys/module_sf_noahdrv.F``). Noah's own net-longwave term
        ``emiss*GLW - emiss*sigma*T1**4`` is the same identity with the reflected
        part already cancelled, not a scheme that ignores it.
    Limitation
        Uses the grid-mean skin temperature where WRF's own LWUPB uses the LSM's
        canopy-adjusted radiative temperature: ~1.7 W/m2 apart over land, ~0.03
        W/m2 over water. Far smaller, WRF's ``STBOLT = 5.67051e-8``
        (``share/module_model_constants.F``; Noah rounds to ``5.67e-8``) against
        the exact CODATA value used here is worth ~0.01 W/m2, 1% of that land
        bias. A model diagnostic, not an observed flux.
    """
    upwelling: NDArray = emiss * STEFAN_BOLTZMANN * tsk**4 + (1.0 - emiss) * glw
    return upwelling


def compute_upwelling_shortwave(albedo: NDArray, swdown: NDArray) -> NDArray:
    """Upwelling (reflected) shortwave radiation at the surface, W/m2 positive upward.

    Formula
        ``SWup = ALBEDO * SWDOWN`` — exactly how WRF forms SWUPB internally,
        which is why this reproduces that field bit-for-bit where it exists.
    Limitation
        ``ALBEDO`` is the broadband all-sky surface albedo; no direct/diffuse or
        spectral split is applied.
    """
    reflected: NDArray = albedo * swdown
    return reflected


def compute_net_shortwave(albedo: NDArray, swdown: NDArray) -> NDArray:
    """Net (absorbed) shortwave radiation at the surface.

    ``SWnet = SWDOWN * (1 - ALBEDO)``, the shortwave the surface keeps, in W/m2
    positive downward (into the surface). Zero at night.
    """
    return swdown * (1.0 - albedo)


def compute_net_longwave(emiss: NDArray, tsk: NDArray, glw: NDArray) -> NDArray:
    """Net longwave radiation at the surface.

    Formula
        ``LWnet = GLW - LWup`` reduces to the ``eps * (GLW - sigma * TSK^4)``
        evaluated here, the reflected ``(1 - eps) * GLW`` term cancelling exactly
        against ``LWup``. Float rounding leaves it agreeing with
        ``GLW - compute_upwelling_longwave(...)`` to within an ulp, not
        bit-for-bit.
    Output
        W/m2 positive downward, almost always negative: the surface is warmer
        than the effective sky and loses longwave day and night.
    """
    return emiss * (glw - STEFAN_BOLTZMANN * tsk**4)


def compute_net_radiation(
    swdown: NDArray, albedo: NDArray, emiss: NDArray, tsk: NDArray, glw: NDArray
) -> NDArray:
    """Net all-wave radiation at the surface.

    Formula
        ``Rn = SWnet + LWnet = SWDOWN * (1 - ALBEDO) + eps * (GLW - sigma * TSK^4)``.
    Output
        W/m2 positive downward: the energy available to drive the sensible
        (HFX), latent (LH) and ground (GRDFLX) heat fluxes.
    Limitation
        Inherits the skin-temperature caveat of
        :func:`compute_upwelling_longwave`, and is not reconcilable cell-by-cell
        with ``NOAHRES`` — Noah's per-tile residual against the canopy
        temperature, not against these grid-mean fields.
    """
    net: NDArray = compute_net_shortwave(albedo, swdown) + compute_net_longwave(emiss, tsk, glw)
    return net


def compute_sky_emissivity(glw: NDArray, t2: NDArray) -> NDArray:
    """Effective (bulk) emissivity of the sky hemisphere.

    Formula
        ``eps_sky = GLW / (sigma * T2^4)`` from ``GLW`` (W/m2) and ``T2`` (K):
        the emissivity a blackbody at screen-level air temperature would need to
        emit the observed downwelling flux.
    Output
        Dimensionless, typically ~0.75 under a dry clear sky and approaching 1
        under a warm overcast, which makes it a cloudiness proxy.
    Limitation
        Takes 2-m air temperature as the radiating temperature of the whole
        column, absorbing any near-surface inversion into the emissivity. Not
        clipped: a first output step carrying ``GLW = 0``, before radiation is
        called, shows up as ``eps_sky = 0``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        emissivity: NDArray = glw / (STEFAN_BOLTZMANN * t2**4)
    return emissivity


def compute_clearness_index(swdown: NDArray, coszen: NDArray, eccentricity: NDArray) -> NDArray:
    """Extraterrestrial clearness index ``kt`` at the surface.

    Formula
        ``kt = SWDOWN / (S0 * E0 * cos(z))`` from ``SWDOWN`` (W/m2), ``COSZEN``
        and the per-step Sun-Earth distance correction — Duffie & Beckman
        eq. (1.10.1), the same definition as :func:`allsky.solar.clearness_index`
        so WRF and the all-sky camera pipeline share one axis. Lower case ``kt``
        is deliberate: Duffie & Beckman sect. 2.9 reserve it for the sub-daily
        index and upper-case ``K_T`` for the daily one, and this field is
        instantaneous. Do not compare it against the "typical 0.25-0.75" figures
        quoted for monthly-mean daily ``K_T``, which include the low-sun hours
        and sit systematically lower.
    Output
        Dimensionless, ~0 under thick cloud and approaching ~0.8 under a high
        clear sky. Cells below :data:`MIN_COSZEN_FOR_CLEARNESS` publish no value
        (``null`` in the values JSON, ``MISSING`` in the cell-series matrix).
    Limitation
        Not clipped at 1: cloud-edge enhancement genuinely produces ``kt > 1``,
        but Duffie & Beckman note very few observations exceed kt = 0.80 at all,
        so sustained values well above it point at the inputs rather than at an
        unusually clear sky. ``COSZEN`` is negative at night, which the elevation
        floor screens out along with the singularity.
    """
    denominator = SOLAR_CONSTANT_WM2 * eccentricity * coszen
    clearness = np.full(swdown.shape, np.nan, dtype=np.float64)
    sun_up = coszen >= MIN_COSZEN_FOR_CLEARNESS
    np.divide(swdown, denominator, out=clearness, where=sun_up)
    return clearness


def extract_upwelling_longwave(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived upwelling longwave radiation (W/m2)."""
    glw = ds.get_variable("GLW")
    values = blank_uninitialised_radiation(
        compute_upwelling_longwave(ds.get_variable("EMISS"), ds.get_variable("TSK"), glw), glw
    )
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_upwelling_shortwave(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived upwelling (reflected) shortwave radiation (W/m2)."""
    values = compute_upwelling_shortwave(ds.get_variable("ALBEDO"), ds.get_variable("SWDOWN"))
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_net_shortwave(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived net shortwave radiation (W/m2)."""
    values = compute_net_shortwave(ds.get_variable("ALBEDO"), ds.get_variable("SWDOWN"))
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_net_longwave(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived net longwave radiation (W/m2, positive downward)."""
    glw = ds.get_variable("GLW")
    values = blank_uninitialised_radiation(
        compute_net_longwave(ds.get_variable("EMISS"), ds.get_variable("TSK"), glw), glw
    )
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_net_radiation(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived net all-wave radiation (W/m2, positive downward)."""
    glw = ds.get_variable("GLW")
    values = blank_uninitialised_radiation(
        compute_net_radiation(
            ds.get_variable("SWDOWN"),
            ds.get_variable("ALBEDO"),
            ds.get_variable("EMISS"),
            ds.get_variable("TSK"),
            glw,
        ),
        glw,
    )
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_sky_emissivity(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived effective sky emissivity (dimensionless)."""
    glw = ds.get_variable("GLW")
    values = blank_uninitialised_radiation(compute_sky_emissivity(glw, ds.get_variable("T2")), glw)
    low, high = _finite_scale_bounds(values, (0.0, 1.0))
    return values, low, high


def extract_clearness_index(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived clearness index ``kt`` (dimensionless).

    Falls back to ``E0 = 1`` when the ``Times`` axis cannot be matched to the
    field's time axis, which costs at most 3.3% rather than failing the export.
    """
    swdown = ds.get_variable("SWDOWN")
    coszen = ds.get_variable("COSZEN")
    times = ds.parse_times()
    if len(times) == swdown.shape[0]:
        stamps = pd.DatetimeIndex([t.replace(tzinfo=None) for t in times])
        eccentricity = eccentricity_correction(stamps)[:, np.newaxis, np.newaxis]
    else:
        logger.warning(
            "Times axis (%d) does not match SWDOWN time axis (%d); "
            "clearness index falls back to E0 = 1",
            len(times),
            swdown.shape[0],
        )
        eccentricity = np.ones((1, 1, 1))
    values = compute_clearness_index(swdown, coszen, eccentricity)
    low, high = _finite_scale_bounds(values, (0.0, 1.0))
    return values, low, high


def compute_air_density(t2: NDArray, psfc: NDArray, q2: NDArray) -> NDArray:
    """Estimate moist-air density at 2 m.

    Formula
        Virtual temperature ``Tv = T2 * (1 + 0.61 * q)`` into the ideal gas law
        ``rho = p / (Rd * Tv)`` with ``Rd = 287.05 J kg-1 K-1``, from ``T2`` (K),
        ``PSFC`` (Pa) and ``Q2`` (kg/kg). Result in kg/m3.
    Limitation
        A near-surface estimate; not density at turbine hub height without a
        vertical thermodynamic profile.
    """
    virtual_temperature = t2 * (1.0 + 0.61 * q2)
    return psfc / (287.05 * virtual_temperature)


def extract_wind_power_density_10m(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Compute wind power density at 10 m.

    Formula
        ``WPD = 0.5 * rho * speed^3`` in W/m2, with
        ``speed = sqrt(U10^2 + V10^2)`` (m/s) and moist-air density from ``T2``
        (K), ``PSFC`` (Pa) and ``Q2`` (kg/kg).
    Limitation
        Available power density in the flow, not turbine output: no rotor area,
        power coefficient, cut-in/cut-out or hub-height extrapolation.
    """
    u10 = ds.get_variable("U10")
    v10 = ds.get_variable("V10")
    t2 = ds.get_variable("T2")
    psfc = ds.get_variable("PSFC")
    q2 = ds.get_variable("Q2")
    speed = np.hypot(u10, v10)
    density = compute_air_density(t2, psfc, q2)
    power_density = 0.5 * (density * np.power(speed, 3))
    p_min, p_max = percentile_scale_bounds(power_density)
    return power_density, p_min, p_max


DEFAULT_STREAM_BLOCK_STEPS = 64


@dataclass(frozen=True)
class WindHeightSeries:
    """Interpolated wind speed series and per-step wind vectors for one height.

    Attributes
    ----------
    target:
        The height the series was interpolated to, meters above ground level.
    vmin, vmax:
        Colour-scale bounds for the speed field, m/s, from
        :func:`percentile_scale_bounds` (so *vmax* is the 98th percentile).
    speed_steps:
        ``(T, ny, nx)`` wind speed in m/s, in the interpolation dtype (float32
        for operational wrfout), spanning the file's whole time axis.
    wind_vectors:
        One entry per time step, in step order: the downsampled arrow payload
        :func:`_package_wind_vectors_step` builds, or ``None`` for a step whose
        packaging failed.
    """

    target: int
    vmin: float
    vmax: float
    speed_steps: NDArray
    wind_vectors: list[dict | None]


def _package_wind_vectors_step(
    u_subgrid: NDArray,
    v_subgrid: NDArray,
    linear_index: NDArray,
) -> dict:
    """Package one timestep's wind vectors from the already-downsampled components.

    *u_subgrid*/*v_subgrid* carry only the cells the payload keeps and
    *linear_index* their full-grid row-major cell ids; :func:`stream_wind_at_heights`
    builds both once per block, because the transcendentals over the full grid
    would discard 15 of every 16 results at the default stride.

    Angles are the bearing the flow blows TOWARD, degrees clockwise from North —
    the meteorological "comes FROM" bearing is ``(angle + 180) % 360``. Rounding
    angles to 1 decimal and magnitudes to 2 is the convention of the standalone
    overlay files (``geojson.create_wind_vectors_json``); anything finer is
    float64 interpolation noise that only inflates the POT_EOLICO values files.
    """
    magnitude = np.hypot(u_subgrid, v_subgrid)
    flow_bearing_deg = np.arctan2(u_subgrid, v_subgrid) * 180.0 / np.pi
    flow_bearing_deg = np.where(flow_bearing_deg < 0, flow_bearing_deg + 360.0, flow_bearing_deg)

    angles_flat = flow_bearing_deg.ravel()
    mags_flat = magnitude.ravel()

    valid = ~np.isnan(angles_flat)

    # float64 before rounding: rounding a float32 array snaps to the nearest
    # float32 (320.6 -> 320.6000061...) and defeats the compact serialization.
    return {
        "downsampled_angles": np.round(angles_flat[valid].astype(np.float64), 1).tolist(),
        "downsampled_magnitudes": np.round(mags_flat[valid].astype(np.float64), 2).tolist(),
        "downsampled_linear_indices": linear_index[valid].tolist(),
    }


def stream_wind_at_heights(
    ds: WRFDataset,
    targets: tuple[int, ...] = (50, 100, 150),
    *,
    block_steps: int = DEFAULT_STREAM_BLOCK_STEPS,
    downsampling: int = 4,
) -> list[WindHeightSeries]:
    """Compute wind speed and wind vectors at *targets* heights, block-streamed.

    Reads U/V/PH/PHB/HGT in ``block_steps``-sized time blocks, so peak memory is
    bounded by the block instead of the file's full time dimension, and takes one
    bracket pass per block and grid: speed on the full grid, u/v on the
    ``downsampling`` subgrid the wind-vector payload keeps. Arithmetic matches
    the eager whole-array path bit-for-bit (float32 chain, same operand order),
    which the byte-diff gates pin. Requires an eager
    :class:`~micrometeorology.wrf.reader.WRFDataset` (uses ``get_variable_block``).

    The staggered U/V are averaged onto the mass points and rotated onto true
    north, as :func:`extract_wind` does at 10 m. The interpolation heights are
    geopotential height ``(PH + PHB) / 9.81`` in meters, averaged from the
    staggered w-levels onto the level centres and referenced to the terrain
    ``HGT``, so *targets* are meters above ground level and not above sea level.

    Parameters
    ----------
    ds:
        Open wrfout carrying U, V, PH, PHB and HGT.
    targets:
        Heights in meters above ground level, one series each.
    block_steps:
        Time steps per streaming block.
    downsampling:
        Stride of the arrow subgrid the wind-vector payload keeps; 1 keeps every
        cell.

    Returns
    -------
    list[WindHeightSeries]
        One series per target, in *targets* order.

    Raises
    ------
    ValueError
        When *block_steps* is not positive.
    """
    from micrometeorology.wrf.interpolation import VerticalInterpolator

    if block_steps <= 0:
        raise ValueError("block_steps must be positive")

    n_t = ds.n_time_steps

    speed_out: dict[int, NDArray] = {}
    vectors_out: dict[int, list[dict | None]] = {t: [] for t in targets}

    for t0 in range(0, n_t, block_steps):
        t1 = min(t0 + block_steps, n_t)
        # Staggering is an allocation plus an in-place second operation; the
        # chained ``(a + b) / 2.0`` form would materialize a second full block.
        u_raw = ds.get_variable_block("U", t0, t1)
        u_c = u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]
        u_c /= 2.0
        del u_raw
        v_raw = ds.get_variable_block("V", t0, t1)
        v_c = v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]
        v_c /= 2.0
        del v_raw
        u_c, v_c = rotate_to_earth_relative(ds, u_c, v_c, t0, t1)

        ph = ds.get_variable_block("PH", t0, t1)
        phb = ds.get_variable_block("PHB", t0, t1)
        height = ph + phb
        height /= 9.81
        del ph, phb
        height_agl = height[:, :-1, :, :] + height[:, 1:, :, :]
        height_agl /= 2.0
        del height
        hgt = ds.get_variable_block("HGT", t0, t1)
        height_agl -= hgt[:, np.newaxis, :, :]
        del hgt

        speed_4d = np.hypot(u_c, v_c)
        ny, nx = speed_4d.shape[2], speed_4d.shape[3]
        linear_index = (
            np.arange(0, ny, downsampling)[:, np.newaxis] * nx
            + np.arange(0, nx, downsampling)[np.newaxis, :]
        ).ravel()

        target_heights = [float(target) for target in targets]
        interpolator = VerticalInterpolator(height_agl, axis=1)
        speed_at_targets = interpolator.interpolate_many(speed_4d, target_heights)

        # Only the wind-vector subgrid of u/v is serialized and the bracket is
        # per-column, so interpolating the strided components is bit-identical at
        # 1/downsampling^2 of the cost; a contiguous copy beats a strided view.
        if downsampling > 1:
            subgrid_interpolator = VerticalInterpolator(
                np.ascontiguousarray(height_agl[:, :, ::downsampling, ::downsampling]), axis=1
            )
            u_subgrid = np.ascontiguousarray(u_c[:, :, ::downsampling, ::downsampling])
            v_subgrid = np.ascontiguousarray(v_c[:, :, ::downsampling, ::downsampling])
        else:
            subgrid_interpolator = interpolator
            u_subgrid, v_subgrid = u_c, v_c
        del u_c, v_c
        u_at_targets = subgrid_interpolator.interpolate_many(u_subgrid, target_heights)
        v_at_targets = subgrid_interpolator.interpolate_many(v_subgrid, target_heights)
        del u_subgrid, v_subgrid

        for target_index, target in enumerate(targets):
            if target not in speed_out:
                speed_out[target] = np.empty((n_t, ny, nx), dtype=speed_4d.dtype)
            speed_out[target][t0:t1] = speed_at_targets[target_index]
            u_3d = u_at_targets[target_index]
            v_3d = v_at_targets[target_index]
            for k in range(t1 - t0):
                try:
                    vectors_out[target].append(
                        _package_wind_vectors_step(u_3d[k], v_3d[k], linear_index)
                    )
                except Exception:  # noqa: BLE001 - one bad step must not sink the whole series
                    logger.warning(
                        "Wind vector packaging failed for step %d at %dm", t0 + k, target
                    )
                    vectors_out[target].append(None)
        del height_agl, speed_4d, interpolator, subgrid_interpolator
        del speed_at_targets, u_at_targets, v_at_targets

    series: list[WindHeightSeries] = []
    for target in targets:
        speed = speed_out[target]
        vmin, vmax = percentile_scale_bounds(speed)
        series.append(
            WindHeightSeries(
                target=target,
                vmin=vmin,
                vmax=vmax,
                speed_steps=speed,
                wind_vectors=vectors_out[target],
            )
        )
    return series
