"""WRF variable extraction and unit conversion.

Consolidates the repeated per-variable extraction logic that was
duplicated across the ``drawmap()`` functions in the legacy scripts.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from micrometeorology.wrf.reader import WRFDataset
from micrometeorology.wrf.safety import assert_reasonable_array_size

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Min / max helpers (preserved from legacy getLowHigh* functions)
# ---------------------------------------------------------------------------


def _drop_spinup_step(value: NDArray) -> NDArray:
    """Drop the spin-up first time step.

    When the time axis has <= 1 entries (single-timestep files) the full
    array is returned instead, so reductions never see an empty tail.
    """
    if value.shape[0] <= 1:
        return value
    return value[1:, :]


def squeeze_array(value: NDArray) -> NDArray:
    """Squeeze an ndarray."""
    return np.squeeze(value)


def materialize_2d(value: NDArray) -> NDArray:
    """Validate and return a 2-D worker payload."""
    squeezed = np.squeeze(value)
    if squeezed.ndim != 2:
        raise ValueError(f"Expected a 2-D worker payload, got shape {squeezed.shape!r}")
    assert_reasonable_array_size(squeezed.shape, squeezed.dtype, context="materialize_2d")
    return np.asarray(squeezed)


def percentile_scale_bounds(variable: NDArray) -> tuple[float, float]:
    """Return color-scale bounds ``(low, high)`` for a 3-D variable, skipping the first step.

    The lower bound is the true minimum, but the **upper bound is the
    98th-percentile saturation cap, not the maximum**: capping there keeps a
    handful of extreme cells from blowing out the map color scale so the bulk of
    the field stays legible. Callers use the pair as ``(vmin, vmax)`` for
    rendering, not as the data's actual range.

    Single-timestep inputs fall back to the full array (see :func:`_drop_spinup_step`).
    """
    flat = _drop_spinup_step(variable).ravel()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: NDArray, v: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays (skip first step).

    Single-timestep inputs fall back to the full arrays (see :func:`_drop_spinup_step`).
    """
    speed = np.hypot(_drop_spinup_step(u).ravel(), _drop_spinup_step(v).ravel())
    return float(np.nanmin(speed)), float(np.nanmax(speed))


def get_low_high_rain(variable: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` of incremental precipitation.

    The input is *cumulative* rain; we compute the per-step increment first.
    """
    arr = np.asarray(variable)
    if arr.ndim < 3:
        flat = arr.ravel()
        return float(np.nanmin(flat)), float(np.nanmax(flat))
    diffs = np.diff(arr, axis=0)
    if diffs.size == 0:
        return 0.0, 0.0
    flat = diffs.ravel()
    return float(np.nanmin(flat)), float(np.nanmax(flat))


# ---------------------------------------------------------------------------
# Variable extractors
# ---------------------------------------------------------------------------


def extract_temperature(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract 2-m temperature (°C).

    Returns ``(temperature_3d, temp_min, temp_max)`` with the raw Kelvin array
    (converted per step by :func:`extract_temperature_step`) and °C bounds.
    Callers that also need surface pressure read PSFC themselves — bundling it
    here forced every temperature export to eagerly load a variable it never
    used.
    """
    t2 = ds.get_variable("T2")  # Kelvin

    t_min, t_max = percentile_scale_bounds(t2)
    t_min -= 273.15
    t_max -= 273.15

    return t2, t_min, t_max


def extract_temperature_step(t2_step: NDArray) -> NDArray:
    """Convert a single time-step of T2 from Kelvin to Celsius."""
    return squeeze_array(t2_step) - 273.15


def extract_skin_temperature(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract WRF surface skin temperature.

    Input variable
        ``TSK`` in Kelvin.
    Formula
        ``TSK_C = TSK_K - 273.15``.
    Output
        Surface/skin temperature in degrees Celsius.
    Limitation
        This is the model skin temperature, not a 2-m air temperature and not
        an observed land-surface temperature product.
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
    """Extract 2-m water-vapor mixing ratio (g/kg).

    WRF metadata describes ``Q2`` as ``QV at 2 M`` with units ``kg kg-1``.
    The exported site variable keeps the legacy ``VAPOR`` id but values are
    converted to g/kg.
    """
    q2 = ds.get_variable("Q2")
    q_min, q_max = percentile_scale_bounds(q2)
    return q2 * 1000.0, q_min * 1000.0, q_max * 1000.0


def compute_relative_humidity(q2: NDArray, t2: NDArray, psfc: NDArray) -> NDArray:
    """Compute 2-m relative humidity from WRF near-surface fields.

    Input variables
        ``Q2`` water-vapor mixing ratio (kg/kg), ``T2`` air temperature (K),
        and ``PSFC`` surface pressure (Pa).
    Formula
        Vapor pressure is estimated as ``e = q * p / (epsilon + q)`` using
        ``epsilon = 0.622``. Saturation vapor pressure over water follows the
        Bolton/Tetens form ``es = 611.2 * exp(17.67 * Tc / (Tc + 243.5))``.
        Relative humidity is ``100 * e / es``.
    Output
        Relative humidity in percent, clipped to the physical display range
        0-100%.
    Limitation
        The calculation assumes Q2 is a mixing ratio, matching WRF's QV
        convention. It is a near-surface diagnostic, not a vertically integrated
        humidity field.
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


def extract_wind(ds: WRFDataset) -> tuple[NDArray, NDArray, float, float]:
    """Extract 10-m U/V wind components and compute speed bounds."""
    u10 = ds.get_variable("U10")
    v10 = ds.get_variable("V10")
    ws_min, ws_max = get_low_high_wind(u10, v10)
    return u10, v10, ws_min, ws_max


def extract_rain(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract total precipitation (convective + non-convective, cumulative)."""
    rainc = ds.get_variable("RAINC")
    rainnc = ds.get_variable("RAINNC")
    total = rainc + rainnc
    r_min, r_max = get_low_high_rain(total)
    return total, r_min, r_max


def extract_rain_step(total: NDArray, i: int) -> NDArray:
    """Compute incremental rain for step *i* from cumulative totals.

    Step 0 has no computable increment at a file/restart boundary and
    publishes zeros; every later step publishes ``total[i] - total[i - 1]``.
    """
    if i == 0:
        return np.zeros_like(np.squeeze(total[i : i + 1, :, :]))
    current: NDArray = np.squeeze(total[i : i + 1, :, :])
    previous: NDArray = np.squeeze(total[i - 1 : i, :, :])
    increment: NDArray = current - previous
    return increment


def extract_scalar(ds: WRFDataset, var_name: str) -> tuple[NDArray, float, float]:
    """Generic extractor for scalar fields (HFX, LH, SWDOWN)."""
    var = ds.get_variable(var_name)
    v_min, v_max = percentile_scale_bounds(var)
    return var, v_min, v_max


# ---------------------------------------------------------------------------
# Derived surface radiation budget
#
# wrfout files carry the DOWNWELLING fluxes as diagnostics (SWDOWN, GLW) but
# the upwelling ones only when the RRTMG bottom-of-atmosphere outputs are
# switched on: LWUPB/SWUPB are present in the 2013 archive runs and absent
# from the 2026 operational runs. Everything below is therefore reconstructed
# from fields every wrfout generation carries — EMISS, TSK, GLW, SWDOWN,
# ALBEDO, T2, COSZEN — so a budget term publishes identically across runs.
#
# The reconstruction is validated against WRF's own fluxes where those exist
# (d02 2013, 9801 cells x 9 steps): LWUP reproduces LWUPB to MAE 0.82 W/m2
# (0.11% of a ~420 W/m2 signal) and SWUP reproduces SWUPB exactly.
# ---------------------------------------------------------------------------

#: Stefan-Boltzmann constant, W m-2 K-4 (CODATA 2018).
STEFAN_BOLTZMANN = 5.670374419e-8

#: Total solar irradiance at 1 AU, W m-2. Kopp & Lean (2011) measured
#: 1360.8 +- 0.5; matches ``allsky.solar.SOLAR_CONSTANT_WM2`` so the WRF and
#: all-sky-camera clearness indices are on one scale. Older references
#: (Duffie & Beckman, quoting the WRC) still carry 1367, which would shift kt
#: by 0.4%.
SOLAR_CONSTANT = 1361.0

#: Sun-elevation floor for the clearness index. Below it the extraterrestrial
#: denominator is small enough that ``kt`` is dominated by the horizon
#: singularity rather than by sky condition, so those cells publish "no value".
#:
#: cos(z) = 0.1736 is a solar zenith angle of 80 degrees (elevation 10), the
#: daytime threshold used by the ARM best-estimate radiation VAP (Shi & Long,
#: DOE/SC-ARM/TR-008); BSRN's component tests tighten further at SZA < 75.
#: Those thresholds exist to bound pyranometer cosine-response error, which
#: model output does not have — but the same grazing geometry is also where
#: WRF's plane-parallel radiative transfer is least trustworthy, so the
#: threshold carries over for a different reason. On a 76-step operational d04
#: run this keeps 30 of the 39 daylight frames and ~77% of their cells.
MIN_COSZEN_FOR_CLEARNESS = 0.1736


def _eccentricity_correction(times: list[datetime]) -> NDArray:
    """Sun-Earth distance correction ``E0 = (r0/r)^2`` per time step (Spencer 1971).

    Same series as :func:`allsky.solar.eccentricity_correction`; duplicated
    rather than imported because ``micrometeorology`` and ``allsky`` are
    independent packages. E0 runs 0.9666 (aphelion, early July) to 1.0351
    (perihelion, early January) — a +-3.4% swing that is not negligible for a
    clearness index, which is why COSZEN alone is not the denominator.
    """
    fractional_year = np.array(
        [
            2.0
            * np.pi
            / 365.0
            * (
                t.timetuple().tm_yday
                - 1.0
                + (t.hour + t.minute / 60.0 + t.second / 3600.0 - 12.0) / 24.0
            )
            for t in times
        ],
        dtype=np.float64,
    )
    correction: NDArray = (
        1.000110
        + 0.034221 * np.cos(fractional_year)
        + 0.001280 * np.sin(fractional_year)
        + 0.000719 * np.cos(2.0 * fractional_year)
        + 0.000077 * np.sin(2.0 * fractional_year)
    )
    return correction


def _finite_scale_bounds(values: NDArray, fallback: tuple[float, float]) -> tuple[float, float]:
    """Percentile scale bounds, falling back when the field is entirely no-value.

    Only the clearness index can reach this: an all-night file leaves every
    cell NaN, ``nanmin``/``nanpercentile`` return NaN, and the values writer
    rejects non-finite bounds outright. Those steps are gated out and never
    published, but the bounds are computed eagerly before the gate runs.
    """
    with warnings.catch_warnings():
        # nanmin/nanpercentile warn (and return NaN) on an all-NaN input; the
        # NaN is the signal we act on below, so the warning is noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = percentile_scale_bounds(values)
    if not (np.isfinite(low) and np.isfinite(high)):
        return fallback
    return low, high


def compute_upwelling_longwave(emiss: NDArray, tsk: NDArray, glw: NDArray) -> NDArray:
    """Upwelling longwave radiation at the surface.

    Input variables
        ``EMISS`` surface emissivity (0-1), ``TSK`` skin temperature (K), and
        ``GLW`` downwelling longwave flux at the ground (W/m2).
    Formula
        ``LWup = eps * sigma * TSK^4 + (1 - eps) * GLW``. The first term is the
        surface's own graybody emission; the second is the fraction of the
        incoming sky radiance a non-black surface reflects. Dropping the
        reflected term is the common mistake and costs ~15 W/m2 here — nearly
        20x the error of the full form.

        This is not an approximation of what WRF does, it is what WRF does.
        RRTMG's ``rtrnmc`` forms the upward radiance per band as
        ``radlu = rad0 + reflect * radld`` with ``rad0 = semiss * B(Ts)`` and
        ``reflect = 1 - semiss`` (``phys/module_ra_rrtmg_lw.F``, under
        "Add in specular reflection of surface downward radiance"), and the
        urban branch of Noah writes the same expression out longhand:
        ``rl_up_rural = -emiss*sigma*tsk**4 - (1.-emiss)*glw``
        (``phys/module_sf_noahdrv.F``). Noah's own net-longwave term
        ``emiss*GLW - emiss*sigma*T1**4`` is the same identity with the
        reflected part already cancelled, not a scheme that ignores it.
    Output
        Upwelling longwave flux in W/m2, positive upward.
    Limitation
        Uses the grid-mean skin temperature. WRF's own LWUPB is computed by the
        radiation scheme from the LSM's canopy-adjusted radiative temperature,
        so over land the two differ by ~1.7 W/m2 (over water, ~0.03 W/m2).
        A second, far smaller discrepancy is the constant itself: WRF carries
        ``STBOLT = 5.67051e-8`` (``share/module_model_constants.F``) and Noah
        rounds to ``5.67e-8``, against the exact CODATA value used here — worth
        ~0.01 W/m2, i.e. 1% of the land bias above.
        This is a model diagnostic, not an observed flux.
    """
    upwelling: NDArray = emiss * STEFAN_BOLTZMANN * tsk**4 + (1.0 - emiss) * glw
    return upwelling


def compute_upwelling_shortwave(albedo: NDArray, swdown: NDArray) -> NDArray:
    """Upwelling (reflected) shortwave radiation at the surface.

    Input variables
        ``ALBEDO`` surface albedo (0-1) and ``SWDOWN`` downwelling shortwave
        flux at the ground (W/m2).
    Formula
        ``SWup = ALBEDO * SWDOWN`` — exactly how WRF forms SWUPB internally,
        which is why this reproduces that field bit-for-bit where it exists.
    Output
        Reflected shortwave flux in W/m2, positive upward.
    Limitation
        ``ALBEDO`` is the broadband all-sky surface albedo; no direct/diffuse
        or spectral split is applied.
    """
    reflected: NDArray = albedo * swdown
    return reflected


def compute_net_shortwave(albedo: NDArray, swdown: NDArray) -> NDArray:
    """Net (absorbed) shortwave radiation at the surface.

    Formula
        ``SWnet = SWDOWN * (1 - ALBEDO)``, the shortwave the surface keeps.
    Output
        Net shortwave flux in W/m2, positive downward (into the surface).
        Zero at night.
    """
    return swdown * (1.0 - albedo)


def compute_net_longwave(emiss: NDArray, tsk: NDArray, glw: NDArray) -> NDArray:
    """Net longwave radiation at the surface.

    Formula
        ``LWnet = GLW - LWup``, which reduces algebraically to
        ``eps * (GLW - sigma * TSK^4)`` — the reflected ``(1 - eps) * GLW``
        term cancels exactly against the same term in ``LWup``. Evaluated in
        the reduced form: it is the identity a reader should see, and it does
        half the arithmetic. Float rounding makes it agree with
        ``GLW - compute_upwelling_longwave(...)`` to within an ulp or so
        rather than bit-for-bit.
    Output
        Net longwave flux in W/m2, positive downward. Almost always negative:
        the surface is warmer than the effective sky and loses longwave both
        day and night.
    """
    return emiss * (glw - STEFAN_BOLTZMANN * tsk**4)


def compute_net_radiation(
    swdown: NDArray, albedo: NDArray, emiss: NDArray, tsk: NDArray, glw: NDArray
) -> NDArray:
    """Net all-wave radiation at the surface.

    Formula
        ``Rn = SWnet + LWnet = SWDOWN * (1 - ALBEDO) + eps * (GLW - sigma * TSK^4)``.
    Output
        Net radiation in W/m2, positive downward. This is the energy actually
        available to drive the sensible (HFX), latent (LH) and ground (GRDFLX)
        heat fluxes.
    Limitation
        Inherits the skin-temperature caveat of
        :func:`compute_upwelling_longwave`. It is not reconcilable cell-by-cell
        with ``NOAHRES``, which is Noah's internal per-tile residual computed
        against the canopy temperature rather than against these grid-mean
        fields.
    """
    net: NDArray = compute_net_shortwave(albedo, swdown) + compute_net_longwave(emiss, tsk, glw)
    return net


def compute_sky_emissivity(glw: NDArray, t2: NDArray) -> NDArray:
    """Effective (bulk) emissivity of the sky hemisphere.

    Input variables
        ``GLW`` downwelling longwave flux (W/m2) and ``T2`` 2-m air
        temperature (K).
    Formula
        ``eps_sky = GLW / (sigma * T2^4)`` — the emissivity a blackbody at
        screen-level air temperature would need to emit the observed
        downwelling flux.
    Output
        Dimensionless effective emissivity, typically ~0.75 under a dry clear
        sky and approaching 1 under a warm overcast, which makes it a
        convenient cloudiness proxy.
    Limitation
        Uses 2-m air temperature as the radiating temperature of the whole
        atmospheric column, so it absorbs any near-surface inversion into the
        emissivity. Values are not clipped: the first output step of a run can
        carry ``GLW = 0`` before radiation is first called, which shows up as
        ``eps_sky = 0``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        emissivity: NDArray = glw / (STEFAN_BOLTZMANN * t2**4)
    return emissivity


def compute_clearness_index(swdown: NDArray, coszen: NDArray, eccentricity: NDArray) -> NDArray:
    """Extraterrestrial clearness index ``kt`` at the surface.

    Input variables
        ``SWDOWN`` downwelling shortwave (W/m2), ``COSZEN`` cosine of the solar
        zenith angle, and the per-step Sun-Earth distance correction.
    Formula
        ``kt = SWDOWN / (S0 * E0 * cos(z))``, the extraterrestrial-normalized
        index of Duffie & Beckman eq. (1.10.1). Same definition as
        :func:`allsky.solar.clearness_index`, so WRF and the all-sky camera
        pipeline can be compared on one axis. Lower case ``kt`` is deliberate:
        Duffie & Beckman sect. 2.9 reserve it for the sub-daily index and
        upper-case ``K_T`` for the daily one, and this field is instantaneous.
        Do not compare these values against the "typical 0.25-0.75" figures
        quoted for monthly-mean daily ``K_T`` — daily integration includes the
        low-sun hours and sits systematically lower.
    Output
        Dimensionless clearness index, ~0 under thick cloud and approaching
        ~0.8 under a high clear sky. Cells where the sun is below
        :data:`MIN_COSZEN_FOR_CLEARNESS` publish no value (``null`` in the
        values JSON, ``MISSING`` in the cell-series matrix).
    Limitation
        Not clipped at 1: cloud-edge enhancement genuinely produces ``kt > 1``.
        Duffie & Beckman note there are very few observations above kt = 0.80
        at all, so sustained field values well above it point at the inputs
        rather than at an unusually clear sky. ``COSZEN`` is negative at night,
        which the elevation floor screens out along with the singularity.
    """
    denominator = SOLAR_CONSTANT * eccentricity * coszen
    clearness = np.full(swdown.shape, np.nan, dtype=np.float64)
    sun_up = coszen >= MIN_COSZEN_FOR_CLEARNESS
    np.divide(swdown, denominator, out=clearness, where=sun_up)
    return clearness


def extract_upwelling_longwave(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived upwelling longwave radiation (W/m2)."""
    values = compute_upwelling_longwave(
        ds.get_variable("EMISS"), ds.get_variable("TSK"), ds.get_variable("GLW")
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
    values = compute_net_longwave(
        ds.get_variable("EMISS"), ds.get_variable("TSK"), ds.get_variable("GLW")
    )
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_net_radiation(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived net all-wave radiation (W/m2, positive downward)."""
    values = compute_net_radiation(
        ds.get_variable("SWDOWN"),
        ds.get_variable("ALBEDO"),
        ds.get_variable("EMISS"),
        ds.get_variable("TSK"),
        ds.get_variable("GLW"),
    )
    low, high = percentile_scale_bounds(values)
    return values, low, high


def extract_sky_emissivity(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract derived effective sky emissivity (dimensionless)."""
    values = compute_sky_emissivity(ds.get_variable("GLW"), ds.get_variable("T2"))
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
        eccentricity = _eccentricity_correction(times)[:, np.newaxis, np.newaxis]
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

    Input variables
        ``T2`` air temperature (K), ``PSFC`` surface pressure (Pa), and ``Q2``
        water-vapor mixing ratio (kg/kg).
    Formula
        Virtual temperature ``Tv = T2 * (1 + 0.61 * q)`` and ideal gas law
        ``rho = p / (Rd * Tv)``, with ``Rd = 287.05 J kg-1 K-1``.
    Output
        Air density in kg/m3.
    Limitation
        This is a near-surface density estimate. It should not be treated as
        density at turbine hub height without a vertical thermodynamic profile.
    """
    virtual_temperature = t2 * (1.0 + 0.61 * q2)
    return psfc / (287.05 * virtual_temperature)


def extract_wind_power_density_10m(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Compute wind power density at 10 m.

    Input variables
        ``U10`` and ``V10`` in m/s, plus ``T2`` (K), ``PSFC`` (Pa), and ``Q2``
        (kg/kg) for moist-air density.
    Formula
        ``speed = sqrt(U10^2 + V10^2)`` and
        ``WPD = 0.5 * rho * speed^3``.
    Output
        Wind power density in W/m2 at 10 m.
    Limitation
        This is available power density in the wind flow, not turbine output.
        It does not include rotor area, power coefficient, cut-in/cut-out, or
        hub-height extrapolation.
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


# ---------------------------------------------------------------------------
# Block-streamed wind-at-height extraction (bounded memory for long files)
# ---------------------------------------------------------------------------

DEFAULT_STREAM_BLOCK_STEPS = 64


@dataclass(frozen=True)
class WindHeightSeries:
    """Interpolated wind speed series and per-step wind vectors for one height."""

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
    *linear_index* their full-grid row-major cell ids; both are built once per
    block by :func:`stream_wind_at_heights`, because computing the
    transcendentals over the full grid would discard 15 of every 16 results at
    the default stride.

    Angles are the bearing the flow blows TOWARD, degrees clockwise from North
    — the meteorological "comes FROM" bearing is ``(angle + 180) % 360``.
    They are rounded to 1 decimal and magnitudes to 2 — the same convention
    as the standalone overlay files (``geojson.create_wind_vectors_json``).
    The front-end only draws arrows from these numbers; anything beyond
    0.1°/0.01 m/s is float64 interpolation noise, and serializing it used to
    inflate every POT_EOLICO values file by ~21%.
    """
    magnitude = np.hypot(u_subgrid, v_subgrid)
    flow_bearing_deg = np.arctan2(u_subgrid, v_subgrid) * 180.0 / np.pi
    flow_bearing_deg = np.where(flow_bearing_deg < 0, flow_bearing_deg + 360.0, flow_bearing_deg)

    angles_flat = flow_bearing_deg.ravel()
    mags_flat = magnitude.ravel()

    valid = ~np.isnan(angles_flat)

    # float64 before rounding: rounding a float32 array snaps to the nearest
    # float32 (320.6 -> 320.6000061...), which would defeat the compact
    # serialization; the standalone overlay path casts the same way.
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

    Reads U/V/PH/PHB/HGT in ``block_steps``-sized time blocks so peak memory is
    bounded by the block size instead of the file's full time dimension, and
    interpolates speed (full grid) and u/v (the ``downsampling`` subgrid the
    wind-vector payload keeps) for all target heights from one bracket pass per
    block and grid. Arithmetic matches the eager whole-array path bit-for-bit
    (float32 chain, same operand order), which the byte-diff gates pin.

    Requires an eager :class:`~micrometeorology.wrf.reader.WRFDataset` (uses
    ``get_variable_block``).
    """
    from micrometeorology.wrf.interpolation import VerticalInterpolator

    if block_steps <= 0:
        raise ValueError("block_steps must be positive")

    n_t = ds.n_time_steps

    speed_out: dict[int, NDArray] = {}
    vectors_out: dict[int, list[dict | None]] = {t: [] for t in targets}

    for t0 in range(0, n_t, block_steps):
        t1 = min(t0 + block_steps, n_t)
        # Each staggering step is split into an allocation plus an in-place
        # second operation: the chained ``(a + b) / 2.0`` form materialized a
        # second full-size block per line for nothing.
        u_raw = ds.get_variable_block("U", t0, t1)
        u_c = u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]
        u_c /= 2.0
        del u_raw
        v_raw = ds.get_variable_block("V", t0, t1)
        v_c = v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]
        v_c /= 2.0
        del v_raw

        ph = ds.get_variable_block("PH", t0, t1)
        phb = ds.get_variable_block("PHB", t0, t1)
        height = ph + phb
        height /= 9.81
        del ph, phb
        height_agl = height[:, :-1, :, :] + height[:, 1:, :, :]
        height_agl /= 2.0
        del height
        hgt = ds.get_variable_block("HGT", t0, t1)
        height_agl -= hgt[:, np.newaxis, :, :]  # level-centered height -> meters AGL
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

        # Only the wind-vector subgrid of u/v is ever serialized, and the
        # bracket is per-column, so interpolating the strided components is
        # bit-identical at 1/downsampling^2 of the cost. A contiguous copy of
        # the subgrid beats handing the interpolator a strided view.
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
                except Exception:
                    logger.warning(
                        "Wind vector packaging failed for step %d at %dm", t0 + k, target
                    )
                    vectors_out[target].append(None)
        del height_agl, speed_4d, interpolator, subgrid_interpolator
        del speed_at_targets, u_at_targets, v_at_targets

    series: list[WindHeightSeries] = []
    for target in targets:
        speed = speed_out[target]
        # Scale bounds follow the site-wide convention (percentile_scale_bounds):
        # skip the spin-up first step and cap the max at the 98th percentile.
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
