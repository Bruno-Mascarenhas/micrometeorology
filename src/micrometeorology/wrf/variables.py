"""WRF variable extraction and unit conversion.

Consolidates the repeated per-variable extraction logic that was
duplicated across the ``drawmap()`` functions in the legacy scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from micrometeorology.wrf.safety import assert_reasonable_array_size

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from micrometeorology.wrf.reader import WRFDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Min / max helpers (preserved from legacy getLowHigh* functions)
# ---------------------------------------------------------------------------


def _tail(value: NDArray) -> NDArray:
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


def get_low_high(variable: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` of a 3-D variable, skipping the first time step.

    Single-timestep inputs fall back to the full array (see :func:`_tail`).
    """
    flat = _tail(variable).ravel()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: NDArray, v: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays (skip first step).

    Single-timestep inputs fall back to the full arrays (see :func:`_tail`).
    """
    speed = np.hypot(_tail(u).ravel(), _tail(v).ravel())
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


def extract_temperature(ds: WRFDataset) -> tuple[NDArray, NDArray, float, float]:
    """Extract 2-m temperature (°C) and surface pressure (hPa).

    Returns ``(temperature_3d, pressure_3d, temp_min, temp_max)`` where
    temperature values are in °C and pressure in hPa.
    """
    t2 = ds.get_variable("T2")  # Kelvin
    psfc = ds.get_variable("PSFC")  # Pa

    t_min, t_max = get_low_high(t2)
    t_min -= 273.15
    t_max -= 273.15

    return t2, psfc / 100.0, t_min, t_max


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
    t_min, t_max = get_low_high(tsk)
    return tsk, t_min - 273.15, t_max - 273.15


def extract_pressure(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract surface pressure (hPa)."""
    psfc = ds.get_variable("PSFC")
    p_min, p_max = get_low_high(psfc)
    return psfc / 100.0, p_min / 100.0, p_max / 100.0


def extract_vapor(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract 2-m water-vapor mixing ratio (g/kg).

    WRF metadata describes ``Q2`` as ``QV at 2 M`` with units ``kg kg-1``.
    The exported site variable keeps the legacy ``VAPOR`` id but values are
    converted to g/kg.
    """
    q2 = ds.get_variable("Q2")
    q_min, q_max = get_low_high(q2)
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
    rh_min, rh_max = get_low_high(rh)
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
    v_min, v_max = get_low_high(var)
    return var, v_min, v_max


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
    p_min, p_max = get_low_high(power_density)
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
    u_target: NDArray,
    v_target: NDArray,
    ny: int,
    nx: int,
    downsampling: int,
) -> dict:
    """Package one timestep's wind vectors.

    Angles are rounded to 1 decimal and magnitudes to 2 — the same convention
    as the standalone overlay files (``geojson.create_wind_vectors_json``).
    The front-end only draws arrows from these numbers; anything beyond
    0.1°/0.01 m/s is float64 interpolation noise, and serializing it used to
    inflate every POT_EOLICO values file by ~21%.
    """
    magnitude = np.hypot(u_target, v_target)
    angle = np.arctan2(u_target, v_target) * 180.0 / np.pi
    angle = np.where(angle < 0, angle + 360.0, angle)

    i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    angles_flat = angle[i_flat, j_flat]
    mags_flat = magnitude[i_flat, j_flat]

    valid = ~np.isnan(angles_flat)
    linear_indices = (i_flat * nx + j_flat)[valid]

    # float64 before rounding: rounding a float32 array snaps to the nearest
    # float32 (320.6 -> 320.6000061...), which would defeat the compact
    # serialization; the standalone overlay path casts the same way.
    return {
        "downsampled_angles": np.round(angles_flat[valid].astype(np.float64), 1).tolist(),
        "downsampled_magnitudes": np.round(mags_flat[valid].astype(np.float64), 2).tolist(),
        "downsampled_linear_indices": linear_indices.tolist(),
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
    interpolates u/v/speed for all target heights from ONE bracket pass per
    block. Arithmetic matches the eager whole-array path bit-for-bit (float32
    chain, same operand order), which the byte-diff gates pin.

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
        u_raw = ds.get_variable_block("U", t0, t1)
        u_c = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0
        del u_raw
        v_raw = ds.get_variable_block("V", t0, t1)
        v_c = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0
        del v_raw

        ph = ds.get_variable_block("PH", t0, t1)
        phb = ds.get_variable_block("PHB", t0, t1)
        height = (ph + phb) / 9.81
        del ph, phb
        height_c = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0
        del height
        hgt = ds.get_variable_block("HGT", t0, t1)
        height_adjusted = height_c - hgt[:, np.newaxis, :, :]
        del height_c, hgt

        speed_4d = np.hypot(u_c, v_c)
        ny, nx = speed_4d.shape[2], speed_4d.shape[3]

        interpolator = VerticalInterpolator(height_adjusted, axis=1)
        for target in targets:
            if target not in speed_out:
                speed_out[target] = np.empty((n_t, ny, nx), dtype=speed_4d.dtype)
            speed_out[target][t0:t1] = interpolator.interpolate(speed_4d, float(target))
            u_3d = interpolator.interpolate(u_c, float(target))
            v_3d = interpolator.interpolate(v_c, float(target))
            for k in range(t1 - t0):
                try:
                    vectors_out[target].append(
                        _package_wind_vectors_step(u_3d[k], v_3d[k], ny, nx, downsampling)
                    )
                except Exception:
                    logger.warning(
                        "Wind vector packaging failed for step %d at %dm", t0 + k, target
                    )
                    vectors_out[target].append(None)
        del u_c, v_c, height_adjusted, speed_4d, interpolator

    series: list[WindHeightSeries] = []
    for target in targets:
        speed = speed_out[target]
        # Scale bounds follow the site-wide convention (get_low_high): skip the
        # spin-up first step and cap the max at the 98th percentile.
        vmin, vmax = get_low_high(speed)
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
