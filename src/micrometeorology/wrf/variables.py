"""WRF variable extraction and unit conversion.

Consolidates the repeated per-variable extraction logic that was
duplicated across the ``drawmap()`` functions in the legacy scripts.
"""

from __future__ import annotations

import logging
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

from micrometeorology.wrf.safety import (
    assert_reasonable_array_size,
    destagger_dataarray,
    safe_binary_op,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from micrometeorology.wrf.reader import WRFReader

logger = logging.getLogger(__name__)

if not TYPE_CHECKING:
    NDArray = Any

WRFArray = Any


# ---------------------------------------------------------------------------
# Min / max helpers (preserved from legacy getLowHigh* functions)
# ---------------------------------------------------------------------------


def _is_xarray(value: object) -> bool:
    return isinstance(value, xr.DataArray)


def _time_dim(value: xr.DataArray) -> str:
    return "Time" if "Time" in value.dims else str(value.dims[0])


def _tail(value: WRFArray) -> WRFArray:
    """Drop the spin-up first time step.

    When the time axis has <= 1 entries (single-timestep files) the full
    array is returned instead, so reductions never see an empty tail.
    """
    if _is_xarray(value):
        time_dim = _time_dim(value)
        if value.sizes[time_dim] <= 1:
            return value
        return value.isel({time_dim: slice(1, None)})
    if value.shape[0] <= 1:
        return value
    return value[1:, :]


def _as_float(value: Any) -> float:
    if _is_xarray(value):
        value = value.compute() if hasattr(value.data, "compute") else value
        return float(value.item())
    return float(value)


def squeeze_array(value: WRFArray) -> WRFArray:
    """Squeeze an ndarray or DataArray without materializing xarray data."""
    if _is_xarray(value):
        return value.squeeze(drop=True)
    return np.squeeze(value)


def materialize_2d(value: WRFArray) -> NDArray:
    """Materialize an ndarray/DataArray at the final 2-D worker payload boundary."""
    squeezed = squeeze_array(value)
    shape = tuple(int(size) for size in squeezed.shape)
    if len(shape) != 2:
        raise ValueError(f"Expected a 2-D worker payload, got shape {shape!r}")
    dtype = squeezed.dtype if _is_xarray(squeezed) else np.asarray(squeezed).dtype
    assert_reasonable_array_size(shape, dtype, context="materialize_2d")
    if _is_xarray(squeezed):
        return np.asarray(squeezed.to_numpy())
    return np.asarray(squeezed)


def materialize_nd(value: WRFArray) -> NDArray:
    """Materialize an ndarray/DataArray without changing dimensionality."""
    shape = tuple(int(size) for size in value.shape)
    dtype = value.dtype if _is_xarray(value) else np.asarray(value).dtype
    assert_reasonable_array_size(shape, dtype, context="materialize_nd")
    if _is_xarray(value):
        return np.asarray(value.to_numpy())
    return np.asarray(value)


def get_low_high(variable: WRFArray) -> tuple[float, float]:
    """Return ``(min, max)`` of a 3-D variable, skipping the first time step.

    Single-timestep inputs fall back to the full array (see :func:`_tail`).
    """
    if _is_xarray(variable):
        tail = _tail(variable)
        return _as_float(tail.min(skipna=True)), _as_float(tail.quantile(0.98, skipna=True))
    flat = _tail(variable).ravel()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: WRFArray, v: WRFArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays (skip first step).

    Single-timestep inputs fall back to the full arrays (see :func:`_tail`).
    """
    if _is_xarray(u) or _is_xarray(v):
        speed = safe_binary_op(
            _tail(u),
            _tail(v),
            np.hypot,
            context="wind speed bounds",
            result_dtype=float,
        )
        return _as_float(speed.min(skipna=True)), _as_float(speed.max(skipna=True))
    flat_u = _tail(u).ravel()
    flat_v = _tail(v).ravel()
    speed = np.hypot(flat_u, flat_v)
    return float(np.nanmin(speed)), float(np.nanmax(speed))


def get_low_high_rain(variable: WRFArray) -> tuple[float, float]:
    """Return ``(min, max)`` of incremental precipitation.

    The input is *cumulative* rain; we compute the per-step increment first.
    """
    if _is_xarray(variable):
        time_dim = _time_dim(variable)
        diffs = variable.diff(dim=time_dim)
        if diffs.size == 0:
            return 0.0, 0.0
        return _as_float(diffs.min(skipna=True)), _as_float(diffs.max(skipna=True))
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


def extract_temperature(ds: WRFReader) -> tuple[WRFArray, WRFArray, float, float]:
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


def extract_temperature_step(t2_step: WRFArray) -> WRFArray:
    """Convert a single time-step of T2 from Kelvin to Celsius."""
    return squeeze_array(t2_step) - 273.15


def extract_skin_temperature(ds: WRFReader) -> tuple[WRFArray, float, float]:
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


def extract_pressure(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract surface pressure (hPa)."""
    psfc = ds.get_variable("PSFC")
    p_min, p_max = get_low_high(psfc)
    return psfc / 100.0, p_min / 100.0, p_max / 100.0


def extract_vapor(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract 2-m water-vapor mixing ratio (g/kg).

    WRF metadata describes ``Q2`` as ``QV at 2 M`` with units ``kg kg-1``.
    The exported site variable keeps the legacy ``VAPOR`` id but values are
    converted to g/kg.
    """
    q2 = ds.get_variable("Q2")
    q_min, q_max = get_low_high(q2)
    return q2 * 1000.0, q_min * 1000.0, q_max * 1000.0


def compute_relative_humidity(q2: WRFArray, t2: WRFArray, psfc: WRFArray) -> WRFArray:
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
    numerator = safe_binary_op(
        q2,
        psfc,
        operator.mul,
        context="relative humidity q2*psfc",
        result_dtype=float,
    )
    vapor_pressure = numerator / (epsilon + q2)
    saturation_pressure = 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = safe_binary_op(
            vapor_pressure,
            saturation_pressure,
            operator.truediv,
            context="relative humidity vapor/saturation pressure",
            result_dtype=float,
        )
        rh = 100.0 * rh
    if _is_xarray(rh):
        return rh.clip(min=0.0, max=100.0)
    return np.clip(rh, 0.0, 100.0)


def extract_relative_humidity(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract derived 2-m relative humidity (%)."""
    q2 = ds.get_variable("Q2")
    t2 = ds.get_variable("T2")
    psfc = ds.get_variable("PSFC")
    rh = compute_relative_humidity(q2, t2, psfc)
    rh_min, rh_max = get_low_high(rh)
    return rh, rh_min, rh_max


def extract_wind(ds: WRFReader) -> tuple[WRFArray, WRFArray, float, float]:
    """Extract 10-m U/V wind components and compute speed bounds."""
    u10 = ds.get_variable("U10")
    v10 = ds.get_variable("V10")
    ws_min, ws_max = get_low_high_wind(u10, v10)
    return u10, v10, ws_min, ws_max


def extract_rain(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract total precipitation (convective + non-convective, cumulative)."""
    rainc = ds.get_variable("RAINC")
    rainnc = ds.get_variable("RAINNC")
    total = safe_binary_op(rainc, rainnc, operator.add, context="total precipitation")
    r_min, r_max = get_low_high_rain(total)
    return total, r_min, r_max


def extract_rain_step(total: WRFArray, i: int) -> WRFArray:
    """Compute incremental rain for step *i* from cumulative totals.

    Step 0 returns zeros: the increment is unknowable at a file/restart
    boundary, and publishing rain accumulated since simulation start would
    massively overstate rainfall. Every later step is ``total[i] - total[i-1]``.
    """
    if _is_xarray(total):
        time_dim = _time_dim(total)
        current = total.isel({time_dim: slice(i, i + 1)})
        if i == 0:
            return xr.zeros_like(squeeze_array(current))
        previous = total.isel({time_dim: slice(i - 1, i)})
        return squeeze_array(current) - squeeze_array(previous)
    if i == 0:
        return np.zeros_like(np.squeeze(total[i : i + 1, :, :]))
    return np.squeeze(total[i : i + 1, :, :]) - np.squeeze(total[i - 1 : i, :, :])


def extract_scalar(ds: WRFReader, var_name: str) -> tuple[WRFArray, float, float]:
    """Generic extractor for scalar fields (HFX, LH, SWDOWN)."""
    var = ds.get_variable(var_name)
    v_min, v_max = get_low_high(var)
    return var, v_min, v_max


def compute_air_density(t2: WRFArray, psfc: WRFArray, q2: WRFArray) -> WRFArray:
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
    virtual_temperature = safe_binary_op(
        t2,
        1.0 + 0.61 * q2,
        operator.mul,
        context="air density virtual temperature",
        result_dtype=float,
    )
    return safe_binary_op(
        psfc,
        287.05 * virtual_temperature,
        operator.truediv,
        context="air density pressure/virtual temperature",
        result_dtype=float,
    )


def extract_wind_power_density_10m(ds: WRFReader) -> tuple[WRFArray, float, float]:
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
    speed = safe_binary_op(u10, v10, np.hypot, context="10m wind speed", result_dtype=float)
    density = compute_air_density(t2, psfc, q2)
    power_density = safe_binary_op(
        density,
        np.power(speed, 3),
        operator.mul,
        context="10m wind power density",
        result_dtype=float,
    )
    power_density = 0.5 * power_density
    p_min, p_max = get_low_high(power_density)
    return power_density, p_min, p_max


# ---------------------------------------------------------------------------
# Height / vertical structure
# ---------------------------------------------------------------------------


def compute_adjusted_heights(ds: WRFReader) -> tuple[WRFArray, WRFArray, WRFArray, WRFArray]:
    """Compute adjusted heights above terrain for vertical interpolation.

    Returns ``(U_central, V_central, height_adjusted, speed_4d)`` where:
    - ``U_central``, ``V_central``: wind components at grid cell centers
    - ``height_adjusted``: height above terrain at layer midpoints
    - ``speed_4d``: resulting wind speed at all levels
    """
    u_raw = ds.get_variable("U")
    v_raw = ds.get_variable("V")

    # Interpolate staggered grids to mass-grid cell centers positionally.
    # This intentionally bypasses xarray label alignment while preserving
    # lazy/dask arrays, because staggered coordinates are adjacent samples, not
    # labels that should be intersected.
    u_central: Any
    if _is_xarray(u_raw):
        u_central = destagger_dataarray(
            cast("xr.DataArray", u_raw),
            staggered_dim="west_east_stag",
            target_dim="west_east",
            context="U wind destagger",
        )
    else:
        u_shape = list(u_raw.shape)
        u_shape[3] -= 1
        assert_reasonable_array_size(u_shape, u_raw.dtype, context="U wind destagger")
        u_central = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0

    v_central: Any
    if _is_xarray(v_raw):
        v_central = destagger_dataarray(
            cast("xr.DataArray", v_raw),
            staggered_dim="south_north_stag",
            target_dim="south_north",
            context="V wind destagger",
        )
    else:
        v_shape = list(v_raw.shape)
        v_shape[2] -= 1
        assert_reasonable_array_size(v_shape, v_raw.dtype, context="V wind destagger")
        v_central = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0

    # Geopotential height
    ph = ds.get_variable("PH")
    phb = ds.get_variable("PHB")
    hgt = ds.get_variable("HGT")

    geopot_total = safe_binary_op(ph, phb, operator.add, context="PH+PHB geopotential")
    height = geopot_total / 9.81

    # Midpoint heights
    height_central: Any
    if _is_xarray(height):
        height_central = destagger_dataarray(
            cast("xr.DataArray", height),
            staggered_dim="bottom_top_stag",
            target_dim="bottom_top",
            context="geopotential height destagger",
        )
    else:
        height_shape = list(height.shape)
        height_shape[1] -= 1
        assert_reasonable_array_size(
            height_shape,
            height.dtype,
            context="geopotential height destagger",
        )
        height_central = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0

    # Adjust for terrain.  Build the expanded HGT array with the exact target
    # dims instead of relying on xarray's automatic alignment/broadcasting.
    height_adjusted: Any
    if _is_xarray(height_central):
        height_central_da = height_central
        hgt_da = cast("xr.DataArray", hgt)
        expected_hgt_dims = (
            height_central_da.dims[0],
            height_central_da.dims[2],
            height_central_da.dims[3],
        )
        if hgt_da.dims != expected_hgt_dims:
            raise ValueError(
                "HGT dimensions do not match height field: "
                f"{hgt_da.dims!r} vs expected {expected_hgt_dims!r}"
            )
        if hgt_da.shape != (
            height_central_da.shape[0],
            height_central_da.shape[2],
            height_central_da.shape[3],
        ):
            raise ValueError(
                "HGT shape does not match height field: "
                f"{hgt_da.shape!r} vs expected "
                f"{(height_central_da.shape[0], height_central_da.shape[2], height_central_da.shape[3])!r}"
            )
        level_dim = height_central_da.dims[1]
        hgt_expanded = hgt_da.expand_dims(
            {level_dim: height_central_da.sizes[level_dim]},
            axis=1,
        ).transpose(*height_central_da.dims)
        height_adjusted = safe_binary_op(
            height_central_da,
            hgt_expanded,
            operator.sub,
            context="height above terrain",
            result_dtype=float,
        )
    else:
        assert_reasonable_array_size(
            height_central.shape,
            height_central.dtype,
            context="height above terrain",
        )
        height_adjusted = height_central - hgt[:, np.newaxis, :, :]

    # Speed at all levels
    speed_4d = safe_binary_op(
        u_central,
        v_central,
        np.hypot,
        context="3D wind speed from destaggered U/V",
        result_dtype=float,
    )

    return u_central, v_central, height_adjusted, speed_4d


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

    Must stay operation-for-operation identical to the numpy branch of
    ``interpolation.compute_wind_vectors_at_height`` (equivalence is pinned
    by tests); the output floats are embedded unrounded in the values JSON.
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

    return {
        "downsampled_angles": angles_flat[valid].tolist(),
        "downsampled_magnitudes": mags_flat[valid].tolist(),
        "downsampled_linear_indices": linear_indices.tolist(),
    }


def stream_wind_at_heights(
    ds: WRFReader,
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

    n_t = ds.n_time_steps  # type: ignore[attr-defined]

    speed_out: dict[int, NDArray] = {}
    vectors_out: dict[int, list[dict | None]] = {t: [] for t in targets}

    for t0 in range(0, n_t, block_steps):
        t1 = min(t0 + block_steps, n_t)
        u_raw = ds.get_variable_block("U", t0, t1)  # type: ignore[attr-defined]
        u_c = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0
        del u_raw
        v_raw = ds.get_variable_block("V", t0, t1)  # type: ignore[attr-defined]
        v_c = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0
        del v_raw

        ph = ds.get_variable_block("PH", t0, t1)  # type: ignore[attr-defined]
        phb = ds.get_variable_block("PHB", t0, t1)  # type: ignore[attr-defined]
        height = (ph + phb) / 9.81
        del ph, phb
        height_c = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0
        del height
        hgt = ds.get_variable_block("HGT", t0, t1)  # type: ignore[attr-defined]
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
