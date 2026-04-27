"""WRF variable extraction and unit conversion.

Consolidates the repeated per-variable extraction logic that was
duplicated across the ``drawmap()`` functions in the legacy scripts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xarray as xr

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
    if _is_xarray(value):
        return value.isel({_time_dim(value): slice(1, None)})
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
    if _is_xarray(squeezed):
        return np.asarray(squeezed.to_numpy())
    return np.asarray(squeezed)


def materialize_nd(value: WRFArray) -> NDArray:
    """Materialize an ndarray/DataArray without changing dimensionality."""
    if _is_xarray(value):
        return np.asarray(value.to_numpy())
    return np.asarray(value)


def get_low_high(variable: WRFArray) -> tuple[float, float]:
    """Return ``(min, max)`` of a 3-D variable, skipping the first time step."""
    if _is_xarray(variable):
        tail = _tail(variable)
        return _as_float(tail.min(skipna=True)), _as_float(tail.quantile(0.98, skipna=True))
    flat = variable[1:, :].ravel()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: WRFArray, v: WRFArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays (skip first step)."""
    if _is_xarray(u) or _is_xarray(v):
        speed = np.hypot(_tail(u), _tail(v))
        return _as_float(speed.min(skipna=True)), _as_float(speed.max(skipna=True))
    flat_u = u[1:, :].ravel()
    flat_v = v[1:, :].ravel()
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


def extract_pressure(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract surface pressure (hPa)."""
    psfc = ds.get_variable("PSFC")
    p_min, p_max = get_low_high(psfc)
    return psfc / 100.0, p_min / 100.0, p_max / 100.0


def extract_vapor(ds: WRFReader) -> tuple[WRFArray, float, float]:
    """Extract 2-m specific humidity (g/kg)."""
    q2 = ds.get_variable("Q2")
    q_min, q_max = get_low_high(q2)
    return q2 * 1000.0, q_min * 1000.0, q_max * 1000.0


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
    total = rainc + rainnc
    r_min, r_max = get_low_high_rain(total)
    return total, r_min, r_max


def extract_rain_step(total: WRFArray, i: int) -> WRFArray:
    """Compute incremental rain for step *i* from cumulative totals."""
    if _is_xarray(total):
        time_dim = _time_dim(total)
        current = total.isel({time_dim: slice(i, i + 1)})
        if i <= 1:
            return squeeze_array(current)
        previous = total.isel({time_dim: slice(i - 1, i)})
        return squeeze_array(current) - squeeze_array(previous)
    if i <= 1:
        return np.squeeze(total[i : i + 1, :, :])
    return np.squeeze(total[i : i + 1, :, :]) - np.squeeze(total[i - 1 : i, :, :])  # type: ignore


def extract_scalar(ds: WRFReader, var_name: str) -> tuple[WRFArray, float, float]:
    """Generic extractor for scalar fields (HFX, LH, SWDOWN)."""
    var = ds.get_variable(var_name)
    v_min, v_max = get_low_high(var)
    return var, v_min, v_max


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

    # Interpolate staggered grid to cell centers
    u_central = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0
    v_central = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0

    # Geopotential height
    ph = ds.get_variable("PH")
    phb = ds.get_variable("PHB")
    hgt = ds.get_variable("HGT")

    geopot_total = ph + phb
    height = geopot_total / 9.81

    # Midpoint heights
    height_central = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0

    # Adjust for terrain — vectorized broadcast (hgt is 3-D: time, ny, nx)
    # height_central is 4-D: time, level, ny, nx
    # Broadcasting hgt[:, np.newaxis, :, :] aligns the level axis automatically
    height_adjusted: Any
    if _is_xarray(height_central):
        height_central_da = cast("xr.DataArray", height_central)
        hgt_da = cast("xr.DataArray", hgt)
        level_dim = height_central_da.dims[1]
        height_adjusted = height_central_da - hgt_da.expand_dims(
            {level_dim: height_central_da.sizes[level_dim]},
            axis=1,
        )
    else:
        height_adjusted = height_central - hgt[:, np.newaxis, :, :]

    # Speed at all levels
    speed_4d = np.hypot(u_central, v_central)

    return u_central, v_central, height_adjusted, speed_4d
