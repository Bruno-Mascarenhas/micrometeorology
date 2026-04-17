"""WRF variable extraction and unit conversion.

Consolidates the repeated per-variable extraction logic that was
duplicated across the ``drawmap()`` functions in the legacy scripts.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from labmim_micrometeorology.wrf.reader import WRFDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Min / max helpers (preserved from legacy getLowHigh* functions)
# ---------------------------------------------------------------------------


def get_low_high(variable: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` of a 3-D variable, skipping the first time step."""
    flat = variable[1:, :].flatten()
    return float(np.nanmin(flat)), float(np.nanpercentile(flat, 98))


def get_low_high_wind(u: NDArray, v: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` wind speed from U/V arrays (skip first step)."""
    flat_u = u[1:, :].flatten()
    flat_v = v[1:, :].flatten()
    speed = np.hypot(flat_u, flat_v)
    return float(np.nanmin(speed)), float(np.nanmax(speed))


def get_low_high_rain(variable: NDArray) -> tuple[float, float]:
    """Return ``(min, max)`` of incremental precipitation.

    The input is *cumulative* rain; we compute the per-step increment first.
    """
    arr = np.asarray(variable)
    if arr.ndim < 3:
        flat = arr.flatten()
        return float(np.nanmin(flat)), float(np.nanmax(flat))
    diffs = np.diff(arr, axis=0)
    if diffs.size == 0:
        return 0.0, 0.0
    flat = diffs.flatten()
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
    return np.squeeze(t2_step) - 273.15


def extract_pressure(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract surface pressure (hPa)."""
    psfc = ds.get_variable("PSFC")
    p_min, p_max = get_low_high(psfc)
    return psfc / 100.0, p_min / 100.0, p_max / 100.0


def extract_vapor(ds: WRFDataset) -> tuple[NDArray, float, float]:
    """Extract 2-m specific humidity (g/kg)."""
    q2 = ds.get_variable("Q2")
    q_min, q_max = get_low_high(q2)
    return q2 * 1000.0, q_min * 1000.0, q_max * 1000.0


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
    """Compute incremental rain for step *i* from cumulative totals."""
    if i <= 1:
        return np.squeeze(total[i : i + 1, :, :])
    return np.squeeze(total[i : i + 1, :, :]) - np.squeeze(total[i - 1 : i, :, :])


def extract_scalar(ds: WRFDataset, var_name: str) -> tuple[NDArray, float, float]:
    """Generic extractor for scalar fields (HFX, LH, SWDOWN)."""
    var = ds.get_variable(var_name)
    v_min, v_max = get_low_high(var)
    return var, v_min, v_max


# ---------------------------------------------------------------------------
# Height / vertical structure
# ---------------------------------------------------------------------------


def compute_adjusted_heights(ds: WRFDataset) -> tuple[NDArray, NDArray, NDArray, NDArray]:
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
    height_adjusted = height_central - hgt[:, np.newaxis, :, :]

    # Speed at all levels
    speed_4d = np.hypot(u_central, v_central)

    return u_central, v_central, height_adjusted, speed_4d
