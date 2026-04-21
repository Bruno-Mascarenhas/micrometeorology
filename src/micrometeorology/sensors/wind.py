"""Wind direction averaging using the vector-mean method.

Wind direction cannot be averaged arithmetically (e.g. 350° and 10° would
give 180° instead of the correct 0°).  The correct approach is to decompose
into U/V components, average those, then recompose.

This module consolidates the logic that was duplicated in at least three
legacy scripts (controle_old.py, graficos1_UFBA_v5.py, graficos3_UFBA_v1.py).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def wind_components(
    speed: NDArray | float,
    direction_deg: NDArray | float,
) -> tuple[NDArray, NDArray]:
    """Decompose wind speed and direction into U and V components.

    Parameters
    ----------
    speed:
        Wind speed (m/s).
    direction_deg:
        Wind direction in degrees (meteorological convention: 0 = from North).

    Returns
    -------
    (u, v):
        Zonal (u) and meridional (v) wind components.
        Follows the convention: u = -speed * sin(dir), v = -speed * cos(dir).
    """
    rad = np.radians(direction_deg)
    u = -np.asarray(speed) * np.sin(rad)
    v = -np.asarray(speed) * np.cos(rad)
    return u, v


def wind_direction_from_components(u: NDArray | float, v: NDArray | float) -> NDArray | float:
    """Compute wind direction element-wise from U/V components.

    Returns the direction in degrees [0, 360).
    """
    alpha = np.arctan2(v, u)
    direction = np.fmod(3.0 * (np.pi / 2.0) - alpha, 2.0 * np.pi) * (180.0 / np.pi)
    return direction % 360.0


def vector_mean_direction(u: NDArray, v: NDArray) -> float:
    """Compute mean wind direction from U/V component arrays.

    Returns the direction in degrees [0, 360).
    """
    u_mean = float(np.nanmean(u))
    v_mean = float(np.nanmean(v))
    return float(wind_direction_from_components(u_mean, v_mean))


def wind_speed_from_components(u: NDArray, v: NDArray) -> NDArray:
    """Compute wind speed from U/V components."""
    return np.hypot(np.asarray(u), np.asarray(v))  # type: ignore
