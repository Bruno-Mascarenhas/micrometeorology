"""Wind direction averaging using the vector-mean method.

Wind direction cannot be averaged arithmetically (e.g. 350° and 10° would
give 180° instead of the correct 0°).  The correct approach is to decompose
into U/V components, average those, then recompose.
"""

import numpy as np
from numpy.typing import NDArray


def wind_components(
    speed: NDArray | float,
    direction_deg: NDArray | float,
) -> tuple[NDArray, NDArray]:
    """Decompose wind speed and direction into U and V components.

    Parameters
    ----------
    speed:
        Wind speed in m/s, shape ``(N,)`` float or a scalar. Pass ``1.0`` to get
        unit vectors, which turns the vector mean into an unweighted one.
    direction_deg:
        Direction the wind blows FROM, in degrees clockwise from true north
        (meteorological convention, 0 = northerly). Shape ``(N,)`` float or a
        scalar, broadcast against *speed*.

    Returns
    -------
    (u, v):
        Zonal and meridional components in m/s, shape ``(N,)`` float, in the
        earth-relative frame where u is eastward and v is northward. Both carry
        the meteorological sign convention ``u = -speed * sin(dir)``,
        ``v = -speed * cos(dir)``, so a 90 deg (easterly) wind gives u < 0.
        ``NaN`` in either input propagates to both outputs.
    """
    rad = np.radians(direction_deg)
    u = -np.asarray(speed) * np.sin(rad)
    v = -np.asarray(speed) * np.cos(rad)
    return u, v


def wind_direction_from_components(u: NDArray | float, v: NDArray | float) -> NDArray | float:
    """Compute wind direction element-wise from U/V components.

    Inverse of :func:`wind_components`, and the recomposition step of the vector
    mean: averaging is done on u and v, never on the bearing itself.

    Parameters
    ----------
    u, v:
        Zonal and meridional components in m/s, shape ``(N,)`` float or scalars,
        in the same earth-relative frame :func:`wind_components` produces. Their
        magnitude sets only whether the result is defined, not its value.

    Returns
    -------
    ndarray or float
        Direction the wind blows FROM, degrees clockwise from true north in
        ``[0, 360)``, shape ``(N,)`` float, or ``NaN`` where the resultant
        vector has zero length. A zero-length resultant carries no directional
        information: ``arctan2(0, 0)`` is 0, which would otherwise publish a
        confident 270° (or 90°, depending on the operands' sign bits) for a calm
        window or a stalled anemometer.
    """
    mathematical_angle = np.arctan2(v, u)
    direction = np.fmod(3.0 * (np.pi / 2.0) - mathematical_angle, 2.0 * np.pi) * (180.0 / np.pi)
    resultant_magnitude = np.hypot(u, v)
    return np.where(resultant_magnitude == 0.0, np.nan, direction % 360.0)


def vector_mean_direction(u: NDArray, v: NDArray) -> float:
    """Compute mean wind direction from U/V component arrays.

    Parameters
    ----------
    u, v:
        Zonal and meridional components in m/s, shape ``(N,)`` float, over the
        window being averaged. ``NaN`` samples are skipped rather than
        propagated, so a partially missing window still yields a bearing.
        Weighting is whatever the components already carry: components built
        from the measured speed give a speed-weighted vector mean, unit-speed
        components give the unweighted directional mean.

    Returns
    -------
    float
        Mean direction the wind blew FROM, degrees clockwise from true north in
        ``[0, 360)``, or ``NaN`` when the mean components cancel to a
        zero-length resultant (see :func:`wind_direction_from_components`).
    """
    u_mean = float(np.nanmean(u))
    v_mean = float(np.nanmean(v))
    return float(wind_direction_from_components(u_mean, v_mean))


def wind_speed_from_components(u: NDArray, v: NDArray) -> NDArray:
    """Recover the scalar wind speed from U/V components.

    Parameters
    ----------
    u, v:
        Zonal and meridional components in m/s, shape ``(N,)`` float.

    Returns
    -------
    ndarray
        Speed in m/s, shape ``(N,)`` float, non-negative by construction.
        Applied to *averaged* components this is the resultant speed, which is
        at most -- and usually below -- the mean of the sampled speeds, since
        directional spread shortens the resultant.
    """
    speed: NDArray = np.hypot(np.asarray(u), np.asarray(v))
    return speed
