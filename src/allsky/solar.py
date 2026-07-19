"""Vectorized NOAA solar position and extraterrestrial irradiance.

Implements the low-cost Fourier-series expansions of Spencer (1971) as
used in the NOAA Global Monitoring Laboratory "General Solar Position
Calculations" sheet: fractional-year angle -> solar declination, equation
of time and Sun-Earth eccentricity correction; local clock time -> true
solar time -> hour angle -> cosine of the solar zenith angle.

All functions are pure numpy/pandas, vectorized over any datetime
sequence convertible to a :class:`pandas.DatetimeIndex`.

Timestamps are **naive local standard time** (the Campbell datalogger
clock — no timezone conversion in v0).  The UTC offset is inferred from
the site longitude as ``round(longitude / 15)`` unless passed explicitly;
for Salvador-BA (longitude -38.51) this yields UTC-3, which is correct
year-round since Brazil abolished DST in 2019.

References
----------
Spencer, J.W. (1971). Fourier series representation of the position of
the sun. *Search* 2(5), 172.
NOAA ESRL/GML, General Solar Position Calculations,
https://gml.noaa.gov/grad/solcalc/solareqns.PDF
Iqbal, M. (1983). *An Introduction to Solar Radiation*. Academic Press.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from allsky.config import SiteConfig

    DatetimeLike = pd.DatetimeIndex | pd.Series | np.ndarray | list | tuple

__all__ = [
    "SOLAR_CONSTANT_WM2",
    "clearness_index",
    "cos_zenith",
    "eccentricity_correction",
    "equation_of_time",
    "extraterrestrial_ghi",
    "hour_angle",
    "solar_azimuth",
    "solar_declination",
    "solar_elevation",
]

#: Total solar irradiance at 1 AU (Kopp & Lean 2011), W m-2.
SOLAR_CONSTANT_WM2 = 1361.0


def _as_datetime_index(timestamps: DatetimeLike) -> pd.DatetimeIndex:
    """Normalize any datetime-like sequence to a naive DatetimeIndex."""
    index = pd.DatetimeIndex(timestamps)
    if index.tz is not None:
        raise ValueError("timestamps must be naive local time (v0 contract); got tz-aware index")
    return index


def _fractional_year(times: pd.DatetimeIndex) -> np.ndarray:
    """Fractional-year angle gamma in radians (NOAA/Spencer).

    Formula
    -------
    ``gamma = 2*pi/365 * (day_of_year - 1 + (hour - 12) / 24)``
    """
    hours = (
        times.hour.to_numpy(dtype=np.float64)
        + times.minute.to_numpy(dtype=np.float64) / 60.0
        + times.second.to_numpy(dtype=np.float64) / 3600.0
    )
    doy = times.dayofyear.to_numpy(dtype=np.float64)
    return 2.0 * np.pi / 365.0 * (doy - 1.0 + (hours - 12.0) / 24.0)


def solar_declination(timestamps: DatetimeLike) -> np.ndarray:
    """Solar declination in **radians** (Spencer 1971 series).

    Formula
    -------
    ``decl = 0.006918 - 0.399912 cos(g) + 0.070257 sin(g) - 0.006758 cos(2g)
    + 0.000907 sin(2g) - 0.002697 cos(3g) + 0.00148 sin(3g)``

    Limitation
    ----------
    Accurate to ~0.01 rad; adequate for irradiance decomposition, not for
    precision tracking applications.
    """
    g = _fractional_year(_as_datetime_index(timestamps))
    declination: np.ndarray = np.asarray(
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    return declination


def equation_of_time(timestamps: DatetimeLike) -> np.ndarray:
    """Equation of time in **minutes** (Spencer 1971 series).

    Formula
    -------
    ``eqtime = 229.18 * (0.000075 + 0.001868 cos(g) - 0.032077 sin(g)
    - 0.014615 cos(2g) - 0.040849 sin(2g))``
    """
    g = _fractional_year(_as_datetime_index(timestamps))
    minutes: np.ndarray = 229.18 * np.asarray(
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    return minutes


def eccentricity_correction(timestamps: DatetimeLike) -> np.ndarray:
    """Sun-Earth distance (eccentricity) correction factor E0 (Spencer 1971).

    Formula
    -------
    ``E0 = (r0/r)^2 = 1.000110 + 0.034221 cos(g) + 0.001280 sin(g)
    + 0.000719 cos(2g) + 0.000077 sin(2g)``
    """
    g = _fractional_year(_as_datetime_index(timestamps))
    eccentricity: np.ndarray = np.asarray(
        1.000110
        + 0.034221 * np.cos(g)
        + 0.001280 * np.sin(g)
        + 0.000719 * np.cos(2 * g)
        + 0.000077 * np.sin(2 * g)
    )
    return eccentricity


def _resolve_utc_offset(longitude: float, utc_offset_hours: float | None) -> float:
    """UTC offset of the local clock: explicit value or the standard meridian."""
    if utc_offset_hours is not None:
        return utc_offset_hours
    return round(longitude / 15.0)


def hour_angle(
    timestamps: DatetimeLike,
    longitude: float,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Solar hour angle in **degrees** (0 at solar noon, negative mornings).

    Formula
    -------
    NOAA chain: ``time_offset = eqtime + 4*longitude - 60*utc_offset`` (min),
    ``tst = 60*h + m + s/60 + time_offset`` (min), ``ha = tst/4 - 180`` (deg).

    Parameters
    ----------
    timestamps:
        Naive local clock times.
    longitude:
        Site longitude in degrees (east positive).
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``longitude`` when None.
    """
    times = _as_datetime_index(timestamps)
    offset = _resolve_utc_offset(longitude, utc_offset_hours)
    eqtime = equation_of_time(times)
    time_offset = eqtime + 4.0 * longitude - 60.0 * offset
    tst = (
        times.hour.to_numpy(dtype=np.float64) * 60.0
        + times.minute.to_numpy(dtype=np.float64)
        + times.second.to_numpy(dtype=np.float64) / 60.0
        + time_offset
    )
    return tst / 4.0 - 180.0


def cos_zenith(
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Cosine of the solar zenith angle, clipped to [-1, 1].

    Formula
    -------
    ``cos(theta_z) = sin(lat) sin(decl) + cos(lat) cos(decl) cos(ha)``
    """
    times = _as_datetime_index(timestamps)
    lat = np.deg2rad(site.latitude)
    decl = solar_declination(times)
    ha = np.deg2rad(hour_angle(times, site.longitude, utc_offset_hours))
    cosz = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    clipped: np.ndarray = np.clip(cosz, -1.0, 1.0)
    return clipped


def solar_elevation(
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Solar elevation angle in **degrees** (negative below the horizon).

    Formula
    -------
    ``elevation = 90 - zenith = arcsin(cos(theta_z))``
    """
    elevation: np.ndarray = np.rad2deg(np.arcsin(cos_zenith(timestamps, site, utc_offset_hours)))
    return elevation


def solar_azimuth(
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Solar azimuth in **degrees clockwise from North**, wrapped to ``[0, 360)``.

    Formula
    -------
    NOAA "deg cw from N" convention.  With zenith ``theta_z``, declination
    ``decl``, latitude ``phi`` and hour angle ``ha``::

        c = (sin(phi) cos(theta_z) - sin(decl)) / (cos(phi) sin(theta_z))
        A = arccos(clip(c, -1, 1))
        azimuth = (A + 180) mod 360   if ha > 0   (afternoon)
        azimuth = (540 - A) mod 360   otherwise   (morning / solar noon)

    In the Southern Hemisphere (this site, ``lat = -13``) the sun crosses
    the northern sky, so azimuth is ~0 deg (North) at winter solar noon and
    ~180 deg (South) at summer solar noon; it rises near 90 deg (East) and
    sets near 270 deg (West).

    Parameters
    ----------
    timestamps:
        Naive local clock times (v0 contract; tz-aware input is rejected).
    site:
        Observation site (latitude/longitude in degrees).
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None (see :func:`hour_angle`).

    Limitation
    ----------
    Undefined exactly at the zenith (``sin(theta_z) = 0``, which occurs at
    this latitude near the two annual zenith-crossing noons); the ratio is
    guarded so the result stays finite there.  Azimuth is returned for every
    row, including night — filter with :func:`solar_elevation` when a horizon
    cut is required.
    """
    times = _as_datetime_index(timestamps)
    lat = np.deg2rad(site.latitude)
    decl = solar_declination(times)
    cosz = cos_zenith(times, site, utc_offset_hours)
    sinz = np.sin(np.arccos(cosz))
    ha = hour_angle(times, site.longitude, utc_offset_hours)

    denom = np.cos(lat) * sinz
    safe = np.abs(denom) > 1e-12
    ratio = np.divide(
        np.sin(lat) * cosz - np.sin(decl),
        denom,
        out=np.zeros_like(cosz),
        where=safe,
    )
    acos_deg = np.rad2deg(np.arccos(np.clip(ratio, -1.0, 1.0)))
    azimuth = np.where(ha > 0.0, acos_deg + 180.0, 540.0 - acos_deg)
    wrapped: np.ndarray = np.mod(azimuth, 360.0)
    return wrapped


def extraterrestrial_ghi(
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Extraterrestrial irradiance on a horizontal plane, W m-2 (>= 0).

    Formula
    -------
    ``E0h = S0 * E0 * cos(theta_z)`` with ``S0 = 1361 W m-2`` (solar
    constant) and ``E0`` the Spencer eccentricity correction; clipped to
    zero when the sun is below the horizon.
    """
    times = _as_datetime_index(timestamps)
    e0h = (
        SOLAR_CONSTANT_WM2
        * eccentricity_correction(times)
        * cos_zenith(times, site, utc_offset_hours)
    )
    clipped: np.ndarray = np.maximum(e0h, 0.0)
    return clipped


def clearness_index(
    ghi: pd.Series | np.ndarray | Sequence[float],
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Clearness index ``kt = GHI / E0h``; NaN when the sun is down.

    Formula
    -------
    ``kt = GHI / E0h`` where ``E0h > 0``, else NaN (undefined at night).
    NaN GHI propagates to NaN kt.

    Limitation
    ----------
    kt is not clipped: near sunrise/sunset (small E0h) sensor noise can
    produce kt > 1.  Downstream consumers should filter by solar
    elevation (see ``LabelConfig.min_solar_elevation_deg``).
    """
    times = _as_datetime_index(timestamps)
    ghi_arr = np.asarray(ghi, dtype=np.float64)
    if ghi_arr.shape != (len(times),):
        raise ValueError(f"ghi length {ghi_arr.shape} does not match {len(times)} timestamps")
    e0h = extraterrestrial_ghi(times, site, utc_offset_hours)
    kt = np.full_like(ghi_arr, np.nan)
    sun_up = e0h > 0.0
    np.divide(ghi_arr, e0h, out=kt, where=sun_up)
    return kt
