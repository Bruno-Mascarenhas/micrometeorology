"""Haurwitz (1945) clear-sky global irradiance and the clear-sky index k*.

Provides the single-parameter Haurwitz clear-sky model for global
horizontal irradiance (GHI) — a function of the solar zenith angle only,
with no aerosol/water-vapour inputs — and the derived **clear-sky index**
``k* = GHI / GHI_cs``.  Unlike the extraterrestrial clearness index ``kt``
(see :func:`labmim_core.solar.clearness_index`), ``k*`` normalizes by a
ground-level clear-sky reference, so clear skies sit near 1.0, broken cloud
scatters around/above 1.0 (cloud enhancement) and overcast skies fall well
below 1.0.

All functions are pure numpy/pandas and vectorized, and they take two different
clocks. :func:`haurwitz_ghi` and :func:`clear_sky_index` read the instrument's
own clock over any datetime sequence convertible to a
:class:`pandas.DatetimeIndex`, sharing the naive local-standard-time contract of
:mod:`labmim_core.solar` (tz-aware input is rejected there).
:func:`clearsky_ghi_and_kt` and :func:`clearsky_diffuse` instead take the
manifest's timezone-aware ``timestamp_utc`` and convert it to the site's clock
themselves, so naive input is what fails there.

References
----------
Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud
density. *Journal of Meteorology* 2(3), 154-166.
doi:10.1175/1520-0469(1945)002<0154:IIRTCA>2.0.CO;2
Reno, M.J., Hansen, C.W., Stein, J.S. (2012). Global horizontal irradiance
clear sky models: implementation and analysis. SAND2012-2389.
"""

from collections.abc import Sequence
from datetime import timedelta, timezone

import numpy as np
import pandas as pd

from labmim_core.site import STATION_UTC_OFFSET_HOURS, SiteConfig
from labmim_core.solar import (
    SOLAR_CONSTANT_WM2,
    DatetimeLike,
    cos_zenith,
    eccentricity_correction,
    solar_elevation_deg,
)

type ArrayLike = Sequence[float] | np.ndarray | pd.Series

__all__ = [
    "HAURWITZ_A_WM2",
    "HAURWITZ_B",
    "clear_sky_index",
    "clearsky_diffuse",
    "clearsky_ghi_and_kt",
    "haurwitz_ghi",
    "haurwitz_ghi_from_cos_zenith",
]

#: Amplitude coefficient of the Haurwitz clear-sky model, W m-2.
HAURWITZ_A_WM2 = 1098.0
#: Optical-depth coefficient of the Haurwitz clear-sky model (dimensionless).
HAURWITZ_B = 0.057


def haurwitz_ghi(
    timestamps: DatetimeLike,
    site: SiteConfig,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance, W m-2 (>= 0).

    Formula
    -------
    ``GHI_cs = 1098 * cos(theta_z) * exp(-0.057 / cos(theta_z))`` when the
    sun is above the horizon (``cos(theta_z) > 0``), else 0.  ``theta_z`` is
    the solar zenith angle from :func:`labmim_core.solar.cos_zenith`.

    Parameters
    ----------
    timestamps:
        Naive local clock times, ``(N,)``.
    site:
        Observation site (latitude/longitude in degrees).
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None.

    Returns
    -------
    numpy.ndarray
        Clear-sky global horizontal irradiance, shape ``(N,)``,
        ``float64``, W m-2, exactly zero at night.

    Limitation
    ----------
    Single-parameter fit (US mid-latitude climatology): it ignores aerosol
    load, water vapour and site elevation, so absolute magnitudes at a humid
    tropical coastal site carry a systematic bias.  It is used here as a
    normalization reference for ``k*``, where that bias largely cancels, not
    as an absolute irradiance predictor.
    """
    return haurwitz_ghi_from_cos_zenith(cos_zenith(timestamps, site, utc_offset_hours))


def haurwitz_ghi_from_cos_zenith(cos_zenith_values: ArrayLike) -> np.ndarray:
    """Haurwitz clear-sky GHI from the cosine of the solar zenith angle.

    Parameters
    ----------
    cos_zenith_values:
        Cosine of the solar zenith angle, shape ``(N,)``, dimensionless.
        Non-positive entries mean the sun is below the horizon.

    Returns
    -------
    numpy.ndarray
        Clear-sky global horizontal irradiance, shape ``(N,)``, W m-2, zero
        wherever the sun is down.
    """
    cosz = np.asarray(cos_zenith_values, dtype=np.float64)
    sun_up = cosz > 0.0
    # Clamp the below-horizon cosines to 1.0 before dividing so the exp never
    # sees 0 or a negative argument; the result is masked back to 0 anyway.
    safe_cosz = np.where(sun_up, cosz, 1.0)
    values = HAURWITZ_A_WM2 * safe_cosz * np.exp(-HAURWITZ_B / safe_cosz)
    ghi_cs: np.ndarray = np.where(sun_up, values, 0.0)
    return ghi_cs


def clear_sky_index(
    ghi: pd.Series | np.ndarray | Sequence[float],
    timestamps: DatetimeLike,
    site: SiteConfig,
    min_elevation_deg: float = 10.0,
    utc_offset_hours: float | None = None,
) -> np.ndarray:
    """Clear-sky index ``k* = GHI / GHI_cs`` (Haurwitz reference); NaN when low.

    Formula
    -------
    ``k* = GHI / GHI_cs`` where the solar elevation is at least
    ``min_elevation_deg`` and ``GHI_cs > 0``; NaN otherwise (sun too low, or
    missing GHI).  ``k*`` is intentionally left unclipped so cloud-enhancement
    events (``k* > 1``) survive.

    Parameters
    ----------
    ghi:
        Measured global horizontal irradiance, shape ``(N,)``, W m-2,
        aligned 1:1 with *timestamps*.
    timestamps:
        Naive local clock times, ``(N,)``.
    site:
        Observation site (latitude/longitude in degrees).
    min_elevation_deg:
        Elevation floor below which ``k*`` is undefined (default 10 deg); at
        low sun the clear-sky reference is small and airmass errors dominate.
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None.

    Returns
    -------
    numpy.ndarray
        Clear-sky index, shape ``(N,)``, ``float64``, dimensionless, NaN
        wherever the sun is too low or the GHI is missing.

    Raises
    ------
    ValueError
        If *ghi* and *timestamps* have different lengths.

    Limitation
    ----------
    Inherits the absolute bias of :func:`haurwitz_ghi`; treat ``k*`` as a
    relative cloud-transmission proxy rather than a calibrated ratio.
    """
    ghi_arr = np.asarray(ghi, dtype=np.float64)
    ghi_cs = haurwitz_ghi(timestamps, site, utc_offset_hours)
    if ghi_arr.shape != ghi_cs.shape:
        raise ValueError(f"ghi shape {ghi_arr.shape} does not match {ghi_cs.shape} timestamps")
    elevation = solar_elevation_deg(timestamps, site, utc_offset_hours)
    valid = (elevation >= min_elevation_deg) & (ghi_cs > 0.0) & np.isfinite(ghi_arr)
    kstar = np.full_like(ghi_arr, np.nan)
    np.divide(ghi_arr, ghi_cs, out=kstar, where=valid)
    return kstar


def clearsky_ghi_and_kt(
    solar_zenith_deg: ArrayLike,
    times: pd.Series,
    utc_offset_hours: float = STATION_UTC_OFFSET_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    """Haurwitz clear-sky GHI and its clearness index, per sample.

    Parameters
    ----------
    solar_zenith_deg:
        ``(N,)`` solar zenith angle in degrees.
    times:
        ``(N,)`` timezone-aware timestamps, one per sample.
    utc_offset_hours:
        Fixed offset of the site's own clock, which the eccentricity correction
        reads a day-of-year from. It defaults to this station's, but it is a
        PARAMETER because this function is shared physics: pinning it to Salvador
        while every sibling in this module takes a site is how a second station —
        Folsom runs at UTC-8 — gets this one's clock without failing anywhere.

    Returns
    -------
    tuple of numpy.ndarray
        ``(ghi_clear, kt_clear)``, both ``(N,)`` float64: clear-sky global
        horizontal irradiance in W m-2, exactly zero where the sun is down, and
        its dimensionless ratio to the extraterrestrial horizontal irradiance,
        which is ``NaN`` there instead.
    """
    cos_zenith_values = np.cos(np.radians(np.asarray(solar_zenith_deg, dtype=np.float64)))
    ghi_clear = haurwitz_ghi_from_cos_zenith(cos_zenith_values)
    local = times.dt.tz_convert(timezone(timedelta(hours=utc_offset_hours)))
    extraterrestrial = (
        SOLAR_CONSTANT_WM2
        * eccentricity_correction(local.dt.tz_localize(None))
        * np.maximum(cos_zenith_values, 0.0)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        kt_clear = np.where(extraterrestrial > 0.0, ghi_clear / extraterrestrial, np.nan)
    return ghi_clear, np.asarray(kt_clear, dtype=np.float64)


def clearsky_diffuse(
    solar_zenith_deg: ArrayLike,
    times: pd.Series,
    utc_offset_hours: float = STATION_UTC_OFFSET_HOURS,
) -> np.ndarray:
    """Diffuse irradiance a cloudless sky would produce, in W m-2.

    Haurwitz clear-sky GHI decomposed by Erbs at the clear-sky clearness index —
    the same reference the evaluator scores ``skill_clearsky`` against, so a run
    that trains on ``DHI / DHI_clear`` is normalized by exactly the baseline it
    is compared to.

    Parameters
    ----------
    solar_zenith_deg:
        ``(N,)`` solar zenith angle in degrees.
    times:
        ``(N,)`` timezone-aware timestamps.
    utc_offset_hours:
        The site's own fixed clock offset; see :func:`clearsky_ghi_and_kt`.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` float64 clear-sky DHI in W m-2, ``NaN`` where the sun is down.
    """
    from allsky.erbs import pseudo_diffuse

    ghi_clear, kt_clear = clearsky_ghi_and_kt(solar_zenith_deg, times, utc_offset_hours)
    return np.asarray(pseudo_diffuse(ghi_clear, kt_clear), dtype=np.float64)
