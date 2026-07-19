"""Haurwitz (1945) clear-sky global irradiance and the clear-sky index k*.

Provides the single-parameter Haurwitz clear-sky model for global
horizontal irradiance (GHI) — a function of the solar zenith angle only,
with no aerosol/water-vapour inputs — and the derived **clear-sky index**
``k* = GHI / GHI_cs``.  Unlike the extraterrestrial clearness index ``kt``
(see :func:`allsky.solar.clearness_index`), ``k*`` normalizes by a
ground-level clear-sky reference, so clear skies sit near 1.0, broken cloud
scatters around/above 1.0 (cloud enhancement) and overcast skies fall well
below 1.0.

All functions are pure numpy/pandas, vectorized over any datetime sequence
convertible to a :class:`pandas.DatetimeIndex`, and share the naive
local-standard-time contract of :mod:`allsky.solar` (tz-aware input is
rejected there).

References
----------
Haurwitz, B. (1945). Insolation in relation to cloudiness and cloud
density. *Journal of Meteorology* 2(3), 154-166.
doi:10.1175/1520-0469(1945)002<0154:IIRTCA>2.0.CO;2
Reno, M.J., Hansen, C.W., Stein, J.S. (2012). Global horizontal irradiance
clear sky models: implementation and analysis. SAND2012-2389.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from allsky.solar import cos_zenith, solar_elevation

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd

    from allsky.config import SiteConfig

    DatetimeLike = pd.DatetimeIndex | pd.Series | np.ndarray | list | tuple

__all__ = [
    "HAURWITZ_A_WM2",
    "HAURWITZ_B",
    "clear_sky_index",
    "haurwitz_ghi",
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
    the solar zenith angle from :func:`allsky.solar.cos_zenith`.

    Parameters
    ----------
    timestamps:
        Naive local clock times.
    site:
        Observation site (latitude/longitude in degrees).
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None.

    Limitation
    ----------
    Single-parameter fit (US mid-latitude climatology): it ignores aerosol
    load, water vapour and site elevation, so absolute magnitudes at a humid
    tropical coastal site carry a systematic bias.  It is used here as a
    normalization reference for ``k*``, where that bias largely cancels, not
    as an absolute irradiance predictor.
    """
    cosz = np.asarray(cos_zenith(timestamps, site, utc_offset_hours), dtype=np.float64)
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
        Measured global horizontal irradiance, W m-2, aligned 1:1 with
        *timestamps*.
    timestamps:
        Naive local clock times.
    site:
        Observation site (latitude/longitude in degrees).
    min_elevation_deg:
        Elevation floor below which ``k*`` is undefined (default 10 deg); at
        low sun the clear-sky reference is small and airmass errors dominate.
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None.

    Limitation
    ----------
    Inherits the absolute bias of :func:`haurwitz_ghi`; treat ``k*`` as a
    relative cloud-transmission proxy rather than a calibrated ratio.
    """
    ghi_arr = np.asarray(ghi, dtype=np.float64)
    ghi_cs = haurwitz_ghi(timestamps, site, utc_offset_hours)
    if ghi_arr.shape != ghi_cs.shape:
        raise ValueError(f"ghi shape {ghi_arr.shape} does not match {ghi_cs.shape} timestamps")
    elevation = solar_elevation(timestamps, site, utc_offset_hours)
    valid = (elevation >= min_elevation_deg) & (ghi_cs > 0.0) & np.isfinite(ghi_arr)
    kstar = np.full_like(ghi_arr, np.nan)
    np.divide(ghi_arr, ghi_cs, out=kstar, where=valid)
    return kstar
