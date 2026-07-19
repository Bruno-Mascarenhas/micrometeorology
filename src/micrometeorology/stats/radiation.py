"""Station-Series solar-radiation indices: clearness index (Kt) and diffuse fraction (Kd).

This module is the **station-Series API**: both functions take and return
aligned :class:`pandas.Series` on the observation :class:`~pandas.DatetimeIndex`,
computing the two dimensionless ratios from *already-measured* irradiance
columns of the processed sensor CSV. They are descriptive quantities for
monitoring and reports, and preserve the public names operational consumers
import.

Relationship to :mod:`allsky` (do **not** merge these)
------------------------------------------------------
The all-sky ML pipeline has overlapping-sounding helpers with different roles;
pick by what you already hold:

- :func:`allsky.solar.clearness_index` computes ``Kt`` from a GHI array **plus
  timestamps and site coordinates**, deriving the extraterrestrial term
  internally via NOAA solar geometry, and returns a NumPy array. Use it in the
  feature pipeline when you only have GHI and need the horizontal-surface ``Kt``.
  :func:`clearness_index` here instead consumes a **precomputed extraterrestrial
  Series** and returns a Series with the station index preserved.
- :func:`allsky.clearsky.clear_sky_index` computes the *clear-sky* index
  ``k* = GHI / GHI_clearsky`` (Haurwitz normalisation), a different denominator
  from ``Kt``'s extraterrestrial one.
- :func:`allsky.erbs.erbs_diffuse_fraction` is the Erbs (1982) correlation that
  **models** ``Kd`` from ``Kt`` (a bootstrap target where no shaded pyranometer
  exists). :func:`diffuse_fraction` here is the **measured** ratio from a real
  diffuse channel (DHI / GHI). Keep the modelled and measured diffuse fractions
  distinct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["clearness_index", "diffuse_fraction"]

# Physically implausible ratios are rejected above this ceiling. The true upper
# bound is ~1 for both indices, but low-sun geometry and sensor time-response
# mismatch inflate the ratio near sunrise/sunset; 1.5 rejects gross outliers
# while tolerating that transient overshoot.
_MAX_RATIO = 1.5


def clearness_index(
    global_radiation: pd.Series,
    extraterrestrial_radiation: pd.Series,
) -> pd.Series:
    r"""Clearness index ``Kt``: measured global over top-of-atmosphere irradiance.

    Formula
    -------
    .. math:: K_t = \frac{S_{w,dw}}{S_{w,top}}

    where ``S_w,dw`` is the measured downwelling global shortwave irradiance and
    ``S_w,top`` the extraterrestrial (top-of-atmosphere) horizontal irradiance.
    ``Kt`` near 0 is fully overcast; near 1 is a clear sky.

    Parameters
    ----------
    global_radiation:
        Measured downwelling global shortwave irradiance (W m-2), indexed by the
        station :class:`~pandas.DatetimeIndex`.
    extraterrestrial_radiation:
        Top-of-atmosphere horizontal irradiance (W m-2) on the same index.

    Returns
    -------
    pandas.Series
        ``Kt`` aligned to the inputs' index. Set to NaN where the
        extraterrestrial term is non-positive (night / low sun) or the ratio
        falls outside ``[0, 1.5]``.

    Limitation
    ----------
    No solar geometry is computed here — the extraterrestrial term must be
    supplied. When you have only GHI and timestamps, use
    :func:`allsky.solar.clearness_index`, which derives that term internally.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = global_radiation / extraterrestrial_radiation
    kt[extraterrestrial_radiation <= 0] = np.nan
    kt[(kt < 0) | (kt > _MAX_RATIO)] = np.nan
    return kt


def diffuse_fraction(
    diffuse_radiation: pd.Series,
    global_radiation: pd.Series,
) -> pd.Series:
    r"""Diffuse fraction ``Kd``: measured diffuse over measured global irradiance.

    Formula
    -------
    .. math:: K_d = \frac{S_{w,dif}}{S_{w,dw}}

    the fraction of downwelling global shortwave that arrives as diffuse (sky)
    radiation rather than direct beam. ``Kd`` near 1 is fully diffuse (overcast);
    small ``Kd`` is a clear, beam-dominated sky.

    Parameters
    ----------
    diffuse_radiation:
        Measured diffuse shortwave irradiance (W m-2), indexed by the station
        :class:`~pandas.DatetimeIndex`.
    global_radiation:
        Measured downwelling global shortwave irradiance (W m-2) on the same
        index.

    Returns
    -------
    pandas.Series
        ``Kd`` aligned to the inputs' index. Set to NaN where global radiation is
        non-positive (night) or the ratio falls outside ``[0, 1.5]``.

    Limitation
    ----------
    This is the **measured** diffuse fraction and needs a real diffuse channel.
    To *model* ``Kd`` from ``Kt`` where no diffuse sensor exists, use
    :func:`allsky.erbs.erbs_diffuse_fraction` (Erbs 1982) instead.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        kd = diffuse_radiation / global_radiation
    kd[global_radiation <= 0] = np.nan
    kd[(kd < 0) | (kd > _MAX_RATIO)] = np.nan
    return kd
