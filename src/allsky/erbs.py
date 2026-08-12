"""Erbs et al. (1982) diffuse-fraction decomposition of global irradiance.

Provides the standard three-piece polynomial correlation ``kd(kt)``
between the hourly diffuse fraction ``kd = DHI / GHI`` and the clearness
index ``kt = GHI / E0h``, plus :func:`pseudo_diffuse` which converts a
GHI series into a **pseudo** diffuse-irradiance target.

.. warning::
   :func:`pseudo_diffuse` produces *pseudo-targets*, not measurements.
   The prepare pipeline reaches for it only when
   ``PrepareTargetsConfig.diffuse_column`` is null; with the default
   ``PSP_Wm2_Avg`` it builds measured diffuse targets instead.  Rows
   built here are flagged ``target_source="erbs_pseudo"``, and a
   regression metric over them scores agreement with the Erbs
   decomposition rather than with a pyranometer.

References
----------
Erbs, D.G., Klein, S.A., Duffie, J.A. (1982). Estimation of the diffuse
radiation fraction for hourly, daily and monthly-average global
radiation. *Solar Energy* 28(4), 293-302.
doi:10.1016/0038-092X(82)90302-4
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

type ErbsInput = pd.Series | np.ndarray | Sequence[float] | float

__all__ = ["erbs_diffuse_fraction", "pseudo_diffuse"]


def erbs_diffuse_fraction(kt: ErbsInput) -> np.ndarray:
    """Diffuse fraction ``kd`` from the clearness index ``kt`` (Erbs 1982).

    Formula
    -------
    - ``kt <= 0.22``:  ``kd = 1.0 - 0.09 kt``
    - ``0.22 < kt <= 0.80``:
      ``kd = 0.9511 - 0.1604 kt + 4.388 kt^2 - 16.638 kt^3 + 12.336 kt^4``
    - ``kt > 0.80``:  ``kd = 0.165``

    Parameters
    ----------
    kt:
        Clearness index ``GHI / E0h``, shape ``(N,)`` (or ``()`` for a
        scalar), dimensionless.  NaN marks a sample where ``kt`` is
        undefined, typically night.

    Returns
    -------
    numpy.ndarray
        Diffuse fraction ``kd = DHI / GHI``, same shape as *kt*,
        ``float64``, dimensionless, clipped to ``[0, 1]``.  NaN input
        yields NaN output.

    Limitation
    ----------
    Hourly correlation fitted on US/Australian stations; applied here to
    5-minute records at a tropical coastal site, so expect scatter.  It
    also cannot capture cloud-enhancement events (kt > 1).
    """
    kt_arr = np.asarray(kt, dtype=np.float64)
    kd = np.full_like(kt_arr, np.nan)

    low = kt_arr <= 0.22
    mid = (kt_arr > 0.22) & (kt_arr <= 0.80)
    high = kt_arr > 0.80

    kd[low] = 1.0 - 0.09 * kt_arr[low]
    kt_mid = kt_arr[mid]
    kd[mid] = 0.9511 - 0.1604 * kt_mid + 4.388 * kt_mid**2 - 16.638 * kt_mid**3 + 12.336 * kt_mid**4
    kd[high] = 0.165

    return np.clip(kd, 0.0, 1.0)


def pseudo_diffuse(ghi: ErbsInput, kt: ErbsInput) -> np.ndarray:
    """Pseudo diffuse irradiance ``DHI = kd(kt) * GHI`` in W m-2.

    Formula
    -------
    ``DHI = erbs_diffuse_fraction(kt) * GHI``

    Parameters
    ----------
    ghi:
        Measured global horizontal irradiance, shape ``(N,)``, W m-2.
    kt:
        Clearness index aligned 1:1 with *ghi*, shape ``(N,)``,
        dimensionless.

    Returns
    -------
    numpy.ndarray
        Pseudo diffuse horizontal irradiance, shape ``(N,)``,
        ``float64``, W m-2.  Since ``kd`` is bounded in [0, 1] the result
        satisfies ``0 <= DHI <= GHI`` for non-negative GHI; NaN in either
        input propagates to NaN.

    Limitation
    ----------
    This is a **pseudo-target**, selected only when no diffuse column is
    configured; dataset rows built from it carry
    ``target_source="erbs_pseudo"``.
    """
    ghi_arr = np.asarray(ghi, dtype=np.float64)
    return erbs_diffuse_fraction(kt) * ghi_arr
