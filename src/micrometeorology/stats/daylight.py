"""The daylight gate every published hourly artifact is conditioned on.

An hourly row is a MEAN over ``[T, T+1h)``, so asking whether the sun was high
enough is a question about the whole window, not about one instant inside it.
Evaluating it at the window's midpoint alone admits the hours the sun rises or
sets in — which is how two artifacts that claim to describe one record end up
describing different populations.

This module owns that gate so the climatology page, the sky page and any figure
answer it identically.  A second copy of the bracket is a second definition of
"daytime", and the only symptom is two row counts that no caveat explains.

Timestamps are naive station-local, as they arrive from the datalogger's own
clock; the offset that turns them into solar geometry enters explicitly as
``utc_offset_hours`` and never from the host's zone.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from labmim_core.site import SiteConfig

__all__ = ["MIN_SOLAR_ELEVATION_DEG", "SELECTION_BRACKET_OFFSETS", "elevation_bounds"]

#: Instants the gate brackets the sun over: the averaging window's own endpoints
#: and its midpoint.  Solar elevation has at most one turning point inside an
#: hour, and the only one that can fall in a window this gate decides is solar
#: midnight (a minimum), so sampling these three brackets the window.
SELECTION_BRACKET_OFFSETS: tuple[pd.Timedelta, ...] = (
    pd.Timedelta(0),
    pd.Timedelta(minutes=30),
    pd.Timedelta(minutes=60),
)

#: Solar elevation above which a shortwave sample counts as daytime: below it the
#: airmass is extreme and relative error swamps the signal, so the clearness index
#: and every shortwave distribution are gated on it.
MIN_SOLAR_ELEVATION_DEG = 10.0


def elevation_bounds(
    times: pd.DatetimeIndex, site: SiteConfig, utc_offset_hours: float
) -> tuple[NDArray, NDArray]:
    """Lowest and highest solar elevation over the hour each stamp labels.

    Parameters
    ----------
    times:
        Naive station-local stamps labelling the START of each averaging window,
        ``(N,)``.
    site:
        Observation site, in degrees.
    utc_offset_hours:
        Fixed offset of the local clock.

    Returns
    -------
    tuple of numpy.ndarray
        Minimum and maximum solar elevation in degrees, ``(N,)`` each, in the
        input's own order.
    """
    from labmim_core.solar import solar_elevation_deg

    elevations = np.stack(
        [
            solar_elevation_deg(times + offset, site, utc_offset_hours)
            for offset in SELECTION_BRACKET_OFFSETS
        ]
    )
    lowest: NDArray = elevations.min(axis=0)
    highest: NDArray = elevations.max(axis=0)
    return lowest, highest
