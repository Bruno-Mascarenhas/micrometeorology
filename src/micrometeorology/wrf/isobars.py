"""Isobars of the sea-level pressure field, as polylines the site draws directly.

The site is given finished lines rather than the pressure grid, for the same
reason the sky page is given finished class fractions: a client tracing its own
contours would be a second numerical path to a published quantity, and the two
would diverge the first time either side changed a level or a smoothing pass.
It also keeps the front-end a path renderer, with no contouring library and no
physics in JavaScript.

The lines are an overlay, independent of whatever field is shaded beneath them,
so they are published once per domain and step and drawn over any of
:data:`ISOBAR_OVERLAY_VARIABLES`.  Isobars over a surface-energy or radiation
field would be decoration: those are governed by land use and the diurnal cycle,
on a scale the pressure pattern says nothing about.

Sea-level reduction, and why surface pressure will not do, is in
:mod:`micrometeorology.wrf.sea_level_pressure`.
"""

from typing import Any

import numpy as np
from contourpy import contour_generator
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from micrometeorology.wrf.safety import assert_reasonable_array_size

#: Output id the per-step files are named with, matching the WIND_VECTORS
#: overlay's spelling: ``{D}_{ID}_{NNN}.json``.
ISOBARS_OUTPUT_ID = "ISOBARS"

#: Shaded base layers the overlay is meaningful over, coarsest reason first: wind
#: and everything derived from it stand in geostrophic balance with the pressure
#: field, so the lines explain the shading; rain follows the convergence the same
#: pattern organises; temperature follows its advection.  Pinned against the id
#: generators by test, since these are a published contract and the generators
#: live in a module that imports this one.
#:
#: ``WIND_VECTORS`` is deliberately absent: it is itself an overlay rather than a
#: field anything is shaded with, so there is no such base layer to draw over.
ISOBAR_OVERLAY_VARIABLES: tuple[str, ...] = (
    "WIND",
    "WIND_POWER_DENSITY_10M",
    "POT_EOLICO_50M",
    "POT_EOLICO_100M",
    "POT_EOLICO_150M",
    "RAIN",
    "TEMP",
)

#: Contour spacings tried from coarsest to finest, all of them intervals synoptic
#: charts are drawn at. Sub-hPa rungs were published once and withdrawn; the
#: measurements are in ``docs/micrometeorology.md``.
ISOBAR_INTERVALS_HPA: tuple[float, ...] = (4.0, 2.0, 1.0)

#: Lines a typical step should carry.  The spacing is chosen against the domain's
#: MEDIAN range rather than its narrowest step: sizing it so the flattest step of
#: all still holds a line would drive the outer domains to a spacing that leaves
#: them nearly blank, which is where isobars carry the most meaning.  A step
#: flatter than the chosen spacing publishes no line at all.
TARGET_LEVELS_PER_STEP = 5

#: Gaussian smoothing applied before contouring, in GRID CELLS, so each domain
#: smooths relative to its own resolution.  The sea-level reduction leaves
#: cell-scale noise over terrain, and contouring it raw produces wiggling closed
#: contours around individual cells: on the operational d01 this cuts 42 line
#: segments to 12 without moving the pattern.
ISOBAR_SMOOTHING_SIGMA_CELLS = 1.5

#: Decimals kept on published coordinates: ~11 m, two orders of magnitude below
#: the finest grid spacing, so the line position is limited by the field and not
#: by the encoding.
ISOBAR_COORDINATE_DECIMALS = 4

#: Decimals kept on a published level, enough for the finest interval above.
ISOBAR_LEVEL_DECIMALS = 2


def smooth_for_contouring(field: NDArray) -> NDArray:
    """Damp the cell-scale noise the sea-level reduction leaves over terrain.

    Parameters
    ----------
    field:
        Sea-level pressure, ``(Y, X)``, hPa.

    Returns
    -------
    numpy.ndarray
        Smoothed field, ``(Y, X)``, hPa, ``float64``.
    """
    return np.asarray(
        gaussian_filter(np.asarray(field, dtype=np.float64), ISOBAR_SMOOTHING_SIGMA_CELLS),
        dtype=np.float64,
    )


def choose_interval_hpa(typical_range_hpa: float) -> float:
    """Coarsest published spacing that still fills a typical step with lines.

    One spacing serves a whole DOMAIN: a spacing that changed between steps would
    make the lines jump under the animation, and the legend would have to change
    with it.  Domains do differ from each other — the innermost spans a fraction
    of the outermost's pressure range — so each publishes its own
    ``metadata.interval`` and the legend follows the domain, not the run.

    Parameters
    ----------
    typical_range_hpa:
        Median peak-to-peak sea-level pressure range over the steps that will be
        published, hPa. Must be finite and positive.

    Returns
    -------
    float
        One of :data:`ISOBAR_INTERVALS_HPA` — the coarsest yielding at least
        :data:`TARGET_LEVELS_PER_STEP` lines on that typical step, or the 1 hPa
        floor when none does, which draws one line on a typical step and none on
        the flattest.

    Raises
    ------
    ValueError
        When the range is not finite and positive. Falling back to the floor
        instead would publish a failed reduction as a plausible single isobar, and
        nothing downstream would report the field as unusable.
    """
    if not np.isfinite(typical_range_hpa) or typical_range_hpa <= 0.0:
        raise ValueError(
            f"typical sea-level pressure range must be finite and positive, "
            f"got {typical_range_hpa!r}"
        )
    for interval in ISOBAR_INTERVALS_HPA:
        if typical_range_hpa / interval >= TARGET_LEVELS_PER_STEP:
            return interval
    return ISOBAR_INTERVALS_HPA[-1]


def isobar_levels_hpa(field: NDArray, interval_hpa: float) -> NDArray:
    """Round pressure levels crossing *field*, empty when none do.

    A field flatter than the spacing yields NO levels rather than a line at its
    midpoint. Such a line would sit at a value the spacing never names, drawn
    only because something had to be drawn: it asserts a gradient the field does
    not resolve. An empty step is the honest publication, and the page renders it
    as a map with no isobars rather than as a map with one meaningless one.

    Parameters
    ----------
    field:
        Sea-level pressure, ``(Y, X)``, hPa.
    interval_hpa:
        Spacing from :func:`choose_interval_hpa`.

    Returns
    -------
    numpy.ndarray
        Levels in hPa, ``(L,)``, ascending, every one strictly inside the field's
        range so each yields a real contour. ``L`` may be zero.

    Raises
    ------
    ValueError
        When the field is entirely non-finite, which is a failed reduction rather
        than a flat one and must not be published as an empty step.
    """
    values = np.asarray(field, dtype=np.float64)
    if not np.isfinite(values).any():
        raise ValueError("sea-level pressure field is all-NaN; the reduction failed")
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    if not high > low:
        return np.empty(0, dtype=np.float64)

    first = np.ceil(low / interval_hpa) * interval_hpa
    levels = np.round(np.arange(first, high, interval_hpa), ISOBAR_LEVEL_DECIMALS)
    return np.asarray(levels[(levels > low) & (levels < high)], dtype=np.float64)


def create_isobars_json(
    lon: NDArray,
    lat: NDArray,
    field: NDArray,
    *,
    levels_hpa: NDArray,
    interval_hpa: float,
    date_time: str,
) -> dict[str, Any]:
    """Build one step's standalone isobar overlay payload.

    Parameters
    ----------
    lon, lat:
        Cell-centre coordinates, ``(Y, X)``, degrees east and north.
    field:
        Sea-level pressure on those cells, ``(Y, X)``, hPa, already smoothed by
        :func:`smooth_for_contouring`.
    levels_hpa:
        Levels to trace, ``(L,)``, hPa, from :func:`isobar_levels_hpa`.
    interval_hpa:
        The run's chosen spacing, published so the legend states it rather than
        inferring it from consecutive levels.
    date_time:
        Already-formatted local stamp, in the spelling every other per-step file
        carries.

    Returns
    -------
    dict[str, Any]
        ``metadata`` with the stamp, unit, interval and smoothing; and
        ``isobars``, one entry per level with its ``level`` in hPa and ``paths``,
        a list of ``[lon, lat]`` polylines. A level whose contour is empty is
        omitted, so every published entry draws something.

    Raises
    ------
    ValueError
        When the coordinate and field shapes disagree.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    values = np.asarray(field, dtype=np.float64)
    if not lon.shape == lat.shape == values.shape:
        raise ValueError(
            f"isobar shapes differ: lon {lon.shape!r}, lat {lat.shape!r}, field {values.shape!r}"
        )
    assert_reasonable_array_size(values.shape, values.dtype, context="isobar generation")

    generator = contour_generator(x=lon, y=lat, z=values, line_type="Separate")
    isobars = []
    for level in np.asarray(levels_hpa, dtype=np.float64):
        paths = [
            np.round(np.asarray(segment, dtype=np.float64), ISOBAR_COORDINATE_DECIMALS).tolist()
            for segment in generator.lines(float(level))
            if np.asarray(segment).shape[0] >= 2
        ]
        if paths:
            isobars.append({"level": float(level), "paths": paths})

    return {
        "metadata": {
            "date_time": date_time,
            "unit": "hPa",
            "interval": round(float(interval_hpa), ISOBAR_LEVEL_DECIMALS),
            "smoothing_sigma_cells": ISOBAR_SMOOTHING_SIGMA_CELLS,
        },
        "isobars": isobars,
    }
