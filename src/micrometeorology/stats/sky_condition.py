"""Sky condition from the clearness index, and the cumulative curve it is read off.

The four sky conditions are Escobedo, Gomes, Oliveira & Soares (2009), Applied
Energy 86(3):299-309, sec. 3.1, with the Portuguese nomenclature of Teramoto &
Escobedo (2012), RBEAA 16(9):985-992.  The bounds themselves are **not** declared
here: they live in :data:`allsky.data.contracts.SKY_CLASS_KT_UPPER_BOUNDS`, which
the all-sky manifest also classifies against, so the two pipelines cannot drift
onto different partitions of the same published quantity.

The upper edge of a band is CLOSED, as published: a Kt sitting exactly on 0.35
is condition I, not II.  :func:`allsky.data.manifest._classify_sky` applies the
same rule to the per-frame manifest, and a cross-consistency test pins the two
against each other on the boundary values themselves.

What differs between the two is the Kt being classified, not the partition.  The
manifest classifies a per-frame Kt on its own solar geometry; the climatology
exporter classifies an HOURLY MEAN whose extraterrestrial denominator carries the
BSRN midpoint correction, gated on solar elevation over the whole averaging
window.  The class fractions published from here therefore do not have to equal
the manifest's ``sky_class`` distribution, and the artifact says so in its
caveats.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from allsky.data.contracts import (
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLASS_NAMES,
    SKY_CLASS_NAMES_PT,
    SKY_CLASS_REFERENCE,
    SKY_CLASS_VALUES,
)

__all__ = [
    "SKY_CONDITION_IDS",
    "classify_sky_condition",
    "cumulative_fractions",
    "sky_condition_summary",
]

#: Roman-numeral slugs of conditions I..IV, in class order.  The published
#: literature numbers the conditions from I; the class integer is that number
#: minus one, and a bare integer crossing that boundary is an off-by-one waiting
#: to happen, so every consumer-facing payload carries both.
SKY_CONDITION_IDS = ("i", "ii", "iii", "iv")


def classify_sky_condition(kt: NDArray) -> NDArray:
    """Sky condition of every clearness index, as the published class integer.

    Parameters
    ----------
    kt:
        Clearness index (global over extraterrestrial horizontal irradiance),
        shape ``(N,)``, dimensionless.

    Returns
    -------
    numpy.ndarray
        Class integers, shape ``(N,)``, dtype ``int64``: ``0`` cloudy, ``1``
        partly cloudy with diffuse dominance, ``2`` partly cloudy with clear
        dominance, ``3`` clear.  Non-finite entries are ``-1``: an unlabelable
        sample is never silently folded into a real class.
    """
    values = np.asarray(kt, dtype=np.float64)
    cloudy, diffuse_dominant, clear_dominant = SKY_CLASS_KT_UPPER_BOUNDS
    labels: NDArray = np.select(
        [values <= cloudy, values <= diffuse_dominant, values <= clear_dominant],
        list(SKY_CLASS_VALUES[:3]),
        default=SKY_CLASS_VALUES[3],
    ).astype(np.int64)
    labels[~np.isfinite(values)] = -1
    return labels


def cumulative_fractions(counts: NDArray | list[int], *, total: int) -> list[float]:
    """Empirical F(x) at each bin's UPPER edge, from the histogram's own counts.

    Derived from the counts the histogram publishes rather than from the sample,
    so the cumulative curve is by construction the running sum of the bars beside
    it: the two cannot disagree on the same page after a gate changes.

    Parameters
    ----------
    counts:
        Per-bin counts, shape ``(B,)``.
    total:
        Denominator of the fraction — the whole subset, including the samples
        that fell outside the binned range, so F is a fraction of the record and
        not of the bars.

    Returns
    -------
    list of float
        ``B`` values in ``[0, 1]``, non-decreasing.  An empty list when *total*
        is zero, since no fraction of nothing is defined.
    """
    if total <= 0:
        return []
    running = np.cumsum(np.asarray(counts, dtype=np.float64))
    return [float(value) for value in running / float(total)]


def sky_condition_summary(kt: NDArray) -> dict[str, Any]:
    """The four conditions with the share of *kt* that falls in each.

    The fraction is published already computed: the page draws what it is given
    and never re-derives a class share by interpolating the cumulative curve,
    which would be a second numerical path to the same number.

    Parameters
    ----------
    kt:
        Clearness index, shape ``(N,)``, dimensionless.  Non-finite entries are
        excluded from both the counts and the denominator.

    Returns
    -------
    dict
        ``kt_upper_bounds``, ``reference`` and ``conditions``: one entry per
        class with ``condition`` (1..4), ``id`` (``i``..``iv``), ``name``,
        ``name_pt``, ``kt_range``, ``count`` and ``fraction``.  ``fraction`` is
        None for every class when no sample is labelable.
    """
    labels = classify_sky_condition(kt)
    labelable = labels >= 0
    total = int(labelable.sum())
    lower_edges = (None, *SKY_CLASS_KT_UPPER_BOUNDS)

    conditions = []
    for class_value in SKY_CLASS_VALUES:
        count = int((labels == class_value).sum())
        conditions.append(
            {
                "condition": class_value + 1,
                "id": SKY_CONDITION_IDS[class_value],
                "name": SKY_CLASS_NAMES[class_value],
                "name_pt": SKY_CLASS_NAMES_PT[class_value],
                "kt_range": [
                    lower_edges[class_value],
                    SKY_CLASS_KT_UPPER_BOUNDS[class_value]
                    if class_value < len(SKY_CLASS_KT_UPPER_BOUNDS)
                    else None,
                ],
                "count": count,
                "fraction": (count / total) if total else None,
            }
        )
    return {
        "kt_upper_bounds": list(SKY_CLASS_KT_UPPER_BOUNDS),
        "reference": SKY_CLASS_REFERENCE,
        "n": total,
        "conditions": conditions,
    }
