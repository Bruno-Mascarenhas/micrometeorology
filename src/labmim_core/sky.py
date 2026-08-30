"""The Escobedo sky-condition partition, and the names it publishes.

Four conditions cut on the clearness index Kt, from Escobedo, Gomes, Oliveira &
Soares (2009), Applied Energy 86(3):299-309, §3.1, with the Portuguese
nomenclature of Teramoto & Escobedo (2012), RBEAA 16(9):985-992.

Both the all-sky classification head and the micrometeorology sky-condition
statistics cut on these same bounds and publish these same names. One owner, or
a bound corrected in one place silently relabels only half the artifacts.

Pure stdlib.
"""

__all__ = [
    "SKY_CLASS_COUNT",
    "SKY_CLASS_KT_UPPER_BOUNDS",
    "SKY_CLASS_MISSING",
    "SKY_CLASS_NAMES",
    "SKY_CLASS_NAMES_PT",
    "SKY_CLASS_REFERENCE",
    "SKY_CLASS_VALUES",
    "SKY_CLEAR",
    "SKY_CLOUDY",
    "SKY_PARTLY_CLOUDY_CLEAR",
    "SKY_PARTLY_CLOUDY_DIFFUSE",
    "sky_class_name",
]


#: Sky-condition classes I..IV, ordered by increasing clearness index so the
#: class integer is the published condition number minus one.
# Escobedo, Gomes, Oliveira & Soares (2009), Applied Energy 86(3):299-309, §3.1;
# Portuguese nomenclature after Teramoto & Escobedo (2012), RBEAA 16(9):985-992.
SKY_CLOUDY = 0
SKY_PARTLY_CLOUDY_DIFFUSE = 1
SKY_PARTLY_CLOUDY_CLEAR = 2
SKY_CLEAR = 3
#: Sentinel for an unlabelable sample (non-finite Kt); ``-1`` in the batch.
SKY_CLASS_MISSING = -1
#: Valid class integers (the missing sentinel is intentionally excluded).
SKY_CLASS_VALUES = (
    SKY_CLOUDY,
    SKY_PARTLY_CLOUDY_DIFFUSE,
    SKY_PARTLY_CLOUDY_CLEAR,
    SKY_CLEAR,
)
#: Machine-readable class names, indexable by the class integer.
SKY_CLASS_NAMES = ("cloudy", "partly_cloudy_diffuse", "partly_cloudy_clear", "clear")
#: The published Portuguese condition names, in the same order.
SKY_CLASS_NAMES_PT = (
    "nebuloso",
    "parcialmente nebuloso com dominancia para o difuso",
    "parcialmente nebuloso com dominancia para o claro",
    "claro",
)
#: Upper Kt bounds of conditions I..III; condition IV is everything above the
#: last bound.  Inclusive upper edges, as published.
SKY_CLASS_KT_UPPER_BOUNDS = (0.35, 0.55, 0.65)
#: Citation recorded in every manifest sidecar so a published figure can be
#: traced back to the classification it used.
SKY_CLASS_REFERENCE = (
    "Escobedo, Gomes, Oliveira & Soares (2009), Applied Energy 86(3):299-309, "
    "sec. 3.1; Portuguese nomenclature after Teramoto & Escobedo (2012), "
    "RBEAA 16(9):985-992"
)
#: Number of sky classes a classification head must emit.
SKY_CLASS_COUNT = len(SKY_CLASS_VALUES)


def sky_class_name(value: int) -> str:
    """Name for a sky-class integer; ``"missing"`` for the ``-1`` sentinel.

    Parameters
    ----------
    value:
        Class integer as stored in the manifest's ``sky_class`` column.

    Returns
    -------
    str
        The machine-readable name from :data:`SKY_CLASS_NAMES`, or ``"missing"``
        for an unlabelable sample.

    Raises
    ------
    ValueError
        If *value* is neither a valid class in :data:`SKY_CLASS_VALUES` nor the
        :data:`SKY_CLASS_MISSING` sentinel.
    """
    if value == SKY_CLASS_MISSING:
        return "missing"
    if value in SKY_CLASS_VALUES:
        return SKY_CLASS_NAMES[value]
    raise ValueError(
        f"invalid sky_class {value!r}; expected one of {SKY_CLASS_VALUES} or "
        f"{SKY_CLASS_MISSING} (missing)"
    )
