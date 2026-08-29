"""Constants that come from the instruments themselves, not from physics.

A number in :mod:`micrometeorology.common.physics` is true everywhere; a number
here is true of the hardware on this tower and changes the day the hardware is
swapped. Keeping the two apart is the point — a gauge replacement must not read
like a correction to a physical constant.

Each entry carries the measurement or the specification it came from, because a
value whose provenance is lost cannot be revised with confidence later.
"""

from __future__ import annotations

__all__ = ["RAIN_TIP_DEPTH_MM"]

#: One tip of the LabMiM tipping-bucket gauge, in millimetres — 0.01 inch.
#:
#: Verified against the archive rather than taken from the datasheet alone:
#: every positive interval total is an integer multiple of it, 21,443 of 21,444
#: positive samples over ten years, worst deviation 4e-06. The one exception is
#: 1.09e9 mm of rain in five minutes on 2018-06-10 09:10, a corrupted field
#: rather than weather.
#:
#: It decides two different things, which is why it was written down twice and
#: has to be written down once: the quantisation check that validates the raw
#: record, and the wet/dry threshold plus narrowest honest histogram bin in the
#: climatology export. A gauge with a different bucket invalidates both at once.
RAIN_TIP_DEPTH_MM = 0.254
