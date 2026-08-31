"""The station's own coordinates and clock offset, declared once.

Both the sensor archive's quality checks and the climatology exporter compute
solar geometry for the same tower, and both read the numbers from here.  Two
copies of a coordinate are two things that can drift: a latitude corrected in
one place and not the other moves sunrise in the QC gate away from sunrise in
the published distributions, with nothing failing.  This module is neither a CLI
nor a sensors module, so both can import it without either depending on the
other, and it imports nothing from this project — the all-sky package reads the
site from here too, which is what keeps that dependency one-directional.
"""

from pydantic import BaseModel

__all__ = ["STATION_SITE", "STATION_UTC_OFFSET_HOURS", "SiteConfig"]


class SiteConfig(BaseModel):
    """Observation site (LabMiM/UFBA, Salvador-BA by default)."""

    #: LabMiM tower, Instituto de Física, UFBA — Ondina, Salvador. These are the
    #: canonical coordinates for the station: every package reads them from
    #: here, so a surveyed correction reaches all of them at once. Repeating
    #: them anywhere else is how the three packages once ended up with three
    #: different latitudes for one tower.
    latitude: float = -13.0055
    longitude: float = -38.5089
    #: Fixed offset of the clock the instrument stamps with, in hours from UTC.
    #: It travels WITH the coordinates because solar geometry needs both, and a
    #: site read from here beside an offset read from a module global is how a
    #: second station would get this one's clock — computing California noon on a
    #: Salvador clock without failing anywhere.
    #:
    #: Fixed, not a named zone: this is the offset of an instrument's own clock,
    #: and a site whose civil time observes DST needs its acquisition convention
    #: declared rather than inferred.
    utc_offset_hours: float = -3.0


#: The station itself, at those default coordinates.
STATION_SITE = SiteConfig()

#: Salvador keeps UTC-3 all year — DST never applied to Bahia and Brazil
#: abolished it in 2019 — so a fixed offset is exact, not an approximation.
STATION_UTC_OFFSET_HOURS = -3.0
