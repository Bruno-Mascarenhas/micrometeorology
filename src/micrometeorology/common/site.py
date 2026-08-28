"""The station's own coordinates and clock offset, declared once.

Both the sensor archive's quality checks and the climatology exporter compute
solar geometry for the same tower, and each used to carry its own copy of the
numbers.  Two copies of a coordinate are two things that can drift: a latitude
corrected in one place and not the other moves sunrise in the QC gate away from
sunrise in the published distributions, with nothing failing.  This module is
neither a CLI nor a sensors module, so both can import it without either
depending on the other.
"""

from allsky.config import SiteConfig

__all__ = ["STATION_SITE", "STATION_UTC_OFFSET_HOURS"]

#: LabMiM tower, Instituto de Física, UFBA — Ondina, Salvador. The numbers live
#: in :class:`allsky.config.SiteConfig`'s defaults, which every package reads
#: through this name; repeating them here is how the three packages ended up
#: with three different latitudes for one tower.
STATION_SITE = SiteConfig()

#: Salvador keeps UTC-3 all year — DST never applied to Bahia and Brazil
#: abolished it in 2019 — so a fixed offset is exact, not an approximation.
STATION_UTC_OFFSET_HOURS = -3.0
