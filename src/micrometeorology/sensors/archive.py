"""The LabMiM station archive: an explicit manifest, staged fixes, one merged frame.

Turning ``data/dados-labmim/`` into a usable database is not a glob. An audit of
every table in the archive (2016-09 to 2026-04) found four ways the obvious
approach silently produces a wrong record:

1. **``*.dat`` drops the rotation files.** Three ``.backup`` tables are the ONLY
   source of an entire austral winter each — JJA 2020, JJA 2022 and June to
   mid-July 2024. A glob that skips them deletes three winters from the record
   without a warning.
2. **The directory holds more than one station.** ``BTS_*`` is a different site
   (CR1000X serial 9429), the ``celsolar`` / ``calibracao`` tables are
   side-by-side instrument campaigns, and the ``solar`` / ``radiacao`` families
   sample at one minute. Merged together they produce a frame that parses
   cleanly and means nothing.
3. **Names lie.** ``dados-labmim/LBM_lenta.dat`` is the RAIN table — TOA5 header
   field 8 reads ``LBM_rain`` — and it is the unique source of February 2019.
4. **Three clock defects cannot be expressed in configuration.** They need the
   bytes fixed before the merge, which is what :func:`stage_archive` does, always
   into a scratch directory: nothing here ever writes to ``data/``.

So this module carries the manifest as data, in ingest order, with each file's
disposition recorded next to it. :func:`verify_frame` then checks the merged
result against the row counts, span and monotonicity the audit measured, so a
future change that quietly drops a file fails loudly instead of publishing a
shorter record.

Relationship to the neighbouring modules
----------------------------------------
- :mod:`micrometeorology.sensors.ingestion` reads and merges individual tables;
  this module decides *which* tables, in what order, and with what repairs.
- :mod:`micrometeorology.sensors.calibration` applies the instrument factors and
  the era-to-era column unification (``sensor_switches``) on top of the frame
  built here.
- :mod:`micrometeorology.sensors.aggregation` collapses it to hourly.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Solar geometry for the night-corruption detector below. The same
# implementation the climatology exporter uses, so "deep night" means the same
# angle in both places.
from allsky.config import SiteConfig
from allsky.solar import cos_zenith, solar_elevation
from micrometeorology.common.paths import ensure_dir
from micrometeorology.sensors.ingestion import merge_dat_files

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_END",
    "ARCHIVE_START",
    "DIFFUSE_RATIO_LIMIT",
    "EXPECTED_LENTA_ROWS",
    "EXPECTED_RAIN_ROWS",
    "LENTA_MANIFEST",
    "NIGHT_CORRUPTION_CHANNELS",
    "NIGHT_CORRUPTION_FLUX_WM2",
    "RAIN_MANIFEST",
    "STATUS_COLUMNS",
    "ArchiveFile",
    "ArchiveReport",
    "build_five_minute_frame",
    "close_net_radiation",
    "mask_impossible_shortwave",
    "mask_night_corrupted_days",
    "mask_sentinels",
    "night_corrupted_days",
    "stage_archive",
    "unshaded_diffuse_days",
    "verify_frame",
]

# Measured over the manifests below. A merge that does not reproduce these has
# lost or gained a file; see verify_frame.
EXPECTED_LENTA_ROWS = 987_969
EXPECTED_RAIN_ROWS = 988_249
ARCHIVE_START = pd.Timestamp("2016-09-29 13:40:00")
ARCHIVE_END = pd.Timestamp("2026-04-24 13:00:00")

# Per-row instrument quality flags. Text, and therefore destroyed by a numeric
# coercion unless named explicitly (see ingestion.read_campbell_dat).
STATUS_COLUMNS = ("MetSENS1_Status", "MetSENS2_Status", "MetSENS_Status")

# Staging directives, resolved by _stage_file.
_CLOCK_PLUS_ONE_HOUR = "clock+1h"
_DROP_LATE_TAIL = "drop-late-tail"
_KEEP_2023_BLOCK = "keep-2023-block"

# The 2020 clock slip: every row stamped at or before this instant is one hour
# early. Verified by RECORD-joining the lenta and rain tables across the window.
_CLOCK_SLIP_LAST = pd.Timestamp("2020-02-28 11:50:00")
# Rows at or after this instant in the 2019 tables are a mis-stamped tail whose
# timestamps the clock-corrected 2020_03 table already carries, cell for cell.
_LATE_TAIL_FIRST = pd.Timestamp("2020-01-07 01:05:00")


@dataclass(frozen=True)
class ArchiveFile:
    """One table of the station record, with why it is in (or how it is repaired).

    Attributes
    ----------
    path:
        Location relative to the data root.
    staging:
        Repair to apply before reading, or ``None`` to read as found.
    note:
        Why this file is in the manifest — usually what would be lost without
        it. Read this before removing an entry.
    """

    path: str
    staging: str | None = None
    note: str = ""


# ---------------------------------------------------------------------------
# The manifests, in ingest order (chronological by first timestamp)
# ---------------------------------------------------------------------------

_D = "dados-labmim"

LENTA_MANIFEST: tuple[ArchiveFile, ...] = (
    ArchiveFile(f"{_D}/LBM_lenta_2016.dat", note="start of record, 2016-09-29"),
    ArchiveFile(f"{_D}/LBM_lenta_2017.dat", note="all of 2017, complete JJA"),
    ArchiveFile(f"{_D}/LBM_lenta_2018_1.dat", note="2018-01..2018-10-16, JJA 2018"),
    ArchiveFile(f"{_D}/LBM_lenta_2018-2019.dat", note="CNR1 commissioning era"),
    ArchiveFile(f"{_D}/LBM_lenta_2019.dat.backup", note="sole source of 2019-03-15 afternoon"),
    ArchiveFile(f"{_D}/LBM_lenta_2019.dat.1.backup", note="sole source of 2019-03-15..18"),
    ArchiveFile(
        f"{_D}/LBM_lenta_2019.dat.2.backup", note="sole source of 2019-03-18..19, WXT arrives"
    ),
    ArchiveFile(f"{_D}/LBM_lenta_2019.dat.3.backup", note="sole source of 2019-03-19..05-31"),
    ArchiveFile(f"{_D}/LBM_lenta_2019_0531.dat", note="2019-05-31 onward"),
    ArchiveFile(f"{_D}/LBM_lenta_2019_0631.dat", note="2019-06 onward"),
    ArchiveFile(f"{_D}/LBM_lenta_2019_1011.dat", note="2019-10 onward, CMP21 diffuse begins"),
    ArchiveFile(
        f"{_D}/LBM_lenta_2019.dat",
        staging=_DROP_LATE_TAIL,
        note="110-row tail is mis-stamped; the clock-fixed 2020_03 table carries it correctly",
    ),
    ArchiveFile(
        f"{_D}/LBM_lenta_2020_03.dat",
        staging=_CLOCK_PLUS_ONE_HOUR,
        note="headerless CSV, and 16855 rows are one hour early",
    ),
    ArchiveFile(f"{_D}/LBM_lenta_2020.dat.backup", note="SOLE SOURCE OF JJA 2020"),
    ArchiveFile(f"{_D}/LBM_lenta_2020.dat", note="rest of 2020"),
    ArchiveFile(f"{_D}/LBM_lenta_2021.dat", note="all of 2021"),
    ArchiveFile(f"{_D}/LBM_lenta_2022.dat.backup", note="SOLE SOURCE OF JJA 2022"),
    ArchiveFile(
        f"{_D}/LBM_lenta_2022.dat", note="rest of 2022 (superset of data/LBM_lenta_2022.dat)"
    ),
    ArchiveFile(f"{_D}/CR5000_LBM_lenta_18-21082023.dat", note="2023-08 spare-logger block"),
    ArchiveFile(f"{_D}/LBM_lenta_2023.dat", note="2023"),
    ArchiveFile(f"{_D}/LBM_lenta_2023_14032024.dat", note="2024-03 handover"),
    ArchiveFile(f"{_D}/LBM_lenta_2024.dat.backup", note="SOLE SOURCE OF JUNE AND 1-19 JULY 2024"),
    ArchiveFile(f"{_D}/LBM_lenta_2024.dat", note="rest of 2024"),
    ArchiveFile(f"{_D}/LBM_lenta_2025.dat.backup", note="2025-03 Gill MetSENS commissioning"),
    ArchiveFile(f"{_D}/LBM_lenta_2025.dat.1.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_D}/LBM_lenta_2025.dat.2.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_D}/LBM_lenta_2025.dat.3.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_D}/LBM_lenta_2025.dat.4.backup", note="2025-03-28..05-14, dual GMX units"),
    ArchiveFile("LBM_lenta_2025.dat", note="v22 era to 2026-04-24; PSP takes over diffuse"),
)

RAIN_MANIFEST: tuple[ArchiveFile, ...] = (
    ArchiveFile(f"{_D}/LBM_rain_2016.dat", note="start of rain record"),
    ArchiveFile(f"{_D}/LBM_rain_2017.dat", note="2017"),
    ArchiveFile(f"{_D}/LBM_rain_2018_2019.dat", note="2018 into 2019"),
    ArchiveFile(
        f"{_D}/LBM_lenta.dat",
        note="MISNAMED: TOA5 field 8 is LBM_rain. Unique source of 2019-01-31..02-26",
    ),
    ArchiveFile(
        f"{_D}/LBM_rain_2019.dat", staging=_DROP_LATE_TAIL, note="same 110-row mis-stamped tail"
    ),
    ArchiveFile(
        f"{_D}/LBM_rain_2020.dat", note="2020 (clock slip is in the lenta table, not here)"
    ),
    ArchiveFile(f"{_D}/LBM_rain_2021.dat", note="2021"),
    ArchiveFile(f"{_D}/LBM_rain_2022.dat", note="2022 (superset of data/LBM_rain_2022.dat)"),
    ArchiveFile(
        f"{_D}/CR5000_LBM_rain_18-21082023.dat",
        staging=_KEEP_2023_BLOCK,
        note="only the 804-row 2023-08 block; 892 scattered pre-2016 rows are a spare logger",
    ),
    ArchiveFile(f"{_D}/LBM_rain_2023.dat", note="2023"),
    ArchiveFile(f"{_D}/LBM_rain2023_14032024.dat", note="2024-03 handover"),
    ArchiveFile(f"{_D}/LBM_rain_2024.dat", note="2024"),
    ArchiveFile("LBM_rain_2025.dat", note="2025 to 2026-04-24"),
)


@dataclass(frozen=True)
class ArchiveReport:
    """What a merged frame actually contains, against what the audit measured."""

    kind: str
    rows: int
    expected_rows: int
    columns: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    duplicated: int
    monotonic: bool
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


# ---------------------------------------------------------------------------
# Staging — repairs that cannot live in configuration
# ---------------------------------------------------------------------------

# A TOA5 file is four header lines then data. Staged copies are rewritten in that
# shape so every consumer can keep using the same reader and the same skiprows.
_TOA5_METADATA = '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","STAGED","0","LBM_staged"'


def _write_toa5(frame: pd.DataFrame, destination: Path, timestamp_column: str) -> None:
    """Write a frame back out in TOA5 shape (metadata, names, units, aggregation)."""
    columns = list(frame.columns)
    with open(destination, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"{_TOA5_METADATA}\n")
        handle.write(",".join(f'"{name}"' for name in columns) + "\n")
        # Units and aggregation rows are placeholders: read_campbell_dat skips
        # both, and no consumer in this package reads them back.
        handle.write(",".join('""' for _ in columns) + "\n")
        handle.write(",".join('""' for _ in columns) + "\n")
    frame.to_csv(destination, mode="a", header=False, index=False, date_format="%Y-%m-%d %H:%M:%S")
    logger.info("staged %s (%d rows)", destination.name, len(frame))
    del timestamp_column


def _read_raw_toa5(path: Path) -> pd.DataFrame:
    """Read a TOA5 table keeping every value as written, timestamps as strings."""
    return pd.read_csv(path, skiprows=[0, 2, 3], low_memory=False, dtype=str)


def _stage_clock_shift(source: Path, destination: Path) -> None:
    """Add one hour to the mis-stamped rows of the headerless 2020_03 table.

    Two defects in one file. It is a plain CSV with a bare column-name line and
    no TOA5 header, so the standard ``skiprows=[0, 2, 3]`` reader would consume
    the names and the first two data rows; and every row up to 2020-02-28 11:50
    is stamped one hour early, which a RECORD-join against the rain table pins
    exactly (the offset is +1 h at RECORD 7901/11932/16539 and 0 by 20294).
    """
    frame = pd.read_csv(source, low_memory=False, dtype=str)
    stamps = pd.to_datetime(frame["TIMESTAMP"], format="ISO8601")
    shifted = stamps.where(stamps > _CLOCK_SLIP_LAST, stamps + pd.Timedelta(hours=1))
    moved = int((shifted != stamps).sum())
    frame["TIMESTAMP"] = shifted.dt.strftime("%Y-%m-%d %H:%M:%S")
    _write_toa5(frame, destination, "TIMESTAMP")
    logger.info("  clock: shifted %d rows by +1h", moved)


def _stage_drop_late_tail(source: Path, destination: Path) -> None:
    """Drop the mis-stamped 110-row tail the clock-corrected table already covers."""
    frame = _read_raw_toa5(source)
    stamps = pd.to_datetime(frame["TIMESTAMP"], format="ISO8601")
    keep = stamps < _LATE_TAIL_FIRST
    _write_toa5(frame.loc[keep], destination, "TIMESTAMP")
    logger.info("  tail: dropped %d late rows", int((~keep).sum()))


def _stage_keep_2023_block(source: Path, destination: Path) -> None:
    """Keep only the August 2023 block of the spare-logger rain table.

    The rest of the file is 892 rows scattered across 2014-2019 with RECORD
    resets, written by a different logger (serial 2727) whose siting cannot be
    verified. They are dropped rather than merged into a published record.
    """
    frame = _read_raw_toa5(source)
    stamps = pd.to_datetime(frame["TIMESTAMP"], format="ISO8601")
    keep = (stamps >= pd.Timestamp("2023-08-01")) & (stamps < pd.Timestamp("2023-09-01"))
    _write_toa5(frame.loc[keep], destination, "TIMESTAMP")
    logger.info(
        "  spare logger: kept %d rows of 2023-08, dropped %d", int(keep.sum()), int((~keep).sum())
    )


_STAGERS = {
    _CLOCK_PLUS_ONE_HOUR: _stage_clock_shift,
    _DROP_LATE_TAIL: _stage_drop_late_tail,
    _KEEP_2023_BLOCK: _stage_keep_2023_block,
}


def stage_archive(
    manifest: tuple[ArchiveFile, ...],
    data_dir: str | Path,
    staging_dir: str | Path,
) -> list[Path]:
    """Resolve a manifest to readable paths, writing repaired copies as needed.

    Parameters
    ----------
    manifest:
        :data:`LENTA_MANIFEST` or :data:`RAIN_MANIFEST`.
    data_dir:
        Root of the archive. **Never written to.**
    staging_dir:
        Scratch directory for the repaired copies. Recreated on every run so a
        stale staged file can never survive a change to the repair logic.

    Returns
    -------
    list[pathlib.Path]
        Paths in ingest order, ready for
        :func:`~micrometeorology.sensors.ingestion.merge_dat_files`.

    Raises
    ------
    FileNotFoundError
        If a manifest entry is missing. A silently shorter record is the failure
        this whole module exists to prevent, so an absent file is fatal rather
        than skipped.
    """
    root = Path(data_dir)
    staged_root = ensure_dir(Path(staging_dir))
    resolved: list[Path] = []

    for entry in manifest:
        source = root / entry.path
        if not source.is_file():
            raise FileNotFoundError(
                f"archive manifest entry missing: {source}\n  ({entry.note})\n"
                "  Every entry is either unique coverage or a documented repair; "
                "dropping one shortens the published record."
            )
        if entry.staging is None:
            resolved.append(source)
            continue
        destination = staged_root / f"{source.name}.staged.dat"
        _STAGERS[entry.staging](source, destination)
        resolved.append(destination)

    return resolved


# ---------------------------------------------------------------------------
# Merge and verification
# ---------------------------------------------------------------------------


def build_five_minute_frame(
    manifest: tuple[ArchiveFile, ...],
    data_dir: str | Path,
    staging_dir: str | Path,
    *,
    sentinel_value: float | None = None,
) -> pd.DataFrame:
    """Merge one manifest into a single 5-minute frame, raw values preserved.

    ``sentinel_value`` defaults to ``None`` here, unlike the reader's own -900:
    that threshold matches nothing in this archive, and leaving it on would only
    suggest that missing data had been handled. Sentinel masking is a separate,
    era-scoped step applied after the merge.
    """
    paths = stage_archive(manifest, data_dir, staging_dir)
    return merge_dat_files(
        paths,
        sentinel_value=sentinel_value,
        text_columns=list(STATUS_COLUMNS),
    )


def verify_frame(frame: pd.DataFrame, kind: str) -> ArchiveReport:
    """Check a merged frame against the row count, span and shape the audit measured.

    This is the guard that turns "the merge still runs" into "the merge still
    captures the whole archive". A file quietly removed from a manifest, a
    staging repair that stops matching its file, or a reader change that eats a
    header row all show up here as a row-count or span mismatch rather than as a
    slightly shorter published distribution.

    Parameters
    ----------
    frame:
        The merged 5-minute frame.
    kind:
        ``"lenta"`` or ``"rain"``, selecting the expected row count.

    Returns
    -------
    ArchiveReport
        ``problems`` is empty when everything matches.
    """
    expected = EXPECTED_LENTA_ROWS if kind == "lenta" else EXPECTED_RAIN_ROWS
    index = frame.index
    first = pd.Timestamp(index.min()) if len(index) else None
    last = pd.Timestamp(index.max()) if len(index) else None
    duplicated = int(index.duplicated().sum())
    monotonic = bool(index.is_monotonic_increasing)

    problems: list[str] = []
    if len(frame) != expected:
        problems.append(
            f"{kind}: {len(frame)} rows, audit measured {expected} ({len(frame) - expected:+d})"
        )
    if first is not None and first != ARCHIVE_START:
        problems.append(f"{kind}: starts {first}, audit measured {ARCHIVE_START}")
    if last is not None and last != ARCHIVE_END:
        problems.append(f"{kind}: ends {last}, audit measured {ARCHIVE_END}")
    if duplicated:
        problems.append(f"{kind}: {duplicated} duplicated timestamps")
    if not monotonic:
        problems.append(f"{kind}: index is not monotonically increasing")

    return ArchiveReport(
        kind=kind,
        rows=len(frame),
        expected_rows=expected,
        columns=len(frame.columns),
        first=first,
        last=last,
        duplicated=duplicated,
        monotonic=monotonic,
        problems=tuple(problems),
    )


# ---------------------------------------------------------------------------
# Sentinel masking — the values a logger writes instead of "missing"
# ---------------------------------------------------------------------------
#
# read_campbell_dat's -900 threshold catches NONE of these. Each entry below was
# found the same way: take the exact-value histogram of a column and look for a
# single value repeating thousands of times. A physical sensor does not report
# -46.8 degC ten thousand times in Salvador.
#
# The split matters. A VALUE rule holds for the whole record, because the value
# is physically impossible. A WINDOW rule is date-scoped because the value is
# legitimate elsewhere: zero is a real wind speed and a real rainfall, so a
# global "mask 0" would delete every calm hour and every dry hour in the record.

# column -> the impossible values it writes when the sensor is absent or faulted
SENTINEL_VALUES: dict[str, tuple[float, ...]] = {
    "PIR1_Wm2_Avg": (-7999.0,),
    "PSP1_Wm2_Avg": (-6673.0,),
    "NRLite_Wm2_Avg": (4268.0, 4320.0, 4367.0, 4420.0),
    "WS_ms_S_WVT": (7999.0, -1000.0),
    "Hamount_WXT_Tot": (7999.0,),
    "Rain_WXT_Tot": (2052.0,),
    "Temp1_Avg": (-100.0,),
    "RH1_Avg": (-100.0,),
    # -46.02 and 989.0 are the near-rail drift values the sensor passes through
    # on its way to the exact rails; they are not temperatures either.
    "AirT_C_Avg": (1000.0, 989.0, -46.8, -46.02),
    "AirT1_C_Avg": (1000.0, 989.0, -46.8, -46.02),
    "AirT2_C_Avg": (1000.0, 989.0, -46.8, -46.02),
    "DP_C_Avg": (1000.0, -273.1),
    "DP1_C_Avg": (1000.0, -273.1),
    "DP2_C_Avg": (1000.0, -273.1),
    "RH": (999.0,),
    "RH1": (999.0,),
    "RH2": (999.0,),
}

# The Eppley case/dome thermistors report kelvin; anything outside this is the
# channel being unwired rather than a temperature.
SENTINEL_RANGES: dict[str, tuple[float, float]] = {
    "T_C1_Avg": (250.0, 330.0),
    "T_D1_Avg": (250.0, 330.0),
}

# (column, value, first, last) — a value that is legitimate outside the window.
SENTINEL_WINDOWS: tuple[tuple[str, float, str, str], ...] = (
    # HMP humidity wrote a fake 0.0 (not NAN) while disconnected.
    ("RH1_Avg", 0.0, "2018-08-27 11:20", "2018-10-16 13:40"),
    # WXT block commissioning: everything reads 0 for two days.
    ("WS_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    ("RH_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    ("Pmb_WXT", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    # WXT dead but still logging zeros for sixteen months.
    ("WS_WXT_Avg", 0.0, "2023-03-10 00:00", "2024-07-19 23:55"),
    ("RH_WXT_Avg", 0.0, "2023-03-10 00:00", "2024-07-19 23:55"),
    ("Pmb_WXT", 0.0, "2023-03-10 00:00", "2024-07-19 23:55"),
    # v22 lost the CMP21 millivolt-to-flux multiplier; the column is a constant 0
    # while the raw mV channel still varies. Diffuse moved to the PSP here.
    ("CMP21_Wm2_Avg", 0.0, "2025-05-14 15:25", "2026-12-31 23:55"),
    # GMX unit-1 humidity rails to 0 after the open-circuit failure.
    ("RH1", 0.0, "2025-12-19 00:00", "2026-12-31 23:55"),
    # The 2019-03 WXT commissioning zeros reach two more columns than the first
    # pass caught. Verified leak: a raw 0.0 on Pmb_WXT_Avg at 2019-03-18 14:25
    # survived masking and fed straight into the unified pressure series.
    ("Pmb_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    ("Temp_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    # MetSENS unit 2 was decommissioned on 2025-05-14 and its channels park on
    # two constants rather than going null.
    ("WS2_ms_GMX", 0.08, "2025-05-14 00:00", "2026-12-31 23:55"),
    ("AirT2_C_Avg", 265.0, "2025-05-14 00:00", "2026-12-31 23:55"),
)

# Periods where an instrument was physically present and reporting, but not
# measuring what its column name claims. Masked wholesale.
#
# The diffuse windows are the highest-stakes entries in this module: an
# unshaded pyranometer reads the GLOBAL flux, so leaving them in publishes
# values up to 1368 W/m2 as "diffuse". They were identified by binning the
# ratio to global BY GLOBAL LEVEL: a shaded diffuse sensor's ratio falls as the
# sky clears (0.48 -> 0.13), an unshaded one stays flat or rises (0.81 -> 0.88).
#
# The list below was re-derived over the whole record with that same criterion
# after the first three entries turned out to cover only part of it: nine
# further multi-day episodes were publishing global irradiance as diffuse, at
# times above the global reading of the same hour (2024-09-16 11:00 published
# Sw_dif 1009.8 against Sw_dw 997.5). :func:`unshaded_diffuse_days` runs the
# criterion at build time so the next episode surfaces as a report line rather
# than waiting for someone to re-audit the ratio by hand.
INVALID_WINDOWS: tuple[tuple[str, str, str, str], ...] = (
    ("CMP21_Wm2_Avg", "2019-09-01 00:00", "2019-10-07 23:55", "PSP/CMP21 unshaded"),
    ("CMP21_Wm2_Avg", "2020-03-06 00:00", "2020-05-31 23:55", "shade ring off for ~87 days"),
    ("CMP21_Wm2_Avg", "2020-08-17 00:00", "2020-08-20 23:55", "ring off: 0.87 -> 0.96 by level"),
    ("CMP21_Wm2_Avg", "2020-09-04 00:00", "2020-09-09 23:55", "ring off: 0.83 -> 0.86 by level"),
    ("CMP21_Wm2_Avg", "2020-09-13 00:00", "2020-09-13 23:55", "ring off again for one day"),
    ("CMP21_Wm2_Avg", "2021-05-31 00:00", "2021-06-08 23:55", "ring off: 0.87 -> 0.97 by level"),
    ("CMP21_Wm2_Avg", "2021-08-05 00:00", "2021-08-09 23:55", "ring off: 0.87 -> 0.96 by level"),
    ("CMP21_Wm2_Avg", "2022-02-24 00:00", "2022-03-01 23:55", "ring off: 0.83 -> 0.98 by level"),
    ("CMP21_Wm2_Avg", "2023-08-07 00:00", "2023-08-13 23:55", "ring off: 0.85 -> 0.94 by level"),
    ("CMP21_Wm2_Avg", "2023-11-06 00:00", "2023-11-08 23:55", "ring off: 0.83 -> 0.77 by level"),
    ("CMP21_Wm2_Avg", "2024-09-12 00:00", "2024-09-17 23:55", "ring off: 0.46 -> 1.01 by level"),
    ("CMP21_Wm2_Avg", "2025-03-09 00:00", "2025-05-14 15:20", "reads 1.2-2.3x global"),
    ("PSP_Wm2_Avg", "2019-09-01 00:00", "2019-10-07 23:55", "unshaded"),
    ("PSP_Wm2_Avg", "2025-03-12 00:00", "2025-05-14 15:20", "unshaded before the handover"),
    # Tipping bucket: 54 consecutive dry days at full instrumentation, inside the
    # wettest months of the year, is a blocked funnel rather than a drought.
    ("PL01_mm_Tot", "2024-02-13 00:00", "2024-04-30 23:55", "gauge suspected blocked"),
)


# Detection constants for the shade-ring check below. Restricted to a clear-sky
# global flux, a properly shaded diffuse sensor reads 0.12-0.22 of the global
# one; every ring-off episode in the record reads 0.83-1.01 at that same level.
#
# The candidate screen sits at 0.55, between the two, but a single day above it
# is not evidence: bright broken cloud raises the diffuse fraction on its own,
# and 46 such days survive across the record with no hardware fault. What
# separates hardware from weather is PERSISTENCE — a ring that falls off stays
# off for days — so a candidate only counts as an episode when it runs for
# three days or reaches a ratio a shaded sensor cannot physically produce.
DIFFUSE_GLOBAL_COLUMN = "CM3Up_Wm2_Avg"
DIFFUSE_CLEAR_SKY_FLOOR = 600.0
DIFFUSE_MIN_SAMPLES_PER_DAY = 20
DIFFUSE_RATIO_LIMIT = 0.55
DIFFUSE_RATIO_CERTAIN = 0.85
DIFFUSE_MIN_EPISODE_DAYS = 3


def unshaded_diffuse_days(
    frame: pd.DataFrame, column: str = "CMP21_Wm2_Avg"
) -> list[tuple[str, float]]:
    """Days where the diffuse channel is still reading the global flux.

    Run this on the frame **after** :func:`mask_sentinels`: an episode already
    covered by :data:`INVALID_WINDOWS` is ``NaN`` by then and drops out on its
    own, so whatever comes back is exactly what the hand-curated list misses.

    A hand-written window table goes stale the moment the ring comes off again,
    and the failure is silent — the column keeps its name and publishes global
    irradiance as diffuse, at times above the global reading of the same hour.
    This turns the next episode into a line in the build report.

    Returns
    -------
    list
        ``(iso date, median clear-sky ratio)`` per offending day, oldest first.
        Empty for the archive as shipped: every episode it detects is masked.
    """
    if column not in frame.columns or DIFFUSE_GLOBAL_COLUMN not in frame.columns:
        return []
    paired = frame[[column, DIFFUSE_GLOBAL_COLUMN]].dropna()
    clear = paired[paired[DIFFUSE_GLOBAL_COLUMN] > DIFFUSE_CLEAR_SKY_FLOOR]
    if clear.empty:
        return []
    ratio = clear[column] / clear[DIFFUSE_GLOBAL_COLUMN]
    daily = ratio.groupby(pd.DatetimeIndex(clear.index).date).agg(["median", "count"])
    candidates = daily[
        (daily["count"] >= DIFFUSE_MIN_SAMPLES_PER_DAY) & (daily["median"] > DIFFUSE_RATIO_LIMIT)
    ]
    if candidates.empty:
        return []

    medians: dict[date, float] = {
        day: float(value) for day, value in zip(candidates.index, candidates["median"], strict=True)
    }
    days = sorted(medians)
    runs: list[list[date]] = [[days[0]]]
    for previous, current in pairwise(days):
        # One clouded-out day inside an episode must not split it; two must.
        if (current - previous).days <= 2:
            runs[-1].append(current)
        else:
            runs.append([current])

    return [
        (str(day), medians[day])
        for run in runs
        for day in run
        if len(run) >= DIFFUSE_MIN_EPISODE_DAYS or medians[day] > DIFFUSE_RATIO_CERTAIN
    ]


# Station coordinates, for the solar geometry the check below needs. Repeated
# here rather than imported from the climatology exporter because a sensors
# module must not depend on a CLI; they are the same numbers and both are the
# station's own.
STATION_SITE = SiteConfig(latitude=-13.0055, longitude=-38.5089)
STATION_UTC_OFFSET_HOURS = -3.0

# Detection constants for the timestamp-corruption check below, measured in
# docs/arqueologia/qc/med-fault-detection.md over the whole record: days with at
# least three DEEP-NIGHT samples of global irradiance above 50 W/m2 number 42
# (1.22% of the record), and the worst of them carries 128 — over ten hours of
# data written on the wrong side of midnight. Deep night is a zenith angle above
# 100 deg, i.e. an elevation below -10: astronomical twilight is long past, so no
# sky state whatsoever puts 50 W/m2 on a pyranometer there.
#
# Whole days, not samples: the audit measured that the DAYTIME half of the same
# day carries the identical shift while wearing ordinary values, so a per-sample
# rule can only ever see half of each episode.
#
# The test is run over EVERY shortwave channel, not just the global one. Keying
# it on ``Sw_dw`` alone reproduces the audit's 42 days but misses ten more that
# only the other pyranometers witness — 2018-08-21..23 and 2018-10-21..23 among
# them, contiguous blocks carrying up to 118 deep-night PAR samples each. The
# channels do not share an outage, so any one of them can be the only survivor
# of a shifted day. Longwave is deliberately absent: a pyrgeometer reads
# 300-400 W/m2 all night by design, so the same threshold there would flag the
# entire record.
NIGHT_CORRUPTION_COLUMNS = ("Sw_dw", "Sw_dif", "Sw_par", "Sw_up")
NIGHT_CORRUPTION_ELEVATION_DEG = -10.0
NIGHT_CORRUPTION_FLUX_WM2 = 50.0
NIGHT_CORRUPTION_MIN_SAMPLES = 3

# Channels the mask covers: every shortwave stream, whose meaning depends
# entirely on the hour that is wrong, plus ``Net_CNR1``. The net is here because
# it is NOT an independent measurement — over 729,225 samples its residual
# against ``Sw_dw - Sw_up + Lw_dw - Lw_up`` never exceeds 8.95 W/m2, so the
# logger computes it from the four components and masking only the shortwave
# ones would leave the net radiation on disk still carrying the corrupted
# contribution.
NIGHT_CORRUPTION_CHANNELS = (*NIGHT_CORRUPTION_COLUMNS, "Net_CNR1")

# BSRN "physically possible" ceiling for global horizontal irradiance
# (Long & Shi 2008): Sa * 1.5 * mu0**1.2 + 100. It is the sun's own geometry as
# the limit, which is what makes it catch what a flat gate cannot — the shipped
# [-20, 1500] rule fires on 6 samples of the whole record while this one finds
# 3,077, of which 2,477 carry full daylight irradiance with the sun below the
# horizon. Deliberately generous where the sun is high (2,150 W/m2 at zenith),
# so genuine cloud-edge enhancement survives; it only bites at low sun, which is
# exactly where a shifted clock puts midday values.
#
# Applied AFTER the whole-day mask above, which removes the gross episodes. What
# is left is the milder form of the same fault: an afternoon that declines
# smoothly and plausibly, an hour or two away from where it happened.
SOLAR_CONSTANT_WM2 = 1367.0
IMPOSSIBLE_SHORTWAVE_CHANNELS = ("Sw_dw", "Net_CNR1")


# ``{unified name: [(source column, inclusive start, inclusive end), ...]}``, as
# ``sensors.calibration.resolve_mapping_windows`` returns it.
SourceWindows = Mapping[str, Sequence[tuple[str, pd.Timestamp, pd.Timestamp]]]


def _mask_column(frame: pd.DataFrame, column: str, hit: NDArray, removed: dict[str, int]) -> None:
    """Blank *column* where *hit* selects a populated sample, tallying into *removed*."""
    if column not in frame.columns:
        return
    selected = hit & frame[column].notna().to_numpy()
    count = int(selected.sum())
    if not count:
        return
    frame.loc[selected, column] = float("nan")
    removed[column] = removed.get(column, 0) + count


def mask_impossible_shortwave(
    frame: pd.DataFrame, sources: SourceWindows | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank global irradiance that the sun's position cannot produce.

    Per SAMPLE, unlike :func:`mask_night_corrupted_days`, because this catches
    the residue rather than the episode. ``Net_CNR1`` follows the same sample for
    the same reason it follows the day: the logger derives it from the four
    components, so leaving it would keep the impossible contribution on disk.

    Pass *sources* (from :func:`~micrometeorology.sensors.calibration.resolve_mapping_windows`)
    so the raw column the unified channel was copied from is blanked on the same
    samples. It is scoped to that column's own era window, because inside it the
    two are the same measurement bit for bit, while outside it the raw column is
    a different instrument that never failed this check.

    Returns
    -------
    tuple
        The masked frame and a ``{column: samples removed}`` tally.
    """
    removed: dict[str, int] = {}
    if "Sw_dw" not in frame.columns:
        return frame, removed
    index = pd.DatetimeIndex(frame.index)
    mu0 = np.clip(cos_zenith(index, STATION_SITE, STATION_UTC_OFFSET_HOURS), 0.0, None)
    ceiling = SOLAR_CONSTANT_WM2 * 1.5 * mu0**1.2 + 100.0
    global_flux = frame["Sw_dw"]
    impossible = (global_flux.notna() & (global_flux > ceiling)).to_numpy()
    if not impossible.any():
        return frame, removed
    for column in IMPOSSIBLE_SHORTWAVE_CHANNELS:
        _mask_column(frame, column, impossible, removed)
        for source, start, end in (sources or {}).get(column, ()):
            within = (index >= start) & (index <= end)
            _mask_column(frame, source, impossible & within, removed)
    return frame, removed


def night_corrupted_days(
    frame: pd.DataFrame, columns: Sequence[str] = NIGHT_CORRUPTION_COLUMNS
) -> list[tuple[str, int]]:
    """Days whose timestamps are shifted, found by irradiance recorded at night.

    Run this on the UNIFIED frame: the corruption spans instrument eras, so the
    era-specific raw aliases each witness only part of it.

    A criterion rather than a hand-written window table, for the same reason
    :func:`unshaded_diffuse_days` is one: 52 dated windows go stale the next time
    the logger's clock slips, and silently, because the values look ordinary.

    Returns
    -------
    list
        ``(iso date, deep-night samples above the flux floor)``, oldest first.
        The count is over all channels, so it measures how much of the day is
        misplaced rather than how one instrument fared.
    """
    present = [column for column in columns if column in frame.columns]
    if not present:
        return []
    index = pd.DatetimeIndex(frame.index)
    deep_night = solar_elevation(index, STATION_SITE, STATION_UTC_OFFSET_HOURS) < (
        NIGHT_CORRUPTION_ELEVATION_DEG
    )
    offending = np.zeros(len(frame), dtype=bool)
    for column in present:
        values = frame[column]
        offending |= (
            values.notna().to_numpy() & deep_night & (values.to_numpy() > NIGHT_CORRUPTION_FLUX_WM2)
        )
    per_day = pd.Series(offending, index=index).groupby(index.date).sum()
    corrupted = per_day[per_day >= NIGHT_CORRUPTION_MIN_SAMPLES]
    return [(str(day), int(count)) for day, count in corrupted.items()]


def mask_night_corrupted_days(
    frame: pd.DataFrame, days: Sequence[tuple[str, int]], sources: SourceWindows | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank :data:`NIGHT_CORRUPTION_CHANNELS` over every day in ``days``.

    The whole day goes, not the samples the detector fired on: what is wrong is
    the clock, so the values are real measurements of another hour and the half
    of the day that still looks plausible is exactly as misplaced as the half
    that does not.

    Pass *sources* (from :func:`~micrometeorology.sensors.calibration.resolve_mapping_windows`)
    to blank the raw columns those channels were copied from as well. Unlike the
    per-sample BSRN mask, this one ignores the era windows and takes every
    source column for the whole day: a slipped clock is a fault of the LOGGER,
    so every solar-geometry-dependent channel it wrote that day is misplaced,
    including the ones that were not the unified source at the time.

    Returns
    -------
    tuple
        The masked frame and a ``{column: samples removed}`` tally, in the shape
        :func:`mask_sentinels` reports.
    """
    removed: dict[str, int] = {}
    if not days:
        return frame, removed
    corrupted = {pd.Timestamp(day).normalize() for day, _count in days}
    within = pd.DatetimeIndex(frame.index).normalize().isin(corrupted)
    for column in NIGHT_CORRUPTION_CHANNELS:
        _mask_column(frame, column, within, removed)
        for source, _start, _end in (sources or {}).get(column, ()):
            _mask_column(frame, source, within, removed)
    return frame, removed


NET_RADIATION_COMPONENTS = ("Sw_dw", "Sw_up", "Lw_dw", "Lw_up")


def close_net_radiation(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Rebuild ``Net_CNR1`` as the sum of the four published components.

    The CNR1 net is not an independent measurement: the logger computes it from
    the same four channels, and over 719,002 samples of the uncalibrated record
    the two agree to 8.95 W/m2. Calibrating one component and not the logger's
    precomputed sum broke that — the residual became a systematic +1.28 W/m2,
    reaching 9.28, so the monitoring chart that invites a reader to add the four
    bars and land on the net line no longer added up.

    Recomputing rather than correcting the sum is what makes the identity exact
    by construction instead of by a second arithmetic that can drift again. It
    also publishes 34,640 five-minute samples of 2018-10 to 2019-03, where the
    four components were recorded but the logger had not yet begun writing a
    net, and drops the 125 where a component is missing and no net is defined.

    Returns the frame, the samples gained and the samples dropped.
    """
    if not all(column in frame.columns for column in NET_RADIATION_COMPONENTS):
        return frame, 0, 0
    down, up, longwave_down, longwave_up = (frame[c] for c in NET_RADIATION_COMPONENTS)
    closed = down - up + longwave_down - longwave_up
    previous = frame["Net_CNR1"] if "Net_CNR1" in frame.columns else pd.Series(np.nan, frame.index)
    gained = int((closed.notna() & previous.isna()).sum())
    dropped = int((closed.isna() & previous.notna()).sum())
    frame["Net_CNR1"] = closed
    return frame, gained, dropped


def mask_sentinels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace every documented sentinel and invalid window with ``NaN``.

    Masking is PER CHANNEL, never per row: when the Gill thermohygrometer railed
    in December 2025 the pressure and wind channels on the same logger stayed
    perfectly good, and dropping whole rows would have thrown them away too.

    Returns
    -------
    tuple
        The masked frame and a ``{column: samples removed}`` tally, which the
        CLI prints so a rule that silently stops matching is visible.
    """
    removed: dict[str, int] = {}

    def _drop(column: str, mask: pd.Series) -> None:
        count = int(mask.sum())
        if count:
            frame.loc[mask, column] = float("nan")
            removed[column] = removed.get(column, 0) + count

    for column, values in SENTINEL_VALUES.items():
        if column in frame.columns:
            _drop(column, frame[column].isin(values))

    for column, (low, high) in SENTINEL_RANGES.items():
        if column in frame.columns:
            series = frame[column]
            _drop(column, series.notna() & ((series < low) | (series > high)))

    for column, value, first, last in SENTINEL_WINDOWS:
        if column not in frame.columns:
            continue
        window = (frame.index >= pd.Timestamp(first)) & (frame.index <= pd.Timestamp(last))
        _drop(column, pd.Series(window, index=frame.index) & (frame[column] == value))

    for column, first, last, _reason in INVALID_WINDOWS:
        if column not in frame.columns:
            continue
        window = (frame.index >= pd.Timestamp(first)) & (frame.index <= pd.Timestamp(last))
        _drop(column, pd.Series(window, index=frame.index) & frame[column].notna())

    return frame, removed
