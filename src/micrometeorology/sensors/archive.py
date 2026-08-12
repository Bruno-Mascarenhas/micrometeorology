"""The LabMiM station archive: an explicit manifest, staged fixes, one merged frame.

Globbing ``data/dados-labmim/`` silently produces a wrong record four ways, per
an audit of every table in the archive (2016-09 to 2026-04):

1. **``*.dat`` drops the rotation files.** Three ``.backup`` tables are the ONLY
   source of an austral winter each — JJA 2020, JJA 2022, June to mid-July 2024.
2. **The directory holds more than one station.** ``BTS_*`` is a different site
   (CR1000X serial 9429), the ``celsolar`` / ``calibracao`` tables are
   side-by-side instrument campaigns, and the ``solar`` / ``radiacao`` families
   sample at one minute.
3. **Names lie.** ``dados-labmim/LBM_lenta.dat`` is the RAIN table — TOA5 header
   field 8 reads ``LBM_rain`` — and it is the unique source of February 2019.
4. **Three clock defects cannot be expressed in configuration.** They need the
   bytes fixed before the merge, which :func:`stage_archive` does into a scratch
   directory: nothing here ever writes to ``data/``.

So the manifest lives here as data, in ingest order — chronological by first
timestamp — each file's disposition next to it, and :func:`verify_frame` checks
the merged result against the row counts, span and monotonicity the audit
measured.

Timestamps are naive station-local throughout, stamped by the logger's own
clock; two of the repairs below exist precisely because that clock has slipped.
Solar geometry therefore needs an explicit offset, pinned here as
:data:`STATION_UTC_OFFSET_HOURS` rather than read from the host time zone.

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

# The same solar geometry the climatology exporter uses, so "deep night" means
# the same angle in both places.
from allsky.config import SiteConfig
from allsky.solar import cos_zenith, solar_elevation_deg
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

# Staging directives, dispatched through _STAGERS in stage_archive.
_CLOCK_PLUS_ONE_HOUR = "clock+1h"
_DROP_LATE_TAIL = "drop-late-tail"
_KEEP_2023_BLOCK = "keep-2023-block"

# The 2020 clock slip: rows stamped at or before this instant are one hour early,
# per a RECORD-join of the lenta and rain tables across the window.
_CLOCK_SLIP_LAST = pd.Timestamp("2020-02-28 11:50:00")
# From here on the 2019 tables carry a mis-stamped tail that the clock-corrected
# 2020_03 table already holds, cell for cell.
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
        What would be lost without this file. Read it before removing an entry.
    """

    path: str
    staging: str | None = None
    note: str = ""


_DIR = "dados-labmim"

LENTA_MANIFEST: tuple[ArchiveFile, ...] = (
    ArchiveFile(f"{_DIR}/LBM_lenta_2016.dat", note="start of record, 2016-09-29"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2017.dat", note="all of 2017, complete JJA"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2018_1.dat", note="2018-01..2018-10-16, JJA 2018"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2018-2019.dat", note="CNR1 commissioning era"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019.dat.backup", note="sole source of 2019-03-15 afternoon"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019.dat.1.backup", note="sole source of 2019-03-15..18"),
    ArchiveFile(
        f"{_DIR}/LBM_lenta_2019.dat.2.backup", note="sole source of 2019-03-18..19, WXT arrives"
    ),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019.dat.3.backup", note="sole source of 2019-03-19..05-31"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019_0531.dat", note="2019-05-31 onward"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019_0631.dat", note="2019-06 onward"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2019_1011.dat", note="2019-10 onward, CMP21 diffuse begins"),
    ArchiveFile(
        f"{_DIR}/LBM_lenta_2019.dat",
        staging=_DROP_LATE_TAIL,
        note="110-row tail is mis-stamped; the clock-fixed 2020_03 table carries it correctly",
    ),
    ArchiveFile(
        f"{_DIR}/LBM_lenta_2020_03.dat",
        staging=_CLOCK_PLUS_ONE_HOUR,
        note="headerless CSV, and 16855 rows are one hour early",
    ),
    ArchiveFile(f"{_DIR}/LBM_lenta_2020.dat.backup", note="SOLE SOURCE OF JJA 2020"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2020.dat", note="rest of 2020"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2021.dat", note="all of 2021"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2022.dat.backup", note="SOLE SOURCE OF JJA 2022"),
    ArchiveFile(
        f"{_DIR}/LBM_lenta_2022.dat", note="rest of 2022 (superset of data/LBM_lenta_2022.dat)"
    ),
    ArchiveFile(f"{_DIR}/CR5000_LBM_lenta_18-21082023.dat", note="2023-08 spare-logger block"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2023.dat", note="2023"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2023_14032024.dat", note="2024-03 handover"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2024.dat.backup", note="SOLE SOURCE OF JUNE AND 1-19 JULY 2024"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2024.dat", note="rest of 2024"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2025.dat.backup", note="2025-03 Gill MetSENS commissioning"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2025.dat.1.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2025.dat.2.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2025.dat.3.backup", note="2025-03 commissioning"),
    ArchiveFile(f"{_DIR}/LBM_lenta_2025.dat.4.backup", note="2025-03-28..05-14, dual GMX units"),
    ArchiveFile("LBM_lenta_2025.dat", note="v22 era to 2026-04-24; PSP takes over diffuse"),
)

RAIN_MANIFEST: tuple[ArchiveFile, ...] = (
    ArchiveFile(f"{_DIR}/LBM_rain_2016.dat", note="start of rain record"),
    ArchiveFile(f"{_DIR}/LBM_rain_2017.dat", note="2017"),
    ArchiveFile(f"{_DIR}/LBM_rain_2018_2019.dat", note="2018 into 2019"),
    ArchiveFile(
        f"{_DIR}/LBM_lenta.dat",
        note="MISNAMED: TOA5 field 8 is LBM_rain. Unique source of 2019-01-31..02-26",
    ),
    ArchiveFile(
        f"{_DIR}/LBM_rain_2019.dat", staging=_DROP_LATE_TAIL, note="same 110-row mis-stamped tail"
    ),
    ArchiveFile(
        f"{_DIR}/LBM_rain_2020.dat", note="2020 (clock slip is in the lenta table, not here)"
    ),
    ArchiveFile(f"{_DIR}/LBM_rain_2021.dat", note="2021"),
    ArchiveFile(f"{_DIR}/LBM_rain_2022.dat", note="2022 (superset of data/LBM_rain_2022.dat)"),
    ArchiveFile(
        f"{_DIR}/CR5000_LBM_rain_18-21082023.dat",
        staging=_KEEP_2023_BLOCK,
        note="only the 804-row 2023-08 block; 892 scattered pre-2016 rows are a spare logger",
    ),
    ArchiveFile(f"{_DIR}/LBM_rain_2023.dat", note="2023"),
    ArchiveFile(f"{_DIR}/LBM_rain2023_14032024.dat", note="2024-03 handover"),
    ArchiveFile(f"{_DIR}/LBM_rain_2024.dat", note="2024"),
    ArchiveFile("LBM_rain_2025.dat", note="2025 to 2026-04-24"),
)


@dataclass(frozen=True)
class ArchiveReport:
    """What a merged frame actually contains, against what the audit measured.

    Attributes
    ----------
    kind:
        ``"lenta"`` or ``"rain"``, the manifest the frame was built from.
    rows, expected_rows:
        Rows merged, and the count the audit measured for that manifest.
    columns:
        Width of the merged frame, which grows over the record as sensors are
        added, so it is reported rather than checked.
    first, last:
        Index bounds as naive station-local timestamps, ``None`` for an empty
        frame.
    duplicated:
        Timestamps appearing more than once. Should be zero: the merge collapses
        overlapping stamps per column.
    monotonic:
        Whether the index increases throughout.
    problems:
        One sentence per mismatch, empty when the frame matches the audit.
    """

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
        """Whether the merge reproduced the audited archive exactly."""
        return not self.problems


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

    Two defects in one file: it is a plain CSV with a bare column-name line, so
    the standard ``skiprows=[0, 2, 3]`` reader would eat the names and the first
    two data rows; and every row up to 2020-02-28 11:50 is one hour early (a
    RECORD-join against rain gives +1 h at RECORD 7901/11932/16539, 0 by 20294).
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

    The rest is 892 rows scattered across 2014-2019 with RECORD resets, written
    by a different logger (serial 2727) whose siting cannot be verified.
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
        Scratch for the repaired copies. Every one of them is rewritten from its
        source on each call, never read back from a previous run, so a stale
        staged file cannot survive a change to the repair logic.

    Returns
    -------
    list[pathlib.Path]
        Paths in ingest order, ready for
        :func:`~micrometeorology.sensors.ingestion.merge_dat_files`.

    Raises
    ------
    FileNotFoundError
        If a manifest entry is missing. Every entry is unique coverage or a
        documented repair, so an absent file is fatal rather than skipped.
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


def build_five_minute_frame(
    manifest: tuple[ArchiveFile, ...],
    data_dir: str | Path,
    staging_dir: str | Path,
    *,
    sentinel_value: float | None = None,
) -> pd.DataFrame:
    """Merge one manifest into a single 5-minute frame, raw values preserved.

    Parameters
    ----------
    manifest:
        :data:`LENTA_MANIFEST` or :data:`RAIN_MANIFEST`.
    data_dir:
        Root of the archive. **Never written to.**
    staging_dir:
        Scratch directory for the repaired copies.
    sentinel_value:
        Threshold passed through to the reader. It defaults to ``None``, unlike
        the reader's own -900, which matches nothing in this archive; sentinel
        masking is a separate, era-scoped step applied after the merge by
        :func:`mask_sentinels`.

    Returns
    -------
    pd.DataFrame
        Every manifest file merged, indexed by naive station-local timestamp at
        the logger's 5-minute cadence, values raw and uncalibrated, with the
        per-row status flags preserved as text.
    """
    paths = stage_archive(manifest, data_dir, staging_dir)
    return merge_dat_files(
        paths,
        sentinel_value=sentinel_value,
        text_columns=list(STATUS_COLUMNS),
    )


def verify_frame(frame: pd.DataFrame, kind: str) -> ArchiveReport:
    """Check a merged frame against the row count, span and shape the audit measured.

    A file dropped from a manifest, a staging repair that stops matching its
    file, or a reader change that eats a header row surfaces as a row-count or
    span mismatch instead of a slightly shorter distribution.

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


# The values a logger writes instead of "missing", per column.
# read_campbell_dat's -900 threshold catches NONE of these; each entry came from
# the exact-value histogram of a column, where a sentinel shows up as one value
# repeating thousands of times. A VALUE rule holds for the whole record because
# the value is physically impossible; a WINDOW rule is date-scoped because the
# value is legitimate elsewhere — zero is a real wind speed and a real rainfall.

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
    # -46.02 and 989.0 are near-rail drift on the way to the exact rails.
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
    # The same 2019-03 WXT commissioning zeros on two more columns: an unmasked
    # 0.0 on Pmb_WXT_Avg at 2019-03-18 14:25 feeds the unified pressure series.
    ("Pmb_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    ("Temp_WXT_Avg", 0.0, "2019-03-18 12:55", "2019-03-19 08:25"),
    # MetSENS unit 2 was decommissioned on 2025-05-14 and its channels park on
    # two constants rather than going null.
    ("WS2_ms_GMX", 0.08, "2025-05-14 00:00", "2026-12-31 23:55"),
    ("AirT2_C_Avg", 265.0, "2025-05-14 00:00", "2026-12-31 23:55"),
)

# Periods where an instrument was present and reporting, but not measuring what
# its column name claims. Masked wholesale.
#
# An unshaded pyranometer reads the GLOBAL flux, so leaving the diffuse windows
# in publishes up to 1368 W/m2 as "diffuse", at times above the same hour's
# global (2024-09-16 11:00: Sw_dif 1009.8 against Sw_dw 997.5). They come from
# binning the ratio to global BY GLOBAL LEVEL — a shaded sensor's ratio falls as
# the sky clears (0.48 -> 0.13), an unshaded one holds or rises (0.81 -> 0.88) —
# the same criterion :func:`unshaded_diffuse_days` runs at build time.
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


# Detection constants for the shade-ring check below. At a clear-sky global flux
# a properly shaded diffuse sensor reads 0.12-0.22 of the global one; every
# ring-off episode in the record reads 0.83-1.01 at that same level. The 0.55
# screen sits between them, but bright broken cloud clears it on 46 days with no
# hardware fault, so a candidate only counts as an episode with PERSISTENCE:
# three days, or a ratio a shaded sensor cannot physically produce.
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

    Run on the frame **after** :func:`mask_sentinels`: an episode already covered
    by :data:`INVALID_WINDOWS` is ``NaN`` by then, so what comes back is exactly
    what the hand-curated list misses — and that list goes stale, silently, the
    next time the ring comes off, since the column keeps its name.

    Parameters
    ----------
    frame:
        5-minute frame in W/m2, holding both *column* and
        :data:`DIFFUSE_GLOBAL_COLUMN`. Missing either, the check reports
        nothing rather than guessing.
    column:
        The diffuse channel to test. Defaults to the CMP21, which held the
        diffuse role until the 2025 handover to the PSP.

    Returns
    -------
    list
        ``(iso date, median clear-sky ratio)`` per offending day, oldest first.
        The ratio is dimensionless, diffuse over global, taken over the samples
        above :data:`DIFFUSE_CLEAR_SKY_FLOOR` only. Empty for the archive as
        shipped: every episode it detects is masked.
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


# Station coordinates for the solar geometry the checks below need, repeated
# rather than imported from the climatology exporter: a sensors module must not
# depend on a CLI. Both copies are the station's own numbers and must stay
# equal — see SITE in cli/export_climatology.py.
STATION_SITE = SiteConfig(latitude=-13.0055, longitude=-38.5089)
STATION_UTC_OFFSET_HOURS = -3.0

# Detection constants for the timestamp-corruption check below, measured in
# docs/arqueologia/qc/med-fault-detection.md: 42 days (1.22% of the record) carry
# at least three DEEP-NIGHT samples of global irradiance above 50 W/m2, the worst
# of them 128. Deep night is a zenith above 100 deg, i.e. elevation below -10 —
# astronomical twilight long past, so no sky state puts 50 W/m2 on a pyranometer.
#
# Flagged per DAY, not per sample: the daytime half of the same day carries the
# identical shift while wearing ordinary values. Over EVERY shortwave channel,
# because ``Sw_dw`` alone reproduces the 42 days but misses ten that only the
# other pyranometers witness (2018-08-21..23 and 2018-10-21..23, up to 118
# deep-night PAR samples each). Longwave is deliberately absent: a pyrgeometer
# reads 300-400 W/m2 all night by design, so the same threshold would flag the
# entire record.
NIGHT_CORRUPTION_COLUMNS = ("Sw_dw", "Sw_dif", "Sw_par", "Sw_up")
NIGHT_CORRUPTION_ELEVATION_DEG = -10.0
NIGHT_CORRUPTION_FLUX_WM2 = 50.0
NIGHT_CORRUPTION_MIN_SAMPLES = 3

# Every shortwave stream plus ``Net_CNR1``. The net is NOT an independent
# measurement — over 729,225 samples its residual against
# ``Sw_dw - Sw_up + Lw_dw - Lw_up`` never exceeds 8.95 W/m2 — so masking only the
# shortwave channels would leave the corrupted contribution inside the net.
NIGHT_CORRUPTION_CHANNELS = (*NIGHT_CORRUPTION_COLUMNS, "Net_CNR1")

# BSRN "physically possible" ceiling for global horizontal irradiance
# (Long & Shi 2008): Sa * 1.5 * mu0**1.2 + 100. Because the limit follows the
# sun's own geometry it catches what a flat gate cannot — the shipped [-20, 1500]
# rule fires on 6 samples of the record, this one on 3,077, of which 2,477 carry
# full daylight irradiance with the sun below the horizon. It stays generous at
# high sun (2,150 W/m2 at zenith) so genuine cloud-edge enhancement survives, and
# bites only at low sun, where a shifted clock puts midday values. Applied AFTER
# the whole-day mask above, so what reaches it is the milder residue of the same
# fault: an afternoon that declines plausibly an hour or two out of place.
SOLAR_CONSTANT_WM2 = 1367.0
IMPOSSIBLE_SHORTWAVE_CHANNELS = ("Sw_dw", "Net_CNR1")


# ``{unified name: [(source column, inclusive start, inclusive end), ...]}``, as
# ``sensors.calibration.resolve_mapping_windows`` returns it.
SourceWindows = Mapping[str, Sequence[tuple[str, pd.Timestamp, pd.Timestamp]]]


def _mask_column(frame: pd.DataFrame, column: str, mask: NDArray, removed: dict[str, int]) -> None:
    """Blank *column* where *mask* selects a populated sample, tallying into *removed*."""
    if column not in frame.columns:
        return
    selected = mask & frame[column].notna().to_numpy()
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
    the residue rather than the episode. ``Net_CNR1`` follows the same sample
    because the logger derives it from the four components.

    Pass *sources* (from :func:`~micrometeorology.sensors.calibration.resolve_mapping_windows`)
    to blank the raw column each unified channel was copied from on the same
    samples, scoped to that column's own era window: inside it the two are the
    same measurement bit for bit, outside it the raw column is a different
    instrument that never failed this check.

    Parameters
    ----------
    frame:
        5-minute frame in W/m2 with a naive station-local index, carrying at
        least ``Sw_dw``. Without it nothing is checked, since the BSRN ceiling
        is defined on the global horizontal flux.
    sources:
        Era windows per unified channel, or ``None`` to mask the unified
        columns alone.

    Returns
    -------
    tuple
        The same frame, mutated in place, and a ``{column: samples removed}``
        tally. An empty tally means every sample sits under the ceiling; on the
        full record it does not, which is the point of the step.
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

    Run on the UNIFIED frame: the corruption spans instrument eras, so the
    era-specific raw aliases each witness only part of it. A criterion rather
    than a table of the 52 dated windows it finds, which would go stale the next
    time the logger's clock slips, and silently: the values look ordinary.

    Parameters
    ----------
    frame:
        Frame in W/m2 with a naive station-local index, against which the sun's
        elevation is computed from :data:`STATION_SITE` and the pinned
        :data:`STATION_UTC_OFFSET_HOURS`.
    columns:
        Shortwave channels to witness the fault on. Longwave must stay out: a
        pyrgeometer reads 300-400 W/m2 all night by design.

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
    deep_night = solar_elevation_deg(index, STATION_SITE, STATION_UTC_OFFSET_HOURS) < (
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

    The whole day goes, not the samples the detector fired on: the clock is what
    is wrong, so the values are real measurements of another hour and the
    plausible-looking half of the day is as misplaced as the rest.

    Pass *sources* (from :func:`~micrometeorology.sensors.calibration.resolve_mapping_windows`)
    to blank the raw columns those channels were copied from as well. Unlike the
    per-sample BSRN mask, this one ignores the era windows: a slipped clock is a
    fault of the LOGGER, so every solar-geometry-dependent channel it wrote that
    day is misplaced, including the ones that were not the unified source then.

    Parameters
    ----------
    frame:
        Frame in W/m2 with a naive station-local index.
    days:
        ``(iso date, sample count)`` pairs as :func:`night_corrupted_days`
        returns them; only the date is read, the count being provenance for the
        reader. An empty sequence masks nothing.
    sources:
        Raw columns per unified channel, or ``None`` to mask the unified
        columns alone.

    Returns
    -------
    tuple
        The same frame, mutated in place, and a ``{column: samples removed}``
        tally, in the shape :func:`mask_sentinels` reports.
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
    the two agree to 8.95 W/m2. Calibrating the components while keeping the
    logger's precomputed sum turns that agreement into a systematic bias, and
    the monitoring chart invites the reader to add the four bars and land on the
    net line. Recomputing keeps the identity exact by construction, where
    correcting the sum would be a second arithmetic free to drift again.

    It also publishes 34,640 five-minute samples of 2018-10 to 2019-03, where the
    components were recorded before the logger began writing a net, and drops the
    125 where a component is missing and no net is defined.

    Parameters
    ----------
    frame:
        Frame in W/m2 holding :data:`NET_RADIATION_COMPONENTS`. Missing any of
        them, the frame is returned untouched: a net rebuilt from three terms
        would be wrong in a way nothing downstream could detect.

    Returns
    -------
    tuple
        The same frame, mutated in place, and the samples gained and dropped —
        gained where the components define a net the logger never wrote,
        dropped where a component is missing so no net is defined.
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
    good.

    Parameters
    ----------
    frame:
        Raw merged frame with a naive station-local index, in the logger's own
        units — the rules are written against those values, so this must run
        before any calibration factor.

    Returns
    -------
    tuple
        The same frame, mutated in place, and a ``{column: samples removed}``
        tally, which the CLI prints so a rule that silently stops matching is
        visible.
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
