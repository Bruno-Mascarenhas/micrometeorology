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
:data:`STATION_UTC_OFFSET_HOURS` rather than read from the host time zone.  The
stamps are also END-stamps — the row at ``t`` averages ``(t - 5 min, t]`` — so
every stage that asks the sun's position asks it at
:func:`averaging_centroid` of the index, never at the closing edge; see
:data:`AVERAGING_CENTROID_OFFSET` and docs/quality-control.md.

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
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# The same solar geometry the climatology exporter uses, so "deep night" means
# the same angle in both places.
from labmim_core.site import STATION_SITE, STATION_UTC_OFFSET_HOURS
from labmim_core.solar import cos_zenith, eccentricity_correction, solar_elevation_deg
from micrometeorology.common import instruments
from micrometeorology.common.paths import ensure_dir
from micrometeorology.sensors.ingestion import merge_dat_files
from micrometeorology.sensors.quality import SAMPLING_INTERVAL

logger = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_END",
    "ARCHIVE_START",
    "DIFFUSE_EXCEEDS_GLOBAL_TOLERANCE",
    "DIFFUSE_RATIO_LIMIT",
    "EXPECTED_LENTA_ROWS",
    "EXPECTED_RAIN_ROWS",
    "LENTA_MANIFEST",
    "NET_RADIATION_COMPONENTS",
    "NIGHT_CORRUPTION_CHANNELS",
    "NIGHT_CORRUPTION_FLUX_WM2",
    "NOCTURNAL_SHORTWAVE_CHANNELS",
    "OFFSET_DRIFT_ALARM_WM2",
    "RAIN_MANIFEST",
    "SAMPLING_INTERVAL",
    "STATUS_COLUMNS",
    "UNGATED_RADIATION_TWINS",
    "ArchiveFile",
    "ArchiveReport",
    "NocturnalOffset",
    "averaging_centroid",
    "blocked_gauge_runs",
    "build_five_minute_frame",
    "close_net_radiation",
    "close_nocturnal_net_radiation",
    "mask_impossible_shortwave",
    "mask_night_corrupted_days",
    "mask_nocturnal_shortwave",
    "mask_sentinels",
    "night_corrupted_days",
    "nocturnal_offset_statistics",
    "stage_archive",
    "station_elevation_deg",
    "unquantised_rain_samples",
    "unshaded_diffuse_days",
    "verify_frame",
    "verify_window",
]

# Measured over the manifests below. A merge that does not reproduce these has
# lost or gained a file; see verify_frame.
EXPECTED_LENTA_ROWS = 1_017_857
EXPECTED_RAIN_ROWS = 1_018_291
ARCHIVE_START = pd.Timestamp("2016-09-29 13:40:00")
ARCHIVE_END = pd.Timestamp("2026-08-12 00:00:00")

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
    ArchiveFile("LBM_lenta_2025.dat", note="v22 era to 2026-08-12; PSP takes over diffuse"),
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
    ArchiveFile("LBM_rain_2025.dat", note="2025 to 2026-08-12"),
)


#: The two manifests this module verifies. A bare ``str`` let any spelling other
#: than ``"lenta"`` pick the RAIN expectation in silence, so a typo compared the
#: slow table against the rain gauge's audited row count.
type ArchiveKind = Literal["lenta", "rain"]


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

    kind: ArchiveKind
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


def _write_toa5(frame: pd.DataFrame, destination: Path) -> None:
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
    _write_toa5(frame, destination)
    logger.info("  clock: shifted %d rows by +1h", moved)


def _stage_drop_late_tail(source: Path, destination: Path) -> None:
    """Drop the mis-stamped 110-row tail the clock-corrected table already covers."""
    frame = _read_raw_toa5(source)
    stamps = pd.to_datetime(frame["TIMESTAMP"], format="ISO8601")
    keep = stamps < _LATE_TAIL_FIRST
    _write_toa5(frame.loc[keep], destination)
    logger.info("  tail: dropped %d late rows", int((~keep).sum()))


def _stage_keep_2023_block(source: Path, destination: Path) -> None:
    """Keep only the August 2023 block of the spare-logger rain table.

    The rest is 892 rows scattered across 2014-2019 with RECORD resets, written
    by a different logger (serial 2727) whose siting cannot be verified.
    """
    frame = _read_raw_toa5(source)
    stamps = pd.to_datetime(frame["TIMESTAMP"], format="ISO8601")
    keep = (stamps >= pd.Timestamp("2023-08-01")) & (stamps < pd.Timestamp("2023-09-01"))
    _write_toa5(frame.loc[keep], destination)
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


def verify_window(frame: pd.DataFrame, kind: ArchiveKind) -> ArchiveReport:
    """Check a rolling window against the invariants that hold for any window.

    :func:`verify_frame` is anchored to the audited historical record — its row
    count, its first stamp, its last — so it cannot judge the operational
    ``--source`` window, which is a handful of days of whatever the logger is
    writing now.  That is why ``--source`` verified nothing at all and
    ``--strict`` could not fail on it.  What still holds regardless of the
    window is the frame's own shape: an index that never goes backwards, no
    stamp appearing twice, and at least one row.

    Parameters
    ----------
    frame:
        The merged 5-minute window.
    kind:
        ``"lenta"`` or ``"rain"``, named in every problem sentence.

    Returns
    -------
    ArchiveReport
        ``expected_rows`` is zero: no audited count applies to a window.
        ``problems`` is empty when the window is well formed.
    """
    index = frame.index
    first = pd.Timestamp(index.min()) if len(index) else None
    last = pd.Timestamp(index.max()) if len(index) else None
    duplicated = int(index.duplicated().sum())
    monotonic = bool(index.is_monotonic_increasing)

    problems: list[str] = []
    if len(frame) == 0:
        problems.append(f"{kind}: the window merged to no row at all")
    if duplicated:
        problems.append(f"{kind}: {duplicated} duplicated timestamps")
    if not monotonic:
        problems.append(f"{kind}: index is not monotonically increasing")

    return ArchiveReport(
        kind=kind,
        rows=len(frame),
        expected_rows=0,
        columns=len(frame.columns),
        first=first,
        last=last,
        duplicated=duplicated,
        monotonic=monotonic,
        problems=tuple(problems),
    )


def verify_frame(frame: pd.DataFrame, kind: ArchiveKind) -> ArchiveReport:
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
    expected = {"lenta": EXPECTED_LENTA_ROWS, "rain": EXPECTED_RAIN_ROWS}[kind]
    index = frame.index
    first = pd.Timestamp(index.min()) if len(index) else None
    last = pd.Timestamp(index.max()) if len(index) else None
    duplicated = int(index.duplicated().sum())
    monotonic = bool(index.is_monotonic_increasing)

    # The station keeps recording after the audit, so the invariant is growth and
    # not equality: the surplus must be the sampling grid between the two ends.
    problems: list[str] = []
    surplus = len(frame) - expected
    if surplus < 0:
        problems.append(
            f"{kind}: {len(frame)} rows, audit measured {expected} ({surplus:+d}); "
            "the record cannot shrink"
        )
    if first is not None and first != ARCHIVE_START:
        problems.append(f"{kind}: starts {first}, audit measured {ARCHIVE_START}")
    if last is not None and last < ARCHIVE_END:
        problems.append(f"{kind}: ends {last}, before the audited {ARCHIVE_END}")
    if last is not None and last > ARCHIVE_END and surplus >= 0:
        grid = int((last - ARCHIVE_END) / SAMPLING_INTERVAL)
        if surplus != grid:
            problems.append(
                f"{kind}: {surplus} rows past the audited end, but {ARCHIVE_END} to {last} "
                f"spans {grid} sampling intervals"
            )
    if duplicated:
        problems.append(f"{kind}: {duplicated} duplicated timestamps")
    if not monotonic:
        problems.append(f"{kind}: index is not monotonically increasing")
    # The surplus rule above already assumes the grid — it converts a span into a
    # row count by dividing by SAMPLING_INTERVAL — but nothing checked that the
    # rows sit ON it. A staging repair that shifts part of a file, or a merge
    # that lands rows off the 5-minute grid, keeps the count and the span intact
    # and only shows up here.
    stamps = pd.DatetimeIndex(index)
    # Through the offset from midnight, not through `asi8`: this index carries
    # microsecond resolution, so the integer view is in microseconds while
    # SAMPLING_INTERVAL.value is nanoseconds, and the comparison would call the
    # whole archive off-grid. Timedelta arithmetic carries its own unit.
    since_midnight = stamps - stamps.normalize()
    off_grid = int((since_midnight % SAMPLING_INTERVAL != pd.Timedelta(0)).sum())
    if off_grid:
        problems.append(f"{kind}: {off_grid} timestamp(s) are not on the {SAMPLING_INTERVAL} grid")

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

# Bounds a working instrument cannot leave; outside them the channel is not
# measuring. Wider than the QC gates in configs/micromet/default.yaml on purpose:
# those remove the implausible, this removes the impossible, and only this one
# runs ahead of the allsky feature build.
#
# The Eppley case/dome thermistors report kelvin, so anything outside 250-330 K
# is the channel being unwired rather than a temperature.
#
# The GMX barometer needs a gate because when MetSENS1 faults it parks
# BP1_mbar_Avg on the same fill value it writes to RH1 and WS1_ms_GMX — 2.62 hPa
# in 420 samples, 0.95 in 11, 157.07 once, 872.14 in two — and every one of them
# passes the -900 sentinel threshold and the finite filter untouched. A range
# rather than the observed values because that fill is a spilled reading, not a
# designed rail, so the next fault can park on a different number. The floor is
# measured, not assumed: across 139,039 readings the record holds 436 samples
# below 950 hPa and then nothing at all until 980, while real pressure at this
# sea-level site spans 1005 to 1022 (1st percentile to maximum, never above
# 1030). At 950 hPa the station would have to stand half a kilometre uphill.
#
# BP1 alone, and deliberately: it is the barometer the allsky feature policy
# reads, and mask_sentinels is the only gate standing between the logger and
# that feature build. BP2/BP_mbar_Avg/Pmb_WXT carry impossible values too (a
# tally of 1504/7/143192 samples), but they reach only the micrometeorology
# chain, which already gates them in configs/micromet/default.yaml. That config
# gates BP1 at the tighter [985, 1030]; the ~11 readings between 980 and 1005
# are implausible rather than impossible and stay on that side of the line.
SENTINEL_RANGES: dict[str, tuple[float, float]] = {
    "T_C1_Avg": (250.0, 330.0),
    "T_D1_Avg": (250.0, 330.0),
    "BP1_mbar_Avg": (950.0, 1100.0),
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

# ``(column, first, last)`` per era, mirroring the ``Sw_dif`` map in
# configs/micromet/calibrations.yaml window for window (pinned by
# test_the_diffuse_eras_mirror_the_calibration_map, because a merged window here
# would hide exactly the unshaded episode the YAML splits it around): the check
# must follow the diffuse ROLE,
# which has moved between three columns, and not one column, which mask_sentinels
# blanks end to end once its instrument retires. Scoped to the era because
# outside its own the same column is a different measurement — the PSP stands
# unshaded beside the CMP21 from 2019-10 to 2025-05, at 0.82-0.86 of global on
# 156 days of the record.
DIFFUSE_CHANNEL_ERAS: tuple[tuple[str, str | None, str | None], ...] = (
    ("PSP1_Wm2_Avg", "2018-11-13 00:00", "2019-02-26 09:30"),
    ("PSP_Wm2_Avg", "2019-03-18 12:55", "2019-08-31 23:55"),
    ("CMP21_Wm2_Avg", "2019-10-01 00:00", "2020-03-06 10:55"),
    ("CMP21_Wm2_Avg", "2020-06-01 00:00", "2025-03-12 13:20"),
    ("PSP_Wm2_Avg", "2025-05-14 15:25", None),
)


def unshaded_diffuse_days(
    frame: pd.DataFrame,
    eras: Sequence[tuple[str, str | None, str | None]] = DIFFUSE_CHANNEL_ERAS,
) -> list[tuple[str, float]]:
    """Days where the diffuse channel is still reading the global flux.

    Run on the frame **after** :func:`mask_sentinels`: an episode already covered
    by :data:`INVALID_WINDOWS` is ``NaN`` by then, so what comes back is exactly
    what the hand-curated list misses — and that list goes stale, silently, the
    next time the ring comes off, since the column keeps its name.

    Parameters
    ----------
    frame:
        5-minute frame in W/m2 with a monotonic naive station-local index,
        holding the tested channels and :data:`DIFFUSE_GLOBAL_COLUMN`. A missing
        column reports nothing rather than guessing.
    eras:
        ``(column, first, last)`` windows to test, each bound an inclusive naive
        local timestamp or ``None`` for open-ended. Defaults to
        :data:`DIFFUSE_CHANNEL_ERAS`, every channel that has carried the diffuse
        role, over the era it carried it.

    Returns
    -------
    list
        ``(iso date, median clear-sky ratio)`` per offending day, oldest first.
        The ratio is dimensionless, diffuse over global, taken over the samples
        above :data:`DIFFUSE_CLEAR_SKY_FLOOR` only. Empty for the archive as
        shipped: every episode it detects is masked.

    Raises
    ------
    TypeError
        If *frame* is not indexed by a :class:`~pandas.DatetimeIndex`. Label
        slicing any other index yields an empty era rather than an error, so
        every check would pass by finding nothing.
    ValueError
        If that index is not monotonic increasing, which label slicing needs.
    """
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"the diffuse check requires a DatetimeIndex, got {type(index).__name__}")
    if not index.is_monotonic_increasing:
        raise ValueError("the diffuse check requires a monotonic increasing DatetimeIndex")
    flagged: list[tuple[str, float]] = []
    for column, first, last in eras:
        flagged.extend(_unshaded_diffuse_days_in_era(frame.loc[first:last], column))
    return sorted(flagged)


def _unshaded_diffuse_days_in_era(frame: pd.DataFrame, column: str) -> list[tuple[str, float]]:
    """Offending days of one diffuse channel over one era of the frame."""
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


#: The one gauge these checks are defined for. Measured before choosing it: over
#: ten years ``PL01_mm_Tot`` carries 21,444 positive samples on a clean 0.254 mm
#: grid, while ``Rain_WXT_Tot`` carries 9 and ``Ramount_Tot`` 6, on scales that
#: are neither each other's nor the bucket's — 547 and 2052 mm in a five-minute
#: total. Whatever those two are, they are not a tipping bucket, and a
#: quantisation check keyed to one would only ever report its own wrong premise.
#: The hail counters ``Hamount_Tot`` and ``Hamount_WXT_Tot`` are out for a
#: simpler reason: they count impacts, not depth.
RAIN_DEPTH_COLUMNS: tuple[str, ...] = ("PL01_mm_Tot",)

#: One tip of the bucket. The value and the measurement behind it live in
#: :mod:`micrometeorology.common.instruments`; the climatology export needs the
#: same number for its wet/dry threshold.
RAIN_TIP_DEPTH_MM = instruments.RAIN_TIP_DEPTH_MM

#: Tolerance the multiple is checked to. Floored at 1e-5 deliberately: at 1e-6
#: four samples fail on float storage alone (8.636001, 9.398001), which is the
#: encoding and not the instrument.
RAIN_QUANTISATION_TOLERANCE = 1e-5

#: Consecutive dry calendar days after which a gauge is presumed blocked rather
#: than the sky dry. Read off a gap: the longest run in ten years is 54 days
#: (2024-02-13..2024-04-06, the funnel that INVALID_WINDOWS covers by hand), and
#: the next longest is 17. Any threshold in [18, 54] finds that episode and
#: nothing else; 30 sits in the middle of the gap.
BLOCKED_GAUGE_MIN_DRY_DAYS = 30


def blocked_gauge_runs(
    frame: pd.DataFrame,
    columns: Sequence[str] = RAIN_DEPTH_COLUMNS,
    min_dry_days: int = BLOCKED_GAUGE_MIN_DRY_DAYS,
) -> list[tuple[str, str, str, int]]:
    """Stretches where a gauge reported no rain for implausibly long.

    A blocked funnel and a dry spell look identical sample by sample — both are
    a run of zeros — and only the LENGTH separates them at this site. Reported
    rather than masked: the run is evidence about the instrument, and which of
    the dry days inside it were genuinely dry is not recoverable from the gauge
    that failed to measure them.

    Run on the frame after :func:`mask_sentinels`, for the same reason
    :func:`unshaded_diffuse_days` is: an episode already covered by
    :data:`INVALID_WINDOWS` is ``NaN`` by then, so what comes back is what the
    hand-curated list misses. That list ages silently — the column keeps its
    name and its zeros the next time the funnel clogs.

    Parameters
    ----------
    frame:
        5-minute frame in mm with a naive station-local index.
    columns:
        Gauge columns to test. A column absent from *frame* reports nothing.
    min_dry_days:
        Consecutive dry calendar days that make a run suspect.

    Returns
    -------
    list
        ``(column, first iso date, last iso date, days)`` per run, oldest first.
    """
    found: list[tuple[str, str, str, int]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        depth = frame[column].dropna()
        if depth.empty:
            continue
        daily = depth.groupby(pd.DatetimeIndex(depth.index).date).max()
        dry = daily <= 0.0
        # A month of missing days between two dry days is not a dry run: without
        # this, `cumsum` bridges the gap and reports the span as one long stretch.
        consecutive = (
            pd.Series(pd.DatetimeIndex(daily.index)).diff().eq(pd.Timedelta(days=1)).to_numpy()
        )
        run_id = pd.Series((~dry).to_numpy() | ~consecutive, index=daily.index).cumsum()
        for _, block in daily[dry].groupby(run_id[dry]):
            span = len(block)
            if span >= min_dry_days:
                found.append((column, str(block.index[0]), str(block.index[-1]), span))
    return sorted(found, key=lambda entry: entry[1])


def unquantised_rain_samples(
    frame: pd.DataFrame, columns: Sequence[str] = RAIN_DEPTH_COLUMNS
) -> dict[str, int]:
    """Positive rain totals that are not whole multiples of one bucket tip.

    A tipping bucket can only report multiples of its own tip, so a total that
    is not one did not come from the bucket: it came from a unit conversion, a
    running accumulator read as an interval total, or a parser. That is a
    different failure family from a broken instrument, and no range, step or
    persistence test looks for it.

    Evaluated BEFORE the range gate, so it sees what that gate is about to remove:
    on this archive it reports one sample, ``1.09e9`` mm of rain in five minutes on
    2018-06-10 09:10. After the gate it would report nothing, which is a less
    useful thing to publish — the corrupted field is the finding.

    Parameters
    ----------
    frame:
        5-minute frame in mm with a naive station-local index.
    columns:
        Gauge columns to test.

    Returns
    -------
    dict
        ``{column: offending samples}``, omitting the columns with none.
    """
    offending: dict[str, int] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        depth = frame[column]
        positive = depth[depth.notna() & (depth > 0.0)].to_numpy(dtype=float)
        if positive.size == 0:
            continue
        tips = positive / RAIN_TIP_DEPTH_MM
        count = int(np.sum(np.abs(tips - np.round(tips)) > RAIN_QUANTISATION_TOLERANCE))
        if count:
            offending[column] = count
    return offending


#: Humidity channels checked against saturation. Every hygrometer at this site
#: should reach it: the coast saturates often enough that a month which never
#: does is the sensor, not the sky.
HUMIDITY_COLUMNS: tuple[str, ...] = (
    "ur",
    "RH1_Avg",
    "RH_WXT_Avg",
    "RelHumidity",
    "RH",
    "RH1",
    "RH2",
)

#: Monthly maximum below which a hygrometer is presumed biased low. Read off a
#: gap, not chosen: over ten years the healthy channels never put a month below
#: 95.0 %RH (``RH1_Avg`` p05 = 95.5 across 97 months, ``RH2`` p05 = 95.0), and the
#: faulty ones never put one above 90.7 (``RH_WXT_Avg`` max = 90.7 across 49
#: months, ``RH1`` max = 87.0). 93.0 sits in the middle of that empty band, so
#: the check separates the two populations with nothing on either shoulder.
HUMIDITY_SATURATION_FLOOR = 93.0

#: Samples a month needs before its maximum means anything.
HUMIDITY_MIN_SAMPLES_PER_MONTH = 2000


def months_never_reaching_saturation(
    frame: pd.DataFrame,
    columns: Sequence[str] = HUMIDITY_COLUMNS,
    floor: float = HUMIDITY_SATURATION_FLOOR,
) -> list[tuple[str, str, float]]:
    """Months whose hygrometer never came near saturation, which it should have.

    The one fault family no gate in this pipeline can otherwise see. A sensor
    reading a steady 10 %RH low moves correctly, never repeats, never steps and
    never leaves its range — range, excursion and persistence tests are all blind
    to it by construction, because each of them asks about the sample's relation
    to its own neighbours. This asks about its relation to PHYSICS instead:
    saturation is an external anchor at 100 %RH, and a coastal tropical site
    reaches it.

    MONTHLY, deliberately. Measured on this archive the daily form flags 31.88%
    of the healthy channel's days as well as 98.48% of the faulty one's — it does
    not separate them. The monthly form separates them completely: zero of 97
    months on the healthy channel, 49 of 49 on the faulty one.

    Reported, never masked. The samples are wrong by an offset, not absent, and
    which offset is not recoverable from the channel that carries it. Masking
    would also delete a year of otherwise usable humidity over a defect a
    recalibration can correct.

    Parameters
    ----------
    frame:
        5-minute frame in %RH with a naive station-local index.
    columns:
        Hygrometer columns to test, including the unified ``ur``.
    floor:
        Monthly maximum below which the month is reported.

    Returns
    -------
    list
        ``(column, month as YYYY-MM, monthly maximum)``, oldest first. Months
        with fewer than :data:`HUMIDITY_MIN_SAMPLES_PER_MONTH` samples are
        skipped: a maximum over a handful of readings says nothing.
    """
    flagged: list[tuple[str, str, float]] = []
    for column in columns:
        if column not in frame.columns:
            continue
        humidity = frame[column].dropna()
        if humidity.empty:
            continue
        grouped = humidity.groupby(pd.DatetimeIndex(humidity.index).to_period("M"))
        monthly = grouped.agg(["max", "count"])
        suspect = monthly[
            (monthly["count"] >= HUMIDITY_MIN_SAMPLES_PER_MONTH) & (monthly["max"] < floor)
        ]
        flagged.extend(
            (column, str(month), float(value))
            for month, value in zip(suspect.index, suspect["max"], strict=True)
        )
    return sorted(flagged, key=lambda entry: (entry[1], entry[0]))


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

# What is MASKED once a day is flagged, which is not the same list as what
# DETECTS it. Longwave cannot detect the episode — a pyrgeometer reads
# 300-400 W/m2 all night by design — but the shifted clock belongs to the whole
# day and to every channel the logger stamped with it. Leaving Lw_dw and Lw_up
# out published 25,999 samples carrying a timestamp already proven wrong, and
# with them 1,123 hours of longwave statistics. ``Net_CNR1`` is here for the
# converse reason: it is not an independent measurement — over 729,225 samples
# its residual against ``Sw_dw - Sw_up + Lw_dw - Lw_up`` never exceeds
# 8.95 W/m2 — so masking only the components would leave the corrupted
# contribution inside the net.
NIGHT_CORRUPTION_CHANNELS = (*NIGHT_CORRUPTION_COLUMNS, "Lw_dw", "Lw_up", "Net_CNR1")

# BSRN "physically possible" ceiling for global horizontal irradiance
# (Long & Shi 2008): Sa * 1.5 * mu0**1.2 + 100. Because the limit follows the
# sun's own geometry it catches what a flat gate cannot — the shipped [-20, 1500]
# rule fires on 6 samples of the record, this one on 3,077, of which 2,477 carry
# full daylight irradiance with the sun below the horizon. It stays generous at
# high sun (2,080-2,220 W/m2 at zenith over the year) so genuine cloud-edge
# enhancement survives, and bites only at low sun, where a shifted clock puts
# midday values. Applied AFTER the whole-day mask above, so what reaches it is
# the milder residue of the same fault: an afternoon that declines plausibly an
# hour or two out of place.
#
# Sa is the constant AT THE EARTH'S ACTUAL DISTANCE, the coefficient below scaled
# by the eccentricity correction: over the year it spans 1321.3 to 1415.0 W/m2, so
# a fixed value would run the ceiling 3.4% high in July and 3.4% low in January.
#
# 1367 is the coefficient the cited method prescribes ("Let Sa = 1367 * E0", Long
# & Shi 2008 pp. 24-25, transcribed in docs/arqueologia/qc/lit-statistical-methods.md),
# NOT a physical constant this repo is free to update: it is deliberately distinct
# from ``labmim_core.solar.SOLAR_CONSTANT_WM2``, the Kopp & Lean TSI that scales
# extraterrestrial irradiance and the clearness index. Two quantities, two names.
BSRN_CEILING_SOLAR_CONSTANT_WM2 = 1367.0

# The remaining three coefficients of the same prescription, named for the same
# reason as the one above: they are transcribed from Long & Shi 2008 pp. 24-25,
# not knobs this repo tunes. The ceiling reads
# ``Sa * GAIN * mu0**EXPONENT + OFFSET``.
BSRN_CEILING_GAIN = 1.5
BSRN_CEILING_MU0_EXPONENT = 1.2
BSRN_CEILING_OFFSET_WM2 = 100.0

#: Radiation channels carrying the sensor's raw count or bridge voltage rather
#: than a flux. They have no range gate, no BSRN envelope and no nocturnal mask,
#: because every one of those is written against the ``_Wm2_`` twin — so they
#: reach the hourly artifact unfiltered, indistinguishable there from the
#: channels that were filtered. Dropped before aggregation for that reason.
UNGATED_RADIATION_TWINS = (
    "CMP21_Avg",
    "CMP22_Avg",
    "CUV5_Avg",
    "NRLite_Avg",
    "PAR_Den_Avg",
    "PIR1_Avg",
    "PSP1_Avg",
    "PSP_Avg",
)

#: BSRN physically-possible ceilings as ``(gain, offset)`` in
#: ``Sa * gain * mu0**1.2 + offset`` (Long & Shi 2008, tab. 1). One pair per
#: component, because a shaded pyranometer and an upward-facing one cannot reach
#: the global's ceiling: reusing the global's pair for all of them is a gate that
#: only ever fires on the global. PAR is the broadband ceiling scaled by the
#: fraction of the solar spectrum the quantum sensor answers to.
_PAR_SPECTRAL_FRACTION = 0.55
BSRN_PPL_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "Sw_dw": (BSRN_CEILING_GAIN, BSRN_CEILING_OFFSET_WM2),
    "Sw_dif": (0.95, 50.0),
    "Sw_up": (1.2, 50.0),
    "Sw_par": (
        BSRN_CEILING_GAIN * _PAR_SPECTRAL_FRACTION,
        BSRN_CEILING_OFFSET_WM2 * _PAR_SPECTRAL_FRACTION,
    ),
}

#: BSRN physically-possible FLOOR, the lower half of the same Long & Shi 2008
#: tab. 1 prescription the ceilings above transcribe: -4 W/m2 for every
#: shortwave component. Its absence is what let the archive publish 36,202
#: negative hours of ``Sw_dw``: a thermopile reads its own zero-offset, and the
#: flat ``[-20, 1500]`` gate in ``default.yaml`` was written wide enough to pass
#: it. PAR is scaled by the same spectral fraction as its ceiling; measured, that
#: gate is inert because the quantum sensor is already clipped at zero upstream,
#: and it is declared anyway so the channel is not the one component of the set
#: with no floor at all.
BSRN_PPL_FLOOR_WM2 = -4.0

#: The BSRN "extremely rare" minimum, the second tier Long & Shi 2008 defines
#: below the physically-possible one. Never a mask here: it is the level the
#: nocturnal-offset monitor reports against.
BSRN_PPL_EXTREMELY_RARE_WM2 = -2.0

#: Shortwave channels blanked while the sun is below the horizon, and the
#: elevation that defines "below". Distinct from
#: :data:`NIGHT_CORRUPTION_COLUMNS`, which uses -10 degrees to find a slipped
#: clock: this one is the horizon itself, because the quantity being rejected is
#: not a fault but the ABSENCE of a measurable signal. A pyranometer at night
#: reports its zero-offset and nothing else — measured on this archive, a median
#: of -1.478 W/m2 on ``Sw_dw`` and +2.620 on ``Sw_up``, the sign following which
#: way the dome faces — so the reading carries instrument state, not irradiance.
#: ``Sw_uv`` is here and in neither list above: 5,175 of its 11,900 valid hours
#: are negative, all nocturnal.
NOCTURNAL_SHORTWAVE_CHANNELS = ("Sw_dw", "Sw_dif", "Sw_up", "Sw_par", "Sw_uv")
NOCTURNAL_ELEVATION_DEG = 0.0

#: Fraction by which the diffuse must exceed the global before the comparison is
#: called a fault. A bare ``>`` was defensible while the diffuse was published as
#: the pyranometer read it; since :func:`~micrometeorology.sensors.calibration.apply_shade_ring_correction`
#: returns the sky the ring occults, the two channels carry the combined error of
#: two instruments plus the isotropic ring model, and equality is no longer
#: exact. Measured on this archive, the bare rule fires on 3,048 samples of which
#: 3,018 exceed the global by less than 2%; at this tolerance 447 remain. The
#: value is the one this repository already measured as the right level in
#: docs/arqueologia/qc/med-fault-detection.md ("Dif/Global > 1.05").
DIFFUSE_EXCEEDS_GLOBAL_TOLERANCE = 1.05

#: Global irradiance above which ``Sw_dif > Sw_dw`` is a real inconsistency
#: rather than thermal offset. Below it the two channels differ by a couple of
#: W/m2 of uncorrected IR loss and the comparison is meaningless: measured, the
#: raw rule fires on 126,404 samples, of which only 30 sit above this level.
#: Those 30 have no physical reading — the witness is 2024-05-15 12:50, where the
#: SHADED CMP21 published 1136.04 W/m2 against a global of 681.11 while the
#: unshaded PSP beside it read 316.66 in the same instant.
DIFFUSE_EXCEEDS_GLOBAL_MIN_WM2 = 200.0


# ``{unified name: [(source column, inclusive start, inclusive end), ...]}``, as
# ``sensors.calibration.resolve_mapping_windows`` returns it.
SourceWindows = Mapping[str, Sequence[tuple[str, pd.Timestamp, pd.Timestamp]]]


#: Shift from a logger stamp to the centre of the interval it averages.
#: The CR5000 END-stamps: the row at ``t`` averages ``(t - 5 min, t]``, measured
#: against the 1-minute truth at RMS 0.083 W/m2 and r = 1.000000 in
#: docs/allsky-label-join.md, which is why the all-sky pipeline carries the same
#: correction as ``PrepareSensorConfig.timestamp_offset_minutes = -2.5``.
#: Solar geometry describes an instant, so it belongs at the interval's centroid,
#: not at its closing edge: evaluated at the raw stamp, a sample within about
#: 5 minutes of sunrise or sunset falls on the wrong side of the horizon — at
#: 2024-06-01 05:55 the raw stamp gives +0.437 deg (daylight) while the centroid
#: 05:52:30 gives -0.125 deg (still night).
AVERAGING_CENTROID_OFFSET = -SAMPLING_INTERVAL / 2


def averaging_centroid(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Move end-stamped rows to the centre of the interval each one averages.

    Parameters
    ----------
    index:
        Naive station-local stamps as the logger wrote them, ``(N,)``.

    Returns
    -------
    pandas.DatetimeIndex
        The same stamps shifted by :data:`AVERAGING_CENTROID_OFFSET`, ``(N,)``.
    """
    return index + AVERAGING_CENTROID_OFFSET


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
    """Blank irradiance outside what the sun's position can produce, both ways.

    Per SAMPLE, unlike :func:`mask_night_corrupted_days`, because this catches
    the residue rather than the episode. ``Net_CNR1`` follows the same sample
    because the logger derives it from the four components.

    The gate has three parts: the geometric ceiling per component, the flat
    :data:`BSRN_PPL_FLOOR_WM2` beneath every one of them, and the sign rule that
    no shortwave flux is zero or negative while the sun is above the horizon.
    A one-sided "physically possible" limit passes half of what the
    prescription rejects: without the floor and the sign rule, negative
    shortwave reaches the published hourly means.

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
    centroid = averaging_centroid(index)
    mu0 = np.clip(cos_zenith(centroid, STATION_SITE, STATION_UTC_OFFSET_HOURS), 0.0, None)
    geometry = (
        BSRN_CEILING_SOLAR_CONSTANT_WM2
        * eccentricity_correction(centroid)
        * mu0**BSRN_CEILING_MU0_EXPONENT
    )

    def _blank(column: str, offending: NDArray) -> None:
        _mask_column(frame, column, offending, removed)
        for source, start, end in (sources or {}).get(column, ()):
            within = (index >= start) & (index <= end)
            _mask_column(frame, source, offending & within, removed)

    daylight = mu0 > 0.0
    for column, (gain, offset) in BSRN_PPL_COEFFICIENTS.items():
        if column not in frame.columns:
            continue
        flux = frame[column]
        floor = BSRN_PPL_FLOOR_WM2 * (_PAR_SPECTRAL_FRACTION if column == "Sw_par" else 1.0)
        # Rayleigh scattering alone puts the diffuse above zero whenever the sun
        # is up, so a non-positive daylight flux is offset or a stuck channel.
        daytime_fault = (
            flux.notna() & ((flux > geometry * gain + offset) | (daylight & (flux <= 0.0)))
        ).to_numpy()
        outside = daytime_fault | (flux.notna() & (flux < floor)).to_numpy()
        if not outside.any():
            continue
        _blank(column, outside)
        # The net is not an independent measurement: the logger derives it from
        # the four components, so a component the sun cannot produce is inside it.
        # Only the daytime verdicts: the floor fires on the nocturnal offset, over
        # a component whose true value is zero, and the night net is longwave.
        if column in NET_RADIATION_COMPONENTS:
            _blank("Net_CNR1", daytime_fault)

    # A shaded sensor measures a subset of what the unshaded one sees, so this is
    # impossible at any level — but only above the offset floor is the comparison
    # meaningful at all, and only beyond the tolerance is it a fault rather than
    # the two instruments disagreeing inside their combined error.
    if "Sw_dif" in frame.columns:
        global_flux, diffuse = frame["Sw_dw"], frame["Sw_dif"]
        inconsistent = (
            global_flux.notna()
            & diffuse.notna()
            & (global_flux > DIFFUSE_EXCEEDS_GLOBAL_MIN_WM2)
            & (diffuse > global_flux * DIFFUSE_EXCEEDS_GLOBAL_TOLERANCE)
        ).to_numpy()
        if inconsistent.any():
            _blank("Sw_dif", inconsistent)
    return frame, removed


def night_corrupted_days(
    frame: pd.DataFrame,
    columns: Sequence[str] = NIGHT_CORRUPTION_COLUMNS,
    elevation_deg: NDArray | None = None,
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
    elevation_deg:
        Solar elevation over *frame*'s index in degrees, ``(N,)``, when the
        caller already holds it; computed here otherwise. A pipeline that runs
        several elevation-gated stages passes one array to all of them so a
        second definition of "night" cannot drift from the first.

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
    if elevation_deg is None:
        elevation_deg = station_elevation_deg(index)
    deep_night = elevation_deg < NIGHT_CORRUPTION_ELEVATION_DEG
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


#: Elevation below which the nocturnal offset is measured, i.e. SZA > 100 deg as
#: docs/arqueologia/qc/lit-radiation-qc.md specifies for the drift monitor. Deeper
#: than :data:`NOCTURNAL_ELEVATION_DEG`, which is the horizon itself: the statistic
#: wants samples with no twilight contamination at all, while the mask wants every
#: sample the sun cannot reach.
OFFSET_MONITOR_ELEVATION_DEG = -10.0

#: Month-on-month change in the nocturnal offset median that raises a drift alarm,
#: against a trailing 12-month baseline (lit-radiation-qc.md). A step in this
#: number warns of ventilator failure, dome degradation or a wiring fault months
#: before it distorts a daytime statistic.
OFFSET_DRIFT_ALARM_WM2 = 1.5
OFFSET_DRIFT_BASELINE_MONTHS = 12


@dataclass(frozen=True, slots=True)
class NocturnalOffset:
    """One channel's thermopile zero-offset, as the drift monitor records it.

    Attributes
    ----------
    night_samples:
        Deep-night samples the statistic was measured on. Zero means the channel
        was measured and had none, which a reader must be able to tell from a
        channel that was never looked at — hence the record exists either way.
    median_wm2, p5_wm2, p95_wm2:
        Offset level and spread, W/m2. ``None`` when *night_samples* is zero.
    fraction_below_bsrn_floor, fraction_below_extremely_rare:
        Share of deep-night samples under the two Long & Shi 2008 minima.
    yearly_median_wm2, monthly_median_wm2:
        Offset median per calendar year and per month, W/m2, keyed by ISO period.
    drift_alarms:
        ``(month, median)`` for each month departing from its trailing baseline
        by more than :data:`OFFSET_DRIFT_ALARM_WM2`.
    """

    night_samples: int
    median_wm2: float | None = None
    p5_wm2: float | None = None
    p95_wm2: float | None = None
    fraction_below_bsrn_floor: float | None = None
    fraction_below_extremely_rare: float | None = None
    yearly_median_wm2: Mapping[str, float] = field(default_factory=dict)
    monthly_median_wm2: Mapping[str, float] = field(default_factory=dict)
    drift_alarms: Sequence[tuple[str, float]] = ()

    def as_report(self) -> dict[str, object]:
        """JSON-ready mapping for ``archive_report.json``."""
        if not self.night_samples:
            return {"night_samples": 0}
        report = asdict(self)
        report["drift_alarms"] = [
            {"month": month, "median_wm2": median} for month, median in self.drift_alarms
        ]
        return report


def nocturnal_offset_statistics(
    frame: pd.DataFrame,
    columns: Sequence[str] = NOCTURNAL_SHORTWAVE_CHANNELS,
    elevation_deg: NDArray | None = None,
) -> dict[str, NocturnalOffset]:
    """Thermopile zero-offset per channel, measured in deep night.

    MUST be called before BOTH :func:`mask_impossible_shortwave` and
    :func:`mask_nocturnal_shortwave`. The second is what makes this function
    exist; the first is what silently empties it, because the BSRN floor removes
    precisely the samples whose share ``fraction_below_bsrn_floor`` reports —
    measured afterwards it is 0.0 for every channel, a statistic that reports
    success while measuring nothing. The nocturnal offset is the only pyranometer health
    diagnostic available without a calibration lab (Dutton et al. 2001; QCRad,
    in Long & Shi 2008), and masking the samples that carry it would retire that
    diagnostic silently. Persisting the statistic keeps the monitor after the
    samples themselves leave the published frame.

    Parameters
    ----------
    frame:
        5-minute frame in W/m2 with a naive station-local index, before any
        nocturnal masking.
    columns:
        Shortwave channels to measure. A column absent from *frame* is skipped.

    Returns
    -------
    dict
        ``{column: NocturnalOffset}``. A channel with no deep-night sample gets a
        record reporting zero rather than being omitted, so a reader can tell
        "measured, nothing there" from "never looked".
    """
    if elevation_deg is None:
        elevation_deg = station_elevation_deg(pd.DatetimeIndex(frame.index))
    deep_night = elevation_deg < OFFSET_MONITOR_ELEVATION_DEG
    statistics: dict[str, NocturnalOffset] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame.loc[deep_night, column].dropna()
        if values.empty:
            statistics[column] = NocturnalOffset(night_samples=0)
            continue
        stamps = pd.DatetimeIndex(values.index)
        monthly = values.groupby(stamps.to_period("M")).median()
        baseline = monthly.rolling(OFFSET_DRIFT_BASELINE_MONTHS, min_periods=3).median().shift(1)
        drift = (monthly - baseline).abs() > OFFSET_DRIFT_ALARM_WM2
        statistics[column] = NocturnalOffset(
            night_samples=int(values.shape[0]),
            median_wm2=float(values.median()),
            p5_wm2=float(values.quantile(0.05)),
            p95_wm2=float(values.quantile(0.95)),
            fraction_below_bsrn_floor=float((values < BSRN_PPL_FLOOR_WM2).mean()),
            fraction_below_extremely_rare=float((values < BSRN_PPL_EXTREMELY_RARE_WM2).mean()),
            yearly_median_wm2={
                str(year): float(median)
                for year, median in values.groupby(stamps.year).median().items()
            },
            monthly_median_wm2={str(month): float(median) for month, median in monthly.items()},
            drift_alarms=[
                (str(month), float(monthly[month])) for month in monthly.index[drift.fillna(False)]
            ],
        )
    return statistics


def station_elevation_deg(index: pd.DatetimeIndex) -> NDArray:
    """Solar elevation over *index*, from the pinned site and UTC offset.

    One spelling for every stage that needs it: a second copy of the call is a
    second definition of "night", free to drift from the first without failing.
    The geometry is evaluated at :func:`averaging_centroid` of *index*, not at
    the logger's own end-stamp.

    Parameters
    ----------
    index:
        Naive station-local stamps as the logger wrote them, ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Degrees above the local horizon at the centre of each averaging
        interval, ``(N,)``.
    """
    return solar_elevation_deg(averaging_centroid(index), STATION_SITE, STATION_UTC_OFFSET_HOURS)


def mask_nocturnal_shortwave(
    frame: pd.DataFrame,
    sources: SourceWindows | None = None,
    elevation_deg: NDArray | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Blank shortwave recorded while the sun is below the horizon.

    What a pyranometer reports at night is its own thermal zero-offset, not
    irradiance, and the sign follows which way the dome faces. The value is
    masked, never clamped to zero: zero is a valid irradiance and would enter
    every downstream mean as an observation. After this stage a shortwave mean is
    a DAYLIGHT mean, and a daily insolation total needs the nocturnal hours
    restored as zero before integrating.

    Must run after :func:`night_corrupted_days`, which finds a slipped logger
    clock BY the irradiance recorded at night. The raw twins are masked over the
    whole record rather than per era window: the sun's elevation belongs to the
    timestamp, not to the instrument.

    Parameters
    ----------
    frame:
        5-minute frame in W/m2 with a naive station-local index, against which
        the sun's elevation is computed from :data:`STATION_SITE` and the pinned
        :data:`STATION_UTC_OFFSET_HOURS` — never the host time zone, which would
        displace the terminator by hours without raising.
    sources:
        Raw columns per unified channel, or ``None`` to mask the unified columns
        alone.

    Returns
    -------
    tuple
        The same frame, mutated in place, and a ``{column: samples masked}``
        tally in the shape :func:`mask_sentinels` reports.
    """
    removed: dict[str, int] = {}
    if elevation_deg is None:
        elevation_deg = station_elevation_deg(pd.DatetimeIndex(frame.index))
    below = elevation_deg < NOCTURNAL_ELEVATION_DEG
    if not below.any():
        return frame, removed
    for column in NOCTURNAL_SHORTWAVE_CHANNELS:
        _mask_column(frame, column, below, removed)
        for source, _start, _end in (sources or {}).get(column, ()):
            _mask_column(frame, source, below, removed)
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


def close_nocturnal_net_radiation(
    frame: pd.DataFrame, elevation_deg: NDArray | None = None
) -> tuple[pd.DataFrame, int]:
    """Recompose the net with the sun down, where the shortwave terms are zero.

    :func:`close_net_radiation` runs before :func:`mask_nocturnal_shortwave` and
    folds the nocturnal thermopile offset into the published net — measured here,
    a median of -3.85 W/m2 against a nocturnal net of -49.1. With the sun below
    the horizon the shortwave terms are ZERO, not the offset, so the nocturnal
    net is the longwave difference alone.

    Parameters
    ----------
    frame:
        Frame in W/m2 with a naive station-local index, holding ``Lw_dw``,
        ``Lw_up`` and ``Net_CNR1``.

    Returns
    -------
    tuple
        The same frame, mutated in place, and the samples recomposed.
    """
    needed = ("Lw_dw", "Lw_up", "Net_CNR1")
    if not all(column in frame.columns for column in needed):
        return frame, 0
    if elevation_deg is None:
        elevation_deg = station_elevation_deg(pd.DatetimeIndex(frame.index))
    below = elevation_deg < NOCTURNAL_ELEVATION_DEG
    longwave = frame["Lw_dw"] - frame["Lw_up"]
    target = below & longwave.notna().to_numpy()
    if not target.any():
        return frame, 0
    frame.loc[target, "Net_CNR1"] = longwave[target]
    return frame, int(target.sum())


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
