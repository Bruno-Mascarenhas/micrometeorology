"""Manifest v2 contracts: column registry, QC flags, portable paths.

This module pins the on-disk schema of the multimodal dataset manifest — the
byte contract every other ``allsky.data`` module (and the training/embedding
stack downstream) codes against.  It is deliberately dependency-light (stdlib
plus typing only): no numpy, pandas or torch, so importing it is cheap and never
pulls a heavy framework.

Three things are fixed here:

- :data:`DATASET_VERSION` and the ordered manifest column -> dtype registry
  (:func:`manifest_column_dtypes`).
- The :class:`QCFlag` bitmask.  The sky-condition class constants and names
  live in :mod:`labmim_core.sky`.
- Portable-path helpers (:func:`to_relative` / :func:`resolve`): manifests store
  image paths as **relative POSIX** strings against a ``data_root``; absolute
  paths are rejected with a clear error so a manifest never bakes in a machine's
  directory layout.
"""

import posixpath
from collections.abc import Mapping, Sequence
from enum import IntFlag
from pathlib import Path, PurePosixPath

__all__ = [
    "DATASET_VERSION",
    "DEGRADABLE_TARGET_COLUMNS",
    "FEATURE_DTYPE",
    "GEOMETRY_COLUMNS",
    "LABELABLE_MIN_ELEVATION_DEG",
    "META_COLUMNS",
    "NS_PER_MINUTE",
    "PROVENANCE_COLUMNS",
    "SPLIT_COLUMN",
    "TARGET_COLUMNS",
    "QCFlag",
    "manifest_column_dtypes",
    "resolve",
    "to_relative",
]

#: On-disk dataset schema version stored in the manifest sidecar meta.
DATASET_VERSION = "2"

#: Pandas dtype used for every engineered feature column.
FEATURE_DTYPE = "float64"

#: Leading identity/metadata columns (ordered).  ``timestamp_utc`` is tz-aware
#: UTC; ``day_id`` is the LOCAL calendar day; ``image_path`` is relative POSIX.
META_COLUMNS: Mapping[str, str] = {
    "sample_id": "string",
    "timestamp_utc": "datetime64[ns, UTC]",
    "day_id": "string",
    "image_path": "string",
    "frame_index": "int64",
    "video": "string",
}

#: Raw solar-geometry columns (degrees).  ``solar_elevation`` / ``solar_zenith``
#: double as engineered features; ``solar_azimuth`` is geometry-only (azimuth is
#: fed to the model as the ``azimuth_sin`` / ``azimuth_cos`` cyclic pair).
GEOMETRY_COLUMNS: Mapping[str, str] = {
    "solar_elevation": "float64",
    "solar_azimuth": "float64",
    "solar_zenith": "float64",
}

#: Trailing target / label columns (ordered).  ``cloud_fraction`` is nullable
#: (all-NaN until ground truth exists); ``qc_flags`` is a :class:`QCFlag`
#: bitmask stored as ``int64``.
TARGET_COLUMNS: Mapping[str, str] = {
    "target_dhi": "float64",
    "target_source": "string",
    "target_kindex": "float64",
    "kindex_kind": "string",
    "target_kt": "float64",
    "sky_class": "int64",
    "cloud_fraction": "float64",
    "qc_flags": "int64",
}

#: Target columns that joined :data:`TARGET_COLUMNS` after the v2 layout was
#: first published.  A manifest built before them is structurally sound and both
#: trains and evaluates; each consumer degrades on its own — the evaluator drops
#: the ``kindex_band`` stratum without ``target_kt``, and the k-index clear-sky
#: baseline is unresolvable without ``kindex_kind`` — so validation reports them
#: as warnings rather than refusing a dataset that works.
DEGRADABLE_TARGET_COLUMNS: tuple[str, ...] = (
    "target_source",
    "kindex_kind",
    "target_kt",
    "cloud_fraction",
)

#: Nanoseconds in a minute. The alignment search and the temporal window both
#: work on int64 nanosecond stamps, and two copies of this factor are two places
#: a window could silently stop meaning minutes.
NS_PER_MINUTE = 60_000_000_000

#: Solar elevation floor, degrees, below which the k-index carries too little
#: signal to label a frame on. One name because three stages read it and each
#: has to read the SAME one: the manifest builder sets ``LOW_SUN`` under it,
#: ``NightFilterConfig`` defaults to it, and ``validate_manifest(strict=True)``
#: reports the rows that survived the build's own floor but sit under this. Three
#: independent literals would let a config change the build and leave the
#: validator checking a band the build no longer produces.
LABELABLE_MIN_ELEVATION_DEG = 10.0

#: Name of the (nullable) split-label column: empty at build, filled in place by
#: :func:`allsky.data.manifest.attach_split_column` from a day-level split.
SPLIT_COLUMN = "split"

#: Trailing provenance columns duplicated constant per row so a manifest is
#: self-describing without its sidecar.  ``dataset_version`` and ``alignment_id``
#: mirror the meta; ``split`` is nullable (``pd.NA`` until a split is attached).
PROVENANCE_COLUMNS: Mapping[str, str] = {
    "dataset_version": "string",
    "alignment_id": "string",
    SPLIT_COLUMN: "string",
}


class QCFlag(IntFlag):
    """Per-sample quality-control bitmask stored in ``qc_flags``.

    Flags are additive: a single ``int64`` column carries any combination.
    ``LOW_SUN``/``SENSOR_GAP``/``ALIGNMENT_FAR``/``KT_ARTIFACT`` are set by the
    manifest builder. ``FRAME_DARK``/``FRAME_SATURATED``/``FRAME_UNREADABLE``
    come from the visual QC pass, and
    ``TIMESTAMP_INTERPOLATED``/``TIMESTAMP_CORRECTED`` from the overlay reader,
    so a sample whose capture time was manufactured rather than read stays
    identifiable after the frame manifest is folded into the dataset.
    """

    NONE = 0
    #: Solar elevation below the k-index elevation floor (target k-index noisy).
    LOW_SUN = 1
    #: The GHI channel — or the configured diffuse channel — was missing on the
    #: paired sensor record. A frame that matched no record within tolerance is
    #: dropped by :func:`allsky.data.manifest.build_manifest`, so it never
    #: reaches a row this flag could mark.
    SENSOR_GAP = 2
    #: Paired sensor record further than the "far" alignment threshold.
    ALIGNMENT_FAR = 4
    #: Clearness/clear-sky index above the physical-plausibility ceiling.
    KT_ARTIFACT = 8
    #: Frame too dark to be usable.
    FRAME_DARK = 16
    #: Frame saturated/over-exposed.
    FRAME_SATURATED = 32
    #: Capture time interpolated from neighbouring frames, not read off this one.
    TIMESTAMP_INTERPOLATED = 64
    #: Capture time re-decided from the capture sequence because the glyphs
    #: this frame carries did not settle a digit on their own.
    TIMESTAMP_CORRECTED = 128
    #: Frame carries no pixels (a truncated decode), so no radiometric quantity
    #: could be measured on it at all.
    FRAME_UNREADABLE = 256


def manifest_column_dtypes(feature_columns: Sequence[str]) -> dict[str, str]:
    """Ordered ``column -> pandas dtype`` map for a manifest with *feature_columns*.

    Column order is canonical and stable: metadata, then raw geometry, then the
    engineered feature columns that are not already provided by geometry
    (``solar_elevation`` / ``solar_zenith`` are shared, so they are not
    duplicated), then the target/label columns, then the constant provenance
    columns (``dataset_version``, ``alignment_id``, ``split``).

    Parameters
    ----------
    feature_columns:
        Engineered feature names in policy order (see
        :func:`allsky.features.policy.resolve_feature_set`).

    Raises
    ------
    ValueError
        If *feature_columns* contains a name that collides with a metadata,
        geometry-azimuth, target or provenance column, or contains duplicates.
    """
    seen: set[str] = set()
    for name in feature_columns:
        if name in seen:
            raise ValueError(f"duplicate feature column {name!r}")
        seen.add(name)

    reserved = (
        set(META_COLUMNS) | {"solar_azimuth"} | set(TARGET_COLUMNS) | set(PROVENANCE_COLUMNS)
    ) & seen
    if reserved:
        raise ValueError(
            f"feature columns collide with reserved manifest columns: {sorted(reserved)}"
        )

    dtypes: dict[str, str] = dict(META_COLUMNS)
    dtypes.update(GEOMETRY_COLUMNS)
    for name in feature_columns:
        if name not in GEOMETRY_COLUMNS:
            dtypes[name] = FEATURE_DTYPE
    dtypes.update(TARGET_COLUMNS)
    dtypes.update(PROVENANCE_COLUMNS)
    return dtypes


def to_relative(path: str | Path, data_root: str | Path) -> str:
    """Convert *path* to a relative POSIX string against *data_root*.

    An already-relative *path* is normalized to POSIX separators; when
    *data_root* is itself relative — both are then relative to the same working
    directory, as with the shipped ``output/allsky/dataset`` default — its
    prefix is stripped too, so ``resolve(to_relative(p, root), root)`` names the
    same file it started from.

    Returns
    -------
    str
        POSIX-separated path relative to *data_root*, as stored in the
        manifest's ``image_path`` column.

    Raises
    ------
    ValueError
        If *path* is absolute and does not live inside *data_root*: a manifest
        must never encode a location outside its data root.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate_posix = PurePosixPath(posixpath.normpath(candidate.as_posix()))
        root_posix = PurePosixPath(posixpath.normpath(Path(data_root).as_posix()))
        if not root_posix.is_absolute():
            try:
                return candidate_posix.relative_to(root_posix).as_posix()
            except ValueError:
                pass  # relative path that is not under the relative root
        return candidate_posix.as_posix()

    root = Path(data_root)
    base = root if root.is_absolute() else root.resolve()
    try:
        relative = candidate.resolve().relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(
            f"path {str(path)!r} is not inside data_root {str(data_root)!r}; "
            "manifest image paths must be relative to the data root"
        ) from exc
    return relative.as_posix()


def resolve(relative: str | Path, data_root: str | Path) -> Path:
    """Resolve a relative POSIX manifest path against *data_root* to a full path.

    Returns
    -------
    pathlib.Path
        *relative* joined onto *data_root* in the host's path flavour.

    Raises
    ------
    ValueError
        If *relative* is an absolute path — manifests must store relative POSIX
        paths so they stay portable across machines.
    """
    text = str(relative)
    if PurePosixPath(text).is_absolute() or Path(text).is_absolute():
        raise ValueError(f"manifest path must be a relative POSIX path, got absolute {relative!r}")
    return Path(data_root) / text
