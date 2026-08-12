"""Manifest v2 contracts: column registry, QC flags, sky classes, paths.

This module pins the on-disk schema of the multimodal dataset manifest — the
byte contract every other ``allsky.data`` module (and the training/embedding
stack downstream) codes against.  It is deliberately dependency-light (stdlib
plus typing only): no numpy, pandas or torch, so importing it is cheap and never
pulls a heavy framework.

Three things are fixed here:

- :data:`DATASET_VERSION` and the ordered manifest column -> dtype registry
  (:func:`manifest_column_dtypes`).
- The :class:`QCFlag` bitmask and the sky-condition class constants/names.
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
    "FEATURE_DTYPE",
    "GEOMETRY_COLUMNS",
    "META_COLUMNS",
    "PROVENANCE_COLUMNS",
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
    "SPLIT_COLUMN",
    "TARGET_COLUMNS",
    "QCFlag",
    "manifest_column_dtypes",
    "resolve",
    "sky_class_name",
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
    manifest builder. ``FRAME_DARK``/``FRAME_SATURATED`` come from the visual QC
    pass, and ``TIMESTAMP_INTERPOLATED``/``TIMESTAMP_CORRECTED`` from the overlay
    reader, so a sample whose capture time was manufactured rather than read
    stays identifiable after the frame manifest is folded into the dataset.
    """

    NONE = 0
    #: Solar elevation below the k-index elevation floor (target k-index noisy).
    LOW_SUN = 1
    #: No sensor record paired within tolerance, or the GHI channel was missing.
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
    same file it started from.  An absolute *path* must live inside *data_root*;
    otherwise a :class:`ValueError` is raised (a manifest must never encode a
    location outside its data root).
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

    Raises
    ------
    ValueError
        If *relative* is an absolute path — manifests must store relative POSIX
        paths so they stay portable across machines.
    """
    text = str(relative)
    if PurePosixPath(text).is_absolute() or Path(text).is_absolute() or text.startswith("/"):
        raise ValueError(f"manifest path must be a relative POSIX path, got absolute {relative!r}")
    return Path(data_root) / text
