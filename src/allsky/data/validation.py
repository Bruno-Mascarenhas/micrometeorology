"""Manifest validation: structural, physical and split-integrity checks.

:func:`validate_manifest` walks the v2 manifest and its sidecar meta and
accumulates every problem into a :class:`ValidationReport` (errors that must
block use, warnings that merely flag suspect rows).  ``strict=True`` promotes
warnings to errors.  Covered failure modes:

- a manifest missing any column the v2 contract declares (the report degrades
  quietly when the target columns the evaluator reads are absent);
- missing image files on disk;
- duplicate ``sample_id`` / ``timestamp_utc``;
- ``timestamp_utc`` not tz-aware;
- NaN/inf in any feature column;
- a sidecar meta declaring no ``feature_columns`` (warning: the two checks that
  walk that list then have nothing to inspect);
- solar elevation below a hard floor (night frames should not be in a dataset);
- invalid targets (``target_dhi`` < 0, ``target_kindex`` out of range,
  ``sky_class`` outside ``{-1, 0, 1, 2}``);
- forbidden (leakage-prone) feature/radiometry columns present;
- day-level split leakage (a ``day_id`` assigned to more than one split);
- a filled ``split`` column disagreeing with the split artifact;
- non-constant ``dataset_version`` / ``alignment_id`` columns, or values that
  disagree with the sidecar meta;
- more than one normalization version in play.

Pure numpy/pandas; importing this module never pulls torch.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.data.contracts import (
    DATASET_VERSION,
    DEGRADABLE_TARGET_COLUMNS,
    META_COLUMNS,
    SPLIT_COLUMN,
    TARGET_COLUMNS,
    QCFlag,
    resolve,
)
from allsky.features.policy import FORBIDDEN_FEATURES
from labmim_core.sky import SKY_CLASS_MISSING, SKY_CLASS_VALUES

__all__ = [
    "STRICT_LOW_SUN_ELEVATION_DEG",
    "ManifestValidationError",
    "ValidationReport",
    "validate_manifest",
]

#: Elevation floor, in degrees, that ``strict=True`` flags rows below. Mirrors
#: the ``min_elevation_deg`` default of
#: :func:`allsky.data.manifest.build_manifest`, the floor that governs
#: :attr:`~allsky.data.contracts.QCFlag.LOW_SUN`.
STRICT_LOW_SUN_ELEVATION_DEG = 10.0


class ManifestValidationError(ValueError):
    """Raised by :meth:`ValidationReport.raise_if_failed` when errors are present."""


@dataclass
class ValidationReport:
    """Accumulated validation errors and warnings for a manifest.

    ``errors`` block use (structural corruption, leakage, invalid targets);
    ``warnings`` flag suspect-but-tolerable conditions.  Both are ordered lists
    of human-readable messages.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no errors were recorded (warnings do not fail validation)."""
        return not self.errors

    def add_error(self, message: str) -> None:
        """Record a blocking error."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Record a non-blocking warning."""
        self.warnings.append(message)

    def raise_if_failed(self) -> None:
        """Raise :class:`ManifestValidationError` if any errors were recorded."""
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ManifestValidationError(
                f"manifest validation failed with {len(self.errors)} error(s):\n  - {joined}"
            )


def validate_manifest(
    manifest: pd.DataFrame,
    meta: dict[str, Any],
    *,
    data_root: str | Path,
    split_artifact: dict[str, Any] | None = None,
    strict: bool = False,
    min_elevation_deg: float = 0.0,
    max_kindex: float = 2.0,
    normalization_versions: Iterable[str] | None = None,
    check_files: bool = True,
) -> ValidationReport:
    """Validate *manifest* against the v2 contract and *meta*.

    Parameters
    ----------
    manifest:
        The manifest DataFrame (columns per
        :func:`allsky.data.contracts.manifest_column_dtypes`).
    meta:
        The sidecar meta dict (provides ``feature_columns`` and
        ``dataset_version``).
    data_root:
        Root that ``image_path`` values resolve against (for existence checks).
    split_artifact:
        Optional loaded split artifact; if given, its day assignment is checked
        for day-level leakage across splits and against the manifest's ``split``
        column. One carrying no day assignment at all is an error, not a pass.
    strict:
        Also record an error for rows in the low-sun band — elevation at or above
        *min_elevation_deg* but below :data:`STRICT_LOW_SUN_ELEVATION_DEG`. It
        promotes nothing: the report's warnings are unaffected.
    min_elevation_deg:
        Elevation floor; rows below it are errors (default ``0`` — only truly
        below-horizon/night rows fail).
    max_kindex:
        Upper bound used for the ``target_kindex`` range check.
    normalization_versions:
        Optional collection of normalization version identifiers; more than one
        distinct value is an error.
    check_files:
        When True, verify each ``image_path`` exists on disk (relative to
        *data_root*).

    Returns
    -------
    ValidationReport
        Populated report; call :meth:`ValidationReport.raise_if_failed` to turn
        errors into an exception.
    """
    report = ValidationReport()
    feature_columns: Sequence[str] = list(meta.get("feature_columns", []))
    if not feature_columns:
        report.add_warning(
            "meta declares no 'feature_columns'; the NaN/inf feature check and the "
            "declared-leaky-feature check were skipped"
        )

    _check_dataset_version(meta, report)
    _check_required_columns(manifest, report)
    _check_duplicates(manifest, report)
    _check_timezone(manifest, report)
    _check_features_finite(manifest, feature_columns, report)
    _check_forbidden_columns(manifest, feature_columns, report)
    _check_elevation(manifest, min_elevation_deg, report, strict=strict)
    _check_targets(manifest, max_kindex, report)
    _check_qc_flags(manifest, report)
    _check_constant_provenance(manifest, meta, report)
    if check_files:
        _check_image_files(manifest, data_root, report)
    if split_artifact is not None:
        if not _day_to_splits(split_artifact):
            report.add_error(
                "the split artifact carries neither an 'assignment' nor a 'splits' map, so "
                "the leakage and split-column checks inspected nothing"
            )
        _check_split_leakage(split_artifact, report)
        _check_split_column(manifest, split_artifact, report)
    _check_normalization_versions(normalization_versions, meta, report)
    return report


def _check_dataset_version(meta: dict[str, Any], report: ValidationReport) -> None:
    version = meta.get("dataset_version")
    if version is None:
        report.add_warning("meta is missing 'dataset_version'")
    elif str(version) != DATASET_VERSION:
        report.add_error(
            f"dataset_version {version!r} does not match the supported version {DATASET_VERSION!r}"
        )


def _check_required_columns(manifest: pd.DataFrame, report: ValidationReport) -> None:
    """Every column the v2 contract declares must be present, or be one that degrades.

    A manifest may carry only part of :data:`~allsky.data.contracts.TARGET_COLUMNS`
    and still be structurally sound row by row, which is why the missing ones used
    to pass unnoticed: the consumers degrade instead of failing.  Without
    ``target_kt`` the evaluator drops the ``kindex_band`` stratum from
    ``stratified.csv``; without ``kindex_kind`` the k-index clear-sky baseline is
    unresolvable and ``rmse_clearsky`` / ``skill_clearsky`` stay ``NaN``.

    Those are reported here, at validate-dataset time, where the column can still
    be rebuilt — but as warnings, because failing them would refuse every
    manifest built before
    :data:`~allsky.data.contracts.DEGRADABLE_TARGET_COLUMNS` joined the schema,
    including ones that train and evaluate today.
    """
    for column in (*META_COLUMNS, *TARGET_COLUMNS):
        if column in manifest.columns:
            continue
        if column in DEGRADABLE_TARGET_COLUMNS:
            report.add_warning(
                f"manifest predates column {column!r}; the consumers that read it degrade"
            )
        else:
            report.add_error(f"manifest is missing required column {column!r}")


def _check_duplicates(manifest: pd.DataFrame, report: ValidationReport) -> None:
    if "sample_id" in manifest.columns:
        dup = manifest["sample_id"].duplicated()
        if dup.any():
            offenders = manifest.loc[dup, "sample_id"].unique().tolist()
            report.add_error(f"duplicate sample_id: {offenders[:10]}")
    if "timestamp_utc" in manifest.columns:
        dup_ts = manifest["timestamp_utc"].duplicated()
        if dup_ts.any():
            report.add_error(f"duplicate timestamp_utc for {int(dup_ts.sum())} row(s)")


def _check_timezone(manifest: pd.DataFrame, report: ValidationReport) -> None:
    if "timestamp_utc" not in manifest.columns:
        return
    dtype = manifest["timestamp_utc"].dtype
    if not isinstance(dtype, pd.DatetimeTZDtype):
        report.add_error(
            f"timestamp_utc must be tz-aware, got dtype {dtype!r} "
            "(naive timestamps lose the UTC contract)"
        )


def _check_features_finite(
    manifest: pd.DataFrame, feature_columns: Sequence[str], report: ValidationReport
) -> None:
    present = [c for c in feature_columns if c in manifest.columns]
    if not present:
        return
    values = manifest.loc[:, present].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        bad = [c for c in present if not np.isfinite(manifest[c].to_numpy(dtype=np.float64)).all()]
        report.add_error(f"non-finite (NaN/inf) values in feature columns: {bad}")


def _check_forbidden_columns(
    manifest: pd.DataFrame, feature_columns: Sequence[str], report: ValidationReport
) -> None:
    present_forbidden = sorted(FORBIDDEN_FEATURES & set(manifest.columns))
    if present_forbidden:
        report.add_error(
            f"forbidden radiometry/target columns present in manifest: {present_forbidden}"
        )
    leaky_features = sorted(
        c for c in feature_columns if c in FORBIDDEN_FEATURES or c.startswith("target_")
    )
    if leaky_features:
        report.add_error(f"declared feature columns are leakage-prone: {leaky_features}")


def _check_elevation(
    manifest: pd.DataFrame,
    min_elevation_deg: float,
    report: ValidationReport,
    *,
    strict: bool,
) -> None:
    if "solar_elevation" not in manifest.columns:
        return
    elevation = manifest["solar_elevation"].to_numpy(dtype=np.float64)
    below = elevation < min_elevation_deg
    if below.any():
        message = (
            f"{int(below.sum())} row(s) below the elevation floor "
            f"{min_elevation_deg} deg (night/below-horizon frames)"
        )
        report.add_error(message)
    if not strict:
        return
    low_sun = (elevation >= min_elevation_deg) & (elevation < STRICT_LOW_SUN_ELEVATION_DEG)
    if low_sun.any():
        report.add_error(
            f"strict: {int(low_sun.sum())} low-sun row(s) "
            f"(elevation < {STRICT_LOW_SUN_ELEVATION_DEG} deg)"
        )


def _check_targets(manifest: pd.DataFrame, max_kindex: float, report: ValidationReport) -> None:
    if "target_dhi" in manifest.columns:
        dhi = manifest["target_dhi"].to_numpy(dtype=np.float64)
        negative = np.isfinite(dhi) & (dhi < 0.0)
        if negative.any():
            report.add_error(f"{int(negative.sum())} row(s) with negative target_dhi")
    if "target_kindex" in manifest.columns:
        kindex = manifest["target_kindex"].to_numpy(dtype=np.float64)
        out_of_range = np.isfinite(kindex) & ((kindex < 0.0) | (kindex > max_kindex))
        if out_of_range.any():
            report.add_error(
                f"{int(out_of_range.sum())} row(s) with target_kindex outside [0, {max_kindex}]"
            )
    if "sky_class" in manifest.columns:
        allowed = {SKY_CLASS_MISSING, *SKY_CLASS_VALUES}
        classes = manifest["sky_class"].to_numpy()
        invalid = ~np.isin(classes, list(allowed))
        if invalid.any():
            offenders = sorted({int(v) for v in classes[invalid]})
            report.add_error(f"sky_class values outside {sorted(allowed)} present: {offenders}")


def _check_qc_flags(manifest: pd.DataFrame, report: ValidationReport) -> None:
    """``qc_flags`` must be a bitmask this build can read.

    The column is an int64 bitmask over :class:`~allsky.data.contracts.QCFlag`,
    and nothing else validated it: a negative value (an unsigned mask written
    through a signed column) or a bit no member declares reaches the evaluator's
    QC stratification, which only asks whether the mask is zero, and lands the
    row in whichever stratum the arithmetic happens to give it.
    """
    if "qc_flags" not in manifest.columns:
        return
    flags = manifest["qc_flags"].dropna()
    if flags.empty:
        return
    values = flags.to_numpy(dtype=np.int64)
    declared = 0
    for member in QCFlag:
        declared |= int(member)
    offending = (values < 0) | ((values & ~declared) != 0)
    if bool(offending.any()):
        report.add_error(
            f"{int(offending.sum())} row(s) carry a qc_flags value outside the declared "
            f"QCFlag bits (0..{declared}): {sorted({int(v) for v in values[offending]})[:5]}"
        )


def _check_image_files(
    manifest: pd.DataFrame, data_root: str | Path, report: ValidationReport
) -> None:
    if "image_path" not in manifest.columns:
        return
    missing: list[str] = []
    for rel in manifest["image_path"]:
        try:
            full = resolve(str(rel), data_root)
        except ValueError as exc:
            report.add_error(str(exc))
            continue
        if not full.exists():
            missing.append(str(rel))
    if missing:
        report.add_error(
            f"{len(missing)} image file(s) missing under {str(data_root)!r}; first: {missing[:5]}"
        )


def _day_to_splits(split_artifact: dict[str, Any]) -> dict[str, set[str]]:
    """Map ``day_id -> set of splits`` from either the assignment or splits form."""
    day_to_splits: dict[str, set[str]] = {}
    assignment = split_artifact.get("assignment")
    if isinstance(assignment, dict):
        for day, split in assignment.items():
            day_to_splits.setdefault(str(day), set()).add(str(split))
    splits = split_artifact.get("splits")
    if isinstance(splits, dict):
        for split, days in splits.items():
            for day in days:
                day_to_splits.setdefault(str(day), set()).add(str(split))
    return day_to_splits


def _check_split_leakage(split_artifact: dict[str, Any], report: ValidationReport) -> None:
    """Flag any day_id assigned to more than one split."""
    leaked = {day: sorted(s) for day, s in _day_to_splits(split_artifact).items() if len(s) > 1}
    if leaked:
        report.add_error(f"split leakage: day_id assigned to multiple splits: {leaked}")


def _check_split_column(
    manifest: pd.DataFrame, split_artifact: dict[str, Any], report: ValidationReport
) -> None:
    """A filled ``split`` column must agree with the split artifact by ``day_id``."""
    if SPLIT_COLUMN not in manifest.columns or "day_id" not in manifest.columns:
        return
    filled = manifest[SPLIT_COLUMN].notna()
    if not bool(filled.any()):
        # A manifest handed a non-empty artifact is meant to carry the labels;
        # entirely unfilled means attach_split_column never ran (or ran against
        # another artifact), and every consumer then silently trains on rows
        # whose split nothing assigned.
        if _day_to_splits(split_artifact):
            report.add_warning(
                "a split artifact was supplied but the manifest's split column is entirely "
                "unfilled; run the splits step, or attach_split_column, before training"
            )
        return
    day_map = {day: min(s) for day, s in _day_to_splits(split_artifact).items()}
    sub = manifest.loc[filled]
    expected = sub["day_id"].astype(str).map(day_map)
    actual = sub[SPLIT_COLUMN].astype(str)
    disagree = expected.notna() & (expected.to_numpy() != actual.to_numpy())
    if bool(disagree.any()):
        days = sorted({str(d) for d in sub.loc[disagree, "day_id"]})
        report.add_error(f"split column disagrees with the split artifact for day(s): {days[:10]}")


def _check_constant_provenance(
    manifest: pd.DataFrame, meta: dict[str, Any], report: ValidationReport
) -> None:
    """Row-constant provenance columns must be constant AND match the meta.

    ``kindex_kind`` and ``target_source`` are here for the same reason the other
    two are: the evaluator reads both from the meta while every metric it
    computes comes from the column, so a manifest whose column and sidecar
    disagree is scored under one parameterization and reported under another.
    """
    for column in ("dataset_version", "alignment_id", "kindex_kind", "target_source"):
        if column not in manifest.columns:
            continue
        values = manifest[column].dropna().unique().tolist()
        if len(values) > 1:
            report.add_error(
                f"{column} column is not constant across rows: {sorted(map(str, values))[:5]}"
            )
            continue
        if not values:
            continue
        actual = str(values[0])
        expected = meta.get(column)
        if expected is not None and str(expected) != actual:
            report.add_error(
                f"{column} column {actual!r} does not match meta {column} {str(expected)!r}"
            )


def _check_normalization_versions(
    normalization_versions: Iterable[str] | None,
    meta: dict[str, Any],
    report: ValidationReport,
) -> None:
    versions: set[str] = set()
    if normalization_versions is not None:
        versions.update(str(v) for v in normalization_versions)
    meta_norm = meta.get("normalization_version")
    if meta_norm is not None:
        versions.add(str(meta_norm))
    if len(versions) > 1:
        report.add_error(f"more than one normalization version in play: {sorted(versions)}")
