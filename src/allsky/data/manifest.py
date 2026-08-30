"""Build and persist the multimodal dataset manifest (v2 parquet + meta sidecar).

:func:`build_manifest` fuses a frame manifest (from
:func:`allsky.video.extract_frames`) with a Campbell sensor frame into the
portable v2 manifest pinned in :mod:`allsky.data.contracts`:

- identity: naive-local frame time -> tz-aware ``timestamp_utc`` (UTC, from the
  fixed UTC-3 America/Bahia clock), ``day_id`` (local calendar day) and
  ``sample_id`` (``allsky-YYYYMMDD-HHMM``, matching the frame filename stem);
- features: solar geometry + cyclic met encodings via
  :func:`allsky.features.build_feature_frame`, screened by the anti-leakage
  :func:`allsky.features.validate_features`;
- targets: ``target_dhi`` (measured diffuse column, or an Erbs pseudo-target
  flagged in ``target_source``), ``target_kindex`` (k\\* via Haurwitz or the
  clearness index k_t), a k-index-binned ``sky_class`` and an all-NaN
  ``cloud_fraction`` placeholder;
- ``qc_flags``: a :class:`~allsky.data.contracts.QCFlag` bitmask (low sun,
  sensor gap, far alignment, k-index artifact).

Diffuse/kt/Erbs derivation reuses the physics helpers in :mod:`allsky.erbs`,
:mod:`labmim_core.solar` and :mod:`allsky.clearsky`.  :func:`write_manifest_parquet`
writes the parquet and its ``<name>.meta.json`` sidecar atomically and records a
content ``manifest_sha256``.
"""

import json
import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.clearsky import clear_sky_index
from allsky.config import (
    SITE_TZ_NAME,
    SITE_UTC_OFFSET_HOURS,
    PrepareConfig,
    SiteConfig,
    manifest_meta_path,
)
from allsky.data.alignment import CenterFrame
from allsky.data.contracts import (
    DATASET_VERSION,
    GEOMETRY_COLUMNS,
    QCFlag,
    manifest_column_dtypes,
    to_relative,
)
from allsky.data.splits import DaySplit
from allsky.erbs import pseudo_diffuse
from allsky.features import build_feature_frame, resolve_feature_set, validate_features
from allsky.provenance import code_version, content_sha256
from labmim_core.atomic import atomic_write, atomic_write_json
from labmim_core.sky import (
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLASS_MISSING,
    SKY_CLASS_NAMES,
    SKY_CLASS_NAMES_PT,
    SKY_CLASS_REFERENCE,
    SKY_CLEAR,
    SKY_CLOUDY,
    SKY_PARTLY_CLOUDY_CLEAR,
    SKY_PARTLY_CLOUDY_DIFFUSE,
)
from labmim_core.solar import clearness_index, solar_azimuth_deg, solar_elevation_deg

logger = logging.getLogger(__name__)

__all__ = [
    "TARGET_SOURCE_ERBS",
    "TARGET_SOURCE_MEASURED",
    "attach_split_column",
    "build_manifest",
    "build_manifest_from_prepare_config",
    "write_manifest_parquet",
]

#: ``target_source`` values written by :func:`build_manifest`.
TARGET_SOURCE_MEASURED = "measured"
TARGET_SOURCE_ERBS = "erbs_pseudo"


def _timezone_meta(site: SiteConfig) -> dict[str, Any]:
    """Sidecar timezone block for *site*.

    The station keeps its named zone; any other site is recorded by its offset
    alone, because a fixed instrument offset is what this pipeline consumes and
    naming a zone it never resolved would be an assertion nobody checked.
    """
    if site.utc_offset_hours == SITE_UTC_OFFSET_HOURS:
        return {"name": SITE_TZ_NAME, "utc_offset_hours": SITE_UTC_OFFSET_HOURS}
    return {"name": None, "utc_offset_hours": float(site.utc_offset_hours)}


def build_manifest(
    frames_manifest: pd.DataFrame,
    sensor_df: pd.DataFrame,
    *,
    site: SiteConfig,
    data_root: str | Path,
    feature_set: str = "safe",
    ghi_column: str = "CM3Up_Wm2_Avg",
    diffuse_column: str | None = "PSP_Wm2_Avg",
    kindex_kind: str = "kstar",
    alignment: CenterFrame | None = None,
    min_elevation_deg: float = 10.0,
    night_min_elevation_deg: float | None = 5.0,
    max_kindex: float | None = None,
    far_distance_minutes: float | None = None,
    extra_features: Iterable[str] = (),
    sensor_timestamp_offset_minutes: float = 0.0,
    config_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the v2 manifest DataFrame and its sidecar meta dict.

    Parameters
    ----------
    frames_manifest:
        Frame manifest with columns ``frame_path``, ``timestamp`` (naive local),
        ``video`` and ``index`` (see :func:`allsky.video.extract_frames`).
    sensor_df:
        Time-indexed Campbell sensor frame carrying the raw logger columns the
        feature policy and targets need (met channels, ``ghi_column`` and the
        optional ``diffuse_column``).
    site:
        Observation site for solar geometry.
    data_root:
        Root the manifest's ``image_path`` values are made relative to.
    feature_set:
        ``"safe"`` (default) or ``"extended"``.
    ghi_column:
        Global-horizontal-irradiance column driving k-index and the Erbs
        pseudo-target.
    diffuse_column:
        Measured diffuse column for ``target_dhi``; ``None`` (or an absent
        column) falls back to Erbs pseudo-targets flagged in ``target_source``.
    kindex_kind:
        ``"kstar"`` (GHI over Haurwitz clear-sky GHI) or ``"kt"`` (clearness
        index).
    alignment:
        Build-time pairing strategy; must be a :class:`CenterFrame` (the
        default) — the windowed strategies pool at dataset level and have no
        ``pair`` method.
    min_elevation_deg:
        Elevation floor below which the k-index is undefined and ``LOW_SUN`` is
        flagged.  Rows here are *kept* (with ``LOW_SUN``) as long as they clear
        *night_min_elevation_deg*.
    night_min_elevation_deg:
        Night threshold: rows with ``solar_elevation`` below this are **dropped**
        before target derivation (a logged count).  ``None`` disables the drop.
        Must be ``<= min_elevation_deg`` for the ``LOW_SUN`` band (night .. floor)
        to be non-empty.
    max_kindex:
        k-index ceiling above which ``KT_ARTIFACT`` is flagged.  ``None`` (default)
        resolves per *kindex_kind*: ``1.2`` for ``"kt"`` and ``1.5`` for
        ``"kstar"`` — cloud enhancement over the Haurwitz clear-sky reference
        legitimately pushes k\\* past 1.2, so a looser ceiling avoids false flags.
    far_distance_minutes:
        Alignment distance above which ``ALIGNMENT_FAR`` is flagged; defaults to
        half the strategy's ``max_distance_minutes``.
    extra_features:
        Extra engineered feature names appended to the resolved set.
    sensor_timestamp_offset_minutes:
        Minutes added to every sensor stamp before pairing, in minutes. The
        Campbell logger end-stamps its block averages, so ``-2.5`` moves a
        5-minute row to the time centroid of the window it actually averages;
        ``0.0`` keeps the raw-stamp join. Shifts only which row a frame is
        paired to and the recorded ``distance_minutes`` — never the values.
    config_sha256:
        Optional hash of the originating config, stored in the meta.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The manifest (columns/dtypes per
        :func:`allsky.data.contracts.manifest_column_dtypes`, with the ``split``
        column left empty for :func:`attach_split_column` to fill) and the
        sidecar meta dict (``manifest_sha256`` is added by
        :func:`write_manifest_parquet`).

    Raises
    ------
    KeyError
        If ``ghi_column``, a configured ``diffuse_column`` or a required feature
        source column is missing from *sensor_df*.
    ValueError
        If *kindex_kind* is not ``"kstar"``/``"kt"``, the resolved feature set
        contains a forbidden (leakage-prone) column, or a configured radiometry
        column is identically zero over every paired row (a dead logger
        channel — see :func:`_reject_dead_channel`).
    """
    if kindex_kind not in ("kstar", "kt"):
        raise ValueError(f"kindex_kind must be 'kstar' or 'kt', got {kindex_kind!r}")

    if max_kindex is None:
        max_kindex = 1.5 if kindex_kind == "kstar" else 1.2

    strategy = alignment if alignment is not None else CenterFrame()
    if not isinstance(strategy, CenterFrame):
        raise TypeError(
            f"build-time alignment must be a CenterFrame, got {type(strategy).__name__}"
        )
    if far_distance_minutes is None:
        far_distance_minutes = strategy.max_distance_minutes / 2.0

    feature_columns = resolve_feature_set(feature_set, extra_features)
    target_source = TARGET_SOURCE_MEASURED if diffuse_column is not None else TARGET_SOURCE_ERBS
    target_columns = [ghi_column] if diffuse_column is None else [ghi_column, diffuse_column]
    validate_features(feature_columns, target_columns=target_columns)

    frames = frames_manifest.sort_values("timestamp").reset_index(drop=True)
    frames["timestamp"] = pd.to_datetime(frames["timestamp"]).astype("datetime64[ns]")
    dup_frames = frames["timestamp"].duplicated(keep="first")
    if dup_frames.any():
        logger.warning(
            "build_manifest: dropped %d duplicate frame timestamps", int(dup_frames.sum())
        )
        frames = frames.loc[~dup_frames].reset_index(drop=True)

    sensors = sensor_df.sort_index()
    sensors = sensors.loc[~sensors.index.duplicated(keep="first")]
    if ghi_column not in sensors.columns:
        raise KeyError(f"sensor frame is missing the GHI column {ghi_column!r}")
    if diffuse_column is not None and diffuse_column not in sensors.columns:
        raise KeyError(f"sensor frame is missing the configured diffuse column {diffuse_column!r}")

    frame_times = pd.DatetimeIndex(frames["timestamp"])
    sensor_index = pd.DatetimeIndex(sensors.index) + pd.Timedelta(
        minutes=sensor_timestamp_offset_minutes
    )

    pairing = strategy.pair(frame_times, sensor_index)
    keep = pairing.matched
    if not keep.any():
        raise ValueError(
            "no frame matched a sensor record within "
            f"{strategy.max_distance_minutes:.1f} min; check the time alignment"
        )
    frames = frames.loc[keep].reset_index(drop=True)
    frame_times = pd.DatetimeIndex(frames["timestamp"])
    matched_pos = pairing.sensor_pos[keep]
    distance_minutes = pairing.distance_minutes[keep]

    paired_sensor = sensors.iloc[matched_pos].copy()
    paired_sensor.index = frame_times

    features = build_feature_frame(
        paired_sensor,
        frame_times,
        site,
        feature_set,
        extra=extra_features,
        utc_offset_hours=float(site.utc_offset_hours),
    )
    finite = np.isfinite(features.to_numpy(dtype=np.float64)).all(axis=1)
    n_nonfinite = int((~finite).sum())
    if n_nonfinite:
        logger.info("build_manifest: dropped %d rows with non-finite features", n_nonfinite)
    frames = frames.loc[finite].reset_index(drop=True)
    frame_times = pd.DatetimeIndex(frames["timestamp"])
    matched_pos = matched_pos[finite]
    distance_minutes = distance_minutes[finite]
    paired_sensor = paired_sensor.loc[finite]
    features = features.loc[finite]
    if len(frames) == 0:
        raise ValueError("no rows survived the finite-feature filter; check sensor coverage")

    utc_offset = float(site.utc_offset_hours)
    elevation = solar_elevation_deg(frame_times, site, utc_offset)

    if night_min_elevation_deg is not None:
        day_mask = elevation >= float(night_min_elevation_deg)
        n_night = int((~day_mask).sum())
        if n_night:
            logger.info(
                "build_manifest: dropped %d night frame(s) below %.1f deg elevation",
                n_night,
                night_min_elevation_deg,
            )
            frames = frames.loc[day_mask].reset_index(drop=True)
            frame_times = pd.DatetimeIndex(frames["timestamp"])
            matched_pos = matched_pos[day_mask]
            distance_minutes = distance_minutes[day_mask]
            paired_sensor = paired_sensor.loc[day_mask]
            features = features.loc[day_mask]
            elevation = elevation[day_mask]
        if len(frames) == 0:
            raise ValueError(
                "no rows survived the night-elevation filter "
                f"(night_min_elevation_deg={night_min_elevation_deg}); check sun coverage"
            )

    azimuth = solar_azimuth_deg(frame_times, site, utc_offset)
    zenith = 90.0 - elevation

    ghi = paired_sensor[ghi_column].to_numpy(dtype=np.float64)
    _reject_dead_channel(ghi, ghi_column, "the live GHI channel is 'CM3Up_Wm2_Avg'")
    kt = clearness_index(ghi, frame_times, site, utc_offset)
    if kindex_kind == "kstar":
        kindex = clear_sky_index(ghi, frame_times, site, min_elevation_deg, utc_offset)
    else:
        kindex = np.asarray(kt, dtype=np.float64)

    if diffuse_column is not None:
        target_dhi = paired_sensor[diffuse_column].to_numpy(dtype=np.float64)
        _reject_dead_channel(
            target_dhi,
            diffuse_column,
            "the live diffuse sensor is 'PSP_Wm2_Avg' (pass diffuse_column=None "
            "for the Erbs pseudo-target)",
        )
    else:
        target_dhi = pseudo_diffuse(ghi, kt)

    sky_class = _classify_sky(kt, labelable=elevation >= min_elevation_deg)
    cloud_fraction = np.full(len(frames), np.nan, dtype=np.float64)

    qc_flags = _qc_flags(
        elevation=elevation,
        ghi=ghi,
        distance_minutes=distance_minutes,
        kindex=kindex,
        min_elevation_deg=min_elevation_deg,
        max_kindex=max_kindex,
        far_distance_minutes=far_distance_minutes,
    )

    sample_id = [f"allsky-{ts:%Y%m%d-%H%M}" for ts in frame_times]
    _check_sample_id_unique(sample_id)
    day_id = frame_times.strftime("%Y-%m-%d")
    site_tz = timezone(timedelta(hours=site.utc_offset_hours))
    timestamp_utc = frame_times.tz_localize(site_tz).tz_convert("UTC").as_unit("ns")
    image_path = [to_relative(path, data_root) for path in frames["frame_path"]]

    dtypes = manifest_column_dtypes(feature_columns)
    data: dict[str, Any] = {
        "sample_id": sample_id,
        "timestamp_utc": timestamp_utc,
        "day_id": list(day_id),
        "image_path": image_path,
        "frame_index": frames["index"].to_numpy(dtype=np.int64),
        "video": frames["video"].astype(str).tolist(),
        "solar_elevation": elevation,
        "solar_azimuth": azimuth,
        "solar_zenith": zenith,
    }
    for name in feature_columns:
        if name not in GEOMETRY_COLUMNS:
            data[name] = features[name].to_numpy(dtype=np.float64)
    n_rows = len(frames)
    data.update(
        {
            "target_dhi": target_dhi,
            "target_source": [target_source] * n_rows,
            "target_kindex": np.asarray(kindex, dtype=np.float64),
            "kindex_kind": [kindex_kind] * n_rows,
            "target_kt": np.asarray(kt, dtype=np.float64),
            "sky_class": sky_class,
            "cloud_fraction": cloud_fraction,
            "qc_flags": qc_flags,
            "dataset_version": [DATASET_VERSION] * n_rows,
            "alignment_id": [strategy.id] * n_rows,
            "split": [None] * n_rows,
        }
    )
    manifest = pd.DataFrame(data, columns=list(dtypes)).astype(dtypes)

    meta = _build_meta(
        alignment_id=strategy.id,
        feature_set=feature_set,
        feature_columns=feature_columns,
        kindex_kind=kindex_kind,
        target_source=target_source,
        config_sha256=config_sha256,
        row_count=len(manifest),
        site=site,
        thresholds={
            "min_elevation_deg": min_elevation_deg,
            "night_min_elevation_deg": night_min_elevation_deg,
            "max_kindex": max_kindex,
            "far_distance_minutes": far_distance_minutes,
            "sensor_timestamp_offset_minutes": sensor_timestamp_offset_minutes,
        },
    )
    logger.info(
        "build_manifest: %d samples, feature_set=%s, kindex=%s, source=%s",
        len(manifest),
        feature_set,
        kindex_kind,
        target_source,
    )
    return manifest, meta


def build_manifest_from_prepare_config(
    frames_manifest: pd.DataFrame,
    sensor_df: pd.DataFrame,
    cfg: PrepareConfig,
    *,
    data_root: str | Path | None = None,
    config_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a manifest from a :class:`~allsky.config.PrepareConfig`.

    Every build parameter is read from *cfg*: the feature set from
    ``cfg.features.feature_set`` (the ``features.set`` YAML key), the GHI column
    from ``cfg.sensor.ghi_column``, plus the site, alignment window, diffuse
    column, k-index kind, sky-class thresholds and the night drop threshold
    (``cfg.night_filter.min_solar_elevation_deg`` -> ``night_min_elevation_deg``).

    Parameters
    ----------
    frames_manifest, sensor_df:
        As in :func:`build_manifest`.
    cfg:
        Prepare configuration carrying every build parameter listed above.
    data_root:
        Directory the manifest's relative ``image_path`` values are resolved
        against; defaults to the config's output ``dataset_dir``.
    config_sha256:
        Optional hash of the originating config, stored in the meta.

    Returns
    -------
    tuple[pandas.DataFrame, dict]
        The manifest and sidecar meta produced by :func:`build_manifest`.
    """
    root = data_root if data_root is not None else cfg.output.dataset_dir
    alignment = CenterFrame(
        window_minutes=cfg.alignment.window_minutes,
        max_distance_minutes=cfg.sensor.tolerance_minutes,
    )
    return build_manifest(
        frames_manifest,
        sensor_df,
        site=cfg.site,
        data_root=root,
        feature_set=cfg.features.feature_set,
        ghi_column=cfg.sensor.ghi_column,
        diffuse_column=cfg.targets.diffuse_column,
        kindex_kind=cfg.targets.kindex_kind,
        alignment=alignment,
        night_min_elevation_deg=cfg.night_filter.min_solar_elevation_deg,
        sensor_timestamp_offset_minutes=cfg.sensor.timestamp_offset_minutes,
        config_sha256=config_sha256,
    )


def write_manifest_parquet(
    manifest: pd.DataFrame,
    meta: dict[str, Any],
    path: str | Path,
) -> dict[str, Any]:
    """Atomically write the manifest parquet and its ``<name>.meta.json`` sidecar.

    A content ``manifest_sha256`` (order-sensitive, parquet-container
    independent) is computed over the manifest and injected into the written
    meta.  Both files are written to a temp name in the same directory and
    ``os.replace``-d into place.

    Parameters
    ----------
    manifest:
        The manifest DataFrame to persist.
    meta:
        Sidecar meta as returned by :func:`build_manifest`; copied, never
        mutated in place.
    path:
        Destination parquet path; the sidecar is written beside it as
        ``<name>.meta.json``.

    Returns
    -------
    dict
        The meta actually written, with ``manifest_sha256`` and ``row_count``
        populated.
    """
    out = Path(path)

    sha = content_sha256(manifest)
    written_meta = {**meta, "manifest_sha256": sha, "row_count": len(manifest)}

    atomic_write(out, lambda tmp: manifest.to_parquet(tmp, index=False))
    atomic_write_json(manifest_meta_path(out), written_meta)

    logger.info(
        "write_manifest_parquet: wrote %s (%d rows, sha256=%s)", out, len(manifest), sha[:12]
    )
    return written_meta


def attach_split_column(
    manifest_path: str | Path,
    split_artifact: DaySplit | dict[str, Any],
) -> dict[str, Any]:
    """Fill the manifest's ``split`` column from a day-level split, atomically.

    The manifest parquet at *manifest_path* is read, its ``split`` column filled
    by joining each row's ``day_id`` to the split assignment (rows whose day is
    absent from the artifact stay ``pd.NA``), the resolved ``split_id`` recorded
    in the sidecar meta, and both files re-written atomically via
    :func:`write_manifest_parquet`.

    .. note::
       Filling ``split`` changes the manifest content, so the re-written
       ``manifest_sha256`` **differs** from the pre-attach hash.  This is by
       design: the hash tracks the exact bytes a checkpoint trained on, and the
       split label is part of those bytes once attached.

    Parameters
    ----------
    manifest_path:
        Path to the manifest parquet (its ``<name>.meta.json`` sidecar is updated
        alongside).
    split_artifact:
        A :class:`allsky.data.splits.DaySplit` or its dict form (``assignment``
        day_id -> split, plus ``split_id``).

    Returns
    -------
    dict
        The re-written meta (new ``manifest_sha256`` + ``split_id``).
    """
    out = Path(manifest_path)
    manifest = pd.read_parquet(out)
    meta_path = manifest_meta_path(out)
    meta: dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)

    assignment, split_id = _split_assignment_and_id(split_artifact)
    day_ids = manifest["day_id"].astype("string")
    manifest = manifest.copy()
    manifest["split"] = day_ids.map(assignment).astype("string")

    meta = {**meta, "split_id": split_id}
    written = write_manifest_parquet(manifest, meta, out)
    logger.info(
        "attach_split_column: filled %d split label(s) (split_id=%s); manifest_sha256 changed",
        int(manifest["split"].notna().sum()),
        str(split_id)[:12],
    )
    return written


def _split_assignment_and_id(
    split_artifact: DaySplit | dict[str, Any],
) -> tuple[Mapping[str, str], str | None]:
    """Extract the ``day_id -> split`` map and ``split_id`` from either form.

    A :class:`~allsky.data.splits.DaySplit` is recognised by its ``assignment``
    attribute; the dict form carries the same two keys.
    """
    assignment = getattr(split_artifact, "assignment", None)
    if assignment is not None:
        split_id = getattr(split_artifact, "split_id", None)
        return {str(k): str(v) for k, v in assignment.items()}, split_id
    if isinstance(split_artifact, dict):
        raw = split_artifact.get("assignment", {})
        return {str(k): str(v) for k, v in raw.items()}, split_artifact.get("split_id")
    raise TypeError(
        f"split_artifact must be a DaySplit or dict, got {type(split_artifact).__name__}"
    )


def _check_sample_id_unique(sample_ids: list[str]) -> None:
    """Raise if minute-resolution ``sample_id`` values collide (naming the minute)."""
    index = pd.Index(sample_ids)
    duplicated = index[index.duplicated(keep=False)]
    if len(duplicated) == 0:
        return
    colliding = sorted(set(duplicated))
    shown = ", ".join(colliding[:10]) + (" ..." if len(colliding) > 10 else "")
    raise ValueError(
        f"duplicate sample_id after minute-resolution binning: {shown}. "
        "sample_id is 'allsky-YYYYMMDD-HHMM' (minute cadence), so two frames in the "
        "same minute collide. Space frames >= 1 min apart, or extend sample_id to "
        "sub-minute resolution before building the manifest."
    )


def _reject_dead_channel(values: np.ndarray, column: str, remedy: str) -> None:
    """Refuse a radiometry channel that is identically zero over every paired row.

    An all-zero irradiance channel is the CR5000 dead-input signature (the CMP21
    diffuse channel logs exactly this).  Accepted as a target it trains a model
    to predict 0 and then reports a near-perfect MAE against it, labelled
    ``target_source="measured"`` and indistinguishable in ``metrics.json`` from
    a real run — so the build refuses it here, the only place that knows which
    logger column was configured.

    Only an *identically zero* channel is refused: an all-NaN channel is a
    sensor gap, which ``QCFlag.SENSOR_GAP`` and the missing-label machinery
    already handle, so it passes through untouched.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0 or np.any(finite != 0.0):
        return
    raise ValueError(
        f"sensor column {column!r} is identically zero across all {finite.size} "
        f"paired rows with a reading (dead logger channel); {remedy}"
    )


def _classify_sky(kt: np.ndarray, *, labelable: np.ndarray) -> np.ndarray:
    """Label each sample with its published sky condition from the clearness index.

    Parameters
    ----------
    kt:
        Clearness index Kt (GHI over extraterrestrial horizontal irradiance),
        shape ``(N,)``, dimensionless.  Non-finite entries are unlabelable.
    labelable:
        Boolean mask, shape ``(N,)``, false where the sun is too low for Kt to
        mean anything (its denominator collapses towards zero near the
        horizon).  Those rows are labelled :data:`SKY_CLASS_MISSING` rather than
        binned, so a horizon artefact never enters training as a real class.

    Returns
    -------
    numpy.ndarray
        Class integers, shape ``(N,)``, dtype ``int64``: ``0`` cloudy, ``1``
        partly cloudy with diffuse dominance, ``2`` partly cloudy with clear
        dominance, ``3`` clear, and :data:`SKY_CLASS_MISSING` where Kt is not
        finite.  The integer is the published condition number minus one.

    Notes
    -----
    The bins are the four sky conditions of Escobedo et al. (2009), applied to
    Kt and not to the clear-sky index: the published bounds are defined on the
    clearness index, so classifying k\\* against them would relabel the record.
    """
    values = np.asarray(kt, dtype=np.float64)
    cloudy, diffuse_dominant, clear_dominant = SKY_CLASS_KT_UPPER_BOUNDS
    labels = np.select(
        [values <= cloudy, values <= diffuse_dominant, values <= clear_dominant],
        [SKY_CLOUDY, SKY_PARTLY_CLOUDY_DIFFUSE, SKY_PARTLY_CLOUDY_CLEAR],
        default=SKY_CLEAR,
    ).astype(np.int64)
    labels[~np.isfinite(values) | ~np.asarray(labelable, dtype=bool)] = SKY_CLASS_MISSING
    return labels


def _qc_flags(
    *,
    elevation: np.ndarray,
    ghi: np.ndarray,
    distance_minutes: np.ndarray,
    kindex: np.ndarray,
    min_elevation_deg: float,
    max_kindex: float,
    far_distance_minutes: float,
) -> np.ndarray:
    """Assemble the per-row :class:`QCFlag` bitmask as int64."""
    n = len(elevation)
    flags = np.zeros(n, dtype=np.int64)
    flags[elevation < min_elevation_deg] |= int(QCFlag.LOW_SUN)
    flags[~np.isfinite(ghi)] |= int(QCFlag.SENSOR_GAP)
    far = np.isfinite(distance_minutes) & (distance_minutes > far_distance_minutes)
    flags[far] |= int(QCFlag.ALIGNMENT_FAR)
    artifact = np.isfinite(kindex) & (kindex > max_kindex)
    flags[artifact] |= int(QCFlag.KT_ARTIFACT)
    return flags


def _build_meta(
    *,
    alignment_id: str,
    feature_set: str,
    feature_columns: list[str],
    kindex_kind: str,
    target_source: str,
    config_sha256: str | None,
    row_count: int,
    site: SiteConfig,
    thresholds: dict[str, float | None],
) -> dict[str, Any]:
    """Assemble the sidecar meta dict (``manifest_sha256`` added on write)."""
    return {
        "dataset_version": DATASET_VERSION,
        "alignment_id": alignment_id,
        "feature_set": feature_set,
        "feature_columns": list(feature_columns),
        "kindex_kind": kindex_kind,
        "sky_classes": {
            "names": list(SKY_CLASS_NAMES),
            "names_pt": list(SKY_CLASS_NAMES_PT),
            "kt_upper_bounds": list(SKY_CLASS_KT_UPPER_BOUNDS),
            "reference": SKY_CLASS_REFERENCE,
        },
        "target_source": target_source,
        "config_sha256": config_sha256,
        "code_version": code_version(),
        "created_at": datetime.now(UTC).isoformat(),
        "row_count": row_count,
        "timezone": _timezone_meta(site),
        "site": {"latitude": site.latitude, "longitude": site.longitude},
        "thresholds": thresholds,
        "manifest_sha256": None,
    }
