"""Captures the camera's live frame and runs a trained checkpoint over it.

Timestamps here are **naive local time** on the camera's own clock, matching the
overlay and datalogger pipelines: :func:`capture_snapshot` prefers the stamp
burned into the frame, falls back to the server's ``Last-Modified`` header
converted through :data:`allsky.config.SITE_TZ`, and only then to the host
clock — recording which of the three it used in the JSON sidecar.

Which feature columns a live frame cannot supply, and what imputing them costs,
are documented in ``docs/allsky-archive.md``.
"""

import datetime as dt
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeIs, runtime_checkable

import numpy as np
import pandas as pd

from allsky.config import (
    SITE_TZ,
    ExperimentConfig,
    PrepareConfig,
    SiteConfig,
    geometry_channels_of,
    image_size_of,
)
from allsky.data.blocks import block_end_of, nearest_to_centroid
from allsky.embeddings.backbone import VisualBackbone
from allsky.frame_pixels import decode_rgb
from allsky.provenance import code_version
from allsky.training.checkpointing import normalizers_from_checkpoint
from labmim_core.atomic import atomic_write, atomic_write_json

if TYPE_CHECKING:
    from micrometeorology.common.config import SensorRangeLimit

logger = logging.getLogger(__name__)

__all__ = [
    "LiveFrameSource",
    "ScalarFeatures",
    "ServedInput",
    "ServedModel",
    "Snapshot",
    "SolarElevationBelowFloorError",
    "StationExport",
    "block_end_of",
    "capture_snapshot",
    "clearsky_dhi_at",
    "clearsky_dhi_series",
    "load_served_model",
    "predict_block",
    "predict_snapshot",
    "read_station_export",
    "shipped_sensor_limits",
    "solar_elevation_at",
]

SNAPSHOT_STEM_FORMAT = "allsky-%Y%m%d-%H%M%S"
#: Pairing window used when the checkpoint records none of its own. Wider than
#: any training default on purpose: a checkpoint written before ``sensor_pairing``
#: existed says nothing about how it paired, so the fallback keeps the behaviour
#: those checkpoints were served under rather than inventing a tighter one.
DEFAULT_SENSOR_TOLERANCE = pd.Timedelta(minutes=15)
SENSOR_TIME_COLUMNS = ("timestamp", "TIMESTAMP", "datetime", "time")
#: What a checkpoint is served on: one capture, or the frames of one datalogger block.
ServedInput = Literal["frame", "block"]
LIVE_FRAME_MAX_AGE = pd.Timedelta(minutes=10)
STORE_RECIPE_KEYS = ("backbone", "pooling", "revision", "dim", "dtype")
EMBEDDING_STORE_DTYPES = ("fp16", "fp32")


@runtime_checkable
class LiveFrameSource(Protocol):
    """The part of :class:`allsky.archive.ArchiveClient` a capture actually uses."""

    base_url: str

    def fetch_live_image(self) -> tuple[bytes, dict[str, str]]:
        """Return the camera's current frame and the response headers."""


@dataclass(frozen=True)
class Snapshot:
    """One captured live frame and its provenance sidecar.

    Attributes
    ----------
    image_path:
        JPEG written as ``allsky-YYYYMMDD-HHMMSS.jpg``, named from
        *captured_at*.
    metadata_path:
        JSON sidecar beside it, recording the capture-time source, the response
        headers and the code version.
    captured_at:
        Naive local capture time.
    """

    image_path: Path
    metadata_path: Path
    captured_at: pd.Timestamp


def _site_now() -> pd.Timestamp:
    """Current time on the camera's own clock, as a naive local timestamp."""
    return pd.Timestamp(dt.datetime.now(tz=SITE_TZ).replace(tzinfo=None)).floor("s")


def _naive_site_time_from_http_date(headers: dict[str, str]) -> pd.Timestamp | None:
    raw = headers.get("last-modified")
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except ValueError:
        logger.warning("unparseable Last-Modified header on the live frame: %r", raw)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return pd.Timestamp(parsed.astimezone(SITE_TZ).replace(tzinfo=None))


def _overlay_timestamp(payload: bytes) -> tuple[pd.Timestamp | None, str | None]:
    from allsky.overlay import read_frame_timestamp

    try:
        reading = read_frame_timestamp(decode_rgb(payload))
    except (OSError, ValueError) as exc:
        logger.warning("could not read the timestamp overlay off the live frame: %s", exc)
        return None, None
    return (pd.Timestamp(reading.timestamp) if reading.timestamp else None), reading.text


def _fresh(candidate: pd.Timestamp | None, now: pd.Timestamp) -> TypeIs[pd.Timestamp]:
    return candidate is not None and abs(candidate - now) <= LIVE_FRAME_MAX_AGE


def capture_snapshot(
    client: LiveFrameSource, out_dir: str | Path, *, timestamp: pd.Timestamp | None = None
) -> Snapshot:
    """Fetch the current frame into *out_dir* alongside a JSON provenance sidecar.

    Parameters
    ----------
    client:
        Anything exposing the camera's live-frame endpoint (in practice an
        :class:`allsky.archive.ArchiveClient`).
    out_dir:
        Directory the JPEG and its sidecar are written into, atomically.
    timestamp:
        Naive local capture time to use verbatim, bypassing the overlay and
        header probes. Left None the capture time is taken from the frame's own
        overlay when it is within :data:`LIVE_FRAME_MAX_AGE` of now, else from
        the server's ``Last-Modified`` under the same freshness test, else from
        the local clock with a warning.

    Returns
    -------
    Snapshot
        Written paths, the naive local capture time and the payload size. The
        ``prediction`` field is always None here.

    Raises
    ------
    ValueError
        If the camera returns an empty payload.
    """
    payload, headers = client.fetch_live_image()
    if not payload:
        raise ValueError("the camera returned an empty live frame")

    now = _site_now()
    overlay_time, overlay_text = _overlay_timestamp(payload)
    server_time = _naive_site_time_from_http_date(headers)
    if timestamp is not None:
        captured, source = timestamp, "argument"
    elif _fresh(overlay_time, now):
        captured, source = overlay_time, "overlay"
    elif _fresh(server_time, now):
        captured, source = server_time, "server-last-modified"
    else:
        logger.warning(
            "neither the frame overlay (%s) nor Last-Modified (%s) is within %s of now — "
            "naming this snapshot from the local clock",
            overlay_time,
            server_time,
            LIVE_FRAME_MAX_AGE,
        )
        captured, source = now, "local-clock"

    directory = Path(out_dir)
    stem = f"{captured:{SNAPSHOT_STEM_FORMAT}}"
    image_path = directory / f"{stem}.jpg"
    atomic_write(image_path, lambda tmp: tmp.write_bytes(payload))

    metadata: dict[str, Any] = {
        "image": image_path.name,
        "captured_at": captured.isoformat(),
        "captured_at_source": source,
        "source_url": f"{client.base_url}image.jpg",
        "bytes": len(payload),
        "content_type": headers.get("content-type"),
        "server_last_modified": headers.get("last-modified"),
        "server_last_modified_as_local": server_time.isoformat() if server_time else None,
        "overlay_stamp": overlay_text,
        "fetched_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "code_version": code_version(),
    }
    metadata_path = atomic_write_json(directory / f"{stem}.json", metadata)
    logger.info("captured live frame %s (%.2f MiB)", image_path.name, len(payload) / (1 << 20))
    return Snapshot(
        image_path=image_path,
        metadata_path=metadata_path,
        captured_at=captured,
    )


def _screened_for_plausibility(
    row: pd.DataFrame, sensor_limits: list[SensorRangeLimit]
) -> pd.DataFrame:
    """Narrow *row* to the columns with a declared plausibility gate, NaN-ing what fails it.

    Parameters
    ----------
    row:
        One-row frame taken from the operator's station export, carrying that
        export's own published units — degC, %, mbar, m s-1, degrees, W m-2 —
        at whatever averaging interval it was written on.
    sensor_limits:
        The declared gates, passed in rather than read off the process-wide
        settings: this is a domain module, and a caller scoring against a
        different station's limits must not have to mutate a global to do it.

    Returns
    -------
    pandas.DataFrame
        The row narrowed to the columns ``sensor_limits`` declares, every
        sample outside its gate replaced by ``NaN``. A column with no declared
        gate is dropped rather than served: :func:`_feature_vector` then imputes
        it at the training mean and names it in ``features.imputed``.

    Raises
    ------
    ValueError
        If the shipped configuration declares no ``sensor_limits``, leaving
        nothing to screen a live reading against.
    """
    from micrometeorology.sensors.ingestion import apply_physical_limits

    limits = sensor_limits
    if not limits:
        raise ValueError(
            "the shipped configuration declares no sensor_limits, so a live sensor reading "
            "cannot be screened for plausibility; restore configs/micromet/default.yaml or "
            "predict without --sensor-csv, which imputes every sensor feature"
        )
    # apply_physical_limits declares its bounds in the logger's raw units and runs
    # ahead of calibration in the archive build. They hold for the calibrated
    # export too because configs/micromet/calibrations.yaml factors only
    # CMP21_Wm2_Avg, CM3Up_Wm2_Avg, PSP1_Wm2_Avg and PSP_Wm2_Avg — broadband
    # radiometry, every one of them a FORBIDDEN_FEATURES column no feature set
    # can read. Raw and calibrated coincide for the channels served here.
    declared = [limit.column for limit in limits if limit.column in row.columns]
    ungated = [
        column
        for column in row.columns
        if column not in declared and bool(row[column].notna().to_numpy()[0])
    ]
    if ungated:
        logger.warning(
            "the sensor export measures %s, which sensor_limits declares no gate for — "
            "imputing them at the training mean rather than serving unscreened readings; "
            "declare their bounds in configs/micromet/default.yaml to use them",
            ", ".join(str(column) for column in ungated),
        )
    measured = row.loc[:, declared].copy()
    was_measured = measured.notna().to_numpy()[0]
    screened = apply_physical_limits(measured, limits)
    refused = [
        column
        for column, before, after in zip(
            declared, was_measured, screened.notna().to_numpy()[0], strict=True
        )
        if before and not after
    ]
    if refused:
        logger.warning(
            "sensor export reads outside the declared plausible range on %s — "
            "imputing those instead of serving them as measurements",
            ", ".join(refused),
        )
    return screened


@dataclass(frozen=True, slots=True)
class StationExport:
    """The operator's station export, parsed once and shared by every reader of one publish.

    Attributes
    ----------
    path:
        The CSV it was read from.
    rows:
        Its rows indexed by naive station-local time, sorted, the unparsable
        stamps dropped. The values are the export's own published units, not
        yet screened: each reader screens the rows it uses.
    """

    path: Path
    rows: pd.DataFrame


def read_station_export(sensor_csv: str | Path) -> StationExport:
    """Read a station export on the two contracts the training path holds it to.

    Its clock is the logger's, i.e. naive site-local, so an export that does
    carry an offset is converted into that zone rather than merely stripped of
    it; a stamp that does not parse drops its row.

    Raises
    ------
    ValueError
        When no column of :data:`SENSOR_TIME_COLUMNS` is present.
    OSError
        When the file cannot be read.
    """
    frame = pd.read_csv(sensor_csv)
    time_column = next((name for name in SENSOR_TIME_COLUMNS if name in frame.columns), None)
    if time_column is None:
        raise ValueError(
            f"{sensor_csv} has no recognisable time column "
            f"(expected one of: {', '.join(SENSOR_TIME_COLUMNS)})"
        )
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
    frame = frame.dropna(subset=[time_column]).set_index(time_column).sort_index()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(SITE_TZ).tz_localize(None)
    frame.index = index
    return StationExport(path=Path(sensor_csv), rows=frame)


def _station_export(sensor_csv: str | Path | StationExport | None) -> StationExport | None:
    if sensor_csv is None or isinstance(sensor_csv, StationExport):
        return sensor_csv
    return read_station_export(sensor_csv)


def _sensor_row_near(
    export: StationExport,
    timestamp: pd.Timestamp,
    tolerance: pd.Timedelta,
    sensor_limits: list[SensorRangeLimit],
    timestamp_offset_minutes: float = 0.0,
) -> tuple[pd.DataFrame, float | None]:
    """Row of *export* nearest *timestamp*, relabelled to it, or an empty frame.

    *timestamp_offset_minutes* is the shift the manifest builder applied to the
    station index before pairing — ``-2.5`` in production, because the CR5000
    end-stamps its five-minute averages. Applied here with the same sign, so a
    live prediction pairs against the same instant training did.

    The chosen row is screened by :func:`_screened_for_plausibility`, which
    assumes the published physical units of the processed station export the
    snapshot command documents — not the logger's raw pre-calibration values,
    and not a 5-minute grid: an hour that railed for part of its samples
    averages to a finite number no sentinel literal matches, and would
    otherwise pass the ``np.isfinite`` screen in :func:`_feature_vector` and be
    served as a measurement.
    """
    frame = export.rows
    if frame.empty:
        return frame, None
    paired_index = pd.DatetimeIndex(frame.index) + pd.Timedelta(minutes=timestamp_offset_minutes)
    position = int(paired_index.get_indexer(pd.DatetimeIndex([timestamp]), method="nearest")[0])
    if position < 0:
        return frame.iloc[0:0], None
    gap = abs(pd.Timestamp(paired_index[position]) - timestamp)
    gap_minutes = gap.total_seconds() / 60.0
    if gap > tolerance:
        logger.warning(
            "nearest sensor row is %s from the frame (tolerance %s) — imputing instead",
            gap,
            tolerance,
        )
        return frame.iloc[0:0], gap_minutes
    # Screened after the row is chosen, not before: every gate is row-wise, so
    # the outcome is identical, and the nearest-row lookup reads the index only.
    row = _screened_for_plausibility(frame.iloc[[position]], sensor_limits)
    row.index = pd.DatetimeIndex([timestamp])
    return row, gap_minutes


def _with_absent_sources_as_nan(sensor: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Add the *feature_set*'s missing source columns to *sensor*, filled with NaN.

    ``build_feature_frame`` raises on a source column it cannot find, which is
    what a dataset build needs — a silently NaN feature column has no place in a
    manifest.  A snapshot is the opposite case: it is documented to impute every
    feature it cannot measure at the training mean, and reaches that path only
    through the NaN this fills in.  Without it a capture with no ``--sensor-csv``
    at all, the very case the fallback exists for, died on a ``KeyError``.
    """
    from allsky.features.policy import resolve_feature_set, source_column

    required = {
        column
        for column in (source_column(name) for name in resolve_feature_set(feature_set))
        if column is not None
    }
    absent = sorted(required - set(sensor.columns))
    if not absent:
        return sensor
    filled = sensor.copy()
    for column in absent:
        filled[column] = np.nan
    return filled


@dataclass(frozen=True, slots=True)
class _SensorPairing:
    """How a live capture is paired with a station row.

    Attributes
    ----------
    tolerance:
        Largest gap still accepted between the frame and the paired row.
    timestamp_offset_minutes:
        Shift applied to the station index before the nearest-row lookup, the
        same one the manifest builder applied.
    from_checkpoint:
        Whether both came from the run's own provenance rather than the
        module defaults.
    """

    tolerance: pd.Timedelta
    timestamp_offset_minutes: float
    from_checkpoint: bool


def _pairing_of(checkpoint: dict[str, Any], override: pd.Timedelta | None) -> _SensorPairing:
    """Resolve the pairing rule for one prediction.

    ``ExperimentConfig`` carries no ``sensor`` section, so both numbers come from
    the checkpoint's own ``sensor_pairing``. An explicit *override* still wins —
    it is the operator's own instruction.
    """
    recorded = checkpoint.get("sensor_pairing") or {}
    raw_tolerance = recorded.get("tolerance_minutes")
    tolerance_minutes = (
        float(raw_tolerance)
        if raw_tolerance is not None and np.isfinite(float(raw_tolerance))
        else None
    )
    offset = float(recorded.get("timestamp_offset_minutes") or 0.0)
    if override is not None:
        return _SensorPairing(override, offset, from_checkpoint=False)
    if tolerance_minutes is None:
        logger.warning(
            "this checkpoint records no sensor pairing; pairing within %s and applying "
            "no timestamp offset, which is not necessarily how it trained",
            DEFAULT_SENSOR_TOLERANCE,
        )
        return _SensorPairing(DEFAULT_SENSOR_TOLERANCE, offset, from_checkpoint=False)
    return _SensorPairing(pd.Timedelta(minutes=tolerance_minutes), offset, from_checkpoint=True)


def _feature_vector(
    timestamp: pd.Timestamp,
    *,
    feature_columns: list[str],
    feature_set: str,
    site: SiteConfig,
    export: StationExport | None,
    tolerance: pd.Timedelta,
    training_means: np.ndarray,
    sensor_limits: list[SensorRangeLimit],
    timestamp_offset_minutes: float = 0.0,
) -> tuple[np.ndarray, list[str], float | None]:
    from allsky.features.engineering import build_feature_frame

    sensor, gap_minutes = (
        _sensor_row_near(export, timestamp, tolerance, sensor_limits, timestamp_offset_minutes)
        if export is not None
        else (pd.DataFrame(index=pd.DatetimeIndex([])), None)
    )
    engineered = build_feature_frame(
        _with_absent_sources_as_nan(sensor, feature_set),
        [timestamp],
        site,
        feature_set,
        utc_offset_hours=float(site.utc_offset_hours),
    )
    unknown = [name for name in feature_columns if name not in engineered.columns]
    if unknown:
        raise ValueError(
            f"the checkpoint expects feature column(s) {unknown} that feature set "
            f"{feature_set!r} does not produce"
        )
    values = engineered.loc[:, feature_columns].to_numpy(dtype=np.float32)[0]
    finite = np.isfinite(values)
    imputed = [name for name, ok in zip(feature_columns, finite, strict=True) if not ok]
    return np.where(finite, values, training_means).astype(np.float32), imputed, gap_minutes


def _image_as_hwc(image_path: str | Path) -> np.ndarray:
    """Read a frame as ``(H, W, 3)`` ``uint8`` RGB — what a visual backbone's transform takes."""
    return decode_rgb(image_path)


def _frame_geometry(checkpoint: dict[str, Any]) -> PrepareConfig | None:
    """The prepare geometry this checkpoint's frames were written through.

    ``None`` when the run's manifest predates the sidecar recording it, in which
    case the live frame is scored as decoded — which is what every checkpoint
    got before, and is wrong for any dataset built with a crop or a pad, so it
    warns rather than passing silently.

    Raises
    ------
    ValueError
        When the recorded geometry names a static mask this machine cannot read:
        the mask zeroes pixels, so scoring without it is a different image.
    """
    recorded = checkpoint.get("frame_geometry")
    if not recorded:
        logger.warning(
            "this checkpoint records no frame geometry; the live frame is scored as "
            "decoded, which is not what the model saw if its dataset was built with a "
            "mask, crop or pad (re-run prepare-local to record it)"
        )
        return None
    geometry = PrepareConfig.model_validate(recorded)
    if geometry.mask.path is not None and not Path(geometry.mask.path).is_file():
        raise ValueError(
            f"the checkpoint's frames were masked with {geometry.mask.path}, which is not "
            "on this machine; scoring without it would feed the model pixels it never saw"
        )
    return geometry


def _image_as_chw(
    image_path: str | Path,
    size: int,
    cfg: ExperimentConfig,
    geometry: PrepareConfig | None = None,
) -> np.ndarray:
    """Decode one live frame the way :class:`MultimodalImageDataset` decodes a training one.

    This is the serving side of the train/serve pair, so the chain has to match
    the dataset's exactly: decode -> prepare geometry -> ``[0, 1]`` -> preprocess
    -> resize -> standardize. Augmentation is training-only and has no
    counterpart here.

    Parameters
    ----------
    image_path:
        Frame to read; any PIL-readable format.
    size:
        Side of the square the model expects, in pixels.
    cfg:
        The checkpoint's own config, which carries the preprocessing settings
        the model was trained under.
    geometry:
        The mask/crop/pad/resize the dataset's own frames were written through,
        from the checkpoint's ``frame_geometry``. ``ExperimentConfig`` cannot
        express it.

    Returns
    -------
    numpy.ndarray
        ``(3, size, size)`` float32, standardized by the DINOv2 channel stats —
        dimensionless, not ``[0, 1]``.
    """
    from allsky.preprocessing import imagenet_standardize

    return imagenet_standardize(_input_frame(image_path, size, cfg, geometry), copy=False)


def _input_frame(
    image_path: str | Path,
    size: int,
    cfg: ExperimentConfig,
    geometry: PrepareConfig | None = None,
) -> np.ndarray:
    """The ``(3, size, size)`` float32 ``[0, 1]`` frame :func:`_image_as_chw` standardizes."""
    from allsky.preprocessing import PreprocessingPipeline, model_input_frame

    return model_input_frame(
        image_path,
        size=size,
        preprocess=PreprocessingPipeline.from_config(cfg),
        geometry=geometry,
    )


class _EmbeddingStoreUnreachableError(ValueError):
    """The embedding store a checkpoint names is not on this machine at all.

    Distinct from a store that is present but unusable: one is a checkpoint that
    travelled away from its store and can still be answered from the recipe the
    checkpoint carries, the other is a store whose own record of how it was
    encoded is broken, where nothing on disk describes the encoding any more.
    """


def embedding_recipe_of(store: str | Path) -> dict[str, Any] | None:
    """The encoding recipe *store* records, or None when it cannot be read.

    Returned as the ``backbone`` provenance an embedding-mode checkpoint carries,
    so a checkpoint copied away from the machine that trained it can still
    rebuild the encoder the model was fitted on.  None when the store or its
    sidecar is unreachable, or when the recipe is incomplete: recording half of
    it would be worse than recording none, since the missing half would be
    guessed at prediction time.

    Returns
    -------
    dict or None
        ``backbone``, ``pooling``, ``revision``, ``dim``, ``dtype`` and, when the
        store recorded it, ``transform``.
    """
    from allsky.embeddings.storage import read_meta

    try:
        meta = read_meta(store)
    except OSError, ValueError:
        return None
    if any(meta.get(key) is None for key in STORE_RECIPE_KEYS):
        return None
    recipe = {key: meta[key] for key in STORE_RECIPE_KEYS}
    if meta.get("transform"):
        recipe["transform"] = meta["transform"]
    return recipe


def _embedding_store_meta(
    cfg: ExperimentConfig, embeddings_dir: str | Path | None
) -> tuple[Path, dict[str, Any]]:
    """Embedding store to encode against, and its provenance sidecar.

    Embedding-mode training never reads ``model.backbone`` /
    ``model.backbone_pooling``: the representation is whatever
    ``precompute-embeddings`` wrote, recorded nowhere but this sidecar. Re-encoding
    a live frame under any other recipe changes the representation silently — the
    checkpoint carries no backbone identity in embedding mode, and for a given
    model ``cls`` and ``mean`` pooling are the same width, so a mismatch
    survives ``load_state_dict`` and comes out as a plausible number.

    Parameters
    ----------
    cfg:
        Config carried by the checkpoint. Its ``data.embeddings_dir`` is
        resolved against ``data.data_root``, which training bakes in as the
        absolute path of the machine that trained — unusable once the
        checkpoint is copied anywhere else.
    embeddings_dir:
        Store to read instead, for exactly that case; None keeps the baked
        path.

    Raises
    ------
    ValueError
        If no embeddings directory is named at all, if the sidecar is not
        readable there, if it omits any part of the encoding recipe, or if it
        records a storage dtype no backbone can be built at.
    """
    from allsky.data.loading import resolve_against_root
    from allsky.embeddings.storage import META_FILENAME, read_meta

    if embeddings_dir is not None:
        store = Path(embeddings_dir)
    elif cfg.data.embeddings_dir is not None:
        store = resolve_against_root(cfg.data.embeddings_dir, Path(cfg.data.data_root))
    else:
        raise ValueError("input_mode='embedding' checkpoint carries no data.embeddings_dir")
    try:
        meta = read_meta(store)
    except FileNotFoundError as exc:
        raise _EmbeddingStoreUnreachableError(
            f"no {META_FILENAME} under {store}: an embedding-mode checkpoint records no "
            "backbone of its own, so without the store's sidecar there is no way to encode "
            "the live frame the way the model was fitted. Point predict_snapshot at the "
            "store that ships with the checkpoint if the trained-on path has moved"
        ) from exc
    absent = sorted(key for key in STORE_RECIPE_KEYS if meta.get(key) is None)
    if absent:
        raise ValueError(f"{store / META_FILENAME} records no {', '.join(absent)}")
    if meta["dtype"] not in EMBEDDING_STORE_DTYPES:
        raise ValueError(
            f"{store / META_FILENAME} records dtype {meta['dtype']!r}; a backbone can only "
            f"be built at one of {', '.join(EMBEDDING_STORE_DTYPES)}"
        )
    return store, meta


def _backbone_matching_recipe(source: str, meta: dict[str, Any], device: str) -> VisualBackbone:
    """Backbone built to *meta*'s recipe, refusing anything it cannot reproduce.

    ``build_backbone`` takes the pooling, the storage dtype and the fake
    backbone's width, so those are forwarded; the pinned revision and the
    embedding width are properties of the built backbone instead, so they are
    compared against what the recipe recorded and a difference is fatal. Both
    would otherwise re-encode the live frame under a recipe the model was never
    fitted on and still produce a vector of the width ``load_state_dict``
    accepts.

    *source* names where the recipe came from — the store's sidecar or the
    checkpoint's own provenance — and appears in the error when the two disagree.
    """
    from allsky.embeddings.backbone import build_backbone

    # The COMPUTE dtype, falling back to the single `dtype` a store written
    # before the two were told apart records — which is the storage one, so an
    # fp32 run of that vintage still rebuilds at fp16 and says so.
    compute_dtype = meta.get("compute_dtype") or meta["dtype"]
    if meta.get("compute_dtype") is None:
        logger.warning(
            "%s records one dtype for both storage and computation; building the backbone "
            "at %s, which is the STORAGE precision",
            source,
            compute_dtype,
        )
    backbone = build_backbone(
        meta["backbone"],
        device=device,
        pooling=meta["pooling"],
        dtype=compute_dtype,
        fake_dim=int(meta["dim"]),
    )
    recorded_transform = meta.get("transform")
    built_transform = getattr(backbone, "transform_description", "")
    mismatched: dict[str, tuple[Any, Any]] = {
        "revision": (meta["revision"], backbone.revision),
        "dim": (int(meta["dim"]), int(backbone.dim)),
    }
    if recorded_transform and built_transform:
        mismatched["transform"] = (recorded_transform, built_transform)
    differing = {key: pair for key, pair in mismatched.items() if pair[0] != pair[1]}
    if differing:
        detail = "; ".join(
            f"{key}: recorded={recorded!r} live={built!r}"
            for key, (recorded, built) in differing.items()
        )
        raise ValueError(
            f"the live backbone cannot reproduce the recipe {source} records "
            f"({detail}), so the frame would be encoded differently from the vectors the "
            "model was fitted on; re-extract the store with this code, or predict with the "
            "checkpoint trained against it"
        )
    return backbone


def shipped_sensor_limits() -> list[SensorRangeLimit]:
    """The plausibility gates the shipped configuration declares.

    Reading the process-wide settings is the CLI's job, not the domain's, so it
    happens here at the public boundary and only when a caller supplied none.
    """
    from micrometeorology.common.config import get_settings

    limits: list[SensorRangeLimit] = get_settings().sensor_limits
    return limits


def _image_input(
    image_path: str | Path,
    image_size: int,
    cfg: ExperimentConfig,
    *,
    timestamp: pd.Timestamp,
    site: SiteConfig,
    geometry: PrepareConfig | None = None,
) -> np.ndarray:
    """The ``(3 + G, S, S)`` ``float32`` tensor the image branch of *cfg* was trained on.

    The three standardized RGB planes come from :func:`_image_as_chw`, through
    the run's own *geometry*; the ``G`` solar-geometry planes are the ones the
    dataset stacks under ``model.geometry_channels``
    (:func:`allsky.geometry.solar_geometry_maps`), built for *timestamp* on the
    site's own clock and the isotropic lens calibration at *image_size*.
    """
    chw = _image_as_chw(image_path, image_size, cfg, geometry)
    return _with_solar_maps(chw, _solar_maps(image_size, cfg, timestamp=timestamp, site=site))


def _solar_maps(
    image_size: int, cfg: ExperimentConfig, *, timestamp: pd.Timestamp, site: SiteConfig
) -> np.ndarray | None:
    """The ``(G, S, S)`` solar-geometry planes of *cfg* at *timestamp*; ``None`` without any."""
    channels = geometry_channels_of(cfg)
    if not channels:
        return None
    from allsky.geometry import solar_geometry_maps
    from allsky.lens import isotropic_calibration
    from labmim_core.solar import solar_azimuth_deg, solar_elevation_deg

    local = pd.DatetimeIndex([timestamp])
    zenith_deg = 90.0 - float(solar_elevation_deg(local, site, site.utc_offset_hours)[0])
    azimuth_deg = float(solar_azimuth_deg(local, site, site.utc_offset_hours)[0])
    return solar_geometry_maps(
        isotropic_calibration(image_size),
        (image_size, image_size),
        sun_zenith_rad=float(np.radians(zenith_deg)),
        sun_azimuth_rad=float(np.radians(azimuth_deg)),
        channels=channels,
    )


def _with_solar_maps(chw: np.ndarray, maps: np.ndarray | None) -> np.ndarray:
    if maps is None:
        return chw
    return np.concatenate([chw, maps], axis=0).astype(np.float32, copy=False)


def clearsky_dhi_series(local: pd.DatetimeIndex, site: SiteConfig) -> np.ndarray:
    """Clear-sky diffuse irradiance (W m-2) ``(N,)`` float64 at naive local instants *local*.

    The reference a ``clearsky_index`` DHI head is trained as a ratio to, so a
    served index times this value is the diffuse irradiance in W m-2.
    """
    from allsky.clearsky import clearsky_diffuse
    from labmim_core.solar import cos_zenith

    zenith_deg = np.degrees(np.arccos(cos_zenith(local, site, site.utc_offset_hours)))
    times = pd.Series(local.tz_localize(site.clock).tz_convert("UTC"))
    return np.asarray(clearsky_diffuse(zenith_deg, times, site.utc_offset_hours), dtype=np.float64)


def clearsky_dhi_at(timestamp: pd.Timestamp, site: SiteConfig) -> float:
    """:func:`clearsky_dhi_series` at one naive local *timestamp*."""
    return float(clearsky_dhi_series(pd.DatetimeIndex([timestamp]), site)[0])


def _refuse_a_windowed_checkpoint(cfg: ExperimentConfig) -> None:
    """Refuse to serve a checkpoint fitted on a window from a single frame.

    A snapshot is one capture, so the batch carries one ``image``/``embedding``
    and never the ``image_seq``/``embedding_seq`` a pooled window is served as.
    The encoders fall back to their single-frame branch when the sequence key is
    absent, so a ``mean_embedding`` or ``attention_pooling`` checkpoint would
    score one frame where it was fitted on up to ``alignment.max_frames`` — no
    error, no warning, and a plausible number. Silence is the one option ruled
    out;
    building the window here needs the frames around the capture, which a live
    snapshot does not have.

    Raises
    ------
    ValueError
        Naming the strategy the checkpoint was trained under.
    """
    strategy = cfg.data.alignment.strategy
    if strategy == "center_frame":
        return
    raise ValueError(
        f"this checkpoint was trained with alignment.strategy={strategy!r}, which pools "
        f"up to {cfg.data.alignment.max_frames} frames over "
        f"{cfg.data.alignment.window_minutes:g} min; a snapshot is a single capture and "
        "scoring it would silently use the model's single-frame path"
    )


@dataclass(frozen=True, slots=True)
class ScalarFeatures:
    """The engineered scalar vector one prediction is fed.

    Attributes
    ----------
    columns:
        Feature names, in the checkpoint's order.
    values:
        ``(F,)`` float32, raw physical units (degrees, m s-1, ...), the
        training mean where a source was missing or refused.
    standardized:
        ``(1, F)`` float32, through the train-split :class:`FeatureNormalizer`.
    imputed:
        Names of the columns that were imputed rather than measured.
    pairing:
        The tolerance and timestamp offset the station row was looked up with.
    gap_minutes:
        Distance in minutes between the capture and the station row used;
        ``None`` when no row was read.
    sensor_csv:
        The station export the row came from, or ``None`` when every sensor
        column was imputed.
    """

    columns: list[str]
    values: np.ndarray
    standardized: np.ndarray
    imputed: list[str]
    pairing: _SensorPairing
    gap_minutes: float | None
    sensor_csv: Path | None

    def record(self, timestamp: pd.Timestamp, feature_set: str) -> dict[str, Any]:
        """The ``features`` block a prediction record publishes."""
        return {
            "timestamp": timestamp.isoformat(),
            "feature_set": feature_set,
            "columns": self.columns,
            "values": [float(value) for value in self.values],
            "imputed": self.imputed,
            "sensor_csv": str(self.sensor_csv) if self.sensor_csv is not None else None,
            "sensor_pairing": {
                "tolerance_minutes": self.pairing.tolerance.total_seconds() / 60.0,
                "timestamp_offset_minutes": self.pairing.timestamp_offset_minutes,
                "from_checkpoint": self.pairing.from_checkpoint,
                "gap_minutes": self.gap_minutes,
            },
        }


@dataclass(frozen=True, slots=True)
class ServedModel:
    """A checkpoint loaded once and ready to score any number of frames.

    Everything :func:`predict_snapshot` used to do per call — read the
    payload, rebuild the architecture, restore the normalizers, resolve the
    frame geometry and, in embedding mode, the encoding recipe — happens once
    in :func:`load_served_model`; the methods here build a batch, run it and
    turn the outputs back into physical units. Publishing several probes of
    one frame (the prediction, an occlusion sweep, a counterfactual) costs one
    load instead of one per forward.

    Attributes
    ----------
    checkpoint:
        The loaded payload, minus nothing: provenance readers take it as is.
    geometry:
        The mask/crop/pad/resize the training frames were written through, or
        ``None`` for a checkpoint that recorded none.
    embedding_backbone:
        In embedding mode, the backbone built to the store's recipe; ``None``
        in image mode and for the scalar-only architectures.
    reads_visual:
        Whether the architecture consumes pixels or an embedding at all, as
        the model registry declares it.
    min_solar_elevation_deg:
        The ``night_filter.min_solar_elevation_deg`` the training manifest
        dropped frames under, when the checkpoint records it; ``None`` for a
        checkpoint written before that provenance existed.
    """

    checkpoint_path: Path
    checkpoint: dict[str, Any]
    cfg: ExperimentConfig
    model: Any
    feature_columns: list[str]
    feature_normalizer: Any
    target_normalizers: dict[str, Any]
    geometry: PrepareConfig | None
    device: str
    embedding_backbone: VisualBackbone | None
    embedding_storage_dtype: str | None
    reads_visual: bool
    min_solar_elevation_deg: float | None

    @property
    def consumes_image(self) -> bool:
        """Whether a forward pass reads pixels (image mode, not a scalar-only architecture)."""
        return self.cfg.data.input_mode == "image" and self.reads_visual

    @property
    def consumes_embedding(self) -> bool:
        """Whether a forward pass reads a precomputed visual vector."""
        return self.cfg.data.input_mode == "embedding" and self.reads_visual

    @property
    def scalar_only(self) -> bool:
        """Whether the architecture ignores every visual input."""
        return not self.reads_visual

    @property
    def image_size(self) -> int:
        """Side of the square input, in pixels."""
        return image_size_of(self.cfg)

    @property
    def serves(self) -> ServedInput:
        """``"frame"`` for a ``center_frame`` checkpoint, ``"block"`` for a windowed one."""
        return "frame" if self.cfg.data.alignment.strategy == "center_frame" else "block"

    @property
    def window_minutes(self) -> float:
        """Width of the block a windowed checkpoint pools frames over, in minutes."""
        return float(self.cfg.data.alignment.window_minutes)

    def scalar_features(
        self,
        timestamp: pd.Timestamp,
        *,
        site: SiteConfig,
        sensor_csv: str | Path | StationExport | None = None,
        tolerance: pd.Timedelta | None = None,
        sensor_limits: list[SensorRangeLimit] | None = None,
    ) -> ScalarFeatures:
        """Engineer and standardize the scalar vector for *timestamp*.

        Columns a live capture cannot supply are imputed at the training mean
        and named in the result; see ``docs/allsky-archive.md``. *sensor_csv*
        may be a path or a :class:`StationExport` already read, so one publish
        parses the export once for every reader.
        """
        export = _station_export(sensor_csv)
        pairing = _pairing_of(self.checkpoint, tolerance)
        values, imputed, gap_minutes = _feature_vector(
            timestamp,
            feature_columns=self.feature_columns,
            feature_set=self.cfg.features.feature_set,
            site=site,
            export=export,
            tolerance=pairing.tolerance,
            training_means=self.feature_normalizer.mean,
            sensor_limits=(sensor_limits if sensor_limits is not None else shipped_sensor_limits()),
            timestamp_offset_minutes=pairing.timestamp_offset_minutes,
        )
        standardized = self.feature_normalizer.transform(
            pd.DataFrame([values], columns=self.feature_columns)
        )
        return ScalarFeatures(
            columns=list(self.feature_columns),
            values=values,
            standardized=np.asarray(standardized, dtype=np.float32),
            imputed=imputed,
            pairing=pairing,
            gap_minutes=gap_minutes,
            sensor_csv=export.path if export is not None else None,
        )

    def input_frame(self, image_path: str | Path) -> np.ndarray:
        """The ``(3, S, S)`` float32 ``[0, 1]`` frame the image branch's input is standardized from."""
        return _input_frame(image_path, self.image_size, self.cfg, self.geometry)

    def planes_of(
        self, frame: np.ndarray, timestamp: pd.Timestamp, *, site: SiteConfig
    ) -> np.ndarray:
        """Standardize a copy of *frame* and append the ``G`` solar-geometry planes of *timestamp*."""
        from allsky.preprocessing import imagenet_standardize

        maps = _solar_maps(self.image_size, self.cfg, timestamp=timestamp, site=site)
        return _with_solar_maps(imagenet_standardize(frame), maps)

    def image_planes(
        self, image_path: str | Path, timestamp: pd.Timestamp, *, site: SiteConfig
    ) -> np.ndarray:
        """The ``(3 + G, S, S)`` float32 standardized planes the image branch reads."""
        return _image_input(
            image_path,
            self.image_size,
            self.cfg,
            timestamp=timestamp,
            site=site,
            geometry=self.geometry,
        )

    def embedding_vector(self, image_path: str | Path) -> np.ndarray:
        """Encode one frame to the ``(1, D)`` float32 vector an embedding-mode model reads.

        Raises
        ------
        ValueError
            For a checkpoint that reads no embedding.
        """
        if self.embedding_backbone is None:
            raise ValueError(f"{self.cfg.name} reads no embedding")
        backbone = self.embedding_backbone
        vector = np.asarray(backbone.encode(backbone.transform([_image_as_hwc(image_path)])))
        if self.embedding_storage_dtype == "fp16":
            vector = vector.astype(np.float16)
        return np.reshape(vector, (1, -1)).astype(np.float32)

    def batch(
        self,
        features: ScalarFeatures,
        *,
        planes: np.ndarray | None = None,
        embedding: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Assemble the single-row batch the model reads, on the served device."""
        import torch

        batch: dict[str, Any] = {
            "features": torch.from_numpy(features.standardized).to(self.device)
        }
        if planes is not None:
            batch["image"] = (
                torch.from_numpy(np.ascontiguousarray(planes)).unsqueeze(0).to(self.device)
            )
        if embedding is not None:
            batch["embedding"] = torch.from_numpy(embedding).to(self.device)
        return batch

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Run the model on *batch* without gradients."""
        import torch

        with torch.no_grad():
            outputs: dict[str, Any] = self.model(batch)
        return outputs

    def physical(
        self, outputs: dict[str, Any], *, timestamp: pd.Timestamp, site: SiteConfig
    ) -> dict[str, Any]:
        """Denormalize one row of *outputs* into the physical-unit prediction record."""
        return _physical_predictions(
            outputs, self.cfg, self.target_normalizers, reference_time=timestamp, site=site
        )

    def physical_values(
        self, outputs: dict[str, Any], name: str, *, timestamp: pd.Timestamp, site: SiteConfig
    ) -> np.ndarray:
        """``(B,)`` float64 physical values of regression head *name* for every row of *outputs*."""
        return _physical_values(
            outputs, name, self.cfg, self.target_normalizers, reference_time=timestamp, site=site
        )

    def record(self) -> dict[str, Any]:
        """The ``model`` block a prediction record publishes."""
        return _model_record(self.checkpoint, self.checkpoint_path, self.cfg, self.device)

    def predict_frame(
        self,
        image_path: str | Path,
        *,
        timestamp: pd.Timestamp,
        site: SiteConfig | None = None,
        sensor_csv: str | Path | StationExport | None = None,
        tolerance: pd.Timedelta | None = None,
        sensor_limits: list[SensorRangeLimit] | None = None,
    ) -> dict[str, Any]:
        """Score one sky image; the record :func:`predict_snapshot` documents.

        Raises
        ------
        ValueError
            For a checkpoint that serves blocks, or one expecting a feature
            column its feature set does not produce.
        """
        if self.serves != "frame":
            _refuse_a_windowed_checkpoint(self.cfg)
        resolved_site = site or SiteConfig()
        features = self.scalar_features(
            timestamp,
            site=resolved_site,
            sensor_csv=sensor_csv,
            tolerance=tolerance,
            sensor_limits=sensor_limits,
        )
        planes = (
            self.image_planes(image_path, timestamp, site=resolved_site)
            if self.consumes_image
            else None
        )
        embedding = self.embedding_vector(image_path) if self.consumes_embedding else None
        outputs = self.forward(self.batch(features, planes=planes, embedding=embedding))
        return {
            "predictions": self.physical(outputs, timestamp=timestamp, site=resolved_site),
            "features": features.record(timestamp, self.cfg.features.feature_set),
            "model": self.record(),
            "image": str(image_path),
        }

    def predict_block(
        self,
        frames: Sequence[tuple[str | Path, pd.Timestamp]],
        *,
        min_solar_elevation_deg: float,
        block_end: pd.Timestamp | None = None,
        site: SiteConfig | None = None,
    ) -> dict[str, Any]:
        """Score one datalogger block from its frames; the record :func:`predict_block` documents.

        Raises
        ------
        ValueError
            If *frames* is empty or none of them falls in the block, or for a
            checkpoint that serves single frames.
        SolarElevationBelowFloorError
            If the representative frame has the sun below *min_solar_elevation_deg*.
        """
        import torch

        from allsky.data.datasets import _subsample_window

        if self.serves != "block":
            _refuse_a_single_frame_checkpoint(self.cfg)
        if not frames:
            raise ValueError("predict_block needs at least one frame")
        alignment = self.cfg.data.alignment
        window_minutes = self.window_minutes
        ordered = sorted(
            ((Path(path), pd.Timestamp(when)) for path, when in frames), key=lambda f: f[1]
        )
        end = (
            pd.Timestamp(block_end)
            if block_end is not None
            else max(block_end_of(when, window_minutes) for _, when in ordered)
        )
        in_block = [f for f in ordered if block_end_of(f[1], window_minutes) == end]
        outside = [f for f in ordered if block_end_of(f[1], window_minutes) != end]
        if not in_block:
            raise ValueError(
                f"none of the {len(ordered)} frame(s) falls in the block "
                f"({end - pd.Timedelta(minutes=window_minutes)}, {end}]"
            )
        representative_time = in_block[
            nearest_to_centroid(
                pd.DatetimeIndex([when for _, when in in_block]), end, window_minutes
            )
        ][1]
        resolved_site = site or SiteConfig()
        elevation_deg = solar_elevation_at(representative_time, resolved_site)
        if elevation_deg < min_solar_elevation_deg:
            raise SolarElevationBelowFloorError(
                representative_time, elevation_deg, float(min_solar_elevation_deg)
            )
        kept = set(_subsample_window(list(range(len(in_block))), alignment.max_frames))
        selected = [f for slot, f in enumerate(in_block) if slot in kept]
        capped = [f for slot, f in enumerate(in_block) if slot not in kept]

        raw_values, imputed, _gap = _feature_vector(
            representative_time,
            feature_columns=self.feature_columns,
            feature_set=self.cfg.features.feature_set,
            site=resolved_site,
            export=None,
            tolerance=DEFAULT_SENSOR_TOLERANCE,
            training_means=self.feature_normalizer.mean,
            sensor_limits=[],
        )
        standardized = self.feature_normalizer.transform(
            pd.DataFrame([raw_values], columns=self.feature_columns)
        )
        maps = _solar_maps(
            self.image_size, self.cfg, timestamp=representative_time, site=resolved_site
        )
        planes = [
            _with_solar_maps(_image_as_chw(path, self.image_size, self.cfg, self.geometry), maps)
            for path, _ in selected
        ]
        sequence = np.zeros((alignment.max_frames, *planes[0].shape), dtype=np.float32)
        mask = np.zeros(alignment.max_frames, dtype=bool)
        for slot, plane in enumerate(planes):
            sequence[slot] = plane
            mask[slot] = True
        batch: dict[str, Any] = {
            "features": torch.from_numpy(standardized).to(self.device),
            "image_seq": torch.from_numpy(sequence).unsqueeze(0).to(self.device),
            "frame_mask": torch.from_numpy(mask).unsqueeze(0).to(self.device),
        }
        outputs = self.forward(batch)

        def _listed(
            entries: list[tuple[Path, pd.Timestamp]], reason: str | None
        ) -> list[dict[str, Any]]:
            return [
                {"path": str(path), "captured_at": when.isoformat()}
                | ({"reason": reason} if reason else {})
                for path, when in entries
            ]

        return {
            "predictions": self.physical(
                outputs, timestamp=representative_time, site=resolved_site
            ),
            "block": {
                "end": end.isoformat(),
                "window_minutes": window_minutes,
                "n_frames": len(selected),
                "frames": _listed(selected, None),
                "representative": representative_time.isoformat(),
                "solar_elevation_deg": elevation_deg,
                "min_solar_elevation_deg": float(min_solar_elevation_deg),
                "ignored": _listed(outside, "outside_block") + _listed(capped, "over_max_frames"),
            },
            "features": {
                "timestamp": representative_time.isoformat(),
                "feature_set": self.cfg.features.feature_set,
                "columns": list(self.feature_columns),
                "values": [float(value) for value in raw_values],
                "imputed": imputed,
            },
            "model": self.record(),
        }


def _night_floor_of(checkpoint: dict[str, Any]) -> float | None:
    recorded = checkpoint.get("night_filter") or {}
    floor = recorded.get("min_solar_elevation_deg")
    return float(floor) if floor is not None else None


def load_served_model(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    trust_checkpoint: bool = False,
    embeddings_dir: str | Path | None = None,
    image_backbone_builder: Callable[[], Any] | None = None,
    expect: ServedInput = "frame",
) -> ServedModel:
    """Load a checkpoint once, ready to score frames or blocks.

    Parameters
    ----------
    checkpoint_path:
        ``best.ckpt`` / ``last.ckpt`` written by ``allsky train``.
    device:
        Torch device the model and, in embedding mode, the backbone run on.
    trust_checkpoint:
        Read with the unrestricted unpickler (own files only).
    embeddings_dir:
        Embedding store overriding the absolute ``data.data_root`` baked into
        an embedding-mode checkpoint; rejected for an image-mode one.
    image_backbone_builder:
        Test seam: builds the image backbone instead of the config's.
    expect:
        What the checkpoint will be served on. ``"frame"`` refuses a windowed
        checkpoint, which would silently score one capture through its
        single-frame path; ``"block"`` refuses a ``center_frame`` or
        embedding-mode one, and warns that a fusion model gets every sensor
        feature imputed since no station export is read per block.

    Returns
    -------
    ServedModel
        The model in eval mode with its normalizers, geometry and recipe.

    Raises
    ------
    ValueError
        A checkpoint the *expect* side refuses, an embedding store that cannot
        be read or reproduced, or *embeddings_dir* for an image-mode checkpoint.
    """
    from allsky.modeling.registry import reads_visual_input, restore_model
    from allsky.training.checkpointing import load_checkpoint

    checkpoint = load_checkpoint(
        checkpoint_path, map_location=device, trust_pickle=trust_checkpoint
    )
    cfg = ExperimentConfig.model_validate(checkpoint["config"])
    if embeddings_dir is not None and cfg.data.input_mode != "embedding":
        raise ValueError(
            f"embeddings_dir was given for an input_mode={cfg.data.input_mode!r} checkpoint, "
            "which encodes the live frame with its own backbone and reads no embedding store"
        )
    if expect == "frame":
        _refuse_a_windowed_checkpoint(cfg)
    else:
        _refuse_a_single_frame_checkpoint(cfg)
        if cfg.model.name != "image_only":
            logger.warning(
                "%s fuses sensor features (model %r); predict_block reads no station export, so "
                "every sensor feature is imputed at its training mean on every block",
                checkpoint_path,
                cfg.model.name,
            )
    feature_columns: list[str] = list(checkpoint["feature_columns"])
    feature_normalizer, target_normalizers = normalizers_from_checkpoint(checkpoint)
    reads_visual = reads_visual_input(cfg.model.name)
    scalar_only = not reads_visual

    embedding_backbone: VisualBackbone | None = None
    storage_dtype: str | None = None
    embedding_dim: int | None = None
    if cfg.data.input_mode == "embedding" and not scalar_only:
        from allsky.embeddings.storage import META_FILENAME

        try:
            store, store_meta = _embedding_store_meta(cfg, embeddings_dir)
            source = str(store / META_FILENAME)
        except _EmbeddingStoreUnreachableError:
            # The store the run trained against is not on this machine. The
            # checkpoint's own copy of its recipe is the only other record of how
            # those vectors were encoded, and a checkpoint written before that
            # copy existed carries none — which is still a refusal, not a guess.
            recorded_recipe = checkpoint.get("backbone") if embeddings_dir is None else None
            if not recorded_recipe:
                raise
            source, store_meta = f"{checkpoint_path} (its own provenance)", dict(recorded_recipe)
            logger.info("embedding store unreachable; encoding to the recipe %s records", source)
        embedding_backbone = _backbone_matching_recipe(source, store_meta, device)
        storage_dtype = str(store_meta.get("storage_dtype") or store_meta["dtype"])
        embedding_dim = int(embedding_backbone.dim)
    elif cfg.data.input_mode == "embedding":
        logger.info(
            "%s reads the scalar vector alone; the live frame is not encoded", cfg.model.name
        )

    model = restore_model(
        cfg,
        checkpoint,
        len(feature_columns),
        embedding_dim=embedding_dim,
        device=device,
        image_backbone_builder=image_backbone_builder,
    )
    model.eval()
    return ServedModel(
        checkpoint_path=Path(checkpoint_path),
        checkpoint=checkpoint,
        cfg=cfg,
        model=model,
        feature_columns=feature_columns,
        feature_normalizer=feature_normalizer,
        target_normalizers=target_normalizers,
        geometry=_frame_geometry(checkpoint) if cfg.data.input_mode == "image" else None,
        device=device,
        embedding_backbone=embedding_backbone,
        embedding_storage_dtype=storage_dtype,
        reads_visual=reads_visual,
        min_solar_elevation_deg=_night_floor_of(checkpoint),
    )


def predict_snapshot(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    timestamp: pd.Timestamp,
    sensor_csv: str | Path | StationExport | None = None,
    tolerance: pd.Timedelta | None = None,
    site: SiteConfig | None = None,
    device: str = "cpu",
    trust_checkpoint: bool = False,
    embeddings_dir: str | Path | None = None,
    sensor_limits: list[SensorRangeLimit] | None = None,
) -> dict[str, Any]:
    """Run a trained checkpoint over one sky image and return physical-unit predictions.

    :func:`load_served_model` then :meth:`ServedModel.predict_frame`; a
    caller scoring more than one frame loads once and calls the method.

    The image is read as ``(3, S, S)`` ``float32`` in ``[0, 1]``, channels-first,
    at the checkpoint's own ``image_size``. Sensor features are engineered for
    *timestamp* under the checkpoint's feature set; any column that comes out
    non-finite is replaced by the training mean and listed in the returned
    ``features.imputed``, so an imputed prediction is never silently equivalent
    to a measured one. Regression heads are denormalized back to physical units.

    Parameters
    ----------
    image_path:
        Sky image to score, in any PIL-readable format.
    checkpoint_path:
        Trained checkpoint carrying the config, feature columns and normalizers.
    timestamp:
        Naive local capture time of the image; drives both the solar geometry
        features and the sensor-row lookup.
    sensor_csv:
        Datalogger export to draw the measured features from. Left None every
        sensor-derived column is imputed.
    tolerance:
        Largest gap between *timestamp* and the nearest sensor row still
        accepted; beyond it the row is discarded and the columns imputed. Left
        None the checkpoint's own ``sensor_pairing`` decides, which is the
        window the run trained under; a checkpoint carrying none falls back to
        :data:`DEFAULT_SENSOR_TOLERANCE` with a warning.
    site:
        Observation site for the solar geometry; defaults to
        :class:`~allsky.config.SiteConfig`. The geometry is built at the site's
        declared ``utc_offset_hours``, the clock the manifest builder trains
        on, never at an offset inferred from the site longitude.
    device:
        Torch device the backbone and model run on.
    trust_checkpoint:
        Allow unpickling a checkpoint that is not weights-only. Leave False for
        any checkpoint whose origin is not your own training run.
    embeddings_dir:
        Embedding store whose sidecar describes how the live frame must be
        encoded, overriding the absolute ``data.data_root`` the training run
        baked into the checkpoint. Point it at the store shipped beside a
        checkpoint that has been copied off the machine it was trained on.
        Rejected for an image-mode checkpoint, which encodes nothing.

    Returns
    -------
    dict
        ``{"predictions", "features", "model", "image"}``. ``predictions`` holds
        whichever heads the checkpoint has: ``dhi`` (diffuse horizontal
        irradiance, W m-2), ``kindex`` (dimensionless), ``cloud_fraction``
        (in [0, 1]), and ``sky_class`` with its ``sky_probabilities`` over
        :data:`labmim_core.sky.SKY_CLASS_NAMES`. ``features`` records the
        raw values fed in and which of them were imputed.

    Raises
    ------
    ValueError
        If the checkpoint expects a feature column its configured feature set
        does not produce, if *embeddings_dir* is given for an image-mode
        checkpoint, or — in embedding mode — if the store it was trained
        against cannot be read for the recipe that encoded it, or the live
        backbone cannot reproduce that recipe.
    """
    served = load_served_model(
        checkpoint_path,
        device=device,
        trust_checkpoint=trust_checkpoint,
        embeddings_dir=embeddings_dir,
    )
    return served.predict_frame(
        image_path,
        timestamp=timestamp,
        site=site,
        sensor_csv=sensor_csv,
        tolerance=tolerance,
        sensor_limits=sensor_limits,
    )


def _physical_predictions(
    outputs: dict[str, Any],
    cfg: ExperimentConfig,
    target_normalizers: dict[str, Any],
    *,
    reference_time: pd.Timestamp,
    site: SiteConfig,
) -> dict[str, Any]:
    """Denormalize one row of model *outputs* into the physical-unit prediction record.

    Parameters
    ----------
    outputs:
        The model's forward result for a batch of one: ``dhi``, ``kindex`` and
        ``cloud_fraction`` as ``(1,)`` normalized float tensors when the head
        exists, ``sky_logits`` as ``(1, K)`` over
        :data:`labmim_core.sky.SKY_CLASS_NAMES`.
    reference_time:
        Naive local time the clear-sky diffuse reference is evaluated at when
        the DHI head was fitted as a clear-sky index; the served row's own
        time, which is the frame's under ``center_frame`` and the block's
        representative frame under ``sensor_block``.
    """
    from labmim_core.sky import SKY_CLASS_NAMES

    predictions: dict[str, Any] = {}
    for name in ("dhi", "kindex", "cloud_fraction"):
        if name in outputs:
            predictions[name] = float(
                _physical_values(
                    outputs, name, cfg, target_normalizers, reference_time=reference_time, site=site
                )[0]
            )
    if "sky_logits" in outputs:
        logits = outputs["sky_logits"].detach().cpu().numpy().reshape(-1)
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        predictions["sky_class"] = SKY_CLASS_NAMES[int(np.argmax(logits))]
        predictions["sky_probabilities"] = {
            name: float(value) for name, value in zip(SKY_CLASS_NAMES, probabilities, strict=True)
        }
    return predictions


def _physical_values(
    outputs: dict[str, Any],
    name: str,
    cfg: ExperimentConfig,
    target_normalizers: dict[str, Any],
    *,
    reference_time: pd.Timestamp,
    site: SiteConfig,
) -> np.ndarray:
    """``(B,)`` float64 physical values of regression head *name*, one per row of *outputs*.

    The denormalization is affine and the clear-sky reference of a
    ``clearsky_index`` DHI head depends on *reference_time* alone, so a batch of
    rows scored at one instant is denormalized in one pass.
    """
    raw = outputs[name].detach().cpu().numpy().reshape(-1).astype(np.float64)
    normalizer = target_normalizers.get(name)
    values = np.asarray(normalizer.denormalize(raw), dtype=np.float64) if normalizer else raw
    if name == "dhi" and cfg.targets.dhi.parameterization == "clearsky_index":
        values = values * clearsky_dhi_at(reference_time, site)
    return values


def _model_record(
    checkpoint: dict[str, Any], checkpoint_path: str | Path, cfg: ExperimentConfig, device: str
) -> dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "name": cfg.name,
        "architecture": cfg.model.name,
        "input_mode": cfg.data.input_mode,
        "device": device,
        # Already loaded, so no extra I/O: without them a published
        # prediction names a checkpoint path and nothing about the dataset
        # or the code that produced it, and the path is the one thing that
        # does not survive the file being copied off this machine.
        "code_version": checkpoint.get("code_version"),
        "dataset_version": checkpoint.get("dataset_version"),
        "split_id": checkpoint.get("split_id"),
        "manifest_sha256": checkpoint.get("manifest_sha256"),
        "dhi_parameterization": cfg.targets.dhi.parameterization,
        "kindex_kind": cfg.targets.kindex.kind,
    }


def solar_elevation_at(timestamp: pd.Timestamp, site: SiteConfig) -> float:
    """Solar elevation at a naive local *timestamp*, degrees above the horizon.

    The one geometry every serving floor is judged against: a block's
    representative frame in :func:`predict_block` and a single live frame in
    the watch, both on the site's declared ``utc_offset_hours`` — the clock
    the manifest's ``night_filter`` dropped frames on.

    Parameters
    ----------
    timestamp:
        Naive local capture time on the camera's clock.
    site:
        Observation site whose latitude, longitude and UTC offset fix the sun.

    Returns
    -------
    float
        Elevation in degrees, negative below the horizon.
    """
    from labmim_core.solar import solar_elevation_deg

    return float(solar_elevation_deg(pd.DatetimeIndex([timestamp]), site, site.utc_offset_hours)[0])


class SolarElevationBelowFloorError(ValueError):
    """The block's representative frame has the sun below the training floor.

    Attributes
    ----------
    elevation_deg:
        Solar elevation at the representative frame, degrees above the horizon.
    floor_deg:
        The ``night_filter.min_solar_elevation_deg`` the manifest dropped
        frames under.
    """

    def __init__(
        self, representative: pd.Timestamp, elevation_deg: float, floor_deg: float
    ) -> None:
        self.elevation_deg = elevation_deg
        self.floor_deg = floor_deg
        super().__init__(
            f"the representative frame at {representative} has the sun {elevation_deg:.1f} deg "
            f"above the horizon, below the {floor_deg:g} deg floor the training manifest "
            "dropped frames under; the model never saw this sky"
        )


def _refuse_a_single_frame_checkpoint(cfg: ExperimentConfig) -> None:
    """The mirror of :func:`_refuse_a_windowed_checkpoint`, for the block path.

    Raises
    ------
    ValueError
        For a ``center_frame`` checkpoint, which was fitted on one frame per
        row and has no pooled path for a window to go through, and for an
        embedding-mode one, whose window is a sequence of stored vectors this
        path does not encode.
    """
    strategy = cfg.data.alignment.strategy
    if strategy == "center_frame":
        raise ValueError(
            "this checkpoint was trained with alignment.strategy='center_frame', one frame "
            "per row; score it with predict_snapshot, which serves a single capture"
        )
    if cfg.data.input_mode != "image":
        raise ValueError(
            f"predict_block serves input_mode='image' checkpoints only; this one is "
            f"{cfg.data.input_mode!r}"
        )
    if strategy != "sensor_block":
        logger.warning(
            "this checkpoint was trained with alignment.strategy=%r, whose window is centred "
            "on each frame; the live block (t - %g min, t] is the logger's window instead",
            strategy,
            cfg.data.alignment.window_minutes,
        )


def predict_block(
    frames: Sequence[tuple[str | Path, pd.Timestamp]],
    checkpoint_path: str | Path,
    *,
    min_solar_elevation_deg: float,
    block_end: pd.Timestamp | None = None,
    site: SiteConfig | None = None,
    device: str = "cpu",
    trust_checkpoint: bool = False,
    image_backbone_builder: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Score one datalogger block from the live frames captured inside it.

    :func:`load_served_model` with ``expect="block"`` then
    :meth:`ServedModel.predict_block`; a watch loads once and calls the method.

    The serving side of ``alignment.strategy='sensor_block'``: the frames whose
    naive local stamp falls in ``(block_end - window_minutes, block_end]`` are
    fed to the model the way :class:`allsky.data.datasets.MultimodalImageDataset`
    feeds a block — ``image_seq`` ``(1, T, 3 + G, S, S)`` float32 in time order,
    zero-padded to ``T = alignment.max_frames`` and evenly subsampled keeping
    the first and last frame when the block holds more, with ``frame_mask``
    ``(1, T)`` bool over the real slots; no ``image`` key is sent. The
    representative frame is the one nearest the block centroid
    (``block_end - window_minutes / 2``, first on a tie), chosen over every
    in-block frame before the cap, as ``representative_rows_per_block`` does.
    Everything the dataset takes from the served row is taken from it: the
    ``G`` solar-geometry planes (the dataset indexes the row's own solar angles
    for every co-frame of the window), the clear-sky diffuse reference of a
    ``clearsky_index`` DHI head, and the sensor features, all of which are
    imputed at the training mean since no station export is read here.

    A block whose representative frame has the sun below
    *min_solar_elevation_deg* is refused before anything is read: the manifest
    dropped every frame under that floor (``night_filter.min_solar_elevation_deg``
    of the prepare config), so the model never saw such a sky, and a
    ``clearsky_index`` DHI head would be scaled by a NaN reference there. The
    checkpoint does not record the floor, hence the parameter.

    Parameters
    ----------
    frames:
        ``(image path, naive local capture time)`` pairs, in any order. Frames
        outside the block are not read, only listed under ``block.ignored``.
    checkpoint_path:
        Image-mode checkpoint trained under a windowed alignment strategy.
    min_solar_elevation_deg:
        The floor the checkpoint's manifest was built with, degrees of solar
        elevation above the horizon.
    block_end:
        Naive local end of the block to score. Left None it is the latest
        frame's own block end, by :func:`block_end_of` under the checkpoint's
        ``window_minutes``.
    site:
        Observation site for the solar geometry; defaults to
        :class:`~allsky.config.SiteConfig`.
    device:
        Torch device the backbone and model run on.
    trust_checkpoint:
        Allow unpickling a checkpoint that is not weights-only.
    image_backbone_builder:
        Injection hook for the visual backbone, as
        :func:`allsky.evaluation.evaluator.evaluate_checkpoint` takes it; None
        builds the backbone the checkpoint's config names.

    Returns
    -------
    dict
        ``predictions`` as :func:`predict_snapshot` returns them; ``block``
        with ``end``, ``window_minutes``, ``n_frames`` fed, ``frames`` fed in
        time order, ``representative``, its ``solar_elevation_deg`` against
        ``min_solar_elevation_deg``, and ``ignored`` (each with its
        ``reason``: ``outside_block`` or ``over_max_frames``); ``features``
        naming the imputed columns; ``model`` as :func:`predict_snapshot`.

    Raises
    ------
    ValueError
        If *frames* is empty or none of them falls in the block, or if the
        checkpoint is ``center_frame`` or not image-mode.
    SolarElevationBelowFloorError
        If the representative frame has the sun below *min_solar_elevation_deg*.
    """
    served = load_served_model(
        checkpoint_path,
        device=device,
        trust_checkpoint=trust_checkpoint,
        image_backbone_builder=image_backbone_builder,
        expect="block",
    )
    return served.predict_block(
        frames, min_solar_elevation_deg=min_solar_elevation_deg, block_end=block_end, site=site
    )
