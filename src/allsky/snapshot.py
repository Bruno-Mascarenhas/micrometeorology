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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeIs, runtime_checkable

import numpy as np
import pandas as pd

from allsky.atomic import atomic_write, atomic_write_json
from allsky.config import SITE_TZ, SITE_UTC_OFFSET_HOURS, ExperimentConfig, SiteConfig
from allsky.embeddings.backbone import VisualBackbone
from allsky.provenance import code_version

logger = logging.getLogger(__name__)

__all__ = [
    "LiveFrameSource",
    "Snapshot",
    "capture_snapshot",
    "predict_snapshot",
]

SNAPSHOT_STEM_FORMAT = "allsky-%Y%m%d-%H%M%S"
DEFAULT_SENSOR_TOLERANCE = pd.Timedelta(minutes=15)
HTTP_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
SENSOR_TIME_COLUMNS = ("timestamp", "TIMESTAMP", "datetime", "time")
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
    """One captured live frame, its provenance sidecar and any prediction over it.

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
    size:
        Size of the fetched JPEG payload in bytes.
    server_last_modified:
        Raw ``Last-Modified`` header as the server sent it, or None.
    prediction:
        Payload from :func:`predict_snapshot`, or None when the capture was not
        scored.
    """

    image_path: Path
    metadata_path: Path
    captured_at: pd.Timestamp
    size: int
    server_last_modified: str | None
    prediction: dict[str, Any] | None = None


def _site_now() -> pd.Timestamp:
    """Current time on the camera's own clock, as a naive local timestamp."""
    return pd.Timestamp(dt.datetime.now(tz=SITE_TZ).replace(tzinfo=None)).floor("s")


def _naive_site_time_from_http_date(headers: dict[str, str]) -> pd.Timestamp | None:
    raw = headers.get("last-modified")
    if not raw:
        return None
    try:
        parsed = dt.datetime.strptime(raw, HTTP_DATE_FORMAT).replace(tzinfo=dt.UTC)
    except ValueError:
        logger.warning("unparseable Last-Modified header on the live frame: %r", raw)
        return None
    return pd.Timestamp(parsed.astimezone(SITE_TZ).replace(tzinfo=None))


def _overlay_timestamp(payload: bytes) -> tuple[pd.Timestamp | None, str | None]:
    import io

    from PIL import Image

    from allsky.overlay import read_frame_timestamp

    try:
        with Image.open(io.BytesIO(payload)) as handle:
            frame = np.asarray(handle.convert("RGB"))
        reading = read_frame_timestamp(frame)
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
        size=len(payload),
        server_last_modified=headers.get("last-modified"),
    )


def _screened_for_plausibility(row: pd.DataFrame) -> pd.DataFrame:
    """Narrow *row* to the columns with a declared plausibility gate, NaN-ing what fails it.

    Parameters
    ----------
    row:
        One-row frame taken from the operator's station export, carrying that
        export's own published units — degC, %, mbar, m s-1, degrees, W m-2 —
        at whatever averaging interval it was written on.

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
    from micrometeorology.common.config import get_settings
    from micrometeorology.sensors.ingestion import apply_physical_limits

    limits = get_settings().sensor_limits
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


def _sensor_row_near(
    sensor_csv: str | Path, timestamp: pd.Timestamp, tolerance: pd.Timedelta
) -> pd.DataFrame:
    """Row of *sensor_csv* nearest *timestamp*, relabelled to it, or an empty frame.

    The export is read on the same two contracts the training path holds it to.
    Its clock is the logger's, i.e. naive site-local, so an export that does
    carry an offset is converted into that zone rather than merely stripped of
    it. Its values are then screened by :func:`_screened_for_plausibility`,
    which assumes the published physical units of the processed station export
    the snapshot command documents — not the logger's raw pre-calibration
    values, and not a 5-minute grid: an hour that railed for part of its
    samples averages to a finite number no sentinel literal matches, and would
    otherwise pass the ``np.isfinite`` screen in :func:`_feature_vector` and be
    served as a measurement.
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
    if frame.empty:
        return frame
    index = frame.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        frame.index = index.tz_convert(SITE_TZ).tz_localize(None)
    position = int(frame.index.get_indexer(pd.DatetimeIndex([timestamp]), method="nearest")[0])
    if position < 0:
        return frame.iloc[0:0]
    gap = abs(pd.Timestamp(frame.index[position]) - timestamp)
    if gap > tolerance:
        logger.warning(
            "nearest sensor row is %s from the frame (tolerance %s) — imputing instead",
            gap,
            tolerance,
        )
        return frame.iloc[0:0]
    # Screened after the row is chosen, not before: every gate is row-wise, so
    # the outcome is identical, and the nearest-row lookup reads the index only.
    row = _screened_for_plausibility(frame.iloc[[position]])
    row.index = pd.DatetimeIndex([timestamp])
    return row


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


def _feature_vector(
    timestamp: pd.Timestamp,
    *,
    feature_columns: list[str],
    feature_set: str,
    site: SiteConfig,
    sensor_csv: str | Path | None,
    tolerance: pd.Timedelta,
    training_means: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    from allsky.features.engineering import build_feature_frame

    sensor = (
        _sensor_row_near(sensor_csv, timestamp, tolerance)
        if sensor_csv is not None
        else pd.DataFrame(index=pd.DatetimeIndex([]))
    )
    engineered = build_feature_frame(
        _with_absent_sources_as_nan(sensor, feature_set),
        [timestamp],
        site,
        feature_set,
        utc_offset_hours=float(SITE_UTC_OFFSET_HOURS),
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
    return np.where(finite, values, training_means).astype(np.float32), imputed


def _image_as_hwc(image_path: str | Path) -> np.ndarray:
    """Read a frame as ``(H, W, 3)`` ``uint8`` RGB — what a visual backbone's transform takes."""
    from PIL import Image

    with Image.open(image_path) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


def _image_as_chw(image_path: str | Path, size: int, cfg: ExperimentConfig) -> np.ndarray:
    """Decode one live frame the way :class:`MultimodalImageDataset` decodes a training one.

    This is the serving side of the train/serve pair, so the chain has to match
    the dataset's exactly: decode -> ``[0, 1]`` -> preprocess at native
    resolution -> resize -> standardize. Augmentation is training-only and has
    no counterpart here.

    Parameters
    ----------
    image_path:
        Frame to read; any PIL-readable format.
    size:
        Side of the square the model expects, in pixels.
    cfg:
        The checkpoint's own config, which carries the preprocessing settings
        the model was trained under.

    Returns
    -------
    numpy.ndarray
        ``(3, size, size)`` float32, standardized by the DINOv2 channel stats —
        dimensionless, not ``[0, 1]``.
    """
    from PIL import Image

    from allsky.preprocessing import PreprocessingPipeline, imagenet_standardize

    with Image.open(image_path) as handle:
        frame = handle.convert("RGB")
    preprocess = PreprocessingPipeline(**cfg.preprocessing.model_dump())
    if preprocess.enabled:
        native = np.asarray(frame, dtype=np.uint8).astype(np.float32) / 255.0
        processed = preprocess(native.transpose(2, 0, 1))
        frame = Image.fromarray((processed.transpose(1, 2, 0) * 255.0).round().astype(np.uint8))
    if frame.size != (size, size):
        frame = frame.resize((size, size), Image.Resampling.BILINEAR)
    scaled = np.asarray(frame, dtype=np.uint8).astype(np.float32) / 255.0
    return imagenet_standardize(np.ascontiguousarray(scaled.transpose(2, 0, 1)))


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
    except FileNotFoundError, OSError, ValueError:
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

    backbone = build_backbone(
        meta["backbone"],
        device=device,
        pooling=meta["pooling"],
        dtype=meta["dtype"],
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


def predict_snapshot(
    image_path: str | Path,
    checkpoint_path: str | Path,
    *,
    timestamp: pd.Timestamp,
    sensor_csv: str | Path | None = None,
    tolerance: pd.Timedelta = DEFAULT_SENSOR_TOLERANCE,
    site: SiteConfig | None = None,
    device: str = "cpu",
    trust_checkpoint: bool = False,
    embeddings_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a trained checkpoint over one sky image and return physical-unit predictions.

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
        accepted; beyond it the row is discarded and the columns imputed.
    site:
        Observation site for the solar geometry; defaults to
        :class:`~allsky.config.SiteConfig`. The geometry is built at the pinned
        :data:`~allsky.config.SITE_UTC_OFFSET_HOURS` the manifest builder trains
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
        :data:`allsky.data.contracts.SKY_CLASS_NAMES`. ``features`` records the
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
    import torch

    from allsky.data.contracts import SKY_CLASS_NAMES
    from allsky.features.normalization import FeatureNormalizer, TargetNormalizer
    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy
    from allsky.training.checkpointing import load_checkpoint
    from allsky.training.engine import _default_image_backbone_builder, _model_param

    checkpoint = load_checkpoint(
        checkpoint_path, map_location=device, trust_pickle=trust_checkpoint
    )
    cfg = ExperimentConfig.model_validate(checkpoint["config"])
    if embeddings_dir is not None and cfg.data.input_mode != "embedding":
        raise ValueError(
            f"embeddings_dir was given for an input_mode={cfg.data.input_mode!r} checkpoint, "
            "which encodes the live frame with its own backbone and reads no embedding store"
        )
    feature_columns: list[str] = list(checkpoint["feature_columns"])
    normalizers = checkpoint["normalizers"]
    feature_normalizer = FeatureNormalizer.from_dict(normalizers["feature_normalizer"])
    target_normalizers = {
        key: TargetNormalizer.from_dict(value)
        for key, value in normalizers["target_normalizers"].items()
    }

    raw_values, imputed = _feature_vector(
        timestamp,
        feature_columns=feature_columns,
        feature_set=cfg.features.feature_set,
        site=site or SiteConfig(),
        sensor_csv=sensor_csv,
        tolerance=tolerance,
        training_means=feature_normalizer.mean,
    )
    standardized = feature_normalizer.transform(pd.DataFrame([raw_values], columns=feature_columns))
    image_size = int(_model_param(cfg, "image_size", 224))

    batch: dict[str, Any] = {"features": torch.from_numpy(standardized).to(device)}
    image_backbone = None
    embedding_dim = None
    if cfg.data.input_mode == "image":
        image_backbone = _default_image_backbone_builder(cfg, device)()
        batch["image"] = (
            torch.from_numpy(_image_as_chw(image_path, image_size, cfg)).unsqueeze(0).to(device)
        )
    else:
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
        backbone = _backbone_matching_recipe(source, store_meta, device)
        # Through transform(), never straight into encode(): the backbone's
        # contract takes a SEQUENCE of (H, W, 3) uint8 HWC frames and does its
        # own resize, ImageNet normalisation and stacking, and that is the
        # recipe precompute-embeddings fed the training store. Handing it the
        # (3, S, S) float array the image branch uses died on the raw ndarray
        # and, had it survived, would have embedded a differently prepared
        # image than the model was fitted on.
        vector = np.asarray(backbone.encode(backbone.transform([_image_as_hwc(image_path)])))
        embedding = np.reshape(vector, (1, -1)).astype(np.float32)
        embedding_dim = int(embedding.shape[1])
        batch["embedding"] = torch.from_numpy(embedding).to(device)

    model = build_model(
        cfg,
        len(feature_columns),
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        temporal_pooling=temporal_pooling_for_strategy(cfg.data.alignment.strategy),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(batch)

    predictions: dict[str, Any] = {}
    for name in ("dhi", "kindex", "cloud_fraction"):
        if name not in outputs:
            continue
        value = float(outputs[name].detach().cpu().numpy().reshape(-1)[0])
        normalizer = target_normalizers.get(name)
        predictions[name] = float(normalizer.denormalize(value)[()]) if normalizer else value
    if "sky_logits" in outputs:
        logits = outputs["sky_logits"].detach().cpu().numpy().reshape(-1)
        weights = np.exp(logits - logits.max())
        probabilities = weights / weights.sum()
        predictions["sky_class"] = SKY_CLASS_NAMES[int(np.argmax(logits))]
        predictions["sky_probabilities"] = {
            name: float(value) for name, value in zip(SKY_CLASS_NAMES, probabilities, strict=True)
        }

    return {
        "predictions": predictions,
        "features": {
            "timestamp": timestamp.isoformat(),
            "feature_set": cfg.features.feature_set,
            "columns": feature_columns,
            "values": [float(value) for value in raw_values],
            "imputed": imputed,
            "sensor_csv": str(sensor_csv) if sensor_csv is not None else None,
        },
        "model": {
            "checkpoint": str(checkpoint_path),
            "name": cfg.name,
            "architecture": cfg.model.name,
            "input_mode": cfg.data.input_mode,
            "device": device,
        },
        "image": str(image_path),
    }
