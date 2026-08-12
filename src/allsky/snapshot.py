"""Captures the camera's live frame and runs a trained checkpoint over it.

Which feature columns a live frame cannot supply, and what imputing them costs,
are documented in ``docs/allsky-archive.md``.
"""

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeIs

import numpy as np
import pandas as pd

from allsky.archive import ArchiveClient
from allsky.atomic import atomic_write, atomic_write_json
from allsky.config import SiteConfig
from allsky.provenance import code_version

logger = logging.getLogger(__name__)

__all__ = [
    "Snapshot",
    "capture_snapshot",
    "predict_snapshot",
]

SNAPSHOT_STEM_FORMAT = "allsky-%Y%m%d-%H%M%S"
DEFAULT_SENSOR_TOLERANCE = pd.Timedelta(minutes=15)
HTTP_DATE_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
SENSOR_TIME_COLUMNS = ("timestamp", "TIMESTAMP", "datetime", "time")
LIVE_FRAME_MAX_AGE = pd.Timedelta(minutes=10)


@dataclass(frozen=True)
class Snapshot:
    image_path: Path
    metadata_path: Path
    captured_at: pd.Timestamp
    size: int
    server_last_modified: str | None
    prediction: dict[str, Any] | None = None


def _naive_local_from_http_date(headers: dict[str, str]) -> pd.Timestamp | None:
    raw = headers.get("last-modified")
    if not raw:
        return None
    try:
        parsed = dt.datetime.strptime(raw, HTTP_DATE_FORMAT).replace(tzinfo=dt.UTC)
    except ValueError:
        logger.warning("unparseable Last-Modified header on the live frame: %r", raw)
        return None
    return pd.Timestamp(parsed.astimezone().replace(tzinfo=None))


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
    client: ArchiveClient, out_dir: str | Path, *, timestamp: pd.Timestamp | None = None
) -> Snapshot:
    """Fetch the current frame into *out_dir* alongside a JSON provenance sidecar."""
    payload, headers = client.fetch_live_image()
    if not payload:
        raise ValueError("the camera returned an empty live frame")

    now = pd.Timestamp.now().floor("s")
    overlay_time, overlay_text = _overlay_timestamp(payload)
    server_time = _naive_local_from_http_date(headers)
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


def _sensor_row_near(
    sensor_csv: str | Path, timestamp: pd.Timestamp, tolerance: pd.Timedelta
) -> pd.DataFrame:
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
        frame.index = index.tz_localize(None)
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
    row = frame.iloc[[position]].copy()
    row.index = pd.DatetimeIndex([timestamp])
    return row


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
    engineered = build_feature_frame(sensor, [timestamp], site, feature_set)
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


def _image_as_chw(image_path: str | Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(image_path) as handle:
        frame = handle.convert("RGB")
    if frame.size != (size, size):
        frame = frame.resize((size, size), Image.Resampling.BILINEAR)
    scaled = np.asarray(frame, dtype=np.uint8).astype(np.float32) / 255.0
    return np.ascontiguousarray(scaled.transpose(2, 0, 1))


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
) -> dict[str, Any]:
    """Run a trained checkpoint over one sky image and return physical-unit predictions."""
    import torch

    from allsky.config import ExperimentConfig
    from allsky.data.contracts import SKY_CLASS_NAMES
    from allsky.features.normalization import FeatureNormalizer, TargetNormalizer
    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy
    from allsky.training.checkpointing import load_checkpoint
    from allsky.training.engine import _default_image_backbone_builder, _model_param

    checkpoint = load_checkpoint(
        checkpoint_path, map_location=device, trust_pickle=trust_checkpoint
    )
    cfg = ExperimentConfig.model_validate(checkpoint["config"])
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
            torch.from_numpy(_image_as_chw(image_path, image_size)).unsqueeze(0).to(device)
        )
    else:
        from allsky.embeddings.backbone import build_backbone

        backbone = build_backbone(
            _model_param(cfg, "backbone", "dinov2_vits14"),
            device=device,
            pooling=_model_param(cfg, "backbone_pooling", "cls"),
        )
        vector = np.asarray(backbone.encode(_image_as_chw(image_path, image_size)))
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
