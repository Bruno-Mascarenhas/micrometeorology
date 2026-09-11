"""``frame.json`` and its three images: the latest scored live frame, explained.

The prediction is the watch's own record for the newest frame it scored, in
the ensemble shape (:func:`allsky.watch.envelope_of` wraps a bare record a
single-member watch wrote before that shape became the only one).
What the publisher adds is computed here from two :class:`ServedModel`
instances loaded once per publish: the attribution member (the occlusion map
and the overlay counterfactual) and the scalars-only control (the "no image"
counterfactual at the same instant).

The images are re-encoded only when the scored frame changes; their content
hashes travel in the document so the page can cache-bust them one by one.
Between changes the sensitivity results are reused from a small state file
under the watch directory, so a publish at night costs no forward pass.

Timestamps are naive station-local, as in every watch record.
"""

import hashlib
import io
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.archive import STATE_SUBDIR
from allsky.attribution import (
    OCCLUSION_STRIDE_PX,
    OCCLUSION_WINDOW_PX,
    OVERLAY_TEXT_BAND_RAW,
    Box,
    frame_geometry_of,
    neutralise_region,
    occlusion_map,
    region_masks,
    render_attribution_rgba,
)
from allsky.config import SiteConfig
from allsky.publish.encoding import (
    DEFAULT_KINDEX_KIND,
    FRAME_SCHEMA,
    INDEX_DECIMALS,
    IRRADIANCE_DECIMALS,
    REFERENCES,
    PublishStamp,
    condition_of,
    document_header,
    kindex_kind_of,
    rounded_or_none,
    sky_conditions_block,
    targets_glossary,
)
from allsky.serving import ServingConfig
from allsky.snapshot import (
    ServedModel,
    Snapshot,
    StationExport,
    clearsky_dhi_at,
    solar_elevation_at,
)
from allsky.watch import (
    FRAMES_SUBDIR,
    PREDICTION_SUFFIX,
    checkpoint_role,
    envelope_of,
    frames_newest_first,
)
from labmim_core.atomic import JsonObjectError, atomic_write_strict_json, read_json_object
from labmim_core.sky import SKY_CLASS_NAMES

logger = logging.getLogger(__name__)

__all__ = [
    "ATTRIBUTION_FILENAME",
    "IMAGE_FILENAME",
    "INPUT_FILENAME",
    "FrameArtifacts",
    "FramePublishError",
    "FrameStatus",
    "LoadedControls",
    "ScoredFrame",
    "build_frame_artifacts",
    "frame_status",
    "latest_scored_frame",
]

IMAGE_FILENAME = "allsky.jpg"
INPUT_FILENAME = "input.jpg"
ATTRIBUTION_FILENAME = "attribution.png"
FRAME_STATE_FILENAME = "publish-frame.json"

PUBLISHED_IMAGE_WIDTH = 1280
IMAGE_JPEG_QUALITY = 85
INPUT_JPEG_QUALITY = 90
HASH_PREFIX_LENGTH = 12
PROBABILITY_DECIMALS = 4
ANGLE_DECIMALS = 2
GRID_DECIMALS = 4
MASS_DECIMALS = 3
DEFAULT_STALE_AFTER_BLOCKS = 3
SEED_SUFFIX = re.compile(r"_s(\d+)$")
#: The camera's web server labels its Last-Modified header with the wrong zone
#: (local time called GMT), so the header sits whole hours away from the
#: overlay stamp; the drift is what remains after that whole-hour label error.
SECONDS_PER_HOUR = 3600.0

STATUS_FRESH = "fresh"
STATUS_NIGHT = "night"
STATUS_NO_SCORED_FRAME = "no_scored_frame"
STATUS_WATCH_STALE = "watch_stale"
STATUS_LABELS_PT: dict[str, str] = {
    STATUS_FRESH: "quadro pontuado há pouco",
    STATUS_NIGHT: "sol abaixo do piso de elevação do modelo; o último quadro pontuado do dia fica em exibição",
    STATUS_NO_SCORED_FRAME: "nenhum quadro foi pontuado ainda",
    STATUS_WATCH_STALE: "o sol está acima do piso e nenhum quadro foi pontuado nos últimos blocos: a captura parece parada",
}

CAVEATS = [
    "A previsão é a média dos membros servidos sobre o quadro JPEG original da câmera; os quadros de treino saíram do timelapse H.264 do dia, e a diferença entre os dois caminhos não foi medida.",
    "O controle sem imagem é a rede só de escalares (geometria solar e anemômetro) treinada sobre o mesmo split; a diferença entre ele e a previsão é o que a imagem acrescenta neste instante. Sem a exportação da estação, o anemômetro entra pela média do treino, e o registro diz quais colunas foram imputadas. A rede servida não lê esses escalares: eles entram só pelo denominador de céu claro da difusa.",
    "A faixa de texto neutralizada é a legenda que a câmera grava no canto (relógio, temperatura do sensor, tempo de exposição), substituída pelo nível médio da rede com o disco do céu intacto; o tempo de exposição correlaciona forte com a difusa, e este número mede quanto a rede se apoia nele.",
    "O mapa de sensibilidade é causal e grosseiro: cada célula é a variação de k* quando uma janela quadrada é substituída pelo nível médio da rede; não é atenção nem segmentação, e a resolução é a da janela.",
    "As probabilidades de condição são as da cabeça de classificação sobre as faixas de Kt de [[escobedo]] aplicadas à média de 5 minutos do piranômetro, que é o rótulo que a rede aprendeu a reproduzir.",
]

FRAME_REFERENCES = {key: REFERENCES[key] for key in ("escobedo", "haurwitz")}


class FramePublishError(ValueError):
    """The watch record cannot be published as the pin's prediction."""


@dataclass(frozen=True, slots=True)
class ScoredFrame:
    """The newest frame the watch scored: its capture, its prediction record and when the record was written."""

    snapshot: Snapshot
    record: dict[str, Any]
    record_written_at_utc: pd.Timestamp

    def camera_clock_offset_s(self) -> float | None:
        """Seconds between the server's Last-Modified (as local time) and the overlay stamp; ``None`` without the header."""
        sidecar = json.loads(self.snapshot.metadata_path.read_text(encoding="utf-8"))
        server_local = sidecar.get("server_last_modified_as_local")
        if not server_local:
            return None
        return float((pd.Timestamp(server_local) - self.snapshot.captured_at).total_seconds())


@dataclass(frozen=True, slots=True)
class FrameStatus:
    """Whether the published frame is current, and why not when it is not.

    Liveness is judged on the host clock — when the newest prediction record
    was written — so a camera whose clock drifts cannot make a healthy watch
    look dead; the capture stamp itself stays the camera's.
    """

    scored: bool
    reason: str
    latest_scored_at: pd.Timestamp | None
    solar_elevation_deg: float
    watch_alive: bool
    camera_clock_offset_s: float | None

    @property
    def camera_clock_drift_s(self) -> float | None:
        """The header-vs-overlay offset with its whole-hour zone-label error removed."""
        if self.camera_clock_offset_s is None:
            return None
        hours = round(self.camera_clock_offset_s / SECONDS_PER_HOUR)
        return self.camera_clock_offset_s - hours * SECONDS_PER_HOUR

    def as_dict(self) -> dict[str, Any]:
        return {
            "scored": self.scored,
            "reason": self.reason,
            "reason_pt": STATUS_LABELS_PT[self.reason],
            "latest_scored_at": (
                self.latest_scored_at.isoformat() if self.latest_scored_at is not None else None
            ),
            "solar_elevation_deg": rounded_or_none(self.solar_elevation_deg, ANGLE_DECIMALS),
            "watch_alive": self.watch_alive,
            "camera_clock_offset_s": rounded_or_none(self.camera_clock_offset_s, 0),
            "camera_clock_drift_s": rounded_or_none(self.camera_clock_drift_s, 0),
        }


@dataclass(frozen=True, slots=True)
class LoadedControls:
    """The two pinned controls, loaded once: the scalars-only MLP and the train means."""

    sensor_only: ServedModel
    climatology: ServedModel


@dataclass(frozen=True, slots=True)
class FrameArtifacts:
    """The document and the images one publish writes; ``images`` is empty when they are unchanged."""

    document: dict[str, Any]
    images: dict[str, bytes]


def latest_scored_frame(watch_dir: str | Path) -> ScoredFrame | None:
    """The newest frame under ``<watch_dir>/frames/`` that has a readable prediction record."""
    frames_dir = Path(watch_dir) / FRAMES_SUBDIR
    if not frames_dir.is_dir():
        return None
    for snapshot in frames_newest_first(frames_dir):
        record_path = frames_dir / f"{snapshot.image_path.stem}{PREDICTION_SUFFIX}"
        if not record_path.is_file():
            continue
        try:
            record: Any = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("skipping unreadable prediction record %s: %s", record_path, exc)
            continue
        if isinstance(record, dict) and isinstance(record.get("predictions"), dict):
            written = pd.Timestamp(record_path.stat().st_mtime, unit="s", tz="UTC")
            return ScoredFrame(
                snapshot=snapshot, record=envelope_of(record), record_written_at_utc=written
            )
    return None


def frame_status(
    latest: ScoredFrame | None,
    *,
    now_local: pd.Timestamp,
    now_host_utc: pd.Timestamp,
    site: SiteConfig,
    min_elevation_deg: float,
    block_minutes: float,
    stale_after_blocks: int = DEFAULT_STALE_AFTER_BLOCKS,
) -> FrameStatus:
    """Judge the newest scored frame against the clock and the sun.

    With the sun above the elevation floor for the whole liveness window, a
    prediction record written more than *stale_after_blocks* blocks before
    *now_host_utc* means the watch stopped scoring; below the floor, or in the
    first blocks after the sun crosses it, an old record is legitimate.

    Parameters
    ----------
    now_local:
        The publish instant on the station's clock, naive, for the sun.
    now_host_utc:
        The publish instant on the host's clock, aware UTC, compared with the
        record's modification time.
    """
    window = pd.Timedelta(minutes=block_minutes * stale_after_blocks)
    elevation = solar_elevation_at(now_local, site)
    daylight = elevation >= min_elevation_deg
    daylight_for_the_whole_window = (
        daylight and solar_elevation_at(now_local - window, site) >= min_elevation_deg
    )
    if latest is None:
        return FrameStatus(
            scored=False,
            reason=STATUS_NO_SCORED_FRAME,
            latest_scored_at=None,
            solar_elevation_deg=elevation,
            watch_alive=not daylight_for_the_whole_window,
            camera_clock_offset_s=None,
        )
    stale = now_host_utc - latest.record_written_at_utc > window
    if daylight_for_the_whole_window and stale:
        reason, alive = STATUS_WATCH_STALE, False
    elif daylight:
        reason, alive = STATUS_FRESH, True
    else:
        reason, alive = STATUS_NIGHT, True
    return FrameStatus(
        scored=True,
        reason=reason,
        latest_scored_at=latest.snapshot.captured_at,
        solar_elevation_deg=elevation,
        watch_alive=alive,
        camera_clock_offset_s=latest.camera_clock_offset_s(),
    )


def _seed_of(name: str) -> int | None:
    match = SEED_SUFFIX.search(name)
    return int(match.group(1)) if match else None


def _digest_of(pin: ServingConfig, digests: dict[Path, str], checkpoint: str) -> str:
    resolved = Path(checkpoint).resolve()
    if resolved not in digests:
        raise FramePublishError(
            f"the watch scored this frame with {checkpoint}, which pin {pin.id} does not serve; "
            "restart the watch with --serving on the current pin"
        )
    return digests[resolved]


def _members(
    pin: ServingConfig, digests: dict[Path, str], record: dict[str, Any]
) -> list[dict[str, Any]]:
    models = {str(model.get("checkpoint")): model for model in record.get("models") or []}
    entries = []
    for member in record["members"]:
        checkpoint = str(member["checkpoint"])
        model = models.get(checkpoint) or {}
        name = str(model.get("name") or Path(checkpoint).parent.name)
        entries.append(
            {
                "name": name,
                "seed": _seed_of(name),
                "role": str(member.get("role") or checkpoint_role(checkpoint)),
                "heads": list(member.get("heads") or []),
                "checkpoint_sha256": _digest_of(pin, digests, checkpoint),
                "code_version": model.get("code_version"),
            }
        )
    return entries


def _kindex_kind(record: dict[str, Any]) -> str:
    kind = kindex_kind_of(record)
    if kind is not None:
        return kind
    logger.warning(
        "the watch record names no kindex_kind; publishing the %s glossary", DEFAULT_KINDEX_KIND
    )
    return DEFAULT_KINDEX_KIND


def _sky_block(predictions: dict[str, Any]) -> dict[str, Any] | None:
    class_name = predictions.get("sky_class")
    if class_name not in SKY_CLASS_NAMES:
        return None
    probabilities = predictions.get("sky_probabilities") or {}
    return {
        **condition_of(SKY_CLASS_NAMES.index(class_name)),
        "probabilities": {
            condition_of(index)["id"]: rounded_or_none(
                probabilities.get(name), PROBABILITY_DECIMALS
            )
            for index, name in enumerate(SKY_CLASS_NAMES)
        },
    }


def _prediction_block(predictions: dict[str, Any]) -> dict[str, Any]:
    return {
        "dhi_w_m2": rounded_or_none(predictions.get("dhi"), IRRADIANCE_DECIMALS),
        "kindex": rounded_or_none(predictions.get("kindex"), INDEX_DECIMALS),
        "sky": _sky_block(predictions),
    }


def _delta(counterfactual: dict[str, Any], base: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, decimals in (("dhi_w_m2", IRRADIANCE_DECIMALS), ("kindex", INDEX_DECIMALS)):
        after, before = counterfactual.get(key), base.get(key)
        out[key] = (
            rounded_or_none(float(after) - float(before), decimals)
            if after is not None and before is not None
            else None
        )
    return out


def _solar_block(
    timestamp: pd.Timestamp, site: SiteConfig, train_max_elevation_deg: float | None
) -> dict[str, Any]:
    from allsky.clearsky import haurwitz_ghi
    from labmim_core.solar import solar_azimuth_deg

    local = pd.DatetimeIndex([timestamp])
    elevation = solar_elevation_at(timestamp, site)
    return {
        "elevation_deg": rounded_or_none(elevation, ANGLE_DECIMALS),
        "azimuth_deg": rounded_or_none(
            float(solar_azimuth_deg(local, site, site.utc_offset_hours)[0]), ANGLE_DECIMALS
        ),
        "clearsky_ghi_w_m2": rounded_or_none(
            float(np.asarray(haurwitz_ghi(local, site, site.utc_offset_hours))[0]),
            IRRADIANCE_DECIMALS,
        ),
        "clearsky_dhi_w_m2": rounded_or_none(clearsky_dhi_at(timestamp, site), IRRADIANCE_DECIMALS),
        "extrapolation": (
            bool(elevation > train_max_elevation_deg)
            if train_max_elevation_deg is not None
            else None
        ),
    }


def _control_test_errors(report_dir: Path) -> dict[str, float | None]:
    """The control's own test error, read beside its live counterfactual; nulls when unreadable."""
    path = Path(report_dir) / "eval_metrics.json"
    try:
        report: Any = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        logger.warning("control report %s is unreadable; publishing its test error as null", path)
        return {"dhi_rmse": None, "kindex_mae": None}
    scores = report.get("global") or {} if isinstance(report, dict) else {}
    return {
        "dhi_rmse": rounded_or_none((scores.get("dhi") or {}).get("rmse"), IRRADIANCE_DECIMALS),
        "kindex_mae": rounded_or_none((scores.get("kindex") or {}).get("mae"), INDEX_DECIMALS),
    }


def _percent(value: float) -> str:
    return f"{round(value * 100):d} %"


def _attribution_summary_pt(mass: dict[str, float], peak_in_disc: bool) -> str:
    where = "no disco do céu" if peak_in_disc else "fora do disco do céu"
    return (
        f"{_percent(mass['disc'])} da sensibilidade no disco do céu, "
        f"{_percent(mass['overlay_band'])} na faixa de texto da câmera, "
        f"{_percent(mass['pad'])} na borda preta; pico {where}."
    )


IMAGE_ALT_PT = "Quadro bruto da câmera all-sky, o JPEG original reduzido para a página"
INPUT_ALT_PT = (
    "Exatamente a imagem que a rede recebeu: recorte, preenchimento preto e redimensionamento"
)
ATTRIBUTION_ALT_PT = (
    "Mapa de sensibilidade à oclusão sobre o quadro: quanto mais opaco, mais a previsão "
    "muda quando aquela região é substituída pelo nível médio da rede"
)


def _short_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:HASH_PREFIX_LENGTH]


def _encode_jpeg(array: np.ndarray, quality: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _encode_png(array: np.ndarray) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(array, "RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _published_size(raw_width: int, raw_height: int) -> tuple[int, int]:
    width = min(PUBLISHED_IMAGE_WIDTH, raw_width)
    return width, round(raw_height * width / raw_width)


def _read_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return read_json_object(path)
    except JsonObjectError:
        return {}


@dataclass(frozen=True, slots=True)
class _Probes:
    """Everything computed by forward passes for one scored frame, cacheable by capture time."""

    images: dict[str, bytes]
    image_meta: dict[str, Any]
    input_meta: dict[str, Any]
    field_of_view: dict[str, float]
    counterfactuals: dict[str, Any]
    attribution: dict[str, Any]

    def cache(self) -> dict[str, Any]:
        return {
            "image": self.image_meta,
            "input": self.input_meta,
            "field_of_view": self.field_of_view,
            "counterfactuals": self.counterfactuals,
            "attribution": self.attribution,
        }


UNMODELLED_OVERLAY_POLICIES = ("crop",)


def _probe(
    scored: ScoredFrame,
    *,
    pin: ServingConfig,
    served: ServedModel,
    controls: LoadedControls,
    site: SiteConfig,
    prediction: dict[str, Any],
    device_note: str,
    occlusion_window_px: int,
    occlusion_stride_px: int,
    sensor_csv: Path | StationExport | None,
) -> _Probes:
    from PIL import Image

    if served.cfg.preprocessing.overlay in UNMODELLED_OVERLAY_POLICIES:
        raise FramePublishError(
            f"the served checkpoint preprocesses with overlay={served.cfg.preprocessing.overlay!r}, "
            "which shifts the camera pixels after the recorded geometry; the published boxes and "
            "the attribution warp do not model it"
        )
    timestamp = scored.snapshot.captured_at
    raw = Image.open(scored.snapshot.image_path).convert("RGB")
    raw_width, raw_height = raw.size
    out_width, out_height = _published_size(raw_width, raw_height)
    frame = frame_geometry_of(
        served.geometry, raw_height=raw_height, raw_width=raw_width, input_size=served.image_size
    )

    features = served.scalar_features(timestamp, site=site)
    input_frame = served.input_frame(scored.snapshot.image_path)
    planes = served.planes_of(input_frame, timestamp, site=site)
    base = served.physical(
        served.forward(served.batch(features, planes=planes)), timestamp=timestamp, site=site
    )
    band_top, band_left, band_h, band_w = OVERLAY_TEXT_BAND_RAW
    band = frame.raw_to_input(Box(band_top, band_left, band_h, band_w))
    overlay = neutralise_region(served, planes, features, box=band, timestamp=timestamp, site=site)
    no_image: dict[str, Any] = {}
    for control_id, control, report in (
        ("sensor_only", controls.sensor_only, pin.controls.sensor_only.report),
        ("climatology", controls.climatology, pin.controls.climatology.report),
    ):
        control_features = control.scalar_features(timestamp, site=site, sensor_csv=sensor_csv)
        said = control.physical(
            control.forward(control.batch(control_features)), timestamp=timestamp, site=site
        )
        block = _prediction_block(said)
        no_image[control_id] = {
            "kind": f"{control_id}_control",
            "member": control.cfg.name,
            **block,
            "delta": _delta(block, prediction),
            "test": _control_test_errors(report),
            "scalars_imputed": list(control_features.imputed),
            "station_export": sensor_csv is not None,
        }
    occlusion = occlusion_map(
        served,
        planes,
        features,
        timestamp=timestamp,
        site=site,
        target=pin.attribution_target,
        window_px=occlusion_window_px,
        stride_px=occlusion_stride_px,
    )
    masks = region_masks(frame)
    mass = masks.mass_of(occlusion.rasterised(frame.input_size))
    peak_row, peak_col = occlusion.peak
    peak_top, peak_left = occlusion.positions[peak_row * occlusion.grid.shape[1] + peak_col]
    peak_centre = (
        int(peak_top + occlusion.window_px // 2),
        int(peak_left + occlusion.window_px // 2),
    )
    peak_in_disc = bool(masks.disc[peak_centre])

    published = np.asarray(raw.resize((out_width, out_height), Image.Resampling.LANCZOS))
    image_bytes = _encode_jpeg(published, IMAGE_JPEG_QUALITY)
    input_bytes = _encode_jpeg(
        np.ascontiguousarray((input_frame.transpose(1, 2, 0) * 255.0).round().astype(np.uint8)),
        INPUT_JPEG_QUALITY,
    )
    attribution_bytes = _encode_png(
        render_attribution_rgba(occlusion, frame, out_width=out_width, out_height=out_height)
    )
    scale = out_width / raw_width
    base_block = _prediction_block(base)
    overlay_block = _prediction_block(overlay)
    return _Probes(
        images={
            IMAGE_FILENAME: image_bytes,
            INPUT_FILENAME: input_bytes,
            ATTRIBUTION_FILENAME: attribution_bytes,
        },
        image_meta={
            "file": IMAGE_FILENAME,
            "sha256_12": _short_hash(image_bytes),
            "width": out_width,
            "height": out_height,
            "original_width": raw_width,
            "original_height": raw_height,
            "source": "camera JPEG",
            "alt_pt": IMAGE_ALT_PT,
        },
        input_meta={
            "file": INPUT_FILENAME,
            "sha256_12": _short_hash(input_bytes),
            "size": served.image_size,
            "geometry": served.geometry.model_dump(include={"crop", "pad", "resize"})
            if served.geometry is not None
            else None,
            "preprocess": served.cfg.preprocessing.model_dump(),
            "content_box": frame.content_box.as_dict(),
            "alt_pt": INPUT_ALT_PT,
        },
        field_of_view=frame.input_to_raw(frame.content_box).scaled(scale).as_dict(),
        counterfactuals={
            "no_image": no_image,
            "overlay_neutralised": {
                "kind": "overlay_text_band_neutralised",
                "member": served.cfg.name,
                "base": base_block,
                **overlay_block,
                "delta": _delta(overlay_block, base_block),
                "band": band.as_dict(),
            },
        },
        attribution={
            "file": ATTRIBUTION_FILENAME,
            "sha256_12": _short_hash(attribution_bytes),
            "alt_pt": ATTRIBUTION_ALT_PT,
            "summary_pt": _attribution_summary_pt(mass, peak_in_disc),
            "method": "occlusion_sensitivity",
            "fill": "network_mean_level",
            "window_px": occlusion.window_px,
            "stride_px": occlusion.stride_px,
            "target": occlusion.target,
            "member": served.cfg.name,
            "device": device_note,
            "base_value": rounded_or_none(
                occlusion.base_value,
                INDEX_DECIMALS if occlusion.target == "kindex" else IRRADIANCE_DECIMALS,
            ),
            "grid_shape": list(occlusion.grid.shape),
            "grid": [
                [rounded_or_none(value, GRID_DECIMALS) for value in row]
                for row in occlusion.grid.tolist()
            ],
            "mass_by_region": {
                key: rounded_or_none(value, MASS_DECIMALS) for key, value in mass.items()
            },
            "peak": {"row": occlusion.peak[0], "col": occlusion.peak[1]},
        },
    )


def _probe_fingerprint(
    pin: ServingConfig,
    digests: dict[Path, str],
    *,
    window_px: int,
    stride_px: int,
    device: str,
    station_export: bool,
) -> dict[str, Any]:
    """Everything the probes depend on besides the frame; a change is a cache miss."""
    return {
        "attribution_member": digests.get(pin.attribution_member.path.resolve()),
        "attribution_target": pin.attribution_target,
        "sensor_only": digests.get(pin.controls.sensor_only.checkpoint.path.resolve()),
        "climatology": digests.get(pin.controls.climatology.checkpoint.path.resolve()),
        "window_px": window_px,
        "stride_px": stride_px,
        "device": device,
        "station_export": station_export,
    }


def _images_present(out_dir: Path, probes: Mapping[str, Any]) -> bool:
    """Whether every image the cached probes name is in *out_dir* with the recorded hash."""
    for key in ("image", "input", "attribution"):
        meta = probes.get(key)
        if not isinstance(meta, dict):
            return False
        path = out_dir / str(meta.get("file", ""))
        if not path.is_file() or _short_hash(path.read_bytes()) != meta.get("sha256_12"):
            return False
    return True


def build_frame_artifacts(
    watch_dir: str | Path,
    *,
    pin: ServingConfig,
    digests: dict[Path, str],
    served: ServedModel,
    controls: LoadedControls,
    stamp: PublishStamp,
    now_local: pd.Timestamp,
    now_host_utc: pd.Timestamp,
    site: SiteConfig,
    out_dir: Path,
    block_minutes: float,
    train_max_elevation_deg: float | None,
    stale_after_blocks: int = DEFAULT_STALE_AFTER_BLOCKS,
    occlusion_window_px: int = OCCLUSION_WINDOW_PX,
    occlusion_stride_px: int = OCCLUSION_STRIDE_PX,
    sensor_csv: Path | StationExport | None = None,
) -> FrameArtifacts:
    """Assemble ``frame.json`` and, when the scored frame changed, its three images.

    Parameters
    ----------
    watch_dir:
        The watch root (``frames/``, ``blocks/``, ``.state/``).
    pin:
        The verified serving pin.
    digests:
        Resolved checkpoint path -> SHA-256, for every checkpoint the pin
        fingerprints.
    served, controls:
        The attribution member and the two controls, loaded once.
    now_local, now_host_utc:
        The publish instant on the station's clock (naive, for the sun) and
        on the host's clock (aware UTC, for liveness).
    out_dir:
        Where the CLI writes the documents: the probe cache is reused only
        while the three images it names are there with the hashes it recorded.
    sensor_csv:
        Station export fed to the controls' scalar vector, so the live
        "no image" reference runs on measured scalars when one is supplied;
        a path, or the export already read so one publish parses it once.
    train_max_elevation_deg:
        Highest solar elevation in the training split, for the
        ``extrapolation`` flag; ``None`` publishes it as unknown.

    Returns
    -------
    FrameArtifacts
        The document, and the image bytes to write before it when the frame
        changed since the last publish.

    Raises
    ------
    FramePublishError
        When the watch scored the frame with checkpoints the pin does not
        serve.
    """
    latest = latest_scored_frame(watch_dir)
    status = frame_status(
        latest,
        now_local=now_local,
        now_host_utc=now_host_utc,
        site=site,
        min_elevation_deg=pin.min_elevation_deg,
        block_minutes=block_minutes,
        stale_after_blocks=stale_after_blocks,
    )
    kindex_kind = _kindex_kind(latest.record) if latest is not None else DEFAULT_KINDEX_KIND
    document: dict[str, Any] = {
        **document_header(FRAME_SCHEMA, stamp),
        "status": status.as_dict(),
        "captured_at": None,
        "captured_at_source": None,
        "image": None,
        "input": None,
        "field_of_view": None,
        "solar": None,
        "prediction": None,
        "counterfactuals": None,
        "attribution": None,
        "members": [],
        "targets": targets_glossary(kindex_kind),
        "sky_conditions": sky_conditions_block(),
        "caveats": CAVEATS,
        "references": FRAME_REFERENCES,
    }
    if latest is None:
        return FrameArtifacts(document=document, images={})

    prediction = _prediction_block(latest.record["predictions"])
    members = _members(pin, digests, latest.record)
    state_path = Path(watch_dir) / STATE_SUBDIR / FRAME_STATE_FILENAME
    state = _read_state(state_path)
    captured = latest.snapshot.captured_at.isoformat()
    fingerprint = _probe_fingerprint(
        pin,
        digests,
        window_px=occlusion_window_px,
        stride_px=occlusion_stride_px,
        device=served.device,
        station_export=sensor_csv is not None,
    )
    images: dict[str, bytes] = {}
    cached = state.get("probes") if isinstance(state.get("probes"), dict) else None
    if (
        cached is not None
        and state.get("captured_at") == captured
        and state.get("fingerprint") == fingerprint
        and _images_present(out_dir, cached)
    ):
        probes_cache: dict[str, Any] = dict(cached)
        logger.info("frame %s already probed; reusing the cached sensitivity results", captured)
    else:
        probes = _probe(
            latest,
            pin=pin,
            served=served,
            controls=controls,
            site=site,
            prediction=prediction,
            device_note=served.device,
            occlusion_window_px=occlusion_window_px,
            occlusion_stride_px=occlusion_stride_px,
            sensor_csv=sensor_csv,
        )
        probes_cache = probes.cache()
        images = probes.images
        state_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_strict_json(
            state_path,
            {"captured_at": captured, "fingerprint": fingerprint, "probes": probes_cache},
        )

    sidecar = json.loads(latest.snapshot.metadata_path.read_text(encoding="utf-8"))
    document.update(
        {
            "captured_at": captured,
            "captured_at_source": sidecar.get("captured_at_source"),
            "image": probes_cache["image"],
            "input": probes_cache["input"],
            "field_of_view": probes_cache["field_of_view"],
            "solar": _solar_block(latest.snapshot.captured_at, site, train_max_elevation_deg),
            "prediction": prediction,
            "counterfactuals": probes_cache["counterfactuals"],
            "attribution": probes_cache["attribution"],
            "members": members,
        }
    )
    return FrameArtifacts(document=document, images=images)
