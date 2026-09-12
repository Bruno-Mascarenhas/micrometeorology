"""frame.json: status, member normalisation, probes, the probe cache and the no-path guarantee."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from allsky.config import SiteConfig
from allsky.publish.encoding import publish_stamp
from allsky.publish.frame import (
    STATUS_FRESH,
    STATUS_NIGHT,
    STATUS_NO_SCORED_FRAME,
    STATUS_WATCH_STALE,
    FramePublishError,
    FrameStatus,
    LoadedControls,
    build_frame_artifacts,
    frame_status,
    latest_scored_frame,
)
from allsky.serving import ServingConfig
from allsky.snapshot import load_served_model, solar_elevation_at
from tests.allsky._block_probe import stub_image_backbone
from tests.allsky._frame_probe import train_frame_probes
from tests.allsky._pins import control, pinned, serving_pin_payload

SITE = SiteConfig()
NOON = pd.Timestamp("2025-03-20T12:00:00")
MIDNIGHT = pd.Timestamp("2025-03-20T00:00:00")
NOON_UTC = pd.Timestamp("2025-03-20T15:00:00", tz="UTC")
MIDNIGHT_UTC = pd.Timestamp("2025-03-20T03:00:00", tz="UTC")
FLOOR_DEG = 10.0
IMAGE_NAMES = {"allsky.jpg", "input.jpg", "attribution.png"}


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    return train_frame_probes(tmp_path_factory.mktemp("probes"))


@pytest.fixture(scope="module")
def models(probes):
    def load(key: str):
        return load_served_model(
            probes[key], trust_checkpoint=True, image_backbone_builder=stub_image_backbone
        )

    return {
        "served": load("probe_s0"),
        "sensor_only": load("sensor_only_s0"),
        "climatology": load("climatology_s0"),
    }


def _controls(models) -> LoadedControls:
    return LoadedControls(sensor_only=models["sensor_only"], climatology=models["climatology"])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _latest(watch: Path):
    latest = latest_scored_frame(watch)
    assert latest is not None
    return latest


def _pin(probes, tmp_path: Path) -> ServingConfig:
    reports = tmp_path / "reports"
    return ServingConfig.model_validate(
        serving_pin_payload(
            frame_checkpoints=[
                pinned(probes["probe_s0"], _sha(probes["probe_s0"]), reports / "member")
            ],
            sensor_only=control(
                probes["sensor_only_s0"], _sha(probes["sensor_only_s0"]), reports / "sensor_only"
            ),
            climatology=control(
                probes["climatology_s0"], _sha(probes["climatology_s0"]), reports / "climatology"
            ),
            dataset=probes["dataset"],
            training_history=reports / "metrics.csv",
            min_elevation_deg=FLOOR_DEG,
            decided_on="2025-03-20",
            label="the probe",
        )
    )


def _digests(pin: ServingConfig) -> dict[Path, str]:
    return {member.path.resolve(): member.sha256 for member in pin.verified_checkpoints}


def _watch_dir(tmp_path: Path, probes, models, *, checkpoint: Path | None = None) -> Path:
    watch = tmp_path / "watch"
    frames = watch / "frames"
    frames.mkdir(parents=True)
    source = min((probes["dataset"] / "frames").glob("*.jpg"))
    stem = "allsky-20250320-120000"
    image = frames / f"{stem}.jpg"
    image.write_bytes(source.read_bytes())
    (frames / f"{stem}.json").write_text(
        json.dumps(
            {
                "image": image.name,
                "captured_at": NOON.isoformat(),
                "captured_at_source": "overlay",
                "server_last_modified_as_local": (NOON + pd.Timedelta(seconds=40)).isoformat(),
            }
        )
    )
    served = models["served"]
    features = served.scalar_features(NOON, site=SITE)
    planes = served.image_planes(image, NOON, site=SITE)
    predictions = served.physical(
        served.forward(served.batch(features, planes=planes)), timestamp=NOON, site=SITE
    )
    record = {
        "predictions": predictions,
        "features": features.record(NOON, "safe"),
        "model": {
            "checkpoint": str(checkpoint or served.checkpoint_path),
            "name": served.cfg.name,
            "kindex_kind": "kstar",
            "code_version": {"git_commit": "abc"},
        },
        "image": str(image),
    }
    (frames / f"{stem}.prediction.json").write_text(json.dumps(record))
    return watch


def _status(latest, *, now_local, now_host_utc):
    return frame_status(
        latest,
        now_local=now_local,
        now_host_utc=now_host_utc,
        site=SITE,
        min_elevation_deg=FLOOR_DEG,
        block_minutes=5.0,
    )


def _sunrise_plus(minutes: float) -> pd.Timestamp:
    probe = pd.Timestamp("2025-03-20T04:00:00")
    while solar_elevation_at(probe, SITE) < FLOOR_DEG:
        probe += pd.Timedelta(minutes=1)
    return probe + pd.Timedelta(minutes=minutes)


def test_no_scored_frame_at_night_says_so_with_the_watch_presumed_alive():
    status = _status(None, now_local=MIDNIGHT, now_host_utc=MIDNIGHT_UTC)

    assert (status.scored, status.reason) == (False, STATUS_NO_SCORED_FRAME)
    assert status.watch_alive is True


def test_no_scored_frame_in_daylight_flags_the_watch():
    status = _status(None, now_local=NOON, now_host_utc=NOON_UTC)

    assert (status.reason, status.watch_alive) == (STATUS_NO_SCORED_FRAME, False)


def test_a_record_written_moments_ago_in_daylight_is_fresh(tmp_path, probes, models):
    latest = _latest(_watch_dir(tmp_path, probes, models))

    status = _status(
        latest,
        now_local=NOON,
        now_host_utc=latest.record_written_at_utc + pd.Timedelta(minutes=1),
    )

    assert (status.reason, status.watch_alive) == (STATUS_FRESH, True)


def test_the_camera_clock_offset_and_drift_come_from_the_capture_sidecar(tmp_path, probes, models):
    latest = _latest(_watch_dir(tmp_path, probes, models))

    status = _status(
        latest,
        now_local=NOON,
        now_host_utc=latest.record_written_at_utc + pd.Timedelta(minutes=1),
    )

    assert status.camera_clock_offset_s == pytest.approx(40.0)
    assert status.as_dict()["camera_clock_drift_s"] == pytest.approx(40.0)


def test_a_whole_hour_header_label_error_is_not_reported_as_drift():
    status = FrameStatus(True, STATUS_FRESH, NOON, 60.0, True, camera_clock_offset_s=-10802.0)

    assert status.as_dict()["camera_clock_drift_s"] == pytest.approx(-2.0)
    assert status.as_dict()["camera_clock_offset_s"] == -10802.0


def test_a_record_written_four_blocks_ago_in_daylight_means_the_watch_is_stale(
    tmp_path, probes, models
):
    latest = _latest(_watch_dir(tmp_path, probes, models))

    status = _status(
        latest,
        now_local=NOON,
        now_host_utc=latest.record_written_at_utc + pd.Timedelta(minutes=20),
    )

    assert (status.reason, status.watch_alive) == (STATUS_WATCH_STALE, False)


def test_an_old_record_at_night_is_the_last_frame_of_the_day_not_a_dead_watch(
    tmp_path, probes, models
):
    latest = _latest(_watch_dir(tmp_path, probes, models))

    status = _status(
        latest,
        now_local=MIDNIGHT,
        now_host_utc=latest.record_written_at_utc + pd.Timedelta(hours=8),
    )

    assert (status.reason, status.watch_alive) == (STATUS_NIGHT, True)


def test_an_old_record_just_after_the_sun_crossed_the_floor_is_not_stale(tmp_path, probes, models):
    latest = _latest(_watch_dir(tmp_path, probes, models))

    status = _status(
        latest,
        now_local=_sunrise_plus(2.0),
        now_host_utc=latest.record_written_at_utc + pd.Timedelta(hours=14),
    )

    assert (status.reason, status.watch_alive) == (STATUS_FRESH, True)


def _build(tmp_path, probes, models, watch, *, out_dir: Path | None = None, pin=None):
    pin = pin or _pin(probes, tmp_path)
    return build_frame_artifacts(
        watch,
        pin=pin,
        digests=_digests(pin),
        served=models["served"],
        controls=_controls(models),
        stamp=publish_stamp(),
        now_local=NOON,
        now_host_utc=_latest(watch).record_written_at_utc + pd.Timedelta(minutes=1),
        site=SITE,
        out_dir=out_dir or tmp_path / "Ceu",
        block_minutes=5.0,
        train_max_elevation_deg=60.0,
        occlusion_window_px=4,
        occlusion_stride_px=4,
    )


def _write_images(out_dir: Path, artifacts) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.images.items():
        (out_dir / name).write_bytes(payload)


def test_a_scored_frame_publishes_prediction_probes_and_three_images(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)

    artifacts = _build(tmp_path, probes, models, watch)

    document = artifacts.document
    assert set(artifacts.images) == IMAGE_NAMES
    assert document["status"]["scored"] is True
    assert document["prediction"]["sky"]["id"] in ("i", "ii", "iii", "iv")
    assert document["attribution"]["grid_shape"] == [2, 2]
    assert document["members"][0]["seed"] == 0


def test_both_controls_answer_without_the_picture_and_say_what_was_imputed(
    tmp_path, probes, models
):
    watch = _watch_dir(tmp_path, probes, models)

    no_image = _build(tmp_path, probes, models, watch).document["counterfactuals"]["no_image"]

    assert set(no_image) == {"sensor_only", "climatology"}
    assert no_image["sensor_only"]["station_export"] is False
    assert no_image["sensor_only"]["delta"].keys() == {"dhi_w_m2", "kindex"}


def test_the_image_hash_in_the_document_is_the_hash_of_the_bytes_to_write(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)

    artifacts = _build(tmp_path, probes, models, watch)

    expected = hashlib.sha256(artifacts.images["allsky.jpg"]).hexdigest()[:12]
    assert artifacts.document["image"]["sha256_12"] == expected


def test_the_second_publish_of_the_same_frame_reuses_the_probes_and_writes_no_image(
    tmp_path, probes, models
):
    watch = _watch_dir(tmp_path, probes, models)
    first = _build(tmp_path, probes, models, watch)
    _write_images(tmp_path / "Ceu", first)

    second = _build(tmp_path, probes, models, watch)

    assert second.images == {}
    assert second.document["attribution"]["grid"] == first.document["attribution"]["grid"]


def test_a_destination_missing_an_image_is_probed_and_written_again(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)
    first = _build(tmp_path, probes, models, watch)
    _write_images(tmp_path / "Ceu", first)
    (tmp_path / "Ceu" / "attribution.png").unlink()

    second = _build(tmp_path, probes, models, watch)

    assert set(second.images) == IMAGE_NAMES


def test_a_second_destination_gets_its_own_images(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)
    first = _build(tmp_path, probes, models, watch, out_dir=tmp_path / "A")
    _write_images(tmp_path / "A", first)

    second = _build(tmp_path, probes, models, watch, out_dir=tmp_path / "B")

    assert set(second.images) == IMAGE_NAMES


def test_a_changed_attribution_target_invalidates_the_probe_cache(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)
    first = _build(tmp_path, probes, models, watch)
    _write_images(tmp_path / "Ceu", first)
    retargeted = _pin(probes, tmp_path).model_copy(update={"attribution_target": "dhi"})

    second = _build(tmp_path, probes, models, watch, pin=retargeted)

    assert set(second.images) == IMAGE_NAMES
    assert second.document["attribution"]["target"] == "dhi"


def test_a_frame_scored_by_a_checkpoint_outside_the_pin_is_refused(tmp_path, probes, models):
    stranger = tmp_path / "stranger" / "best.ckpt"
    stranger.parent.mkdir()
    stranger.write_bytes(b"other weights")
    watch = _watch_dir(tmp_path, probes, models, checkpoint=stranger)

    with pytest.raises(FramePublishError, match="does not serve"):
        _build(tmp_path, probes, models, watch)


def _strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def test_the_document_carries_no_filesystem_path(tmp_path, probes, models):
    watch = _watch_dir(tmp_path, probes, models)

    document = _build(tmp_path, probes, models, watch).document

    assert not [text for text in _strings(document) if text.startswith(("/", "output/"))]


def test_without_any_scored_frame_the_document_is_still_written_with_empty_slots(
    tmp_path, probes, models
):
    watch = tmp_path / "watch"
    (watch / "frames").mkdir(parents=True)
    pin = _pin(probes, tmp_path)

    artifacts = build_frame_artifacts(
        watch,
        pin=pin,
        digests=_digests(pin),
        served=models["served"],
        controls=_controls(models),
        stamp=publish_stamp(),
        now_local=NOON,
        now_host_utc=NOON_UTC,
        site=SITE,
        out_dir=tmp_path / "Ceu",
        block_minutes=5.0,
        train_max_elevation_deg=None,
    )

    assert artifacts.images == {}
    assert artifacts.document["status"]["reason"] == STATUS_NO_SCORED_FRAME
    assert artifacts.document["prediction"] is None
