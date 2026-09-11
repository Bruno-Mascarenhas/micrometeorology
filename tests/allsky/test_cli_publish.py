"""``allsky publish-site``: refusal scopes, write order and exit codes, with the builders stubbed."""

import hashlib
import json
import os
import time
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from allsky.cli import app
from allsky.publish.frame import FrameArtifacts

runner = CliRunner()


def _pin(tmp_path: Path) -> Path:
    def checkpoint(name: str) -> dict:
        path = tmp_path / name / "best.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    payload = {
        "serving": True,
        "id": "probe",
        "label": "the probe",
        "frame_checkpoints": [checkpoint("member")],
        "min_elevation_deg": 10.0,
        "controls": {
            "sensor_only": {
                "checkpoint": checkpoint("sensor"),
                "report": str(tmp_path / "r" / "s"),
            },
            "climatology": {"checkpoint": checkpoint("clim"), "report": str(tmp_path / "r" / "c")},
        },
        "reports": {
            "dataset": str(tmp_path / "dataset"),
            "members": [str(tmp_path / "r" / "m")],
            "training_history": str(tmp_path / "r" / "metrics.csv"),
        },
        "selection": {"criterion": "the probe", "decided_on": "2026-09-11"},
    }
    pin = tmp_path / "pin.yaml"
    pin.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return pin


def _frame(alive: bool) -> FrameArtifacts:
    document = {
        "schema": "labmim-allsky-frame-v2",
        "status": {
            "reason": "fresh" if alive else "watch_stale",
            "reason_pt": "…",
            "watch_alive": alive,
        },
    }
    return FrameArtifacts(
        document=document, images={"allsky.jpg": b"jpeg", "attribution.png": b"png"}
    )


@pytest.fixture
def stubbed(monkeypatch):
    import allsky.publish.frame as frame_module
    import allsky.publish.model_card as card_module
    import allsky.publish.timeline as timeline_module
    import allsky.snapshot as snapshot_module

    calls: dict[str, object] = {"frame_alive": True, "card_error": None, "timeline_error": None}

    def build_timeline(*_args, **_kwargs):
        if calls["timeline_error"] is not None:
            raise timeline_module.TimelineError(str(calls["timeline_error"]))
        return {"schema": "labmim-allsky-timeline-v1"}

    def build_model_card(*_a, **_k):
        if calls["card_error"] is not None:
            raise card_module.ModelCardError(str(calls["card_error"]))
        return {"schema": "labmim-allsky-model-v1"}

    monkeypatch.setattr(timeline_module, "build_timeline", build_timeline)
    monkeypatch.setattr(snapshot_module, "load_served_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        frame_module, "build_frame_artifacts", lambda *_a, **_k: _frame(bool(calls["frame_alive"]))
    )
    monkeypatch.setattr(card_module, "checkpoint_metadata", lambda *_a, **_k: object())
    monkeypatch.setattr(card_module, "build_model_card", build_model_card)
    monkeypatch.setattr("allsky.publish.dataset.train_max_solar_elevation_deg", lambda *_a: 60.0)
    return calls


def _invoke(tmp_path: Path, *extra: str):
    (tmp_path / "watch" / "frames").mkdir(parents=True, exist_ok=True)
    return runner.invoke(
        app,
        [
            "publish-site",
            "--serving",
            str(_pin(tmp_path)),
            "--watch-dir",
            str(tmp_path / "watch"),
            "--out",
            str(tmp_path / "Ceu"),
            "--no-trust-checkpoint",
            *extra,
        ],
    )


@pytest.mark.usefixtures("stubbed")
def test_a_fresh_watch_publishes_every_document_and_exits_zero(tmp_path):
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    names = sorted(path.name for path in (tmp_path / "Ceu").iterdir())
    assert names == ["allsky.jpg", "attribution.png", "frame.json", "model.json", "timeline.json"]


@pytest.mark.usefixtures("stubbed")
def test_the_images_land_before_the_documents_and_frame_json_last(tmp_path):
    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    out = tmp_path / "Ceu"
    order = [
        (out / name).stat().st_mtime_ns
        for name in ("allsky.jpg", "attribution.png", "timeline.json", "model.json", "frame.json")
    ]
    assert order == sorted(order)


def test_a_stale_watch_still_publishes_but_exits_two(tmp_path, stubbed):
    stubbed["frame_alive"] = False

    result = _invoke(tmp_path)

    assert result.exit_code == 2
    assert (tmp_path / "Ceu" / "frame.json").is_file()


def test_a_model_card_refusal_keeps_the_previous_card_and_publishes_the_rest(tmp_path, stubbed):
    out = tmp_path / "Ceu"
    out.mkdir()
    (out / "model.json").write_text('{"schema":"previous"}')
    stubbed["card_error"] = "control evaluated on another split"

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert json.loads((out / "model.json").read_text())["schema"] == "previous"
    assert (out / "frame.json").is_file()


def test_a_timeline_refusal_keeps_the_previous_timeline_and_publishes_the_rest(tmp_path, stubbed):
    out = tmp_path / "Ceu"
    out.mkdir()
    (out / "timeline.json").write_text('{"schema":"previous"}')
    stubbed["timeline_error"] = "export without the diffuse column"

    result = _invoke(tmp_path)

    assert result.exit_code == 0, result.output
    assert json.loads((out / "timeline.json").read_text())["schema"] == "previous"
    assert (out / "frame.json").is_file()


@pytest.mark.usefixtures("stubbed")
def test_a_rewritten_pinned_checkpoint_blocks_the_whole_publish(tmp_path):
    pin = _pin(tmp_path)
    (tmp_path / "member" / "best.ckpt").write_bytes(b"rewritten")
    (tmp_path / "watch" / "frames").mkdir(parents=True)

    result = runner.invoke(
        app,
        [
            "publish-site",
            "--serving",
            str(pin),
            "--watch-dir",
            str(tmp_path / "watch"),
            "--out",
            str(tmp_path / "Ceu"),
        ],
    )

    assert result.exit_code == 1
    assert not (tmp_path / "Ceu").exists()


@pytest.mark.usefixtures("stubbed")
def test_a_directory_without_frames_or_blocks_is_not_a_watch_directory(tmp_path):
    (tmp_path / "watch").mkdir()

    result = runner.invoke(
        app,
        [
            "publish-site",
            "--serving",
            str(_pin(tmp_path)),
            "--watch-dir",
            str(tmp_path / "watch"),
            "--out",
            str(tmp_path / "Ceu"),
        ],
    )

    assert result.exit_code == 1
    assert not (tmp_path / "Ceu").exists()


@pytest.mark.usefixtures("stubbed")
def test_a_retention_window_not_longer_than_the_timeline_is_refused(tmp_path):
    result = _invoke(tmp_path, "--days", "3", "--prune-frames-days", "3")

    assert result.exit_code == 1


@pytest.mark.usefixtures("stubbed")
def test_frames_older_than_the_retention_window_are_pruned_and_blocks_kept(tmp_path):
    frames = tmp_path / "watch" / "frames"
    blocks = tmp_path / "watch" / "blocks"
    frames.mkdir(parents=True)
    blocks.mkdir(parents=True)
    old = frames / "allsky-20260101-120000.jpg"
    old.write_bytes(b"x")
    ancient = time.time() - 30 * 86400
    os.utime(old, (ancient, ancient))
    fresh = frames / "allsky-20260911-120000.jpg"
    fresh.write_bytes(b"y")
    block = blocks / "20260101-1200.prediction.json"
    block.write_text("{}")
    os.utime(block, (ancient, ancient))

    result = _invoke(tmp_path, "--prune-frames-days", "14")

    assert result.exit_code == 0, result.output
    assert not old.exists()
    assert fresh.exists()
    assert block.exists()
