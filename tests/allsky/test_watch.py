"""The ``allsky watch`` loop: frames in, one block record out, all state on disk.

Captures are scripted through the real :func:`allsky.snapshot.capture_snapshot`
with the ``timestamp=`` bypass, so the sidecars the resume path reads are the
ones production writes. Block checkpoints are real files carrying a config,
so the start-up inspection is the real one; predictions are stubbed at the two
entry points ``run_watch`` calls, except in the end-to-end test that scores a
block through a trained probe.
"""

import io
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from allsky.archive import ArchiveError
from allsky.cli import app
from allsky.snapshot import Snapshot, SolarElevationBelowFloorError, block_end_of, capture_snapshot
from allsky.watch import (
    BLOCKS_SUBDIR,
    FRAMES_SUBDIR,
    ensemble_prediction,
    frames_on_disk,
    run_watch,
)
from tests.allsky import _archive_fake as fake
from tests.allsky._block_probe import stub_image_backbone, train_block_probe

DAY = "2026-09-06"
NOON_BLOCK = "20260906-1205"
NOON_FLOOR_DEG = 10.0
runner = CliRunner()


def _tiny_jpeg() -> bytes:
    buffer = io.BytesIO()
    pixels = np.random.default_rng(7).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


JPEG = _tiny_jpeg()


def _at(clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{DAY} {clock}")


class _Camera:
    base_url = "https://camera.invalid/"

    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}

    def fetch_live_image(self) -> tuple[bytes, dict[str, str]]:
        return JPEG, self.headers


class _Feed:
    """One scripted capture per poll: a stamp to file the frame under, or an error.

    The camera's clock follows the script — each poll reads the stamp it just
    captured — except the last poll, which reads *now*; that is where a test
    places the clock relative to the grace. *clocks* overrides the whole
    sequence.
    """

    def __init__(
        self,
        frames_dir: Path,
        script: list[str | Exception],
        now: str,
        clocks: list[str] | None = None,
    ) -> None:
        self._frames_dir = frames_dir
        self._script = list(script)
        self._clocks = clocks or _clocks_following(script, now)
        self.polls = 0

    def __call__(self) -> Snapshot:
        item = self._script[self.polls]
        self.polls += 1
        if isinstance(item, Exception):
            raise item
        return capture_snapshot(_Camera(), self._frames_dir, timestamp=_at(item))

    def now(self) -> pd.Timestamp:
        return _at(self._clocks[self.polls - 1])


def _clocks_following(script: list[str | Exception], now: str) -> list[str]:
    clocks: list[str] = []
    for item in script:
        clocks.append(item if isinstance(item, str) else (clocks[-1] if clocks else now))
    clocks[-1] = now
    return clocks


def _stub_block_predictions(
    monkeypatch: pytest.MonkeyPatch, per_checkpoint: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    import allsky.watch as watch_module

    calls: list[dict[str, Any]] = []

    def fake_predict_block(
        frames: Any, checkpoint_path: Path, *, block_end: pd.Timestamp, **_: Any
    ) -> dict[str, Any]:
        calls.append({"frames": list(frames), "checkpoint": checkpoint_path, "end": block_end})
        predictions = (per_checkpoint or {}).get(
            checkpoint_path.name,
            {
                "dhi": 100.0,
                "sky_class": "clear",
                "sky_probabilities": {"clear": 0.6, "cloudy": 0.4},
            },
        )
        return {
            "predictions": predictions,
            "block": {"end": block_end.isoformat(), "window_minutes": 5.0, "n_frames": len(frames)},
            "model": {"checkpoint": str(checkpoint_path)},
        }

    monkeypatch.setattr(watch_module, "predict_block", fake_predict_block)
    return calls


def _stub_frame_predictions(monkeypatch: pytest.MonkeyPatch) -> list[pd.Timestamp]:
    import allsky.watch as watch_module

    calls: list[pd.Timestamp] = []

    def fake_predict_snapshot(
        _image: Path, _checkpoint: Path, *, timestamp: pd.Timestamp, **_: Any
    ) -> dict[str, Any]:
        calls.append(timestamp)
        return {"predictions": {"dhi": 42.0}, "features": {"imputed": []}}

    monkeypatch.setattr(watch_module, "predict_snapshot", fake_predict_snapshot)
    return calls


def _block_config(*, window_minutes: float, strategy: str) -> dict[str, Any]:
    return {
        "experiment": True,
        "seed": 0,
        "output_dir": "out",
        "data": {
            "manifest": "manifest.parquet",
            "data_root": "data",
            "split_artifact": "splits.json",
            "input_mode": "image",
            "alignment": {
                "strategy": strategy,
                "window_minutes": window_minutes,
                "max_frames": 5,
                "one_sample_per_block": strategy == "sensor_block",
            },
        },
        "features": {"set": "safe"},
        "targets": {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
        "model": {"name": "image_only", "image_size": 8},
        "train": {"epochs": 1, "batch_size": 8, "num_workers": 0, "device": "cpu"},
    }


def _checkpoint(
    tmp_path: Path,
    name: str = "block.ckpt",
    *,
    window_minutes: float = 5.0,
    strategy: str = "sensor_block",
) -> Path:
    import torch

    from allsky.config import ExperimentConfig

    cfg = ExperimentConfig.model_validate(
        _block_config(window_minutes=window_minutes, strategy=strategy)
    )
    path = tmp_path / name
    torch.save({"config": cfg.model_dump()}, path)
    return path


def _watch(
    tmp_path: Path,
    script: list[str | Exception],
    *,
    now: str,
    clocks: list[str] | None = None,
    checkpoints: tuple[Path, ...] = (),
    checkpoint_frame: Path | None = None,
    min_frames: int = 3,
    block_minutes: float = 5.0,
    min_solar_elevation_deg: float | None = NOON_FLOOR_DEG,
    sleep: Callable[[float], None] = lambda _seconds: None,
    **kwargs: Any,
) -> int:
    out_dir = tmp_path / "watch"
    feed = _Feed(out_dir / FRAMES_SUBDIR, script, now, clocks)
    return run_watch(
        feed,
        out_dir,
        checkpoint_frame=checkpoint_frame,
        checkpoint_blocks=checkpoints,
        block_minutes=block_minutes,
        min_frames=min_frames,
        grace_seconds=90.0,
        min_solar_elevation_deg=min_solar_elevation_deg,
        clock=feed.now,
        sleep=sleep,
        max_polls=len(script),
        **kwargs,
    )


def _block_record(tmp_path: Path, name: str) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(
        (tmp_path / "watch" / BLOCKS_SUBDIR / name).read_text(encoding="utf-8")
    )
    return loaded


def _block_records(tmp_path: Path) -> list[str]:
    return sorted(path.name for path in (tmp_path / "watch" / BLOCKS_SUBDIR).glob("*"))


@pytest.mark.parametrize(
    ("stamp", "end"),
    [
        ("12:00:00", "12:00:00"),
        ("12:00:01", "12:05:00"),
        ("12:04:59", "12:05:00"),
        ("12:05:00", "12:05:00"),
    ],
)
def test_a_frame_is_filed_under_the_block_whose_end_it_rounds_up_to(stamp: str, end: str) -> None:
    assert block_end_of(_at(stamp), 5.0) == _at(end)


def test_a_block_closes_when_a_frame_stamped_after_it_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    checkpoint = _checkpoint(tmp_path)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:04:00", "12:05:00", "12:06:00"],
        now="12:06:00",
        checkpoints=(checkpoint,),
    )

    assert [call["end"] for call in calls] == [_at("12:05:00")]
    assert [when for _, when in calls[0]["frames"]] == [
        _at("12:01:00"),
        _at("12:02:00"),
        _at("12:03:00"),
        _at("12:04:00"),
        _at("12:05:00"),
    ]
    record = _block_record(tmp_path, f"{NOON_BLOCK}.prediction.json")
    assert record["predictions"]["dhi"] == 100.0
    assert record["closed_by"] == "later_frame"


def test_a_block_stays_open_without_a_later_frame_inside_the_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:04:00", "12:05:00"],
        now="12:06:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert calls == []
    assert list((tmp_path / "watch" / BLOCKS_SUBDIR).glob("*")) == []


def test_a_block_closes_by_grace_when_the_camera_stalls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:04:00", ArchiveError("camera down")],
        now="12:06:30",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert [call["end"] for call in calls] == [_at("12:05:00")]
    assert len(calls[0]["frames"]) == 4
    assert _block_record(tmp_path, f"{NOON_BLOCK}.prediction.json")["closed_by"] == "grace"


def test_a_frame_arriving_after_its_block_closed_is_indexed_with_a_warning_and_not_fed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _stub_block_predictions(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        _watch(
            tmp_path,
            ["12:01:00", "12:02:00", "12:03:00", ArchiveError("camera down"), "12:04:00"],
            now="12:07:00",
            clocks=["12:01:00", "12:02:00", "12:03:00", "12:06:35", "12:07:00"],
            checkpoints=(_checkpoint(tmp_path),),
        )

    assert [len(call["frames"]) for call in calls] == [3]
    assert f"falls in block {NOON_BLOCK}, which is already closed" in caplog.text
    assert [record.captured_at for record in frames_on_disk(tmp_path / "watch" / FRAMES_SUBDIR)][
        -1
    ] == _at("12:04:00")


def test_a_block_with_too_few_frames_is_skipped_and_never_revisited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    checkpoint = _checkpoint(tmp_path)

    _watch(tmp_path, ["12:03:00", "12:04:00"], now="12:06:30", checkpoints=(checkpoint,))
    skipped = _block_record(tmp_path, f"{NOON_BLOCK}.skipped.json")
    _watch(tmp_path, ["12:07:00"], now="12:07:00", checkpoints=(checkpoint,))

    assert skipped["reason"] == "insufficient_frames"
    assert skipped["n_frames"] == 2
    assert calls == []
    assert _block_record(tmp_path, f"{NOON_BLOCK}.skipped.json") == skipped


def test_a_capture_gap_leaves_an_empty_record_per_missing_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:22:00", "12:23:00", "12:24:00", "12:26:00"],
        now="12:26:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert _block_records(tmp_path) == [
        "20260906-1205.prediction.json",
        "20260906-1210.skipped.json",
        "20260906-1215.skipped.json",
        "20260906-1220.skipped.json",
        "20260906-1225.prediction.json",
    ]
    gap = _block_record(tmp_path, "20260906-1215.skipped.json")
    assert gap == {
        "block_end": "2026-09-06T12:15:00",
        "closed_by": "later_frame",
        "n_frames": 0,
        "reason": "insufficient_frames",
        "min_frames": 3,
    }
    assert [call["end"] for call in calls] == [_at("12:05:00"), _at("12:25:00")]


def test_a_frame_older_than_the_last_closed_block_still_gets_its_block_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)

    _watch(
        tmp_path,
        ["12:21:00", "12:22:00", "12:23:00", "12:26:00", "12:15:30"],
        now="12:26:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert [call["end"] for call in calls] == [_at("12:25:00")]
    assert _block_records(tmp_path) == [
        "20260906-1220.skipped.json",
        "20260906-1225.prediction.json",
    ]
    assert _block_record(tmp_path, "20260906-1220.skipped.json")["n_frames"] == 1


def test_the_watch_resumes_from_the_sidecars_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    checkpoint = _checkpoint(tmp_path)
    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:04:00"],
        now="12:04:30",
        checkpoints=(checkpoint,),
    )
    assert calls == []

    _watch(tmp_path, ["12:06:00"], now="12:06:00", checkpoints=(checkpoint,))

    assert [call["end"] for call in calls] == [_at("12:05:00")]
    frames_dir = tmp_path / "watch" / FRAMES_SUBDIR
    assert [(path.parent, when) for path, when in calls[0]["frames"]] == [
        (frames_dir, _at("12:01:00")),
        (frames_dir, _at("12:02:00")),
        (frames_dir, _at("12:03:00")),
        (frames_dir, _at("12:04:00")),
    ]


def test_a_repeated_stamp_is_scored_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scored = _stub_frame_predictions(monkeypatch)

    _watch(
        tmp_path,
        ["12:01:00", "12:01:00", "12:02:00"],
        now="12:02:00",
        checkpoint_frame=_checkpoint(tmp_path, "frame.ckpt"),
    )

    assert scored == [_at("12:01:00"), _at("12:02:00")]


def test_a_new_frame_is_scored_into_its_own_prediction_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _stub_frame_predictions(monkeypatch)
    _watch(tmp_path, ["12:01:00"], now="12:01:00", checkpoint_frame=_checkpoint(tmp_path))
    frames_dir = tmp_path / "watch" / FRAMES_SUBDIR
    (prediction,) = list(frames_dir.glob("*.prediction.json"))

    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        resumed = frames_on_disk(frames_dir)

    assert json.loads(prediction.read_text(encoding="utf-8"))["predictions"]["dhi"] == 42.0
    assert [record.captured_at for record in resumed] == [_at("12:01:00")]
    assert "unreadable frame sidecar" not in caplog.text


def test_a_frame_whose_prediction_raises_is_kept_and_the_watch_goes_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import allsky.watch as watch_module

    def refusing_predict_snapshot(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("no such checkpoint")

    monkeypatch.setattr(watch_module, "predict_snapshot", refusing_predict_snapshot)

    polls = _watch(
        tmp_path, ["12:01:00", "12:02:00"], now="12:02:00", checkpoint_frame=_checkpoint(tmp_path)
    )

    frames_dir = tmp_path / "watch" / FRAMES_SUBDIR
    assert polls == 2
    assert list(frames_dir.glob("*.prediction.json")) == []
    assert [record.captured_at for record in frames_on_disk(frames_dir)] == [
        _at("12:01:00"),
        _at("12:02:00"),
    ]


def test_two_block_checkpoints_are_averaged_and_the_class_is_the_argmax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_block_predictions(
        monkeypatch,
        {
            "a.ckpt": {
                "dhi": 100.0,
                "sky_class": "clear",
                "sky_probabilities": {"clear": 0.6, "cloudy": 0.4},
            },
            "b.ckpt": {
                "dhi": 200.0,
                "sky_class": "cloudy",
                "sky_probabilities": {"clear": 0.3, "cloudy": 0.7},
            },
        },
    )

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:06:00"],
        now="12:06:00",
        checkpoints=(_checkpoint(tmp_path, "a.ckpt"), _checkpoint(tmp_path, "b.ckpt")),
    )

    record = _block_record(tmp_path, f"{NOON_BLOCK}.prediction.json")
    assert record["predictions"]["dhi"] == pytest.approx(150.0)
    assert record["predictions"]["sky_probabilities"] == pytest.approx(
        {"clear": 0.45, "cloudy": 0.55}
    )
    assert record["predictions"]["sky_class"] == "cloudy"
    assert [model["checkpoint"] for model in record["models"]] == [
        str(tmp_path / "a.ckpt"),
        str(tmp_path / "b.ckpt"),
    ]


def test_an_ensemble_averages_a_head_over_the_members_that_carry_it() -> None:
    merged = ensemble_prediction(
        [
            {"predictions": {"dhi": 10.0, "kindex": 0.2}, "block": {"end": "x"}, "model": {}},
            {"predictions": {"dhi": 30.0}, "block": {"end": "x"}, "model": {}},
        ]
    )

    assert merged["predictions"] == {"dhi": pytest.approx(20.0), "kindex": pytest.approx(0.2)}


def test_an_ensemble_needs_at_least_one_member() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        ensemble_prediction([])


def test_a_recorded_block_is_never_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    recorded = tmp_path / "watch" / BLOCKS_SUBDIR / f"{NOON_BLOCK}.prediction.json"
    recorded.parent.mkdir(parents=True)
    recorded.write_text('{"sentinel": true}', encoding="utf-8")

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:06:00"],
        now="12:06:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert calls == []
    assert recorded.read_text(encoding="utf-8") == '{"sentinel": true}'


def test_a_skipped_block_is_not_rescored_once_it_holds_enough_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    skipped = tmp_path / "watch" / BLOCKS_SUBDIR / f"{NOON_BLOCK}.skipped.json"
    skipped.parent.mkdir(parents=True)
    skipped.write_text('{"sentinel": true}', encoding="utf-8")

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:06:00"],
        now="12:06:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    assert calls == []
    assert skipped.read_text(encoding="utf-8") == '{"sentinel": true}'
    assert not (skipped.parent / f"{NOON_BLOCK}.prediction.json").exists()


def test_a_capture_failure_is_logged_and_polling_continues(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        polls = _watch(tmp_path, [ArchiveError("camera down"), "12:01:00"], now="12:01:00")

    assert polls == 2
    assert "capture failed on poll 1" in caplog.text
    assert [
        record.captured_at for record in frames_on_disk(tmp_path / "watch" / FRAMES_SUBDIR)
    ] == [_at("12:01:00")]


def test_the_watch_sleeps_the_poll_interval_between_polls(tmp_path: Path) -> None:
    naps: list[float] = []

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00"],
        now="12:03:00",
        sleep=naps.append,
        poll_seconds=20.0,
    )

    assert naps == [20.0, 20.0]


def test_a_failing_block_prediction_is_recorded_instead_of_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import allsky.watch as watch_module

    attempts: list[pd.Timestamp] = []

    def refusing_predict_block(_frames: Any, _checkpoint: Path, *, block_end: Any, **_: Any) -> Any:
        attempts.append(block_end)
        raise RuntimeError("CUDA error: device-side assert triggered")

    monkeypatch.setattr(watch_module, "predict_block", refusing_predict_block)
    checkpoint = _checkpoint(tmp_path)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:06:00", "12:07:00"],
        now="12:07:00",
        checkpoints=(checkpoint,),
    )

    skipped = _block_record(tmp_path, f"{NOON_BLOCK}.skipped.json")
    assert skipped["reason"] == "prediction_failed"
    assert "device-side assert" in skipped["error"]
    assert attempts == [_at("12:05:00")]


def test_a_block_with_the_sun_below_the_floor_is_recorded_as_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import allsky.watch as watch_module

    def night_predict_block(*_args: Any, **_kwargs: Any) -> Any:
        raise SolarElevationBelowFloorError(_at("22:02:30"), -60.0, NOON_FLOOR_DEG)

    monkeypatch.setattr(watch_module, "predict_block", night_predict_block)

    _watch(
        tmp_path,
        ["22:01:00", "22:02:00", "22:03:00", "22:06:00"],
        now="22:06:00",
        checkpoints=(_checkpoint(tmp_path),),
    )

    skipped = _block_record(tmp_path, "20260906-2205.skipped.json")
    assert skipped["reason"] == "below_elevation_floor"
    assert skipped["solar_elevation_deg"] == -60.0
    assert skipped["min_solar_elevation_deg"] == NOON_FLOOR_DEG
    assert skipped["n_frames"] == 3


def test_a_block_checkpoint_pooling_another_window_is_refused_at_start_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _stub_block_predictions(monkeypatch)
    ten_minute = _checkpoint(tmp_path, "ten.ckpt", window_minutes=10.0)

    with pytest.raises(ValueError, match="pools a 10 min window"):
        _watch(
            tmp_path,
            ["12:01:00", "12:02:00", "12:03:00", "12:06:00"],
            now="12:06:00",
            checkpoints=(ten_minute,),
        )

    assert calls == []
    assert not (tmp_path / "watch" / BLOCKS_SUBDIR).exists()


def test_a_center_frame_checkpoint_is_refused_at_start_up(tmp_path: Path) -> None:
    single = _checkpoint(tmp_path, "single.ckpt", strategy="center_frame")

    with pytest.raises(ValueError, match="center_frame"):
        _watch(tmp_path, ["12:01:00"], now="12:01:00", checkpoints=(single,))

    assert not (tmp_path / "watch" / FRAMES_SUBDIR).exists()


def test_a_block_checkpoint_needs_the_elevation_floor(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_solar_elevation_deg"):
        _watch(
            tmp_path,
            ["12:01:00"],
            now="12:01:00",
            checkpoints=(_checkpoint(tmp_path),),
            min_solar_elevation_deg=None,
        )


def test_a_block_is_scored_end_to_end_through_a_trained_probe(tmp_path: Path) -> None:
    probe = train_block_probe(tmp_path)

    _watch(
        tmp_path,
        ["12:01:00", "12:02:00", "12:03:00", "12:06:00"],
        now="12:06:00",
        checkpoints=(probe,),
        trust_checkpoint=True,
        image_backbone_builder=stub_image_backbone,
    )

    record = _block_record(tmp_path, f"{NOON_BLOCK}.prediction.json")
    assert np.isfinite(record["predictions"]["dhi"])
    assert record["predictions"]["sky_class"] in record["predictions"]["sky_probabilities"]
    assert record["block"]["n_frames"] == 3
    assert record["block"]["representative"] == "2026-09-06T12:02:00"
    assert record["block"]["solar_elevation_deg"] > NOON_FLOOR_DEG
    assert record["closed_by"] == "later_frame"


def test_a_stale_frame_named_from_the_host_clock_is_deleted_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import allsky.snapshot as snapshot_module

    calls = _stub_block_predictions(monkeypatch)
    host_clock = iter([_at("12:00:00"), _at("12:00:20"), _at("12:00:40")])
    monkeypatch.setattr(snapshot_module, "_site_now", lambda: next(host_clock))
    out_dir = tmp_path / "watch"
    frames_dir = out_dir / FRAMES_SUBDIR

    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        run_watch(
            lambda: capture_snapshot(_Camera(), frames_dir),
            out_dir,
            checkpoint_blocks=(_checkpoint(tmp_path),),
            min_solar_elevation_deg=NOON_FLOOR_DEG,
            clock=lambda: _at("12:01:00"),
            sleep=lambda _seconds: None,
            max_polls=3,
        )

    assert caplog.text.count("the camera has not advanced") == 3
    assert sorted(frames_dir.iterdir()) == []
    assert calls == []


def test_a_frame_stamped_by_the_server_clock_stays_on_disk_but_is_not_filed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import allsky.snapshot as snapshot_module

    scored = _stub_frame_predictions(monkeypatch)
    monkeypatch.setattr(snapshot_module, "_site_now", lambda: _at("12:00:00"))
    out_dir = tmp_path / "watch"
    frames_dir = out_dir / FRAMES_SUBDIR
    camera = _Camera({"last-modified": "Sun, 06 Sep 2026 15:00:10 GMT"})

    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        run_watch(
            lambda: capture_snapshot(camera, frames_dir),
            out_dir,
            checkpoint_frame=_checkpoint(tmp_path, "frame.ckpt"),
            clock=lambda: _at("12:01:00"),
            sleep=lambda _seconds: None,
            max_polls=1,
        )

    assert "came from the server-last-modified" in caplog.text
    assert [path.name for path in sorted(frames_dir.iterdir())] == [
        "allsky-20260906-120010.jpg",
        "allsky-20260906-120010.json",
    ]
    assert scored == []
    assert frames_on_disk(frames_dir) == []


def test_resuming_skips_an_unreadable_sidecar_and_one_whose_image_is_gone(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    frames_dir = tmp_path / "frames"
    kept = capture_snapshot(_Camera(), frames_dir, timestamp=_at("12:02:00"))
    orphan = capture_snapshot(_Camera(), frames_dir, timestamp=_at("12:01:00"))
    orphan.image_path.unlink()
    (frames_dir / "allsky-20260906-120000.json").write_text("not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="allsky.watch"):
        resumed = frames_on_disk(frames_dir)

    assert resumed == [kept]
    assert "skipping unreadable frame sidecar" in caplog.text
    assert f"its image {orphan.image_path.name} is gone" in caplog.text


def test_watch_archives_the_live_frame_from_the_mirror(tmp_path: Path) -> None:
    mirror = fake.ArchiveMirror(tmp_path / "site")
    mirror.publish_image(JPEG)
    out_dir = tmp_path / "watch"
    try:
        result = runner.invoke(
            app,
            [
                "watch",
                "--out",
                str(out_dir),
                "--base-url",
                mirror.base_url,
                "--insecure",
                "--max-polls",
                "1",
                "--poll-seconds",
                "0",
            ],
        )
    finally:
        mirror.close()

    assert result.exit_code == 0, result.output
    assert "watch finished after 1 poll(s)" in result.output
    (image,) = list((out_dir / FRAMES_SUBDIR).glob("*.jpg"))
    assert image.read_bytes() == JPEG


def test_watch_exits_with_code_one_when_the_ca_bundle_cannot_be_read(tmp_path: Path) -> None:
    bad_pem = tmp_path / "bad.pem"
    bad_pem.write_text("not a certificate", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "watch",
            "--out",
            str(tmp_path / "watch"),
            "--base-url",
            "https://camera.invalid/",
            "--ca-file",
            str(bad_pem),
            "--max-polls",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)


def test_watch_exits_with_code_one_when_a_block_checkpoint_lacks_its_floor(
    tmp_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "watch",
            "--out",
            str(tmp_path / "watch"),
            "--base-url",
            "http://127.0.0.1:9/",
            "--insecure",
            "--checkpoint-block",
            str(_checkpoint(tmp_path)),
            "--max-polls",
            "1",
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)


def test_watch_stops_cleanly_on_ctrl_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import allsky.snapshot as snapshot_module

    def interrupted_capture(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr(snapshot_module, "capture_snapshot", interrupted_capture)

    result = runner.invoke(
        app,
        [
            "watch",
            "--out",
            str(tmp_path / "watch"),
            "--base-url",
            "http://127.0.0.1:9/",
            "--insecure",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "watch stopped" in result.output


def test_watch_is_listed_in_the_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "watch" in result.output
