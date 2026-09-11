"""``timeline.json``: a regular axis with holes, screened measurements and the live comparison."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from allsky.publish import timeline as timeline_module
from allsky.publish.encoding import publish_stamp
from allsky.publish.timeline import TimelineError, build_timeline
from allsky.serving import ServingConfig
from allsky.snapshot import solar_elevation_at
from labmim_core.site import SiteConfig
from labmim_core.sky import SKY_CLASS_NAMES
from tests.allsky._pins import control, pinned, serving_pin_payload

DAY = "2026-09-10"
NOW = pd.Timestamp(f"{DAY} 12:17:00")
SITE = SiteConfig()
DIGEST = "c" * 64
NOON_POSITION = 12 * 12
CLEAR_PROBABILITIES = {
    "cloudy": 0.1,
    "partly_cloudy_diffuse": 0.1,
    "partly_cloudy_clear": 0.2,
    "clear": 0.6,
}
CLOUDY_PROBABILITIES = {
    "cloudy": 0.7,
    "partly_cloudy_diffuse": 0.2,
    "partly_cloudy_clear": 0.05,
    "clear": 0.05,
}


def _pin(tmp_path: Path, *, decided_on: str = "2026-09-09") -> ServingConfig:
    checkpoint = tmp_path / "serving" / "best.ckpt"
    return ServingConfig.model_validate(
        serving_pin_payload(
            frame_checkpoints=[pinned(checkpoint, DIGEST, tmp_path / "eval-test")],
            sensor_only=control(checkpoint, DIGEST, tmp_path / "so"),
            climatology=control(checkpoint, DIGEST, tmp_path / "cl"),
            dataset=tmp_path / "dataset",
            training_history=tmp_path / "metrics.csv",
            decided_on=decided_on,
        )
    )


def _write_block(watch: Path, stem: str, kind: str, payload: dict[str, Any]) -> None:
    blocks = watch / "blocks"
    blocks.mkdir(parents=True, exist_ok=True)
    (blocks / f"{stem}.{kind}.json").write_text(json.dumps(payload), encoding="utf-8")


def _prediction(
    end: str, dhi: float, sky_class: str, probabilities: dict[str, float], **extra: Any
) -> dict[str, Any]:
    return {
        "block_end": end,
        "closed_by": "later_frame",
        "n_frames": 5,
        "source": "frame_aggregate",
        "predictions": {
            "dhi": dhi,
            "kindex": 0.912345,
            "sky_class": sky_class,
            "sky_probabilities": probabilities,
        },
        "frame_aggregate": {"frames": [{"path": "/watch/frames/allsky-1.jpg", "captured_at": end}]},
        **extra,
    }


def _skipped(end: str, reason: str, n_frames: int) -> dict[str, Any]:
    return {"block_end": end, "closed_by": "later_frame", "n_frames": n_frames, "reason": reason}


def _watch_dir(tmp_path: Path) -> Path:
    watch = tmp_path / "watch"
    _write_block(
        watch,
        "20260909-1200",
        "prediction",
        _prediction("2026-09-09T12:00:00", 200.0, "clear", CLEAR_PROBABILITIES),
    )
    _write_block(
        watch,
        "20260910-1200",
        "prediction",
        _prediction(f"{DAY}T12:00:00", 120.456, "clear", CLEAR_PROBABILITIES),
    )
    _write_block(
        watch, "20260910-1205", "skipped", _skipped(f"{DAY}T12:05:00", "insufficient_frames", 1)
    )
    _write_block(
        watch,
        "20260910-1215",
        "prediction",
        _prediction(
            f"{DAY}T12:15:00",
            110.0,
            "cloudy",
            CLOUDY_PROBABILITIES,
            source="block_model",
            block_model={"models": [{"checkpoint": "/serving/best.ckpt", "kindex_kind": "kstar"}]},
        ),
    )
    (watch / "frames").mkdir()
    return watch


def _export(tmp_path: Path, rows: str) -> Path:
    path = tmp_path / "station.csv"
    path.write_text(f"timestamp,PSP_Wm2_Avg,CM3Up_Wm2_Avg\n{rows}", encoding="utf-8")
    return path


def _build(tmp_path: Path, watch: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "pin": _pin(tmp_path),
        "stamp": publish_stamp(),
        "now_local": NOW,
        "days": 1,
        "site": SITE,
    }
    arguments.update(overrides)
    return build_timeline(watch, **arguments)


def _strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [text for value in node.values() for text in _strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _strings(value)]
    return [node] if isinstance(node, str) else []


def test_the_axis_runs_from_the_window_midnight_to_the_last_closed_block(tmp_path):
    watch = _watch_dir(tmp_path)

    axis = _build(tmp_path, watch)["axis"]

    assert axis == {"start": f"{DAY}T00:00:00", "step_minutes": 5.0, "count": NOON_POSITION + 4}


def test_blocks_without_a_record_are_null_between_scored_ones(tmp_path):
    watch = _watch_dir(tmp_path)

    document = _build(tmp_path, watch)

    dhi = document["series"]["dhi_w_m2"]
    assert dhi[NOON_POSITION : NOON_POSITION + 4] == [120.46, None, None, 110.0]
    assert all(value is None for value in dhi[:NOON_POSITION])
    assert document["source"][NOON_POSITION : NOON_POSITION + 4] == [
        "frame_aggregate",
        None,
        None,
        "block_model",
    ]
    assert document["series"]["condition"][NOON_POSITION : NOON_POSITION + 4] == [4, None, None, 1]
    assert document["series"]["p_iv"][NOON_POSITION] == 0.6


def test_skipped_blocks_carry_their_reason_and_its_label(tmp_path):
    watch = _watch_dir(tmp_path)

    document = _build(tmp_path, watch)

    assert document["skipped"] == [{"t": f"{DAY}T12:05:00", "reason": "insufficient_frames"}]
    assert document["reason_labels_pt"]["insufficient_frames"].startswith("quadros de menos")
    assert document["series"]["n_frames"][NOON_POSITION + 1] == 1


def test_measured_is_null_without_an_export(tmp_path):
    watch = _watch_dir(tmp_path)

    document = _build(tmp_path, watch)

    assert document["measured"] is None
    assert document["live"] is None
    assert all(value is None for value in document["series"]["measured_dhi_w_m2"])
    assert document["measured_status"] == {
        "available": False,
        "reason": "no_export",
        "last_row_at": None,
        "source_label": timeline_module.MEASURED_SOURCE_LABEL,
    }


def test_measured_rows_are_matched_to_the_block_ending_at_their_stamp(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(
        tmp_path,
        f"{DAY} 12:00:00,118.2,600.0\n{DAY} 12:10:00,125.0,610.0\n{DAY} 12:15:00,131.0,620.0\n",
    )

    document = _build(tmp_path, watch, sensor_csv=export)

    measured = document["series"]["measured_dhi_w_m2"]
    assert measured[NOON_POSITION : NOON_POSITION + 4] == [118.2, None, 125.0, 131.0]
    assert document["measured"] == {
        "source_column": "PSP_Wm2_Avg",
        "screening": "sentinels + sensor_limits",
        "n": 3,
    }
    assert document["measured_status"]["reason"] == "ok"
    assert document["measured_status"]["last_row_at"] == f"{DAY}T12:15:00"


def test_a_railed_export_row_becomes_null(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, f"{DAY} 12:00:00,118.2,600.0\n{DAY} 12:05:00,-7999,600.0\n")

    document = _build(tmp_path, watch, sensor_csv=export)

    assert document["series"]["measured_dhi_w_m2"][NOON_POSITION : NOON_POSITION + 2] == [
        118.2,
        None,
    ]
    assert document["measured"]["n"] == 1


def test_an_export_that_ends_before_the_window_is_reported_stale(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, "2026-09-08 12:00:00,118.2,600.0\n")

    document = _build(tmp_path, watch, sensor_csv=export)

    assert document["measured_status"] == {
        "available": False,
        "reason": "export_stale",
        "last_row_at": "2026-09-08T12:00:00",
        "source_label": timeline_module.MEASURED_SOURCE_LABEL,
    }
    assert document["measured"]["n"] == 0


def test_an_export_column_without_a_declared_limit_is_not_published(tmp_path, monkeypatch):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, f"{DAY} 12:00:00,118.2,600.0\n")
    monkeypatch.setattr(timeline_module, "shipped_sensor_limits", list)

    document = _build(tmp_path, watch, sensor_csv=export)

    assert document["measured"] is None
    assert document["measured_status"]["reason"] == "no_export"


def test_an_export_without_the_diffuse_column_is_refused(tmp_path):
    watch = _watch_dir(tmp_path)
    export = tmp_path / "station.csv"
    export.write_text(f"timestamp,CM3Up_Wm2_Avg\n{DAY} 12:00:00,600.0\n", encoding="utf-8")

    with pytest.raises(TimelineError, match="PSP_Wm2_Avg"):
        _build(tmp_path, watch, sensor_csv=export)


def test_the_live_comparison_covers_every_block_since_the_pin_was_decided(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(
        tmp_path,
        f"2026-09-09 12:00:00,190.0,600.0\n{DAY} 12:00:00,118.2,600.0\n{DAY} 12:15:00,131.0,620.0\n",
    )
    errors = np.array([200.0 - 190.0, 120.456 - 118.2, 110.0 - 131.0])

    live = _build(tmp_path, watch, sensor_csv=export)["live"]

    assert live["since"] == "2026-09-09T00:00:00"
    assert (live["n_days"], live["n_blocks"]) == (2, 3)
    assert live["dhi"]["mbe"] == pytest.approx(errors.mean(), abs=0.01)
    assert live["dhi"]["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)), abs=0.01)
    assert live["kindex"] == {"mae": None}
    assert live["sky"] == {"balanced_accuracy": None}


def test_blocks_before_the_pin_decision_stay_out_of_the_live_comparison(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, f"2026-09-09 12:00:00,190.0,600.0\n{DAY} 12:00:00,118.2,600.0\n")

    live = _build(tmp_path, watch, sensor_csv=export, pin=_pin(tmp_path, decided_on=DAY))["live"]

    assert (live["n_days"], live["n_blocks"]) == (1, 1)


def test_extrapolation_flags_the_blocks_above_the_training_elevation(tmp_path):
    watch = _watch_dir(tmp_path)

    flags = _build(tmp_path, watch, train_max_elevation_deg=59.8)["series"]["extrapolation"]

    assert flags[NOON_POSITION] is True
    assert flags[1] is False


def test_extrapolation_is_null_without_a_training_elevation(tmp_path):
    watch = _watch_dir(tmp_path)

    flags = _build(tmp_path, watch)["series"]["extrapolation"]

    assert all(flag is None for flag in flags)


def test_the_solar_envelope_is_the_watch_geometry_at_the_block_centroid(tmp_path):
    watch = _watch_dir(tmp_path)
    centroid = pd.Timestamp(f"{DAY} 11:57:30")

    series = _build(tmp_path, watch)["series"]

    assert series["solar_elevation_deg"][NOON_POSITION] == round(
        solar_elevation_at(centroid, SITE), 2
    )
    assert (
        series["clearsky_ghi_w_m2"][NOON_POSITION] > series["clearsky_dhi_w_m2"][NOON_POSITION] > 0
    )
    assert series["clearsky_dhi_w_m2"][0] is None
    assert series["solar_elevation_deg"][0] < 0


def test_days_summarise_scored_skipped_frames_and_the_condition_share(tmp_path):
    watch = _watch_dir(tmp_path)

    days = _build(tmp_path, watch)["days"]

    assert days == [
        {
            "date": f"{DAY}T00:00:00",
            "blocks_scored": 2,
            "blocks_skipped": 1,
            "frames": 11,
            "condition_share": {"i": 0.5, "ii": 0.0, "iii": 0.0, "iv": 0.5},
        }
    ]


def test_latest_names_the_last_scored_block_and_the_last_record_status(tmp_path):
    watch = _watch_dir(tmp_path)
    _write_block(
        watch, "20260910-1220", "skipped", _skipped(f"{DAY}T12:20:00", "below_elevation_floor", 4)
    )

    latest = _build(tmp_path, watch, now_local=pd.Timestamp(f"{DAY} 12:23:00"))["latest"]

    assert latest == {
        "last_scored_block": f"{DAY}T12:15:00",
        "last_block_status": "skipped",
        "reason": "below_elevation_floor",
    }


def test_no_string_in_the_timeline_is_a_filesystem_path(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, f"{DAY} 12:00:00,118.2,600.0\n")

    document = _build(tmp_path, watch, sensor_csv=export, train_max_elevation_deg=59.8)

    assert [text for text in _strings(document) if text.startswith(("/", "output/"))] == []


def test_the_kindex_kind_falls_back_to_the_newest_frame_prediction(tmp_path):
    watch = tmp_path / "watch"
    frames = watch / "frames"
    frames.mkdir(parents=True)
    for stem, kind in (("allsky-20260910-120030", "kstar"), ("allsky-20260910-121530", "kt")):
        (frames / f"{stem}.json").write_text(
            json.dumps(
                {
                    "captured_at": f"{DAY}T12:00:30",
                    "captured_at_source": "overlay",
                    "image": f"{stem}.jpg",
                }
            ),
            encoding="utf-8",
        )
        (frames / f"{stem}.prediction.json").write_text(
            json.dumps(
                {"predictions": {"dhi": 1.0}, "model": {"checkpoint": "/x", "kindex_kind": kind}}
            ),
            encoding="utf-8",
        )

    document = _build(tmp_path, watch)

    assert document["targets"]["kindex"]["kind"] == "kt"
    assert document["latest"] == {
        "last_scored_block": None,
        "last_block_status": None,
        "reason": None,
    }


def test_a_watch_without_blocks_yet_publishes_an_empty_axis(tmp_path):
    watch = tmp_path / "watch"
    watch.mkdir()

    document = _build(tmp_path, watch)

    assert document["axis"]["count"] == NOON_POSITION + 4
    assert document["days"][0]["blocks_scored"] == 0
    assert document["targets"]["kindex"]["kind"] == "kstar"


def test_a_record_naming_an_unknown_sky_class_is_refused(tmp_path):
    watch = _watch_dir(tmp_path)
    _write_block(
        watch,
        "20260910-1210",
        "prediction",
        _prediction(f"{DAY}T12:10:00", 100.0, "overcast", dict.fromkeys(SKY_CLASS_NAMES, 0.25)),
    )

    with pytest.raises(TimelineError, match="overcast"):
        _build(tmp_path, watch)


def test_a_timezone_aware_now_is_refused(tmp_path):
    watch = _watch_dir(tmp_path)

    with pytest.raises(TimelineError, match="naive"):
        _build(tmp_path, watch, now_local=NOW.tz_localize("UTC"))


def test_a_missing_watch_directory_is_refused(tmp_path):
    with pytest.raises(TimelineError, match="not a directory"):
        _build(tmp_path, tmp_path / "nowhere")


def test_an_export_whose_rows_all_fail_screening_is_reported_without_a_last_row(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(tmp_path, f"{DAY} 12:05:00,-7999,500\n{DAY} 12:10:00,-7999,500\n")

    document = _build(tmp_path, watch, sensor_csv=export)

    assert document["measured_status"]["reason"] == "no_valid_rows"
    assert document["measured_status"]["last_row_at"] is None
    assert document["measured"] is None
    assert "None" not in " ".join(document["caveats"])


def test_an_hourly_export_is_refused_as_the_block_mean(tmp_path):
    watch = _watch_dir(tmp_path)
    export = _export(
        tmp_path,
        f"{DAY} 10:00:00,120,500\n{DAY} 11:00:00,140,600\n{DAY} 12:00:00,150,700\n{DAY} 13:00:00,130,650\n",
    )

    document = _build(tmp_path, watch, sensor_csv=export)

    assert document["measured_status"]["reason"] == "interval_mismatch"
    assert document["measured_status"]["available"] is False
    assert document["measured"] is None


def test_a_two_day_axis_starts_at_the_earlier_midnight_and_leaves_the_night_null(tmp_path):
    watch = _watch_dir(tmp_path)

    document = _build(tmp_path, watch, days=2)

    axis = document["axis"]
    assert axis["start"] == f"{pd.Timestamp(DAY) - pd.Timedelta(days=1):%Y-%m-%dT00:00:00}"
    night = slice(0, 6 * 12)
    assert all(value is None for value in document["series"]["dhi_w_m2"][night])
    assert len(document["days"]) == 2
