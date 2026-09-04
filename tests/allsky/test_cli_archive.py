"""Tests for allsky.cli.archive — the dedup planner and both commands, offline.

``sync-archive`` and ``snapshot`` are driven against a local HTTP mirror with
``--insecure``, because the default context build repairs the archive's TLS
chain by fetching the CA intermediate over the network.
"""

import datetime as dt
import http.client
import io
import json
import logging
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner, Result

from allsky.archive import (
    LEDGER_FILENAME,
    ArchiveClient,
    ArchiveEntry,
    ArchiveError,
    Ledger,
    ledger_lock,
)
from allsky.cli import app
from allsky.cli.archive import (
    FRAMES_SUBDIR,
    REMOTE_FRAMES,
    REMOTE_SNAPSHOTS,
    REMOTE_VIDEOS,
    STATE_SUBDIR,
    VIDEOS_SUBDIR,
    DayPlan,
    TimestampSource,
    UploadChoice,
    _plan_day,
)
from allsky.config import SITE_TZ
from allsky.drive import DriveTarget
from allsky.overlay import MANIFEST_FILENAME
from allsky.snapshot import capture_snapshot
from tests.allsky import _archive_fake as fake

runner = CliRunner()

DAY_OLD = "20260809"
DAY_NEW = "20260810"
DAY_LATER = "20260811"
LIVE_IMAGE = b"\xff\xd8\xff\xe0live-frame-bytes"
FRAME_HEIGHT = 272
FRAME_WIDTH = 480
STAMPS = {
    DAY_OLD: ("20260809120000", "20260809120100", "20260809120200"),
    DAY_NEW: ("20260810120000", "20260810120100", "20260810120200"),
    DAY_LATER: ("20260811120000", "20260811120100", "20260811120200"),
}
TARGET = DriveTarget(remote="labmim")


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / STATE_SUBDIR / LEDGER_FILENAME)


@pytest.fixture
def entry() -> ArchiveEntry:
    return fake.archive_entry(DAY_NEW)


@pytest.fixture
def mirror(tmp_path: Path) -> Iterator[fake.ArchiveMirror]:
    """An archive serving two days of real overlay-stamped timelapses."""
    served = fake.ArchiveMirror(tmp_path / "site")
    staging = tmp_path / "staging"
    staging.mkdir()
    for day in (DAY_OLD, DAY_NEW):
        served.publish_video_file(_timelapse(staging, day))
    served.publish_image(LIVE_IMAGE)
    try:
        yield served
    finally:
        served.close()


@pytest.fixture
def rclone_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / "rclone-invocations.log"
    fake.write_fake_rclone(binaries, log)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    return log


def _timelapse(directory: Path, day: str) -> Path:
    return fake.write_overlay_video(
        directory / f"allsky-{day}.mp4", STAMPS[day], height=FRAME_HEIGHT, width=FRAME_WIDTH
    )


def _sync(data_dir: Path, mirror: fake.ArchiveMirror, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "sync-archive",
            "--data-dir",
            str(data_dir),
            "--base-url",
            mirror.base_url,
            "--insecure",
            "--retries",
            "1",
            "--delay",
            "0",
            *extra,
        ],
    )


def _reload(data_dir: Path) -> Ledger:
    return Ledger.load(data_dir / STATE_SUBDIR / LEDGER_FILENAME)


def _plan(
    entry: ArchiveEntry,
    ledger: Ledger,
    root: Path,
    *,
    target: DriveTarget | None = None,
    extract: bool = False,
    step: int = 1,
    resize: int | None = None,
    timestamps: TimestampSource = TimestampSource.overlay,
    upload: UploadChoice = UploadChoice.none,
) -> DayPlan:
    return _plan_day(
        entry,
        ledger=ledger,
        root=root,
        target=target,
        extract=extract,
        step=step,
        resize=resize,
        timestamps=timestamps,
        upload=upload,
    )


def _mark_fully_mirrored(ledger: Ledger, root: Path, entry: ArchiveEntry) -> None:
    fake.record_downloaded_day(ledger, root, entry.key)
    ledger.record_frames(
        entry.key,
        directory=f"{FRAMES_SUBDIR}/{entry.key}",
        count=3,
        step=1,
        resize=None,
        timestamps="overlay",
    )
    ledger.record_upload(entry.key, TARGET.path(REMOTE_VIDEOS, entry.filename), kind="video")
    ledger.record_upload(entry.key, TARGET.path(REMOTE_FRAMES, entry.key), kind="frames")


def test_a_day_the_ledger_has_never_seen_is_planned_for_download(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    plan = _plan(entry, ledger, tmp_path)
    assert plan.download is True
    assert plan.has_work is True
    assert (plan.extract, plan.upload_video, plan.upload_frames) == (False, False, False)


def test_a_day_already_downloaded_extracted_and_uploaded_everywhere_has_no_work_left(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    _mark_fully_mirrored(ledger, tmp_path, entry)
    plan = _plan(entry, ledger, tmp_path, target=TARGET, extract=True, upload=UploadChoice.both)

    assert plan.has_work is False


def test_asking_for_upload_after_the_fact_replans_an_already_held_day(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    fake.record_downloaded_day(ledger, tmp_path, entry.key)
    assert _plan(entry, ledger, tmp_path).has_work is False

    later = _plan(entry, ledger, tmp_path, target=TARGET, upload=UploadChoice.videos)
    assert later.has_work is True
    assert later.download is False
    assert later.upload_video is True


@pytest.mark.parametrize(("step", "resize"), [(2, None), (1, 224)])
def test_changing_the_extraction_parameters_replans_extraction_only(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path, step: int, resize: int | None
):
    _mark_fully_mirrored(ledger, tmp_path, entry)
    plan = _plan(
        entry,
        ledger,
        tmp_path,
        target=TARGET,
        extract=True,
        step=step,
        resize=resize,
        upload=UploadChoice.videos,
    )

    assert plan.extract is True
    assert plan.download is False
    assert plan.upload_video is False


def test_frames_extracted_from_the_other_clock_are_planned_for_extraction_again(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    _mark_fully_mirrored(ledger, tmp_path, entry)
    plan = _plan(entry, ledger, tmp_path, extract=True, timestamps=TimestampSource.config)

    assert plan.extract is True


def test_frames_whose_clock_the_ledger_never_recorded_are_planned_for_extraction_again(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    _mark_fully_mirrored(ledger, tmp_path, entry)
    del (ledger.frames(entry.key) or {})["timestamps"]
    plan = _plan(entry, ledger, tmp_path, extract=True)

    assert plan.extract is True


def test_frames_are_only_planned_for_upload_once_they_exist_or_are_about_to(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    fake.record_downloaded_day(ledger, tmp_path, entry.key)
    without_extract = _plan(entry, ledger, tmp_path, target=TARGET, upload=UploadChoice.frames)
    assert without_extract.upload_frames is False

    with_extract = _plan(
        entry, ledger, tmp_path, target=TARGET, extract=True, upload=UploadChoice.frames
    )
    assert with_extract.extract is True
    assert with_extract.upload_frames is True


def test_a_pruned_day_is_not_fetched_again_just_because_its_bytes_are_gone(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    result = fake.record_downloaded_day(ledger, tmp_path, entry.key)
    ledger.record_upload(entry.key, TARGET.path(REMOTE_VIDEOS, entry.filename), kind="video")
    ledger.mark_pruned(entry.key)
    result.path.unlink()

    plan = _plan(entry, ledger, tmp_path, target=TARGET, upload=UploadChoice.videos)
    assert plan.has_work is False


def test_a_pruned_day_is_fetched_again_when_a_new_destination_still_wants_it(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    result = fake.record_downloaded_day(ledger, tmp_path, entry.key)
    ledger.mark_pruned(entry.key)
    result.path.unlink()

    plan = _plan(entry, ledger, tmp_path, target=TARGET, upload=UploadChoice.videos)
    assert plan.download is True
    assert plan.upload_video is True


def test_a_locally_damaged_video_is_refetched_only_when_some_step_still_needs_its_bytes(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    result = fake.record_downloaded_day(ledger, tmp_path, entry.key)
    result.path.write_bytes(b"truncated on disk")

    assert _plan(entry, ledger, tmp_path).download is False
    assert _plan(entry, ledger, tmp_path, extract=True).download is True


def test_describe_names_every_step_the_plan_still_owes(entry: ArchiveEntry):
    plan = DayPlan(entry=entry, download=True, extract=False, upload_video=True, upload_frames=True)
    assert plan.describe() == f"{entry.filename}: download, upload-video, upload-frames"


def test_a_plan_that_owes_nothing_reports_no_work(entry: ArchiveEntry):
    """The only case of the disjunction worth a test: every planner test above
    already covers a plan that owes something, and enumerating the other four
    combinations pinned an `or` no realistic edit gets wrong.
    """
    plan = DayPlan(
        entry=entry, download=False, extract=False, upload_video=False, upload_frames=False
    )

    assert plan.has_work is False


def test_a_first_sync_mirrors_every_published_day_and_the_next_one_finds_nothing_to_do(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    first = _sync(data, mirror)
    assert first.exit_code == 0, first.output
    assert "2 day(s) processed, 0 failed" in first.output

    videos = sorted((data / VIDEOS_SUBDIR).glob("*.mp4"))
    assert [path.name for path in videos] == [f"allsky-{DAY_OLD}.mp4", f"allsky-{DAY_NEW}.mp4"]
    assert videos[1].read_bytes() == (mirror.videos_dir / f"allsky-{DAY_NEW}.mp4").read_bytes()
    ledger = _reload(data)
    assert sorted(ledger.entries) == [DAY_OLD, DAY_NEW]
    assert ledger.has_video(DAY_NEW, local_root=data) is True

    mirror.requests.clear()
    second = _sync(data, mirror)
    assert second.exit_code == 0, second.output
    assert "nothing to do" in second.output
    assert mirror.video_requests() == []


def test_the_ledger_keeps_a_purged_day_while_recording_a_newly_published_one(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    assert _sync(data, mirror).exit_code == 0
    purged = _reload(data).video(DAY_OLD)
    assert purged is not None

    mirror.unpublish_video(f"allsky-{DAY_OLD}.mp4")
    mirror.publish_video_file(_timelapse(tmp_path / "staging", DAY_LATER))
    later = _sync(data, mirror)

    assert later.exit_code == 0, later.output
    assert "1 day(s) processed" in later.output
    reloaded = _reload(data)
    assert sorted(reloaded.entries) == [DAY_OLD, DAY_NEW, DAY_LATER]
    assert reloaded.video(DAY_OLD) == purged


def test_list_only_reports_the_plan_and_transfers_nothing(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    result = _sync(data, mirror, "--list")

    assert result.exit_code == 0, result.output
    assert f"allsky-{DAY_NEW}.mp4: download" in result.output
    assert mirror.video_requests() == []
    assert not (data / VIDEOS_SUBDIR).exists()


def test_since_and_until_narrow_the_days_the_run_considers(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    result = _sync(data, mirror, "--since", "2026-08-10", "--until", "2026-08-10")

    assert result.exit_code == 0, result.output
    assert "published: 2 day(s); selected: 1" in result.output
    assert sorted(_reload(data).entries) == [DAY_NEW]


def test_a_malformed_since_date_is_rejected_before_anything_is_fetched(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    result = _sync(tmp_path / "data", mirror, "--since", "10/08/2026")
    assert result.exit_code != 0
    assert "must be YYYY-MM-DD" in result.output


def test_limit_caps_how_many_days_one_run_transfers(mirror: fake.ArchiveMirror, tmp_path: Path):
    data = tmp_path / "data"
    result = _sync(data, mirror, "--limit", "1")

    assert result.exit_code == 0, result.output
    assert "limited to 1 day(s) this run" in result.output
    assert sorted(_reload(data).entries) == [DAY_OLD]


def test_extraction_names_frames_from_the_overlay_and_a_repeat_run_re_uses_them(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    first = _sync(data, mirror, "--extract", "--since", "2026-08-10")
    assert first.exit_code == 0, first.output

    frames_dir = data / FRAMES_SUBDIR / DAY_NEW
    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == [
        f"allsky-{DAY_NEW}-1200.jpg",
        f"allsky-{DAY_NEW}-1201.jpg",
        f"allsky-{DAY_NEW}-1202.jpg",
    ]
    manifest = pd.read_parquet(frames_dir / MANIFEST_FILENAME)
    assert manifest["timestamp"].iloc[0] == pd.Timestamp("2026-08-10 12:00:00")

    recorded = _reload(data).frames(DAY_NEW)
    assert recorded is not None
    assert recorded["timestamps"] == "overlay"
    assert recorded["count"] == 3
    assert recorded["dir"] == f"{FRAMES_SUBDIR}/{DAY_NEW}"

    mirror.requests.clear()
    second = _sync(data, mirror, "--extract", "--since", "2026-08-10")
    assert "nothing to do" in second.output
    assert mirror.video_requests() == []


def _prepare_config(path: Path, clock: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'video:\n  timestamps: "{clock}"\n', encoding="utf-8")
    return path


def test_the_config_timestamp_model_names_frames_from_a_fixed_start_time_instead(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    modelled = _prepare_config(tmp_path / "configs" / "modelled.yaml", "modelled")
    result = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(modelled),
        "--since",
        "2026-08-10",
    )

    assert result.exit_code == 0, result.output
    frames_dir = data / FRAMES_SUBDIR / DAY_NEW
    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == [
        f"allsky-{DAY_NEW}-0600.jpg",
        f"allsky-{DAY_NEW}-0601.jpg",
        f"allsky-{DAY_NEW}-0602.jpg",
    ]
    recorded = _reload(data).frames(DAY_NEW)
    assert recorded is not None
    # The modelled clock is filed with a digest of its own cadence fields: two
    # configs that both say "modelled" describe different frames whenever
    # start_time or minutes_per_frame differs.
    assert recorded["timestamps"].startswith("config:")


def test_a_config_that_names_the_overlay_clock_extracts_under_the_overlay_clock(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    overlay_config = _prepare_config(tmp_path / "configs" / "overlay.yaml", "overlay")

    result = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(overlay_config),
        "--since",
        "2026-08-10",
    )

    assert result.exit_code == 0, result.output
    frames_dir = data / FRAMES_SUBDIR / DAY_NEW
    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == [
        f"allsky-{DAY_NEW}-1200.jpg",
        f"allsky-{DAY_NEW}-1201.jpg",
        f"allsky-{DAY_NEW}-1202.jpg",
    ]


def test_the_ledger_records_the_clock_the_config_chose_not_the_flag(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    overlay_config = _prepare_config(tmp_path / "configs" / "overlay.yaml", "overlay")

    result = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(overlay_config),
        "--since",
        "2026-08-10",
    )

    assert result.exit_code == 0, result.output
    recorded = _reload(data).frames(DAY_NEW)
    assert recorded is not None
    assert recorded["timestamps"] == "overlay"


def test_a_day_extracted_through_a_config_naming_the_overlay_is_not_extracted_again(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    overlay_config = _prepare_config(tmp_path / "configs" / "overlay.yaml", "overlay")
    first = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(overlay_config),
        "--since",
        "2026-08-10",
    )
    assert first.exit_code == 0, first.output

    second = _sync(data, mirror, "--extract", "--since", "2026-08-10")

    assert second.exit_code == 0, second.output
    assert "nothing to do" in second.output


def _re_extract_under_the_config_clock(data: Path, mirror: fake.ArchiveMirror) -> Path:
    assert _sync(data, mirror, "--extract", "--since", "2026-08-10").exit_code == 0
    modelled = _prepare_config(data.parent / "configs" / "modelled.yaml", "modelled")
    second = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(modelled),
        "--since",
        "2026-08-10",
    )
    assert second.exit_code == 0, second.output
    return data / FRAMES_SUBDIR / DAY_NEW


def test_re_extracting_a_day_under_the_other_clock_leaves_none_of_the_first_clocks_frames(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"

    frames_dir = _re_extract_under_the_config_clock(data, mirror)

    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == [
        f"allsky-{DAY_NEW}-0600.jpg",
        f"allsky-{DAY_NEW}-0601.jpg",
        f"allsky-{DAY_NEW}-0602.jpg",
    ]


def test_the_recorded_frame_count_matches_what_a_re_extraction_left_on_disk(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"

    frames_dir = _re_extract_under_the_config_clock(data, mirror)

    recorded = _reload(data).frames(DAY_NEW)
    assert recorded is not None
    assert recorded["count"] == len(list(frames_dir.glob("*.jpg")))


def _fail_re_extracting_a_day_whose_overlay_cannot_be_read(
    data: Path, mirror: fake.ArchiveMirror, tmp_path: Path
) -> Path:
    """Extract DAY_LATER under the modelled clock, then fail re-extracting it under the overlay.

    The published video for that day carries another day's burned-in stamps, so
    the overlay reader refuses it — the permanent, un-retryable fault the frames
    already on disk must survive.
    """
    mislabelled = fake.write_overlay_video(
        tmp_path / "staging" / f"allsky-{DAY_LATER}.mp4",
        STAMPS[DAY_OLD],
        height=FRAME_HEIGHT,
        width=FRAME_WIDTH,
    )
    mirror.publish_video_file(mislabelled)
    modelled_config = _prepare_config(tmp_path / "configs" / "modelled.yaml", "modelled")
    modelled = _sync(
        data,
        mirror,
        "--extract",
        "--timestamps",
        "config",
        "-c",
        str(modelled_config),
        "--since",
        "2026-08-11",
    )
    assert modelled.exit_code == 0, modelled.output

    failed = _sync(data, mirror, "--extract", "--since", "2026-08-11")

    assert failed.exit_code == 0, failed.output
    return data / FRAMES_SUBDIR / DAY_LATER


def test_a_re_extraction_that_fails_keeps_the_frames_the_previous_one_left(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"

    frames_dir = _fail_re_extracting_a_day_whose_overlay_cannot_be_read(data, mirror, tmp_path)

    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == [
        f"allsky-{DAY_LATER}-0600.jpg",
        f"allsky-{DAY_LATER}-0601.jpg",
        f"allsky-{DAY_LATER}-0602.jpg",
    ]


def test_a_re_extraction_that_fails_keeps_the_record_of_the_frames_it_could_not_replace(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"

    _fail_re_extracting_a_day_whose_overlay_cannot_be_read(data, mirror, tmp_path)

    recorded = _reload(data).frames(DAY_LATER)
    assert recorded is not None
    assert recorded["timestamps"].startswith("config:")
    assert recorded["count"] == 3


def test_a_re_extraction_that_fails_leaves_no_half_written_frame_set_behind(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"

    frames_dir = _fail_re_extracting_a_day_whose_overlay_cannot_be_read(data, mirror, tmp_path)

    assert [path.name for path in frames_dir.parent.iterdir()] == [DAY_LATER]


def test_a_day_whose_overlay_can_never_be_read_is_recorded_and_not_decoded_again(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    _fail_re_extracting_a_day_whose_overlay_cannot_be_read(data, mirror, tmp_path)

    assert (
        _reload(data).extraction_faulted(DAY_LATER, step=1, resize=None, timestamps="overlay")
        is not None
    )

    again = _sync(data, mirror, "--extract", "--since", "2026-08-11")

    assert again.exit_code == 0, again.output
    assert "nothing to do" in again.output


def test_a_video_that_decodes_to_no_frame_keeps_the_frames_the_previous_run_left(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data = tmp_path / "data"
    assert _sync(data, mirror, "--extract", "--since", "2026-08-10").exit_code == 0
    frames_dir = data / FRAMES_SUBDIR / DAY_NEW
    before = sorted(path.name for path in frames_dir.glob("*.jpg"))
    assert before

    monkeypatch.setattr("allsky.overlay._sampled_reads", lambda *_args, **_kwargs: iter(()))
    empty = _sync(data, mirror, "--extract", "--step", "2", "--since", "2026-08-10")

    assert empty.exit_code == 0, empty.output
    assert sorted(path.name for path in frames_dir.glob("*.jpg")) == before


def test_re_extracting_a_day_queues_its_frames_for_upload_again(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    _mark_fully_mirrored(ledger, tmp_path, entry)

    plan = _plan(
        entry,
        ledger,
        tmp_path,
        target=TARGET,
        extract=True,
        timestamps=TimestampSource.config,
        upload=UploadChoice.frames,
    )

    assert (plan.extract, plan.upload_frames) == (True, True)


def test_days_a_previous_release_clocked_ambiguously_are_named_before_any_work_starts(
    mirror: fake.ArchiveMirror, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    data = tmp_path / "data"
    assert _sync(data, mirror, "--extract", "--since", "2026-08-10").exit_code == 0
    ledger = _reload(data)
    (ledger.frames(DAY_NEW) or {})["timestamps"] = "config"
    ledger.save()

    with caplog.at_level(logging.WARNING, logger="allsky.cli.archive"):
        result = _sync(data, mirror, "--extract", "--since", "2026-08-10")

    assert result.exit_code == 0, result.output
    assert any(
        "recorded 'config' as the clock" in record.getMessage() and DAY_NEW in record.getMessage()
        for record in caplog.records
    )


def test_recording_a_new_frame_set_forgets_the_upload_of_the_one_it_replaced(
    entry: ArchiveEntry, ledger: Ledger, tmp_path: Path
):
    _mark_fully_mirrored(ledger, tmp_path, entry)
    frames_destination = TARGET.path(REMOTE_FRAMES, entry.key)
    assert ledger.uploaded(entry.key, frames_destination) is True

    ledger.record_frames(
        entry.key,
        directory=f"{FRAMES_SUBDIR}/{entry.key}",
        count=4,
        step=1,
        resize=None,
        timestamps="config",
    )

    assert ledger.uploaded(entry.key, frames_destination) is False
    assert ledger.uploaded(entry.key, TARGET.path(REMOTE_VIDEOS, entry.filename)) is True


def test_changing_the_step_re_extracts_without_asking_the_server_for_the_video_again(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    assert _sync(data, mirror, "--extract", "--since", "2026-08-10").exit_code == 0

    mirror.requests.clear()
    second = _sync(data, mirror, "--extract", "--step", "2", "--since", "2026-08-10")

    assert second.exit_code == 0, second.output
    assert mirror.video_requests() == []
    recorded = _reload(data).frames(DAY_NEW)
    assert recorded is not None
    assert recorded["step"] == 2
    assert recorded["count"] == 2


def test_a_day_the_index_advertises_without_a_file_is_reported_as_a_failure(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    mirror.advertise(f"allsky-{DAY_LATER}.mp4")
    result = _sync(data, mirror)

    assert result.exit_code == 1
    assert "2 day(s) processed, 1 failed" in result.output
    assert sorted(_reload(data).entries) == [DAY_OLD, DAY_NEW]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--upload", "videos"), "needs --drive-remote"),
        (("--prune-uploaded",), "only makes sense together with --upload"),
        (("--upload", "frames", "--drive-remote", "labmim"), "needs --extract"),
    ],
)
def test_incoherent_upload_flag_combinations_are_rejected(
    mirror: fake.ArchiveMirror, tmp_path: Path, extra: tuple[str, ...], message: str
):
    result = _sync(tmp_path / "data", mirror, *extra)
    assert result.exit_code != 0
    assert message in result.output


def test_uploading_records_the_destination_so_the_next_run_leaves_it_alone(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    data = tmp_path / "data"
    first = _sync(data, mirror, "--upload", "videos", "--drive-remote", "labmim")
    assert first.exit_code == 0, first.output

    invocations = fake.rclone_invocations(rclone_log)
    assert invocations[0].startswith("lsd labmim:")
    assert sum(1 for line in invocations if line.startswith("copyto")) == 2
    assert _reload(data).uploaded(DAY_NEW, TARGET.path(REMOTE_VIDEOS, f"allsky-{DAY_NEW}.mp4"))

    second = _sync(data, mirror, "--upload", "videos", "--drive-remote", "labmim")
    assert "nothing to do" in second.output


def test_an_upload_that_failed_is_retried_next_run_without_fetching_the_video_again(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    data = tmp_path / "data"
    fake.write_fake_rclone(rclone_log.parent / "bin", rclone_log, fail_on="copyto")

    failed = _sync(data, mirror, "--upload", "videos", "--drive-remote", "labmim")
    assert failed.exit_code == 1
    assert "0 day(s) processed, 2 failed" in failed.output
    stalled = _reload(data)
    assert stalled.uploaded(DAY_NEW, TARGET.path(REMOTE_VIDEOS, f"allsky-{DAY_NEW}.mp4")) is False
    assert stalled.has_video(DAY_NEW, local_root=data) is True

    fake.write_fake_rclone(rclone_log.parent / "bin", rclone_log)
    mirror.requests.clear()
    retried = _sync(data, mirror, "--upload", "videos", "--drive-remote", "labmim")

    assert retried.exit_code == 0, retried.output
    assert "2 day(s) processed, 0 failed" in retried.output
    assert mirror.video_requests() == []
    assert _reload(data).uploaded(DAY_NEW, TARGET.path(REMOTE_VIDEOS, f"allsky-{DAY_NEW}.mp4"))


@pytest.mark.usefixtures("rclone_log")
def test_pruning_deletes_the_local_video_but_keeps_what_the_ledger_knows_about_it(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    result = _sync(
        data,
        mirror,
        "--upload",
        "videos",
        "--drive-remote",
        "labmim",
        "--prune-uploaded",
    )

    assert result.exit_code == 0, result.output
    assert list((data / VIDEOS_SUBDIR).glob("*.mp4")) == []
    stored = _reload(data).video(DAY_NEW)
    assert stored is not None
    assert stored["pruned"] is True
    assert stored["sha256"]

    assert (
        "nothing to do"
        in _sync(data, mirror, "--upload", "videos", "--drive-remote", "labmim").output
    )


def test_extra_rclone_flags_reach_every_invocation(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    """The flag has to reach the transfers, not merely every line that happened.

    ``rclone_invocations`` returns [] when the log is absent and ``all([])`` is
    True, so a run that uploaded nothing satisfied this on its own; and the
    uploader appends the extra args to every command including the ``lsd``
    pre-flight, so 'every line carries it' held without a single copy running.
    """
    result = _sync(
        tmp_path / "data",
        mirror,
        "--upload",
        "videos",
        "--drive-remote",
        "labmim",
        "--rclone-arg",
        "--transfers=8",
    )

    assert result.exit_code == 0, result.output
    invocations = fake.rclone_invocations(rclone_log)
    copies = [line for line in invocations if line.startswith("copyto")]
    assert copies
    assert all(line.endswith("--transfers=8") for line in invocations)


def test_a_second_sync_cannot_start_while_another_holds_the_ledger_lock(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    (data / STATE_SUBDIR).mkdir(parents=True)
    with ledger_lock(data / STATE_SUBDIR / LEDGER_FILENAME):
        result = _sync(data, mirror)

    assert result.exit_code != 0
    assert isinstance(result.exception, ArchiveError)
    assert "already running" in str(result.exception)
    assert mirror.video_requests() == []


def test_snapshot_writes_the_live_frame_and_a_provenance_sidecar(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    out_dir = tmp_path / "snapshots"
    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(out_dir),
            "--base-url",
            mirror.base_url,
            "--insecure",
        ],
    )

    assert result.exit_code == 0, result.output
    (image,) = list(out_dir.glob("*.jpg"))
    (sidecar,) = list(out_dir.glob("*.json"))
    assert image.read_bytes() == LIVE_IMAGE
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    assert metadata["bytes"] == len(LIVE_IMAGE)
    assert metadata["image"] == image.name
    assert metadata["content_type"] == "image/jpeg"
    assert metadata["captured_at_source"] == "server-last-modified"
    assert metadata["source_url"].endswith("/image.jpg")


def test_a_non_finite_prediction_is_refused_instead_of_published_as_a_bare_nan(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def predict_a_diverged_model(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "predictions": {"dhi": float("nan")},
            "features": {"imputed": []},
        }

    monkeypatch.setattr("allsky.snapshot.predict_snapshot", predict_a_diverged_model)
    checkpoint = tmp_path / "diverged.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    out_dir = tmp_path / "snapshots"

    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(out_dir),
            "--base-url",
            mirror.base_url,
            "--insecure",
            "--checkpoint",
            str(checkpoint),
        ],
    )

    assert result.exit_code == 1
    assert list(out_dir.glob("*.prediction.json")) == []


def _snapshot_recording_prediction_kwargs(
    mirror: fake.ArchiveMirror,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *extra_options: str,
) -> tuple[Result, dict[str, object]]:
    recorded: dict[str, object] = {}

    def predict_recording_its_kwargs(*_args: object, **kwargs: object) -> dict[str, object]:
        recorded.update(kwargs)
        return {"predictions": {"dhi": 137.0}, "features": {"imputed": []}}

    monkeypatch.setattr("allsky.snapshot.predict_snapshot", predict_recording_its_kwargs)
    checkpoint = tmp_path / "moved.ckpt"
    checkpoint.write_bytes(b"checkpoint-bytes")

    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(tmp_path / "snapshots"),
            "--base-url",
            mirror.base_url,
            "--insecure",
            "--checkpoint",
            str(checkpoint),
            *extra_options,
        ],
    )
    return result, recorded


def test_the_embeddings_dir_flag_overrides_the_store_baked_into_the_checkpoint(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = tmp_path / "embeddings"
    store.mkdir()

    result, recorded = _snapshot_recording_prediction_kwargs(
        mirror, tmp_path, monkeypatch, "--embeddings-dir", str(store)
    )

    assert result.exit_code == 0, result.output
    assert recorded["embeddings_dir"] == store


def test_a_snapshot_without_the_embeddings_dir_flag_keeps_the_baked_store(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    result, recorded = _snapshot_recording_prediction_kwargs(mirror, tmp_path, monkeypatch)

    assert result.exit_code == 0, result.output
    assert recorded["embeddings_dir"] is None


def _die_mid_stream(*_args: object, **_kwargs: object) -> None:
    raise http.client.IncompleteRead(b"first-chunk", 4096)


def test_a_day_whose_body_dies_mid_stream_is_one_failed_day_not_an_aborted_run(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(ArchiveClient, "download", _die_mid_stream)

    result = _sync(tmp_path / "data", mirror)

    assert isinstance(result.exception, SystemExit)
    assert "0 day(s) processed, 2 failed" in result.output


def test_a_snapshot_whose_body_dies_mid_stream_exits_one_instead_of_crashing(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("allsky.snapshot.capture_snapshot", _die_mid_stream)

    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(tmp_path / "snapshots"),
            "--base-url",
            mirror.base_url,
            "--insecure",
        ],
    )

    assert isinstance(result.exception, SystemExit)
    assert result.exit_code == 1


def test_both_archive_commands_are_registered_on_the_allsky_app():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sync-archive" in result.output
    assert "snapshot" in result.output


class _StubLiveCamera:
    """Serves one live frame with fixed headers, so only the clock varies."""

    base_url = "https://camera.invalid/"

    def __init__(self, payload: bytes, headers: dict[str, str]) -> None:
        self._payload = payload
        self._headers = headers

    def fetch_live_image(self) -> tuple[bytes, dict[str, str]]:
        return self._payload, self._headers


def _overlay_jpeg(stamp: dt.datetime) -> bytes:
    frame = fake.render_overlay_frame(stamp.strftime("%Y%m%d%H%M%S"))
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


@pytest.mark.parametrize("zone", ["America/Bahia", "UTC", "Asia/Tokyo"])
def test_the_capture_clock_is_the_sites_not_the_hosts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, zone: str
):
    site_now = dt.datetime.now(tz=SITE_TZ).replace(microsecond=0)
    naive_site_now = site_now.replace(tzinfo=None)
    camera = _StubLiveCamera(
        _overlay_jpeg(naive_site_now),
        {
            "last-modified": site_now.astimezone(dt.UTC).strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "content-type": "image/jpeg",
        },
    )
    monkeypatch.setenv("TZ", zone)
    time.tzset()
    try:
        snapshot = capture_snapshot(camera, tmp_path / zone.replace("/", "_"))
    finally:
        monkeypatch.delenv("TZ", raising=False)
        time.tzset()

    assert snapshot.captured_at == pd.Timestamp(naive_site_now)
    metadata = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
    assert metadata["captured_at_source"] == "overlay"
    assert metadata["server_last_modified_as_local"] == naive_site_now.isoformat()


def test_two_modelled_cadences_are_different_clocks_in_the_ledger():
    """`_clock` collapsed every `modelled` config to the bare string "config", so
    the resume gate answered "already extracted" for a day whose frames carry the
    other cadence — and the frames on disk are named after their own timestamps,
    so the two sets would have survived side by side."""
    from allsky.cli.archive import _clock_record
    from allsky.config import VideoConfig

    one_per_minute = _clock_record(VideoConfig(timestamps="modelled", minutes_per_frame=1.0))
    two_per_minute = _clock_record(VideoConfig(timestamps="modelled", minutes_per_frame=0.5))
    later_start = _clock_record(
        VideoConfig(timestamps="modelled", minutes_per_frame=1.0, start_time="07:00")
    )

    assert one_per_minute != two_per_minute
    assert one_per_minute != later_start
    assert _clock_record(VideoConfig(timestamps="overlay")) == "overlay"


def test_the_same_modelled_cadence_is_the_same_clock():
    """Otherwise every run would re-extract every day."""
    from allsky.cli.archive import _clock_record
    from allsky.config import VideoConfig

    first = _clock_record(VideoConfig(timestamps="modelled", minutes_per_frame=1.0))
    second = _clock_record(VideoConfig(timestamps="modelled", minutes_per_frame=1.0))

    assert first == second


class TestTheCaptureClockFallbackChain:
    """Three sources in order — the burned-in overlay, the server's
    Last-Modified, then the host clock — and only the first was ever exercised.
    The last is the exact regression the docs warn about: a container in UTC
    names the snapshot three hours off and nothing fails."""

    @staticmethod
    def _capture(tmp_path: Path, payload: bytes, headers: dict[str, str]):
        from allsky.snapshot import capture_snapshot

        return capture_snapshot(_StubLiveCamera(payload, headers), tmp_path)

    def _sidecar(self, snapshot) -> dict:
        loaded: dict = json.loads(snapshot.metadata_path.read_text(encoding="utf-8"))
        return loaded

    def test_a_fresh_overlay_wins(self, tmp_path: Path):
        now = dt.datetime.now(tz=SITE_TZ).replace(microsecond=0, tzinfo=None)
        snapshot = self._capture(tmp_path, _overlay_jpeg(now), {})

        assert self._sidecar(snapshot)["captured_at_source"] == "overlay"

    def test_a_stale_overlay_falls_to_a_fresh_last_modified(self, tmp_path: Path):
        stale = dt.datetime.now(tz=SITE_TZ).replace(microsecond=0, tzinfo=None) - dt.timedelta(
            days=3
        )
        fresh = dt.datetime.now(tz=dt.UTC).replace(microsecond=0)
        headers = {"last-modified": fresh.strftime("%a, %d %b %Y %H:%M:%S GMT")}

        snapshot = self._capture(tmp_path, _overlay_jpeg(stale), headers)

        assert self._sidecar(snapshot)["captured_at_source"] == "server-last-modified"

    def test_a_stale_overlay_and_an_unparseable_header_fall_to_the_host_clock(
        self, tmp_path: Path, caplog
    ):
        stale = dt.datetime.now(tz=SITE_TZ).replace(microsecond=0, tzinfo=None) - dt.timedelta(
            days=3
        )

        with caplog.at_level("WARNING"):
            snapshot = self._capture(tmp_path, _overlay_jpeg(stale), {"last-modified": "nonsense"})

        assert self._sidecar(snapshot)["captured_at_source"] == "local-clock"
        assert "local clock" in caplog.text

    def test_an_empty_payload_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="empty live frame"):
            self._capture(tmp_path, b"", {})


def test_uploading_both_mirrors_the_frames_and_records_the_destination(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    """`--upload both` is the cron recipe's own invocation and no test drove it:
    the frames branch could have stopped mirroring, or stopped recording what it
    mirrored, with every assertion in this file still green."""
    data = tmp_path / "data"

    result = _sync(data, mirror, "--extract", "--upload", "both", "--drive-remote", "labmim")

    assert result.exit_code == 0, result.output
    invocations = fake.rclone_invocations(rclone_log)
    frames = [line for line in invocations if line.startswith("sync")]
    assert frames, invocations
    for line in frames:
        assert "--include *.jpg" in line
        assert f"{REMOTE_FRAMES}/" in line
    ledger = _reload(data)
    assert ledger.uploaded(DAY_NEW, TARGET.path(REMOTE_FRAMES, DAY_NEW))
    assert ledger.frames(DAY_NEW) is not None


@pytest.mark.usefixtures("rclone_log")
def test_a_second_upload_both_run_leaves_the_frames_alone(
    mirror: fake.ArchiveMirror, tmp_path: Path
):
    data = tmp_path / "data"
    first = _sync(data, mirror, "--extract", "--upload", "both", "--drive-remote", "labmim")
    assert first.exit_code == 0, first.output

    second = _sync(data, mirror, "--extract", "--upload", "both", "--drive-remote", "labmim")

    assert "nothing to do" in second.output


def test_a_snapshot_publishes_its_frame_and_sidecar_to_drive(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    """The whole Drive publish path of `snapshot` — the branch the hourly cron
    runs — was never executed: it could have stopped uploading, or uploaded to
    the wrong day folder, with every snapshot test still green."""
    out_dir = tmp_path / "snapshots"

    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(out_dir),
            "--base-url",
            mirror.base_url,
            "--insecure",
            "--drive-remote",
            "labmim",
        ],
    )

    assert result.exit_code == 0, result.output
    copied = [line for line in fake.rclone_invocations(rclone_log) if line.startswith("copyto")]
    # The JPEG and its provenance sidecar; no checkpoint, so no prediction.
    assert len(copied) == 2
    day = f"{next(out_dir.glob('*.jpg')).stem.split('-')[1]}"
    assert all(f"{REMOTE_SNAPSHOTS}/{day}/" in line for line in copied), copied


def test_a_snapshot_whose_upload_fails_exits_one(
    mirror: fake.ArchiveMirror, tmp_path: Path, rclone_log: Path
):
    """The frame is on disk either way; what must not happen is exit 0 while the
    site never received it."""
    fake.write_fake_rclone(rclone_log.parent / "bin", rclone_log, fail_on="copyto")
    out_dir = tmp_path / "snapshots"

    result = runner.invoke(
        app,
        [
            "snapshot",
            "--out",
            str(out_dir),
            "--base-url",
            mirror.base_url,
            "--insecure",
            "--drive-remote",
            "labmim",
        ],
    )

    assert result.exit_code == 1
    assert list(out_dir.glob("*.jpg")), "the capture itself must survive the failed upload"
