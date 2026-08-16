"""Tests for allsky.drive — remote-path composition and the rclone wrapper.

No real ``rclone`` is involved: an executable stand-in written into ``tmp_path``
and put first on PATH records every invocation, so the tests can count processes
as well as inspect argv.
"""

import os
import shutil
from pathlib import Path

import pytest

from allsky.drive import DriveTarget, RcloneError, RcloneUploader
from tests.allsky import _archive_fake as fake

LOG_NAME = "rclone-invocations.log"


@pytest.fixture
def target() -> DriveTarget:
    return DriveTarget(remote="labmim")


@pytest.fixture
def rclone_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a logging stand-in for rclone first on PATH; yields the invocation log."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    log = tmp_path / LOG_NAME
    fake.write_fake_rclone(binaries, log)
    fake.write_fake_rclone(binaries, log, name="rclone-broken", exit_code=3)
    monkeypatch.setenv("PATH", f"{binaries}{os.pathsep}{os.environ['PATH']}")
    return log


@pytest.mark.parametrize(
    ("root", "parts", "expected"),
    [
        (
            "LabMiM/allsky",
            ("videos", "allsky-20260810.mp4"),
            "gd:LabMiM/allsky/videos/allsky-20260810.mp4",
        ),
        ("LabMiM/allsky", (), "gd:LabMiM/allsky"),
        ("", ("frames", "20260810"), "gd:frames/20260810"),
        ("/LabMiM/allsky/", ("frames",), "gd:LabMiM/allsky/frames"),
        ("LabMiM/allsky", ("", "snapshots"), "gd:LabMiM/allsky/snapshots"),
    ],
)
def test_drive_target_composes_the_remote_path(root: str, parts: tuple[str, ...], expected: str):
    assert DriveTarget(remote="gd", root=root).path(*parts) == expected


def test_a_missing_rclone_binary_explains_how_to_install_and_configure_it(
    target: DriveTarget, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    uploader = RcloneUploader(target=target)

    with pytest.raises(RcloneError, match="not found on PATH") as raised:
        uploader.executable()

    assert "rclone config" in str(raised.value)
    assert "rclone lsd labmim:" in str(raised.value)


@pytest.mark.usefixtures("rclone_log")
def test_the_resolved_binary_is_looked_up_once_and_then_reused(
    target: DriveTarget, monkeypatch: pytest.MonkeyPatch
):
    uploader = RcloneUploader(target=target)
    resolved = uploader.executable()
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    assert uploader.executable() == resolved


def test_uploading_a_whole_directory_costs_a_single_rclone_process(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    frames = tmp_path / "frames" / "20260810"
    frames.mkdir(parents=True)
    for minute in range(40):
        (frames / f"allsky-20260810-12{minute:02d}.jpg").write_bytes(b"jpeg")

    destination = RcloneUploader(target=target).upload_dir(
        frames, "frames", "20260810", pattern="*.jpg"
    )

    invocations = fake.rclone_invocations(rclone_log)
    assert len(invocations) == 1
    assert destination == "labmim:LabMiM/allsky/frames/20260810"
    assert invocations[0].split() == [
        "copy",
        str(frames),
        destination,
        "--checksum",
        "--include",
        "*.jpg",
    ]


def test_uploading_a_directory_without_a_pattern_passes_no_include_filter(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "one.jpg").write_bytes(b"jpeg")

    RcloneUploader(target=target).upload_dir(frames, "frames")
    assert "--include" not in fake.rclone_invocations(rclone_log)[0]


def test_uploading_a_file_sends_copyto_with_the_composed_destination(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    video = tmp_path / "allsky-20260810.mp4"
    video.write_bytes(b"timelapse")

    destination = RcloneUploader(target=target).upload_file(video, "videos", "allsky-20260810.mp4")

    assert destination == "labmim:LabMiM/allsky/videos/allsky-20260810.mp4"
    assert fake.rclone_invocations(rclone_log) == [
        f"copyto {video} {destination} --checksum",
    ]


def test_extra_arguments_are_appended_to_every_invocation(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    video = tmp_path / "allsky-20260810.mp4"
    video.write_bytes(b"timelapse")
    uploader = RcloneUploader(target=target, extra_args=("--transfers", "8"))

    uploader.upload_file(video, "videos", video.name)

    (invocation,) = fake.rclone_invocations(rclone_log)
    assert invocation.endswith("--checksum --transfers 8")


def test_uploading_something_that_is_not_a_file_never_reaches_rclone(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    with pytest.raises(RcloneError, match="nothing to upload"):
        RcloneUploader(target=target).upload_file(tmp_path / "absent.mp4", "videos", "absent.mp4")
    assert fake.rclone_invocations(rclone_log) == []


def test_uploading_a_directory_that_does_not_exist_never_reaches_rclone(
    target: DriveTarget, rclone_log: Path, tmp_path: Path
):
    with pytest.raises(RcloneError, match="nothing to upload"):
        RcloneUploader(target=target).upload_dir(tmp_path / "absent", "frames")
    assert fake.rclone_invocations(rclone_log) == []


@pytest.mark.usefixtures("rclone_log")
def test_a_non_zero_exit_becomes_an_rclone_error_carrying_the_binarys_complaint(
    target: DriveTarget, tmp_path: Path
):
    video = tmp_path / "allsky-20260810.mp4"
    video.write_bytes(b"timelapse")
    uploader = RcloneUploader(target=target, binary="rclone-broken")

    with pytest.raises(RcloneError, match="rclone copyto exited 3: fake rclone refused"):
        uploader.upload_file(video, "videos", video.name)


def test_check_probes_the_remote_root_before_any_transfer(target: DriveTarget, rclone_log: Path):
    RcloneUploader(target=target).check()
    assert fake.rclone_invocations(rclone_log) == ["lsd labmim: --max-depth 1"]


@pytest.mark.usefixtures("rclone_log")
def test_check_reports_an_unconfigured_remote_as_an_rclone_error(target: DriveTarget):
    with pytest.raises(RcloneError, match="rclone lsd exited 3"):
        RcloneUploader(target=target, binary="rclone-broken").check()
