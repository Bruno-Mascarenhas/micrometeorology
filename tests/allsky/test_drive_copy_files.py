"""``RcloneUploader.copy_files``: the publisher's upload step, with the rclone process stubbed."""

from pathlib import Path

import pytest

from allsky.drive import DriveTarget, RcloneError, RcloneUploader


def _uploader(calls: list[list[str]]) -> RcloneUploader:
    uploader = RcloneUploader(DriveTarget(remote="ftp", root="site/Ceu"))

    def record(args: list[str]) -> str:
        calls.append(list(args))
        return ""

    uploader._run = record  # type: ignore[method-assign]
    return uploader


def test_files_are_copied_one_by_one_in_the_given_order(tmp_path):
    for name in ("allsky.jpg", "frame.json"):
        (tmp_path / name).write_bytes(b"x")
    calls: list[list[str]] = []

    destination = _uploader(calls).copy_files(tmp_path, ["allsky.jpg", "frame.json"])

    assert destination == "ftp:site/Ceu"
    assert [call[0] for call in calls] == ["copyto", "copyto"]
    assert [Path(call[1]).name for call in calls] == ["allsky.jpg", "frame.json"]
    assert [call[2] for call in calls] == ["ftp:site/Ceu/allsky.jpg", "ftp:site/Ceu/frame.json"]


def test_a_missing_file_refuses_before_any_transfer(tmp_path):
    (tmp_path / "allsky.jpg").write_bytes(b"x")
    calls: list[list[str]] = []

    with pytest.raises(RcloneError, match=r"frame\.json"):
        _uploader(calls).copy_files(tmp_path, ["allsky.jpg", "frame.json"])

    assert calls == []


def test_sync_is_never_used(tmp_path):
    (tmp_path / "frame.json").write_bytes(b"x")
    calls: list[list[str]] = []

    _uploader(calls).copy_files(tmp_path, ["frame.json"])

    assert all("sync" not in call for call in calls)
