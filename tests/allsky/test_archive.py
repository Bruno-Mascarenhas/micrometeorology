"""Tests for allsky.archive against a local HTTP mirror — no network is touched.

Every request in this module goes to a ``ThreadingHTTPServer`` bound to an
ephemeral 127.0.0.1 port, so the HTTPS-only client is driven with
``allow_plaintext=True``; the one test that proves the opt-in is required leaves
it off.
"""

import hashlib
import json
import ssl
from collections.abc import Iterator
from pathlib import Path

import pytest

from allsky import archive
from allsky.archive import (
    ARCHIVE_BASE_URL,
    ArchiveClient,
    ArchiveError,
    Ledger,
    build_ssl_context,
    ledger_lock,
    parse_catalog,
)
from tests.allsky import _archive_fake as fake

DAY_OLD = "20260809"
DAY_NEW = "20260810"
PAYLOAD_OLD = b"O" * 77
PAYLOAD_NEW = b"N" * 4096
LIVE_IMAGE = b"\xff\xd8\xff\xe0live-frame-bytes"


@pytest.fixture
def mirror(tmp_path: Path) -> Iterator[fake.ArchiveMirror]:
    """A served archive tree holding two days and the live frame."""
    served = fake.ArchiveMirror(tmp_path / "site")
    served.publish_video(f"allsky-{DAY_OLD}.mp4", PAYLOAD_OLD)
    served.publish_video(f"allsky-{DAY_NEW}.mp4", PAYLOAD_NEW)
    served.publish_image(LIVE_IMAGE)
    try:
        yield served
    finally:
        served.close()


@pytest.fixture
def client(mirror: fake.ArchiveMirror) -> ArchiveClient:
    return ArchiveClient(mirror.base_url, allow_plaintext=True, retries=1, delay=0.0, timeout=10.0)


def _system_pem() -> str:
    certificates = ssl.create_default_context().get_ca_certs(binary_form=True)
    if not certificates:
        pytest.skip("no CA certificates in the system trust store to build a PEM from")
    return ssl.DER_cert_to_PEM_cert(certificates[0])


def test_parse_catalog_collapses_the_two_links_each_day_gets_on_the_index_page():
    page = fake.build_index_html([f"allsky-{DAY_NEW}.mp4"])
    assert page.count(f"allsky-{DAY_NEW}.mp4") == 2
    assert [entry.key for entry in parse_catalog(page)] == [DAY_NEW]


def test_parse_catalog_returns_days_oldest_first_whatever_order_the_page_lists_them():
    page = fake.build_index_html(
        [f"allsky-{DAY_NEW}.mp4", "allsky-20250101.mp4", f"allsky-{DAY_OLD}.mp4"]
    )
    assert [entry.key for entry in parse_catalog(page)] == ["20250101", DAY_OLD, DAY_NEW]


@pytest.mark.parametrize("impossible", ["20261332", "20260230", "20261301", "20260000"])
def test_parse_catalog_skips_a_filename_whose_date_cannot_exist(impossible: str):
    page = fake.build_index_html([f"allsky-{impossible}.mp4", f"allsky-{DAY_NEW}.mp4"])
    assert [entry.key for entry in parse_catalog(page)] == [DAY_NEW]


def test_parse_catalog_resolves_filenames_against_the_videos_directory():
    (entry,) = parse_catalog(fake.build_index_html([f"allsky-{DAY_NEW}.mp4"]), "https://host/root/")
    assert entry.url == f"https://host/root/videos/allsky-{DAY_NEW}.mp4"
    assert entry.filename == f"allsky-{DAY_NEW}.mp4"
    assert entry.date.isoformat() == "2026-08-10"


def test_parse_catalog_returns_nothing_for_a_page_that_lists_no_timelapses():
    assert parse_catalog("<html><body>maintenance</body></html>") == ()


def test_list_videos_reads_the_served_index_and_orders_it_oldest_first(client: ArchiveClient):
    assert [entry.key for entry in client.list_videos()] == [DAY_OLD, DAY_NEW]


def test_list_videos_ignores_a_link_the_server_advertises_without_a_file_behind_it(
    mirror: fake.ArchiveMirror, client: ArchiveClient
):
    mirror.advertise("allsky-20261332.mp4")
    assert [entry.key for entry in client.list_videos()] == [DAY_OLD, DAY_NEW]


def test_download_writes_the_payload_and_reports_its_size_and_sha256(
    client: ArchiveClient, tmp_path: Path
):
    entry = client.list_videos()[1]
    destination = tmp_path / "videos"
    result = client.download(entry, destination)

    assert result.downloaded is True
    assert result.path == destination / f"allsky-{DAY_NEW}.mp4"
    assert result.path.read_bytes() == PAYLOAD_NEW
    assert result.size == len(PAYLOAD_NEW)
    assert result.sha256 == hashlib.sha256(PAYLOAD_NEW).hexdigest()
    assert result.last_modified is not None


def test_download_leaves_the_destination_directory_free_of_temporary_files(
    client: ArchiveClient, tmp_path: Path
):
    destination = tmp_path / "videos"
    client.download(client.list_videos()[0], destination)
    assert [path.name for path in destination.iterdir()] == [f"allsky-{DAY_OLD}.mp4"]


def test_download_raises_when_the_server_sends_fewer_bytes_than_it_announced(
    mirror: fake.ArchiveMirror, client: ArchiveClient, tmp_path: Path
):
    entry = client.list_videos()[1]
    mirror.truncate[mirror.video_url_path(entry.filename)] = 512
    destination = tmp_path / "videos"

    with pytest.raises(ArchiveError, match=r"truncated: got 512 bytes, server announced 4096"):
        client.download(entry, destination)

    assert list(destination.iterdir()) == []


def test_download_of_a_day_the_server_no_longer_serves_fails_loudly(
    mirror: fake.ArchiveMirror, client: ArchiveClient, tmp_path: Path
):
    entry = client.list_videos()[0]
    (mirror.videos_dir / entry.filename).unlink()

    with pytest.raises(ArchiveError, match="failed after 1 attempt"):
        client.download(entry, tmp_path / "videos")


def test_the_client_refuses_a_file_url_because_its_opener_carries_no_file_handler(
    client: ArchiveClient,
):
    with pytest.raises(ArchiveError, match="unknown url type: file"):
        client.fetch_text("file:///etc/passwd")


def test_plaintext_http_is_refused_unless_the_caller_opts_in(mirror: fake.ArchiveMirror):
    https_only = ArchiveClient(mirror.base_url, retries=1, delay=0.0, timeout=10.0)
    with pytest.raises(ArchiveError, match="unknown url type: http"):
        https_only.list_videos()


def test_fetch_live_image_returns_the_bytes_and_lowercased_headers(client: ArchiveClient):
    payload, headers = client.fetch_live_image()
    assert payload == LIVE_IMAGE
    assert headers["content-type"] == "image/jpeg"
    assert list(headers) == [key.lower() for key in headers]


def test_the_client_normalises_a_base_url_without_a_trailing_slash(mirror: fake.ArchiveMirror):
    trimmed = ArchiveClient(
        mirror.base_url.rstrip("/"), allow_plaintext=True, retries=1, delay=0.0, timeout=10.0
    )
    assert [entry.key for entry in trimmed.list_videos()] == [DAY_OLD, DAY_NEW]


def test_a_ledger_round_trips_through_save_and_load(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    ledger.record_frames(
        DAY_NEW, directory="frames/20260810", count=7, step=3, resize=224, timestamps="overlay"
    )
    ledger.record_upload(DAY_NEW, "gd:LabMiM/allsky/videos/x.mp4", kind="video")
    ledger.save()

    reloaded = Ledger.load(tmp_path / "ledger.json")
    stored = reloaded.video(DAY_NEW)
    assert stored == ledger.video(DAY_NEW)
    assert stored is not None
    assert stored["sha256"] == result.sha256
    assert reloaded.frames_match(DAY_NEW, step=3, resize=224) is True
    assert reloaded.uploaded(DAY_NEW, "gd:LabMiM/allsky/videos/x.mp4") is True
    assert reloaded.last_modified(DAY_NEW) == fake.LAST_MODIFIED


def test_loading_a_missing_ledger_starts_an_empty_one(tmp_path: Path):
    ledger = Ledger.load(tmp_path / "nothing-here.json")
    assert ledger.entries == {}
    assert ledger.video(DAY_NEW) is None
    assert ledger.frames(DAY_NEW) is None


def test_loading_a_json_file_that_is_not_a_ledger_is_refused(tmp_path: Path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"version": 1, "days": []}), encoding="utf-8")
    with pytest.raises(ArchiveError, match="not an all-sky archive ledger"):
        Ledger.load(path)


def test_loading_a_ledger_written_by_another_version_is_refused(tmp_path: Path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"version": 99, "entries": {}}), encoding="utf-8")
    with pytest.raises(ArchiveError, match="ledger version 99"):
        Ledger.load(path)


def test_has_video_is_false_once_the_recorded_file_is_deleted(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    assert ledger.has_video(DAY_NEW, local_root=tmp_path) is True

    result.path.unlink()
    assert ledger.has_video(DAY_NEW, local_root=tmp_path) is False


def test_has_video_is_false_when_the_file_on_disk_has_a_different_size(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    result.path.write_bytes(b"a shorter file")
    assert ledger.has_video(DAY_NEW, local_root=tmp_path) is False


def test_has_video_reports_the_record_alone_when_no_local_root_is_given(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    result.path.unlink()
    assert ledger.has_video(DAY_NEW) is True
    assert ledger.has_video("20200101") is False


@pytest.mark.parametrize(("step", "resize"), [(1, None), (2, 224), (3, None), (3, 448), (1, 224)])
def test_frames_match_is_false_whenever_step_or_resize_differ_from_the_record(
    tmp_path: Path, step: int, resize: int | None
):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_frames(
        DAY_NEW, directory="frames/20260810", count=4, step=3, resize=224, timestamps="overlay"
    )
    assert ledger.frames_match(DAY_NEW, step=3, resize=224) is True
    assert ledger.frames_match(DAY_NEW, step=step, resize=resize) is False


def test_frames_match_is_false_for_a_day_that_was_never_extracted(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    assert ledger.frames_match(DAY_NEW, step=1, resize=None) is False


def test_uploaded_tracks_each_destination_on_its_own(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_upload(DAY_NEW, "gd:LabMiM/allsky/videos/allsky-20260810.mp4", kind="video")
    assert ledger.uploaded(DAY_NEW, "gd:LabMiM/allsky/videos/allsky-20260810.mp4") is True
    assert ledger.uploaded(DAY_NEW, "gd:LabMiM/allsky/frames/20260810") is False
    assert ledger.uploaded(DAY_OLD, "gd:LabMiM/allsky/videos/allsky-20260810.mp4") is False


def test_uploaded_is_false_for_the_empty_destination_a_missing_drive_target_produces(
    tmp_path: Path,
):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_upload(DAY_NEW, "", kind="video")
    assert ledger.uploaded(DAY_NEW, "") is False


def test_mark_pruned_reports_the_video_as_gone_but_keeps_everything_it_knew(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    ledger.mark_pruned(DAY_NEW)

    assert ledger.has_video(DAY_NEW, local_root=tmp_path) is False
    assert ledger.has_video(DAY_NEW) is True
    stored = ledger.video(DAY_NEW)
    assert stored is not None
    assert stored["sha256"] == result.sha256
    assert stored["pruned"] is True
    assert "pruned_at" in stored


def test_saving_a_later_day_keeps_the_entries_for_days_the_server_has_purged(tmp_path: Path):
    path = tmp_path / "ledger.json"
    first = Ledger(path)
    purged = fake.record_downloaded_day(first, tmp_path, DAY_OLD)
    first.save()

    second = Ledger.load(path)
    fake.record_downloaded_day(second, tmp_path, DAY_NEW)
    second.save()

    reloaded = Ledger.load(path)
    assert sorted(reloaded.entries) == [DAY_OLD, DAY_NEW]
    stored = reloaded.video(DAY_OLD)
    assert stored is not None
    assert stored["sha256"] == purged.sha256


def test_recorded_paths_are_stored_relative_to_the_root_and_resolved_back_against_it(
    tmp_path: Path,
):
    ledger = Ledger(tmp_path / "ledger.json")
    result = fake.record_downloaded_day(ledger, tmp_path, DAY_NEW)
    stored = ledger.video(DAY_NEW)
    assert stored is not None
    assert stored["path"] == f"videos/allsky-{DAY_NEW}.mp4"
    assert ledger.video_path(DAY_NEW, root=tmp_path) == result.path
    assert ledger.video_path(DAY_NEW) == Path(f"videos/allsky-{DAY_NEW}.mp4")


def test_frames_dir_resolves_the_recorded_relative_directory_against_the_root(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_frames(
        DAY_NEW, directory="frames/20260810", count=2, step=1, resize=None, timestamps="overlay"
    )
    assert ledger.frames_dir(DAY_NEW, root=tmp_path) == tmp_path / "frames" / "20260810"
    assert ledger.frames_dir(DAY_OLD, root=tmp_path) is None


def _take_and_release_the_ledger_lock(path: Path) -> None:
    with ledger_lock(path):
        pass


def test_a_second_ledger_lock_is_refused_while_the_first_is_still_held(tmp_path: Path):
    path = tmp_path / "state" / "ledger.json"
    with ledger_lock(path), pytest.raises(ArchiveError, match="already running"):
        _take_and_release_the_ledger_lock(path)


def test_the_ledger_lock_is_released_when_its_block_exits(tmp_path: Path):
    path = tmp_path / "state" / "ledger.json"
    with ledger_lock(path):
        pass
    with ledger_lock(path):
        pass
    assert (tmp_path / "state" / "ledger.json.lock").is_file()


def test_the_ledger_lock_creates_the_state_directory_it_locks_in(tmp_path: Path):
    path = tmp_path / "brand" / "new" / "ledger.json"
    with ledger_lock(path):
        assert path.parent.is_dir()


def test_the_insecure_context_turns_verification_off(tmp_path: Path):
    context = build_ssl_context(state_dir=tmp_path, insecure=True)
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False


def test_a_cached_intermediate_keeps_the_context_build_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(archive, "AIA_INTERMEDIATE_URL", "http://127.0.0.1:1/unreachable.crt")
    cache = tmp_path / archive.INTERMEDIATE_CACHE_FILENAME
    cache.write_text(_system_pem(), encoding="ascii")

    context = build_ssl_context(state_dir=tmp_path, timeout=1.0)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.get_ca_certs()


def test_a_supplied_ca_file_short_circuits_the_intermediate_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(archive, "AIA_INTERMEDIATE_URL", "http://127.0.0.1:1/unreachable.crt")
    pem = tmp_path / "roots.pem"
    pem.write_text(_system_pem(), encoding="ascii")

    context = build_ssl_context(state_dir=tmp_path, ca_file=pem, timeout=1.0)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert not (tmp_path / archive.INTERMEDIATE_CACHE_FILENAME).exists()


def test_the_published_base_url_still_points_at_the_planetarium_over_https():
    assert ARCHIVE_BASE_URL == "https://allsky.planetario.ufba.br/"
