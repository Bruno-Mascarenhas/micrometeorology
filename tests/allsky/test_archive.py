"""Tests for allsky.archive against a local HTTP mirror — no network is touched.

Every request in this module goes to a ``ThreadingHTTPServer`` bound to an
ephemeral 127.0.0.1 port, so the HTTPS-only client is driven with
``allow_plaintext=True``; the one test that proves the opt-in is required leaves
it off.

``AIA_INTERMEDIATE_PEM`` below is the genuine *RNP ICPEdu GR46 OV TLS CA 2025*
certificate the archive's leaf names in its AIA extension, embedded so the
accept path of the pinned digest is exercised without a fetch. A certificate
taken from the system trust store stands in for the self-signed CA an on-path
attacker would answer the plaintext fetch with.
"""

import errno
import hashlib
import http.client
import json
import ssl
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
AIA_INTERMEDIATE_PEM = """\
-----BEGIN CERTIFICATE-----
MIIGnjCCBIagAwIBAgIRAISsNxNp8rOJbBB1nGV+D9EwDQYJKoZIhvcNAQEMBQAw
RjELMAkGA1UEBhMCQkUxGTAXBgNVBAoTEEdsb2JhbFNpZ24gbnYtc2ExHDAaBgNV
BAMTE0dsb2JhbFNpZ24gUm9vdCBSNDYwHhcNMjUxMTE5MDMyNzU1WhcNMzAxMTE5
MDAwMDAwWjBpMQswCQYDVQQGEwJCUjExMC8GA1UEChMoUkVERSBOQUNJT05BTCBE
RSBFTlNJTk8gRSBQRVNRVUlTQSAtIFJOUDEnMCUGA1UEAxMeUk5QIElDUEVkdSBH
UjQ2IE9WIFRMUyBDQSAyMDI1MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKC
AgEAqBd4pyjCSQPAFcGq8km8PmRE/BoAJdPvYWKz6gr6x4sXfY40iCWxWLgZkGuN
+OwfXRmCXC1hIb2/WWf8Nl92SSfTb/J0D7KVSksnTrXtxyxXSaEnlKEVKltoTVSC
yEop+pZf6SwuK0l66xvmM9dt/xGgngIvpeaxIIMAAgNDp52PlSrJh5IKzD/FJ4s0
Rq9Aiz3bXLYXCMCa+8WSSVfxJWYgK9IdrTAsu1T24c+xj9TvMcbUgzTDZLhvsRwj
FcY332r9XszFkDqZxCSnjb/ztYa5jFrNGGHlmygMUmBCHGqcsqet/trLKjtaGoqO
JwD0Kk/FAsqF4aGUH7hwIwRuD6ce4BjmPbv870jGdhlRyaMITLbP1YwPDOuoRkrI
utjL30dyHZWiJq1WuYbcgWODH8QwF5BUyxdHpGsv/QL2GXn/Z+GFgdO+4dBnGz7S
AokPYKmMQwKdyn3hzNgShDSFq0sj+vhT3fAXSlth1xjCvdDz+ttccinn71FV5Ahu
S4n9gw8RmSlJoYhr1pzn4Aryi6+31gJVPA8UyTfVh9MbBi7BqPmMnYeZYj7Pr6o0
k8wsz5RA+HL5AbS1QouWcFFe4LEKWqXh67xFONVfi1bO/Y31mDU3fhSzW8tx6bkm
s7N2YunWsPgrOGTcBKbrQTseAShEamJoenImGL4fbP8zZF8CAwEAAaOCAWIwggFe
MA4GA1UdDwEB/wQEAwIBhjATBgNVHSUEDDAKBggrBgEFBQcDATASBgNVHRMBAf8E
CDAGAQH/AgEAMB0GA1UdDgQWBBSUsMlPweBs787GK2yztMsiinZJtzAfBgNVHSME
GDAWgBQDXKtzgYeozLCm1ZTiNpZJ/wWZLDB7BggrBgEFBQcBAQRvMG0wLgYIKwYB
BQUHMAGGImh0dHA6Ly9vY3NwLmdsb2JhbHNpZ24uY29tL3Jvb3RyNDYwOwYIKwYB
BQUHMAKGL2h0dHA6Ly9zZWN1cmUuZ2xvYmFsc2lnbi5jb20vY2FjZXJ0L3Jvb3Ry
NDYuY3J0MDYGA1UdHwQvMC0wK6ApoCeGJWh0dHA6Ly9jcmwuZ2xvYmFsc2lnbi5j
b20vcm9vdHI0Ni5jcmwwLgYDVR0gBCcwJTAIBgZngQwBAgIwDAYKKwYBBAGgMgoB
AjALBgkrBgEEAaAyARQwDQYJKoZIhvcNAQEMBQADggIBABAGOWuTA3iFMqABnBPR
rb1YNwcRqQE1Tvz+9f7a0YUWk+wNH3yPVAqd3i9abWtRT6nNBpu49XNAqWSebXpR
9J5Im5NJruhYxqtd5c6Y04GdoS/JSh6HmJxiBS2Tv5E+Z9yqkBgt41VhxszYEbus
qL/56foMry+EWvXRJ0nA9P0GxdFmbTD4DkLFq++E15FBeRf7uRGUe46bZN345OQ7
jhN2eZwpHzIwTrfaPwHxkFThXU8KPgClmq77JJ/NzlAAPEjpr+TF6K17lzBZS1o1
Lrk+EFDOGVGc0Ncad7JcFldwmLd0C9B73jq0t5opQcl/u1wdIanosZ418vCJp25K
QW1BCDCHenf+6mA2VptyuywHcDqM75rSGi5wxbE4IxkMxrl89HsQKfRSJbDT/0v5
gGlMw/UaCMP6O/4nwHrmXQJ2CVS7M4ucbUbm+ZSqOWgkDCTVvXyyIzWFpWxMk7UH
XBTei1qG19/PbShKTgYcgod0ReTr4osyARZ5T7jZqe8UKW1DUXcVasukATKryagk
NRRjRmM9YasyVMO3SVhTFp49aqAe173TKd2yatDmxlvB9s5fZ5pz5ptdyQFTyv+N
yLDZd5PHiO/JWBFL3g2XVFgKmDVlKsBkqLs/NR18/RWv0d7YKTVVfhc64gdXQAMt
Lyso4S/KfU1hV0inHITEfil4
-----END CERTIFICATE-----
"""


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


def test_a_truncated_re_download_leaves_the_valid_video_it_failed_to_replace(
    mirror: fake.ArchiveMirror, client: ArchiveClient, tmp_path: Path
):
    entry = client.list_videos()[1]
    destination = tmp_path / "videos"
    client.download(entry, destination)
    mirror.truncate[mirror.video_url_path(entry.filename)] = 100

    with pytest.raises(ArchiveError, match="truncated"):
        client.download(entry, destination)

    assert (destination / entry.filename).read_bytes() == PAYLOAD_NEW


class _BodyDiesMidStream:
    """A response whose headers arrive whole and whose body raises on the first read."""

    def __init__(self, response: Any) -> None:
        self._response = response
        self.headers = response.headers

    def read(self, *_size: int) -> bytes:
        raise http.client.IncompleteRead(b"", len(PAYLOAD_NEW))

    def close(self) -> None:
        self._response.close()


def _break_the_first_video_body(client: ArchiveClient, monkeypatch: pytest.MonkeyPatch) -> None:
    opener = client._opener
    original = opener.open
    broken: list[str] = []

    def open_and_break_the_first_video(url: str, timeout: float | None = None) -> Any:
        response = original(url, timeout=timeout)
        if url.endswith(".mp4") and not broken:
            broken.append(url)
            return _BodyDiesMidStream(response)
        return response

    monkeypatch.setattr(opener, "open", open_and_break_the_first_video)


def test_a_body_that_dies_mid_stream_is_retried_rather_than_failing_the_download(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client = ArchiveClient(
        mirror.base_url, allow_plaintext=True, retries=2, backoff=0.0, delay=0.0, timeout=10.0
    )
    entry = client.list_videos()[1]
    _break_the_first_video_body(client, monkeypatch)

    result = client.download(entry, tmp_path / "videos")

    assert result.path.read_bytes() == PAYLOAD_NEW
    assert result.sha256 == hashlib.sha256(PAYLOAD_NEW).hexdigest()


class _BodyEndsEarly:
    """A response whose headers announce the whole video and whose body stops short.

    ``http.client.HTTPResponse.read(amt)`` deliberately does not raise on a body
    that ends before its ``Content-Length``: it closes the connection and returns
    ``b""``.  A proxy or idle timeout closing cleanly mid-video therefore reaches
    the client as a silent short read, which is what this reproduces.
    """

    def __init__(self, response: Any, keep: int) -> None:
        self._response = response
        self.headers = response.headers
        self._remaining = keep

    def read(self, *size: int) -> bytes:
        if self._remaining <= 0:
            return b""
        chunk = bytes(
            self._response.read(min(size[0], self._remaining) if size else self._remaining)
        )
        self._remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self._response.close()


def test_a_body_that_ends_early_without_raising_is_retried_rather_than_failing_the_download(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client = ArchiveClient(
        mirror.base_url, allow_plaintext=True, retries=2, backoff=0.0, delay=0.0, timeout=10.0
    )
    entry = client.list_videos()[1]
    opener = client._opener
    original = opener.open
    shortened: list[str] = []

    def open_and_shorten_the_first_video(url: str, timeout: float | None = None) -> Any:
        response = original(url, timeout=timeout)
        if url.endswith(".mp4") and not shortened:
            shortened.append(url)
            return _BodyEndsEarly(response, 512)
        return response

    monkeypatch.setattr(opener, "open", open_and_shorten_the_first_video)

    result = client.download(entry, tmp_path / "videos")

    assert result.path.read_bytes() == PAYLOAD_NEW
    assert result.sha256 == hashlib.sha256(PAYLOAD_NEW).hexdigest()


def test_a_destination_that_cannot_be_written_fails_at_once_instead_of_refetching(
    mirror: fake.ArchiveMirror, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    client = ArchiveClient(
        mirror.base_url, allow_plaintext=True, retries=3, backoff=0.0, delay=0.0, timeout=10.0
    )
    entry = client.list_videos()[1]

    def _refuse_to_write(*_args: Any, **_kwargs: Any) -> Path:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("allsky.archive.atomic_write", _refuse_to_write)
    mirror.requests.clear()

    with pytest.raises(OSError, match="No space left on device"):
        client.download(entry, tmp_path / "videos")

    assert [line for line in mirror.requests if ".mp4" in line] != []
    assert len([line for line in mirror.requests if ".mp4" in line]) == 1


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
    assert reloaded.frames_match(DAY_NEW, step=3, resize=224, timestamps="overlay") is True
    assert reloaded.uploaded(DAY_NEW, "gd:LabMiM/allsky/videos/x.mp4") is True
    assert stored["last_modified"] == fake.LAST_MODIFIED


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


@pytest.mark.parametrize(("step", "resize"), [(2, 224), (3, None), (3, 448)])
def test_frames_match_is_false_whenever_step_or_resize_differ_from_the_record(
    tmp_path: Path, step: int, resize: int | None
):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_frames(
        DAY_NEW, directory="frames/20260810", count=4, step=3, resize=224, timestamps="overlay"
    )
    assert ledger.frames_match(DAY_NEW, step=3, resize=224, timestamps="overlay") is True
    assert ledger.frames_match(DAY_NEW, step=step, resize=resize, timestamps="overlay") is False


def test_frames_match_is_false_when_another_clock_stamped_the_record(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.record_frames(
        DAY_NEW, directory="frames/20260810", count=4, step=3, resize=224, timestamps="overlay"
    )

    matched = ledger.frames_match(DAY_NEW, step=3, resize=224, timestamps="config")

    assert matched is False


def test_frames_match_is_false_for_a_day_that_was_never_extracted(tmp_path: Path):
    ledger = Ledger(tmp_path / "ledger.json")
    assert ledger.frames_match(DAY_NEW, step=1, resize=None, timestamps="overlay") is False


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


def test_the_insecure_context_turns_verification_off(tmp_path: Path):
    context = build_ssl_context(state_dir=tmp_path, insecure=True)
    assert context.verify_mode == ssl.CERT_NONE
    assert context.check_hostname is False


def test_a_cached_intermediate_keeps_the_context_build_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(archive, "AIA_INTERMEDIATE_URL", "http://127.0.0.1:1/unreachable.crt")
    cache = tmp_path / archive.INTERMEDIATE_CACHE_FILENAME
    cache.write_text(AIA_INTERMEDIATE_PEM, encoding="ascii")

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


def test_an_intermediate_served_over_plaintext_is_refused_when_its_digest_is_not_the_pinned_one(
    tmp_path: Path, mirror: fake.ArchiveMirror, monkeypatch: pytest.MonkeyPatch
):
    (mirror.root / "forged.crt").write_text(_system_pem(), encoding="ascii")
    monkeypatch.setattr(archive, "AIA_INTERMEDIATE_URL", f"{mirror.base_url}forged.crt")
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    with pytest.raises(ArchiveError, match="pinned"):
        build_ssl_context(state_dir=state_dir, timeout=10.0)

    assert not (state_dir / archive.INTERMEDIATE_CACHE_FILENAME).exists()


def test_a_cached_intermediate_is_refused_when_its_digest_is_not_the_pinned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(archive, "AIA_INTERMEDIATE_URL", "http://127.0.0.1:1/unreachable.crt")
    cache = tmp_path / archive.INTERMEDIATE_CACHE_FILENAME
    cache.write_text(_system_pem(), encoding="ascii")

    with pytest.raises(ArchiveError, match="pinned"):
        build_ssl_context(state_dir=tmp_path, timeout=1.0)


def test_the_published_base_url_still_points_at_the_planetarium_over_https():
    assert ARCHIVE_BASE_URL == "https://allsky.planetario.ufba.br/"
