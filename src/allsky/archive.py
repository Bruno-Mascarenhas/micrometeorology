"""Mirrors the Salvador all-sky camera archive at ``allsky.planetario.ufba.br``.

Protocol notes, the broken TLS chain and the ledger's dedup contract are
documented in ``docs/allsky-archive.md``.
"""

import datetime as dt
import fcntl
import hashlib
import http.client
import json
import logging
import re
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urljoin, urlsplit

from allsky.atomic import atomic_write, atomic_write_json

logger = logging.getLogger(__name__)

__all__ = [
    "AIA_INTERMEDIATE_URL",
    "ARCHIVE_BASE_URL",
    "LEDGER_FILENAME",
    "USER_AGENT",
    "ArchiveClient",
    "ArchiveEntry",
    "ArchiveError",
    "DownloadResult",
    "Ledger",
    "build_ssl_context",
    "ledger_lock",
    "parse_catalog",
]

ARCHIVE_BASE_URL = "https://allsky.planetario.ufba.br/"
VIDEO_PATH = "videos/"
LIVE_IMAGE_PATH = "image.jpg"
AIA_INTERMEDIATE_URL = "http://secure.globalsign.com/cacert/rnpicpedugr46ovtlsca2025.crt"
INTERMEDIATE_CACHE_FILENAME = "rnp-icpedu-gr46-2025.pem"
USER_AGENT = "labmim-allsky-archive/1.0 (LabMiM/UFBA; +https://labmim.if.ufba.br)"
LEDGER_FILENAME = "archive-ledger.json"
LEDGER_VERSION = 1

_VIDEO_FILENAME_RE = re.compile(r"allsky-(\d{8})\.mp4")
_DOWNLOAD_CHUNK_BYTES = 1 << 20


class ArchiveError(RuntimeError):
    """A request against the all-sky archive failed after every retry."""


def _https_only_opener(
    context: ssl.SSLContext | None = None, *, allow_plaintext: bool = False
) -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPSHandler(context=context))
    if allow_plaintext:
        opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    opener.add_handler(urllib.request.HTTPRedirectHandler())
    return opener


def _fetch_aia_intermediate(url: str, *, timeout: float) -> bytes:
    opener = _https_only_opener(allow_plaintext=True)
    opener.addheaders = [("User-Agent", USER_AGENT)]
    try:
        with opener.open(url, timeout=timeout) as response:
            payload = bytes(response.read())
    except (urllib.error.URLError, OSError) as exc:
        raise ArchiveError(
            f"could not fetch the CA intermediate from {url}: {exc}. Without it the archive's "
            "TLS chain cannot be verified; pass --ca-file with a PEM copy, or --insecure."
        ) from exc
    if b"BEGIN CERTIFICATE" in payload:
        return payload
    return ssl.DER_cert_to_PEM_cert(payload).encode("ascii")


def build_ssl_context(
    *,
    state_dir: str | Path | None = None,
    ca_file: str | Path | None = None,
    insecure: bool = False,
    timeout: float = 30.0,
) -> ssl.SSLContext:
    """Build a verifying context, repairing the chain the archive host serves."""
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return context
    if ca_file is not None:
        context.load_verify_locations(cafile=str(ca_file))
        return context

    cache = Path(state_dir) / INTERMEDIATE_CACHE_FILENAME if state_dir is not None else None
    if cache is not None and cache.is_file():
        pem = cache.read_bytes()
    else:
        pem = _fetch_aia_intermediate(AIA_INTERMEDIATE_URL, timeout=timeout)
        if cache is not None:
            atomic_write(cache, lambda tmp: tmp.write_bytes(pem))
            logger.info("cached the archive CA intermediate at %s", cache)
    context.load_verify_locations(cadata=pem.decode("ascii"))
    return context


@dataclass(frozen=True)
class ArchiveEntry:
    date: dt.date
    filename: str
    url: str

    @property
    def key(self) -> str:
        return self.date.strftime("%Y%m%d")


@dataclass(frozen=True)
class DownloadResult:
    entry: ArchiveEntry
    path: Path
    size: int
    sha256: str
    last_modified: str | None
    downloaded: bool


def parse_catalog(html: str, base_url: str = ARCHIVE_BASE_URL) -> tuple[ArchiveEntry, ...]:
    """Extract the archived timelapse days from the ``videos/`` listing, oldest first."""
    videos_url = urljoin(base_url, VIDEO_PATH)
    entries: dict[str, ArchiveEntry] = {}
    for match in _VIDEO_FILENAME_RE.finditer(html):
        stamp = match.group(1)
        try:
            date = dt.date.fromisoformat(stamp)
        except ValueError:
            logger.warning("skipping archive filename with an impossible date: %s", match.group(0))
            continue
        entries[stamp] = ArchiveEntry(
            date=date, filename=match.group(0), url=urljoin(videos_url, match.group(0))
        )
    return tuple(entries[key] for key in sorted(entries))


class ArchiveClient:
    """Fetches the archive over HTTPS only, with retries and a polite delay."""

    def __init__(
        self,
        base_url: str = ARCHIVE_BASE_URL,
        *,
        context: ssl.SSLContext | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        backoff: float = 2.0,
        delay: float = 1.0,
        allow_plaintext: bool = False,
    ) -> None:
        self.base_url = base_url if base_url.endswith("/") else f"{base_url}/"
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff = backoff
        self.delay = delay
        self._opener = _https_only_opener(context, allow_plaintext=allow_plaintext)

    @contextmanager
    def _open(self, url: str, headers: dict[str, str] | None = None) -> Generator[Any]:
        self._opener.addheaders = [("User-Agent", USER_AGENT), *(headers or {}).items()]
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self._opener.open(url, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code == http.client.NOT_MODIFIED:
                    yield exc
                    return
                last = exc
                if exc.code < 500 and exc.code != http.client.TOO_MANY_REQUESTS:
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
            else:
                try:
                    yield response
                finally:
                    response.close()
                return
            if attempt < self.retries:
                pause = self.backoff * attempt
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                    url,
                    attempt,
                    self.retries,
                    last,
                    pause,
                )
                time.sleep(pause)
        raise ArchiveError(f"GET {url} failed after {self.retries} attempt(s): {last}")

    def fetch_text(self, path: str) -> str:
        with self._open(urljoin(self.base_url, path)) as response:
            payload = bytes(response.read())
            charset = response.headers.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="replace")

    def fetch_binary(self, path: str) -> tuple[bytes, dict[str, str]]:
        with self._open(urljoin(self.base_url, path)) as response:
            payload = bytes(response.read())
            headers = {key.lower(): value for key, value in response.headers.items()}
        return payload, headers

    def list_videos(self) -> tuple[ArchiveEntry, ...]:
        entries = parse_catalog(self.fetch_text(VIDEO_PATH), self.base_url)
        logger.info(
            "archive lists %d day(s)%s",
            len(entries),
            f" ({entries[0].date} .. {entries[-1].date})" if entries else "",
        )
        return entries

    def download(
        self, entry: ArchiveEntry, dest_dir: str | Path, *, if_modified_since: str | None = None
    ) -> DownloadResult:
        """Stream one day's timelapse into *dest_dir*, atomically, verifying its length."""
        destination = Path(dest_dir) / entry.filename
        headers = {"If-Modified-Since": if_modified_since} if if_modified_since else None
        digest = hashlib.sha256()
        written = 0

        with self._open(entry.url, headers) as response:
            if getattr(response, "code", None) == http.client.NOT_MODIFIED:
                logger.info("%s unchanged on the server (304)", entry.filename)
                return DownloadResult(
                    entry=entry,
                    path=destination,
                    size=destination.stat().st_size if destination.is_file() else 0,
                    sha256="",
                    last_modified=if_modified_since,
                    downloaded=False,
                )
            raw_length = response.headers.get("Content-Length")
            declared = int(raw_length) if raw_length and raw_length.isdigit() else None
            last_modified = response.headers.get("Last-Modified")

            def _stream(tmp: Path) -> None:
                nonlocal written
                with open(tmp, "wb") as handle:
                    while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)

            atomic_write(destination, _stream)

        if declared is not None and written != declared:
            destination.unlink(missing_ok=True)
            raise ArchiveError(
                f"{entry.filename} truncated: got {written} bytes, server announced {declared}"
            )
        logger.info("downloaded %s (%.1f MiB)", entry.filename, written / (1 << 20))
        if self.delay:
            time.sleep(self.delay)
        return DownloadResult(
            entry=entry,
            path=destination,
            size=written,
            sha256=digest.hexdigest(),
            last_modified=last_modified,
            downloaded=True,
        )

    def fetch_live_image(self) -> tuple[bytes, dict[str, str]]:
        return self.fetch_binary(LIVE_IMAGE_PATH)


def _utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")


class Ledger:
    """Append-only record of the days downloaded, extracted and uploaded."""

    def __init__(self, path: str | Path, payload: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._payload: dict[str, Any] = payload or {
            "version": LEDGER_VERSION,
            "source": ARCHIVE_BASE_URL,
            "entries": {},
        }

    @classmethod
    def load(cls, path: str | Path) -> Self:
        ledger_path = Path(path)
        if not ledger_path.is_file():
            return cls(ledger_path)
        with open(ledger_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "entries" not in payload:
            raise ArchiveError(f"{ledger_path} is not an all-sky archive ledger")
        if payload.get("version") != LEDGER_VERSION:
            raise ArchiveError(
                f"{ledger_path} has ledger version {payload.get('version')!r}, this build writes "
                f"{LEDGER_VERSION}. Move it aside to rebuild from scratch."
            )
        return cls(ledger_path, payload)

    def save(self) -> Path:
        self._payload["updated_at"] = _utc_now()
        return atomic_write_json(self.path, self._payload)

    @property
    def entries(self) -> dict[str, Any]:
        record: dict[str, Any] = self._payload["entries"]
        return record

    def entry(self, key: str) -> dict[str, Any]:
        record = self.entries.setdefault(key, {})
        if not isinstance(record, dict):
            raise ArchiveError(f"ledger entry {key!r} is corrupt: expected an object")
        return record

    def video(self, key: str) -> dict[str, Any] | None:
        record = self.entries.get(key)
        if not isinstance(record, dict):
            return None
        stored: dict[str, Any] | None = record.get("video")
        return stored

    def frames(self, key: str) -> dict[str, Any] | None:
        record = self.entries.get(key)
        if not isinstance(record, dict):
            return None
        stored: dict[str, Any] | None = record.get("frames")
        return stored

    def video_path(self, key: str, *, root: str | Path | None = None) -> Path | None:
        stored = self.video(key)
        if stored is None or not stored.get("path"):
            return None
        return _resolved(Path(stored["path"]), root)

    def frames_dir(self, key: str, *, root: str | Path | None = None) -> Path | None:
        stored = self.frames(key)
        if stored is None or not stored.get("dir"):
            return None
        return _resolved(Path(stored["dir"]), root)

    def has_video(self, key: str, *, local_root: str | Path | None = None) -> bool:
        """True when *key* was downloaded and, when *local_root* is given, is still on disk."""
        stored = self.video(key)
        if stored is None:
            return False
        if local_root is None:
            return True
        if stored.get("pruned"):
            return False
        path = _resolved(Path(stored.get("path", "")), local_root)
        return path.is_file() and path.stat().st_size == stored.get("size", -1)

    def frames_match(self, key: str, *, step: int, resize: int | None) -> bool:
        """True when *key* was already extracted with these very parameters."""
        stored = self.frames(key)
        if stored is None:
            return False
        return stored.get("step") == step and stored.get("resize") == resize

    def last_modified(self, key: str) -> str | None:
        stored = self.video(key)
        return None if stored is None else stored.get("last_modified")

    def uploaded(self, key: str, destination: str) -> bool:
        record = self.entries.get(key)
        if not isinstance(record, dict) or not destination:
            return False
        uploads: dict[str, Any] = record.get("uploads", {})
        return destination in uploads

    def record_video(self, result: DownloadResult, *, root: str | Path | None = None) -> None:
        path = result.path
        if root is not None:
            try:
                path = path.relative_to(Path(root))
            except ValueError:
                path = result.path
        self.entry(result.entry.key)["video"] = {
            "filename": result.entry.filename,
            "path": path.as_posix(),
            "size": result.size,
            "sha256": result.sha256,
            "last_modified": result.last_modified,
            "url": result.entry.url,
            "downloaded_at": _utc_now(),
        }

    def record_frames(
        self,
        key: str,
        *,
        directory: str | Path,
        count: int,
        step: int,
        resize: int | None,
        timestamps: str,
    ) -> None:
        self.entry(key)["frames"] = {
            "dir": Path(directory).as_posix(),
            "count": count,
            "step": step,
            "resize": resize,
            "timestamps": timestamps,
            "extracted_at": _utc_now(),
        }

    def record_upload(self, key: str, destination: str, **details: Any) -> None:
        uploads: dict[str, Any] = self.entry(key).setdefault("uploads", {})
        uploads[destination] = {"uploaded_at": _utc_now(), **details}

    def mark_pruned(self, key: str) -> None:
        stored = self.video(key)
        if stored is not None:
            stored["pruned"] = True
            stored["pruned_at"] = _utc_now()


def _resolved(path: Path, root: str | Path | None) -> Path:
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


@contextmanager
def ledger_lock(path: str | Path) -> Generator[None]:
    """Hold an exclusive advisory lock so a cron run and a manual run cannot interleave."""
    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ArchiveError(
                f"another all-sky sync is already running (lock held on {lock_path})"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def archive_host(base_url: str = ARCHIVE_BASE_URL) -> str:
    return urlsplit(base_url).hostname or base_url
