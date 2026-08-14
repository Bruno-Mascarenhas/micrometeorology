"""Offline stand-ins for the all-sky archive: a local mirror and stamped frames.

The mirror is a ``ThreadingHTTPServer`` on an ephemeral 127.0.0.1 port serving a
real directory tree, so ``ArchiveClient`` exercises its own HTTP path with no
network. The frame renderer paints the timestamp overlay with the glyph bank
:mod:`allsky.overlay` matches against, which is what lets the reader be tested
without a byte of camera footage.
"""

import datetime as dt
import functools
import hashlib
import http.server
import logging
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from allsky import overlay
from allsky.archive import ARCHIVE_BASE_URL, ArchiveEntry, DownloadResult, Ledger

logger = logging.getLogger(__name__)

OVERLAY_BACKGROUND = (40, 90, 20)
OVERLAY_TEXT = (0, 0, 255)
FRAME_HEIGHT = 1080
FRAME_WIDTH = 1920
VIDEO_FPS = 5

INDEX_FILENAME = "index.html"
VIDEOS_DIRNAME = "videos"
LIVE_IMAGE_NAME = "image.jpg"


def render_overlay_frame(
    stamp: str,
    *,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
    exemplar: int = 0,
    background: tuple[int, int, int] = OVERLAY_BACKGROUND,
) -> np.ndarray:
    """Paint ``YYYYMMDDHHMMSS`` into a frame using the reader's own glyph exemplars."""
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :] = background
    cells = zip(stamp, overlay.DIGIT_CELL_LEFT_EDGES, strict=True)
    for digit, left in cells:
        exemplars = overlay._BANK[digit]
        glyph = exemplars[exemplar % len(exemplars)]
        cell = frame[overlay.OVERLAY_ROW_SLICE, left : left + overlay.DIGIT_CELL_WIDTH]
        cell[glyph] = OVERLAY_TEXT
    return frame


def write_overlay_video(
    path: str | Path,
    stamps: Sequence[str],
    *,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> Path:
    """Encode one stamped frame per entry of *stamps*, losslessly so the glyphs survive."""
    frames = np.stack([render_overlay_frame(s, height=height, width=width) for s in stamps])
    out = Path(path)
    iio.imwrite(
        out,
        frames,
        fps=VIDEO_FPS,
        codec="libx264rgb",
        pixelformat="rgb24",
        macro_block_size=1,
        output_params=["-crf", "0"],
    )
    return out


def build_index_html(filenames: Sequence[str]) -> str:
    """Render the ``videos/`` listing, linking every day twice as the real page does."""
    cards = "\n".join(
        f"    <div class='card'>"
        f"<a href='./{name}'><video class='thumb' preload='metadata' src='./{name}'></video></a>"
        f"<p class='day'>{name[7:11]}-{name[11:13]}-{name[13:15]}</p></div>"
        for name in filenames
    )
    return (
        "<!doctype html>\n<html>\n<head><title>All-Sky Timelapses</title></head>\n"
        f"<body>\n  <h1>Timelapses</h1>\n  <div class='grid'>\n{cards}\n  </div>\n</body>\n</html>\n"
    )


def _handler_factory(
    directory: Path, requests: list[str], truncate: dict[str, int]
) -> Callable[..., http.server.SimpleHTTPRequestHandler]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            requests.append(f"{self.path} {code} {size}")

        def copyfile(self, source, outputfile):
            keep = truncate.get(self.path)
            if keep is None:
                super().copyfile(source, outputfile)
                return
            outputfile.write(source.read()[:keep])

    return functools.partial(Handler, directory=str(directory))


class ArchiveMirror:
    """A served copy of the archive tree, plus the request log and truncation hooks."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.videos_dir = self.root / VIDEOS_DIRNAME
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.requests: list[str] = []
        self.truncate: dict[str, int] = {}
        self._listed: list[str] = []
        self._write_index()
        self._server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), _handler_factory(self.root, self.requests, self.truncate)
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        host, port = self._server.socket.getsockname()[:2]
        return f"http://{host}:{port}/"

    def video_url_path(self, filename: str) -> str:
        return f"/{VIDEOS_DIRNAME}/{filename}"

    def publish_video(self, filename: str, payload: bytes) -> Path:
        path = self.videos_dir / filename
        path.write_bytes(payload)
        self.advertise(filename)
        return path

    def publish_video_file(self, source: str | Path) -> Path:
        origin = Path(source)
        return self.publish_video(origin.name, origin.read_bytes())

    def publish_image(self, payload: bytes) -> Path:
        path = self.root / LIVE_IMAGE_NAME
        path.write_bytes(payload)
        return path

    def advertise(self, filename: str) -> None:
        """List *filename* on the index page whether or not a file backs it."""
        if filename not in self._listed:
            self._listed.append(filename)
            self._write_index()

    def unpublish_video(self, filename: str) -> None:
        (self.videos_dir / filename).unlink(missing_ok=True)
        if filename in self._listed:
            self._listed.remove(filename)
            self._write_index()

    def video_requests(self) -> list[str]:
        return [line for line in self.requests if line.split(" ", 1)[0].endswith(".mp4")]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def _write_index(self) -> None:
        page = build_index_html(sorted(self._listed))
        (self.videos_dir / INDEX_FILENAME).write_text(page, encoding="utf-8")


LAST_MODIFIED = "Sun, 09 Aug 2026 03:04:05 GMT"


def archive_entry(key: str, base_url: str = ARCHIVE_BASE_URL) -> ArchiveEntry:
    filename = f"allsky-{key}.mp4"
    return ArchiveEntry(
        date=dt.date.fromisoformat(key), filename=filename, url=f"{base_url}videos/{filename}"
    )


def record_downloaded_day(
    ledger: Ledger, root: str | Path, key: str, *, payload: bytes = b"timelapse-bytes"
) -> DownloadResult:
    """Put a day's mp4 on disk under *root* and record it in *ledger* as the CLI would."""
    entry = archive_entry(key)
    path = Path(root) / VIDEOS_DIRNAME / entry.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    result = DownloadResult(
        entry=entry,
        path=path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        last_modified=LAST_MODIFIED,
        downloaded=True,
    )
    ledger.record_video(result, root=Path(root))
    return result


RCLONE_REFUSAL = "fake rclone refused"


def write_fake_rclone(
    directory: str | Path,
    log_path: str | Path,
    *,
    name: str = "rclone",
    exit_code: int = 0,
    fail_on: str | None = None,
) -> Path:
    """Write an executable stand-in for ``rclone`` that logs its argv before exiting.

    Without *fail_on* every invocation exits *exit_code*; with it, only that
    subcommand fails, which is how a run that reaches the remote but cannot copy
    to it is reproduced.
    """
    if fail_on is None:
        verdict = f'test {exit_code} -eq 0 || echo "{RCLONE_REFUSAL}" >&2\nexit {exit_code}\n'
    else:
        verdict = (
            f'if [ "$1" = "{fail_on}" ]; then\n'
            f'  echo "{RCLONE_REFUSAL}" >&2\n'
            f"  exit {exit_code or 3}\n"
            "fi\nexit 0\n"
        )
    script = Path(directory) / name
    script.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$*" >> {Path(log_path)}\n{verdict}', encoding="utf-8"
    )
    script.chmod(0o700)
    return script


def rclone_invocations(log_path: str | Path) -> list[str]:
    path = Path(log_path)
    if not path.is_file():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
