"""All-sky video frame access and extraction.

Videos are one-day timelapse files named by date (``allsky-YYYYMMDD.mp4``):
frame 0 is captured at :attr:`~allsky.config.VideoConfig.start_time` local
time and each subsequent frame advances
:attr:`~allsky.config.VideoConfig.minutes_per_frame` minutes of real time.

All timestamps produced here are **naive local times**, matching the sensor
logger convention (no timezone conversion).  Videos are always decoded
as a stream (:func:`imageio.v3.imiter`) — a full one-day 1080p video is never
loaded into memory at once.
"""

import datetime as dt
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from allsky.config import VideoConfig
from allsky.frame_pixels import resize_bilinear
from micrometeorology.common.timeparse import parse_naive_timestamp

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.parquet"

JPEG_QUALITY = 92

MANIFEST_COLUMNS = ("frame_path", "timestamp", "video", "index")


@dataclass(frozen=True)
class FrameRecord:
    """A single decoded video frame with its wall-clock timestamp.

    Attributes
    ----------
    index:
        Zero-based frame position in the source video.
    timestamp:
        Naive local wall-clock time of the frame
        (``start_time + index * minutes_per_frame``).
    image:
        ``uint8`` RGB array of shape ``(height, width, 3)``.
    """

    index: int
    timestamp: pd.Timestamp
    image: np.ndarray


def _start_timestamp(date: dt.date, cfg: VideoConfig) -> pd.Timestamp:
    """Naive local timestamp of frame 0 for a video recorded on *date*."""
    return pd.Timestamp(f"{date.isoformat()} {cfg.start_time}")


def video_date(path: str | Path, cfg: VideoConfig) -> dt.date:
    """Parse the recording date from an all-sky video filename.

    Parameters
    ----------
    path:
        Video file path; only the stem is used (e.g. ``allsky-20260625``).
    cfg:
        Video config providing ``filename_date_format``.

    Returns
    -------
    datetime.date
        Local calendar date the video covers.

    Raises
    ------
    ValueError
        If the filename stem does not match ``filename_date_format``.
    """
    stem = Path(path).stem
    try:
        # This pipeline is naive local by construction, hence
        # :func:`parse_naive_timestamp` rather than pandas: it applies the strftime
        # format with ``strptime``'s match-or-raise semantics while staying naive
        # (see its module docstring for the pandas escape hatches it closes).
        return parse_naive_timestamp(stem, cfg.filename_date_format).date()
    except ValueError as exc:
        raise ValueError(
            f"Video filename {stem!r} does not match the configured "
            f"date format {cfg.filename_date_format!r}"
        ) from exc


def iter_frames(path: str | Path, cfg: VideoConfig, *, step: int = 1) -> Iterator[FrameRecord]:
    """Stream every *step*-th frame of an all-sky video.

    Frames are decoded one at a time via :func:`imageio.v3.imiter`; the video
    is never fully loaded into memory.  Skipped frames are still decoded (the
    codec requires it) but immediately discarded.

    Parameters
    ----------
    path:
        Video file whose name encodes the recording date.
    cfg:
        Video config providing the time mapping.
    step:
        Yield one frame every *step* source frames (frame indices
        ``0, step, 2*step, ...``).

    Yields
    ------
    FrameRecord
        Frame index, naive local timestamp and ``uint8`` RGB image.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    date = video_date(path, cfg)
    start = _start_timestamp(date, cfg)
    for index, image in enumerate(iio.imiter(path)):
        if index % step:
            continue
        timestamp = start + pd.Timedelta(minutes=index * cfg.minutes_per_frame)
        yield FrameRecord(
            index=index,
            timestamp=timestamp,
            image=np.asarray(image, dtype=np.uint8),
        )


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    cfg: VideoConfig,
    step: int = 1,
    resize: int | tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Extract every *step*-th frame to JPEG and write a frame manifest.

    JPEGs named ``allsky-YYYYMMDD-HHMM.jpg`` (quality ``92``) are written to
    *out_dir* together with a ``manifest.parquet`` describing them.  The
    manifest (also returned) has columns ``frame_path`` (path as written,
    resolving relative to the caller's working directory), ``timestamp``
    (naive local), ``video`` (source filename) and ``index`` (frame position).

    Parameters
    ----------
    path:
        Source video whose filename encodes the recording date.
    out_dir:
        Output directory, created if missing.  The manifest is overwritten on
        every call — use one directory per video (or per extraction run).
    cfg:
        Video config providing the time mapping.
    step:
        Keep one frame every *step* source frames.
    resize:
        Optional output size — ``int`` for square, ``(width, height)`` tuple
        otherwise.  ``None`` keeps the native resolution.

    Limitation
    ----------
    Filenames carry minute resolution: with ``minutes_per_frame < 1`` two
    frames can map to the same name and overwrite each other.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_name = Path(path).name

    rows: list[dict[str, object]] = []
    for record in iter_frames(path, cfg, step=step):
        image = record.image if resize is None else resize_bilinear(record.image, resize)
        frame_file = out_path / f"allsky-{record.timestamp:%Y%m%d-%H%M}.jpg"
        iio.imwrite(frame_file, image, quality=JPEG_QUALITY)
        rows.append(
            {
                "frame_path": str(frame_file),
                "timestamp": record.timestamp,
                "video": video_name,
                "index": record.index,
            }
        )

    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"]).astype("datetime64[ns]")
    manifest["index"] = manifest["index"].astype("int64")
    manifest_file = out_path / MANIFEST_FILENAME
    manifest.to_parquet(manifest_file, index=False)
    logger.info("Extracted %d frames from %s to %s", len(manifest), video_name, out_path)
    return manifest
