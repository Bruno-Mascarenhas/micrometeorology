"""All-sky video frame access and extraction.

Videos are one-day timelapse files named by date (``allsky-YYYYMMDD.mp4``):
frame 0 is captured at :attr:`~allsky.config.VideoConfig.start_time` local
time and each subsequent frame advances
:attr:`~allsky.config.VideoConfig.minutes_per_frame` minutes of real time.

All timestamps produced here are **naive local times**, matching the sensor
logger convention (no timezone conversion in v0).  Videos are always decoded
as a stream (:func:`imageio.v3.imiter`) — a full one-day 1080p video is never
loaded into memory at once.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd

from allsky.config import AllSkyConfig, VideoConfig

logger = logging.getLogger(__name__)

#: Filename of the parquet manifest written next to extracted JPEG frames.
MANIFEST_FILENAME = "manifest.parquet"

#: JPEG quality used by :func:`extract_frames`.
JPEG_QUALITY = 92

#: Column order of the frame manifest returned by :func:`extract_frames`.
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


def _video_cfg(cfg: VideoConfig | AllSkyConfig) -> VideoConfig:
    """Accept either the root config or its ``video`` section."""
    return cfg.video if isinstance(cfg, AllSkyConfig) else cfg


def _start_timestamp(date: dt.date, cfg: VideoConfig) -> pd.Timestamp:
    """Naive local timestamp of frame 0 for a video recorded on *date*."""
    return pd.Timestamp(f"{date.isoformat()} {cfg.start_time}")


def video_date(path: str | Path, cfg: VideoConfig | AllSkyConfig) -> dt.date:
    """Parse the recording date from an all-sky video filename.

    Parameters
    ----------
    path:
        Video file path; only the stem is used (e.g. ``allsky-20260625``).
    cfg:
        Video config (or root config) providing ``filename_date_format``.

    Raises
    ------
    ValueError
        If the filename stem does not match ``filename_date_format``.
    """
    vcfg = _video_cfg(cfg)
    stem = Path(path).stem
    try:
        return datetime.strptime(stem, vcfg.filename_date_format).date()
    except ValueError as exc:
        raise ValueError(
            f"Video filename {stem!r} does not match the configured "
            f"date format {vcfg.filename_date_format!r}"
        ) from exc


def frame_timestamps(n: int, date: dt.date, cfg: VideoConfig | AllSkyConfig) -> pd.DatetimeIndex:
    """Wall-clock timestamps of the first *n* frames of a video for *date*.

    Formula
    -------
    ``timestamp(i) = date + start_time + i * minutes_per_frame`` (naive local
    time), for ``i = 0 .. n-1``.

    Limitation
    ----------
    The mapping assumes the camera never skips frames; dropped frames in the
    timelapse would shift every subsequent timestamp.
    """
    vcfg = _video_cfg(cfg)
    start = _start_timestamp(date, vcfg)
    offsets = pd.to_timedelta(np.arange(n) * vcfg.minutes_per_frame, unit="min")
    return pd.DatetimeIndex(start + offsets, name="timestamp")


def iter_frames(
    path: str | Path, cfg: VideoConfig | AllSkyConfig, *, step: int = 1
) -> Iterator[FrameRecord]:
    """Stream every *step*-th frame of an all-sky video.

    Frames are decoded one at a time via :func:`imageio.v3.imiter`; the video
    is never fully loaded into memory.  Skipped frames are still decoded (the
    codec requires it) but immediately discarded.

    Parameters
    ----------
    path:
        Video file whose name encodes the recording date.
    cfg:
        Video config (or root config) providing the time mapping.
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
    vcfg = _video_cfg(cfg)
    date = video_date(path, vcfg)
    start = _start_timestamp(date, vcfg)
    for index, image in enumerate(iio.imiter(path)):
        if index % step:
            continue
        timestamp = start + pd.Timedelta(minutes=index * vcfg.minutes_per_frame)
        yield FrameRecord(
            index=index,
            timestamp=timestamp,
            image=np.asarray(image, dtype=np.uint8),
        )


def _resize_image(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Bilinear-resize an RGB image to *size* (``int`` means square)."""
    from PIL import Image

    if isinstance(size, int):
        size = (size, size)
    resized = Image.fromarray(image).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized)


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    cfg: VideoConfig | AllSkyConfig,
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
        Video config (or root config) providing the time mapping.
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
        image = record.image if resize is None else _resize_image(record.image, resize)
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
