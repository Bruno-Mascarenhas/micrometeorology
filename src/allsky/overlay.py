"""Reads the capture timestamp the all-sky camera burns into every video frame.

The stamp is a fourteen-digit ``YYYYMMDDHHMMSS`` run of blue-on-dark glyphs in a
fixed band at the top of the frame. Each digit cell is matched against a packed
bank of exemplars over a small horizontal shift window, and the cells the pixels
leave ambiguous are re-decided against the neighbouring frames, whose capture
times can only increase.

Every timestamp this module produces is **naive local time**: it is the camera's
own clock, read back off the image, and re-stamping it as UTC would invent a
precision the acquisition does not have. The UTC boundary is the manifest layer,
which converts using :data:`allsky.config.SITE_UTC_OFFSET_HOURS`.

Frames are ``(H, W, 3)`` ``uint8`` RGB arrays in the camera's native
resolution — the overlay is read before any crop or resize, since the digit
cell geometry is pinned to raw pixel columns.
"""

import base64
import datetime as dt
import itertools
import logging
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from allsky.config import VideoConfig
from allsky.data.contracts import QCFlag
from allsky.video import JPEG_QUALITY, MANIFEST_FILENAME
from allsky.video import MANIFEST_COLUMNS as VIDEO_MANIFEST_COLUMNS

logger = logging.getLogger(__name__)

__all__ = [
    "OverlayReading",
    "OverlayTimestampError",
    "extract_frames_for",
    "extract_frames_with_overlay_timestamps",
    "read_frame_timestamp",
    "read_video_timestamps",
]

OVERLAY_ROW_SLICE = slice(6, 26)
DIGIT_CELL_LEFT_EDGES = (16, 32, 48, 64, 80, 96, 112, 128, 156, 172, 196, 212, 236, 252)
DIGIT_CELL_WIDTH = 14
DIGIT_CELL_SEARCH_PAD = 4
DIGIT_MATCH_SHIFTS = range(-DIGIT_CELL_SEARCH_PAD + 1, DIGIT_CELL_SEARCH_PAD)
TEXT_BLUE_FLOOR = 180
TEXT_RED_CEILING = 60
TEXT_GREEN_CEILING = 60
OVERLAY_STAMP_DIGITS = 14

MIN_CAPTURE_INTERVAL = dt.timedelta(seconds=20)
MAX_CAPTURE_INTERVAL = dt.timedelta(minutes=10)
MAX_UNREADABLE_FRACTION = 0.2
AMBIGUOUS_SCORE_MARGIN = 32
MAX_ALTERNATIVE_COMBINATIONS = 64
MIN_CELL_INK_PIXELS = 20

GLYPH_EXEMPLAR_COUNTS = (38, 3, 22, 4, 8, 9, 16, 7, 16, 2)
PACKED_GLYPH_BANK = (
    "eNqll01q20AUx588JQpY8VwgJRcoxOBFBVXiQJa9jMHbLgw5QK6SI0xRoYu61QUKMXibhVJ3Uaia6XvzIY00ksal"
    "myiWfvq/r3lvRgAJk7yaZ0KyipdcLKFidMFfaVYe8BmMIWm5x2fRMSongA8z/AVbNk/EW/hOlwX+wpslPmMjKplV"
    "iR2V1VmPymREZZ7tSIWF3WWCt901iOtuecrke0J+w9MUn72DJ/aAyM3XafYBkfHU1RGBb6iTF/Ll1iIPs9qXiUFu"
    "JYxHtJdxBaGgxxG8GUqdQbyIFs3FRASCgczlKDJexn1oYWbHLMznkKE0aGiuDcH/GVIFYOFW69QoPs7QNfzyFgOM"
    "Lcxg0JSXuNON/1zGzPriTYagyvU/ZbeO6FXYXQj7ApCBMoAX6k3v0kZosScw71xcZDWAzExEB7bldBvFyQKaoVel"
    "lE03DiNqkL2YiJTNGqEUyQKHRxBRKnXDDiF0CSJf/hDCtlOLlHxTsR0iU/mIEfFQRJieCBHfENUb/0NDk7C70RGG"
    "wtnVqatGDW0GkJVTALAFIEP97h7ny50crzT4vkRC6VnEWy8x/hFx191WdnEdKhUmLkylw2sXRDB1vFwKNGT7Cscd"
    "/aM7xGyfbYQ630PwxsZVYU5Hqi2LqUHRNuQi9T6tVyt5h2bzgmfLSh4oIea9AaQkadPVAZUGMWlwkfvKTgqNRI1K"
    "zpNeQ1i9tsqzFB7iGTrC3Y/obkQI3aRuRBkC0IkmdWldo5wmQ4PoGi3gU1PGS6si+U+tMlWIKuMPvRgGDdlKlzyI"
    "4IbgIKUUbqV3eidZ3ilk56uYIw5vq/QFHVmVsk+ldfTDjR0foC+f80eOZ9rWZmN3WIMUhLwWa3/Lmpf5ZkZIjkUq"
    "L8Q62sY1cmIM5Y6htorpxtoXUlldtQ2xumGNL6pDu77kMknmC9QzKllSLkTRIJOhoOMhRPly5u3TuTwn6dpQmuwW"
    "vruj2dWDzFNx3YVu6gree5hNkpWjQoijcsP68rLrOXmMBG36yB40zNARdnbiehUIHTzEnEqon6jXzbTpQzi8UTeq"
    "5SDCtAqOlX2NzGhzUDtJ7CBMPDKL6IU/g9Rxt2SbiBrWIBul0iD0zo5B+4hDeZH3LzT0aZAvzTeJ3sqvEPmmET6A"
    "OCqF7CDN+QUzb7JkvsfU8Chkj8p9V8V3t9dQANHb5+pqOGh1OEEHM7Emd7WKXZi4oBuVsdQBnCqVc7FAFaz7vgk6"
    "JZVC9kTE+3zpIFUXmYVSB+FKq8+WQqb4GY/NdamXqQka3cVXXpwjDg5AEiuKJQZGi7j+4Ggd8s/FukHwK2uHz/4C"
    "+hRjoA=="
)

FRAME_FILENAME_FORMAT = "allsky-%Y%m%d-%H%M"
#: Overlay-timestamped extraction writes the frame manifest of
#: :mod:`allsky.video` with one extra column carrying the timestamp provenance,
#: so both extraction paths produce a manifest the builder reads the same way.
MANIFEST_COLUMNS = (*VIDEO_MANIFEST_COLUMNS, "qc_frame_flags")


class OverlayTimestampError(ValueError):
    """The burned-in timestamps could not be read well enough to trust."""


@dataclass(frozen=True)
class OverlayReading:
    """One frame's overlay read, and how much of it the pixels actually settled.

    Attributes
    ----------
    index:
        Zero-based frame position in the source video, or ``-1`` for a read
        taken outside a video (a single live frame).
    timestamp:
        Naive local capture time, or ``None`` when the stamp could not be read
        or parsed at all.
    text:
        The fourteen digits as read, verbatim — kept even when *timestamp* was
        overridden, so a disputed read stays auditable.
    interpolated:
        The stamp was unreadable and *timestamp* was interpolated between the
        nearest readable neighbours.
    corrected:
        The stamp parsed but contradicted its neighbours, and *timestamp* was
        re-decided from the runner-up glyph candidates.
    """

    index: int
    timestamp: dt.datetime | None
    text: str
    interpolated: bool = False
    corrected: bool = False


def _glyph_bank() -> dict[str, np.ndarray]:
    height = OVERLAY_ROW_SLICE.stop - OVERLAY_ROW_SLICE.start
    total = sum(GLYPH_EXEMPLAR_COUNTS)
    bits = np.unpackbits(
        np.frombuffer(zlib.decompress(base64.b64decode(PACKED_GLYPH_BANK)), dtype=np.uint8)
    )
    exemplars = bits[: total * height * DIGIT_CELL_WIDTH].reshape(-1, height, DIGIT_CELL_WIDTH)
    bank: dict[str, np.ndarray] = {}
    offset = 0
    for digit, count in zip("0123456789", GLYPH_EXEMPLAR_COUNTS, strict=True):
        bank[digit] = exemplars[offset : offset + count].astype(bool)
        offset += count
    return bank


_BANK = _glyph_bank()


def _text_mask(frame: np.ndarray) -> np.ndarray:
    band = np.asarray(frame[OVERLAY_ROW_SLICE, :], dtype=np.int16)
    if band.ndim != 3 or band.shape[2] < 3:
        raise OverlayTimestampError("overlay reading needs an RGB frame")
    red, green, blue = band[..., 0], band[..., 1], band[..., 2]
    return (blue > TEXT_BLUE_FLOOR) & (red < TEXT_RED_CEILING) & (green < TEXT_GREEN_CEILING)


def _cell_scores(mask: np.ndarray, left: int) -> list[tuple[str, int]]:
    """Every digit scored against one cell, best first, over the shift search."""
    strip = mask[:, left - DIGIT_CELL_SEARCH_PAD : left + DIGIT_CELL_WIDTH + DIGIT_CELL_SEARCH_PAD]
    best: dict[str, int] = {}
    for shift in DIGIT_MATCH_SHIFTS:
        start = DIGIT_CELL_SEARCH_PAD + shift
        candidate = strip[:, start : start + DIGIT_CELL_WIDTH]
        if candidate.shape[1] != DIGIT_CELL_WIDTH:
            continue
        for digit, exemplars in _BANK.items():
            score = int((exemplars == candidate).sum(axis=(1, 2)).max())
            if score > best.get(digit, -1):
                best[digit] = score
    return sorted(best.items(), key=lambda item: -item[1])


def _cell_alternatives(scores: list[tuple[str, int]]) -> tuple[str, ...]:
    """Digits scoring within :data:`AMBIGUOUS_SCORE_MARGIN` of the best one.

    A cell whose neighbour's ink reaches into the shift window can leave the
    true digit tied with a lookalike (``8`` against ``6`` is the pair this font
    produces), so the runners-up are kept for the temporal pass to choose
    between.
    """
    if not scores:
        return ()
    top = scores[0][1]
    return tuple(digit for digit, score in scores if top - score <= AMBIGUOUS_SCORE_MARGIN)


def read_frame_timestamp(frame: np.ndarray) -> OverlayReading:
    """Read one frame's burned-in ``YYYYMMDD HH:MM:SS`` stamp.

    Parameters
    ----------
    frame:
        ``(H, W, 3)`` ``uint8`` RGB frame at the camera's native resolution,
        in image (row, column) coordinates.

    Returns
    -------
    OverlayReading
        Index ``-1`` (no video position is known here) and a naive local
        ``timestamp``, which is ``None`` when no overlay is present or the
        digits do not form a real date. Neither ``interpolated`` nor
        ``corrected`` is ever set: a single frame has no neighbours to be
        re-decided against.

    Raises
    ------
    OverlayTimestampError
        If the frame is not RGB, or is too narrow to contain the digit band.
    """
    return _read_frame(frame)[0]


def _read_frame(frame: np.ndarray) -> tuple[OverlayReading, tuple[tuple[str, ...], ...]]:
    mask = _text_mask(frame)
    if mask.shape[1] <= DIGIT_CELL_LEFT_EDGES[-1] + DIGIT_CELL_WIDTH:
        raise OverlayTimestampError(
            f"frame is only {mask.shape[1]} px wide; the timestamp overlay needs more than "
            f"{DIGIT_CELL_LEFT_EDGES[-1] + DIGIT_CELL_WIDTH}"
        )
    if not _cells_carry_ink(mask):
        return OverlayReading(index=-1, timestamp=None, text=""), ()
    per_cell = [_cell_scores(mask, left) for left in DIGIT_CELL_LEFT_EDGES]
    text = "".join(scores[0][0] if scores else "" for scores in per_cell)
    alternatives = tuple(_cell_alternatives(scores) for scores in per_cell)
    return OverlayReading(index=-1, timestamp=parse_overlay_stamp(text), text=text), alternatives


def _cells_carry_ink(mask: np.ndarray) -> bool:
    """Whether every digit cell holds enough text pixels to be a digit at all.

    Without this an overlay-free frame matches whichever exemplars are closest
    to blank and yields ``11111111111111`` — a *parseable* date in the year 1111
    — instead of an honest failure.
    """
    return all(
        int(mask[:, left : left + DIGIT_CELL_WIDTH].sum()) >= MIN_CELL_INK_PIXELS
        for left in DIGIT_CELL_LEFT_EDGES
    )


def parse_overlay_stamp(text: str) -> dt.datetime | None:
    """Parse a 14-digit ``YYYYMMDDHHMMSS`` reading as naive local time, or None if impossible."""
    if len(text) != OVERLAY_STAMP_DIGITS or not text.isdigit():
        return None
    try:
        return dt.datetime.fromisoformat(f"{text[:8]}T{text[8:]}")
    except ValueError:
        return None


def _plausible(stamp: dt.datetime, video_date: dt.date) -> bool:
    return stamp.date() in (video_date, video_date + dt.timedelta(days=1))


def _bracketing_stamps(
    readings: list[OverlayReading], position: int
) -> tuple[dt.datetime | None, dt.datetime | None]:
    before = readings[position - 1].timestamp if position > 0 else None
    after = readings[position + 1].timestamp if position + 1 < len(readings) else None
    return before, after


def _candidate_texts(alternatives: tuple[tuple[str, ...], ...]) -> Iterator[str]:
    """Every reading the pixels support, cheapest-doubt cells varied first.

    The cells are ranked by how close their runner-up came, and alternatives are
    admitted from the most doubtful down until
    :data:`MAX_ALTERNATIVE_COMBINATIONS` would be exceeded, so a frame with many
    marginal cells still yields the candidates most likely to be the misread one
    instead of being skipped for being too branchy.
    """
    ranked = sorted(range(len(alternatives)), key=lambda cell: -len(alternatives[cell]))
    admitted: dict[int, tuple[str, ...]] = {}
    combinations = 1
    for cell in ranked:
        options = alternatives[cell]
        if len(options) < 2:
            continue
        if combinations * len(options) > MAX_ALTERNATIVE_COMBINATIONS:
            break
        admitted[cell] = options
        combinations *= len(options)
    per_cell = [
        admitted.get(cell, alternatives[cell][:1] or ("",)) for cell in range(len(alternatives))
    ]
    for choice in itertools.product(*per_cell):
        yield "".join(choice)


def _correct_against_neighbours(
    readings: list[OverlayReading], alternatives: list[tuple[tuple[str, ...], ...]]
) -> list[OverlayReading]:
    """Re-decide readings the neighbouring frames contradict.

    Where a cell's runner-up scores within :data:`AMBIGUOUS_SCORE_MARGIN` of the
    winner, the pixels do not settle the digit on their own; the capture
    sequence does, because timestamps only ever increase. A frame that lands
    outside its neighbours' bracket is re-read from the alternatives, keeping
    the candidate nearest the midpoint of that bracket.
    """
    corrected = list(readings)
    for position, reading in enumerate(readings):
        before, after = _bracketing_stamps(readings, position)
        stamp = reading.timestamp
        if before is None or after is None or before >= after:
            continue
        if stamp is not None and before < stamp < after:
            continue
        midpoint = before + (after - before) / 2
        best: dt.datetime | None = None
        for text in _candidate_texts(alternatives[position]):
            candidate = parse_overlay_stamp(text)
            if candidate is None or not before < candidate < after:
                continue
            if best is None or abs(candidate - midpoint) < abs(best - midpoint):
                best = candidate
        if best is not None and best != stamp:
            logger.info(
                "frame %d: overlay read %r contradicts its neighbours; the capture sequence "
                "supports %s",
                reading.index,
                reading.text,
                best,
            )
            corrected[position] = OverlayReading(
                index=reading.index,
                timestamp=best,
                text=reading.text,
                interpolated=reading.interpolated,
                corrected=True,
            )
    return corrected


def _repair(readings: list[OverlayReading]) -> list[OverlayReading]:
    known = [position for position, item in enumerate(readings) if item.timestamp is not None]
    if not known:
        return readings
    repaired = list(readings)
    for position, item in enumerate(readings):
        if item.timestamp is not None:
            continue
        before = [k for k in known if k < position]
        after = [k for k in known if k > position]
        if not before or not after:
            continue
        low, high = before[-1], after[0]
        start = readings[low].timestamp
        end = readings[high].timestamp
        if start is None or end is None:
            continue
        fraction = (position - low) / (high - low)
        repaired[position] = OverlayReading(
            index=item.index,
            timestamp=start + (end - start) * fraction,
            text=item.text,
            interpolated=True,
        )
    return repaired


def read_video_timestamps(
    path: str | Path, *, step: int = 1, video_date: dt.date | None = None
) -> list[OverlayReading]:
    """Read the overlay timestamp of every *step*-th frame, repairing isolated failures.

    Three passes run over the day: each sampled frame is read on its own, reads
    that contradict their neighbours are re-decided from the runner-up glyph
    candidates, then frames still without a stamp are linearly interpolated
    between the nearest readable ones. A read landing outside *video_date* or
    the day after is discarded before any of that.

    Parameters
    ----------
    path:
        One-day timelapse video, named ``allsky-YYYYMMDD`` when *video_date* is
        not given.
    step:
        Read one frame every *step* source frames; every frame is still decoded,
        as the codec requires.
    video_date:
        Local calendar day the video covers; parsed from the filename when None.

    Returns
    -------
    list of OverlayReading
        One entry per sampled frame, in capture order, carrying naive local
        timestamps. Entries flagged ``interpolated`` or ``corrected`` did not
        come from their own pixels alone.

    Raises
    ------
    ValueError
        If *step* is below 1.
    OverlayTimestampError
        If the filename encodes no date, if more than
        :data:`MAX_UNREADABLE_FRACTION` of the sampled frames are unreadable, or
        if the recovered timestamps run backwards.
    """
    import imageio.v3 as iio

    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    day = video_date if video_date is not None else _video_date_from_name(path)
    readings: list[OverlayReading] = []
    alternatives: list[tuple[tuple[str, ...], ...]] = []
    for index, frame in enumerate(iio.imiter(path)):
        if index % step:
            continue
        reading, cell_alternatives = _read_frame(np.asarray(frame))
        stamp = reading.timestamp
        if stamp is not None and not _plausible(stamp, day):
            logger.warning(
                "frame %d of %s reads %r, outside %s/%s",
                index,
                Path(path).name,
                reading.text,
                day,
                day + dt.timedelta(days=1),
            )
            stamp = None
        readings.append(OverlayReading(index=index, timestamp=stamp, text=reading.text))
        alternatives.append(cell_alternatives)

    readings = _correct_against_neighbours(readings, alternatives)

    unreadable = sum(1 for item in readings if item.timestamp is None)
    if readings and unreadable / len(readings) > MAX_UNREADABLE_FRACTION:
        raise OverlayTimestampError(
            f"{unreadable} of {len(readings)} frames in {Path(path).name} have an unreadable "
            "timestamp overlay — refusing to timestamp this video from a guess"
        )
    repaired = _repair(readings)
    _check_monotonic(repaired, Path(path).name)
    if unreadable:
        logger.warning(
            "%s: %d of %d overlay reads failed and were interpolated",
            Path(path).name,
            unreadable,
            len(readings),
        )
    return repaired


def _check_monotonic(readings: list[OverlayReading], name: str) -> None:
    """Reject a sequence that goes backwards, and report the odd cadence it keeps.

    Only the ordering is fatal. The camera really does emit occasional extra
    captures seconds apart, and really does stop for hours, so an interval
    outside the usual band is reported rather than refused — refusing it would
    reject sound data over a cadence the camera never promised.
    """
    stamped = [item for item in readings if item.timestamp is not None]
    unusual = 0
    for earlier, later in itertools.pairwise(stamped):
        first, second = earlier.timestamp, later.timestamp
        if first is None or second is None:
            continue
        if second <= first:
            raise OverlayTimestampError(
                f"{name}: overlay timestamps go backwards at frame {later.index} "
                f"({first} -> {second})"
            )
        gap = (second - first) / (later.index - earlier.index)
        if not MIN_CAPTURE_INTERVAL <= gap <= MAX_CAPTURE_INTERVAL:
            unusual += 1
    if unusual:
        logger.info(
            "%s: %d frame gap(s) outside %.0fs..%.0fs — the camera's own capture cadence",
            name,
            unusual,
            MIN_CAPTURE_INTERVAL.total_seconds(),
            MAX_CAPTURE_INTERVAL.total_seconds(),
        )


def _video_date_from_name(path: str | Path) -> dt.date:
    stem = Path(path).stem
    digits = stem.rsplit("-", 1)[-1]
    try:
        return dt.date.fromisoformat(digits)
    except ValueError as exc:
        raise OverlayTimestampError(
            f"cannot tell which day {stem!r} covers; expected an allsky-YYYYMMDD name"
        ) from exc


def _resized(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    from PIL import Image

    target = (size, size) if isinstance(size, int) else size
    return np.asarray(Image.fromarray(image).resize(target, Image.Resampling.BILINEAR))


def _frames_with_timestamps(
    path: str | Path, readings: list[OverlayReading], step: int
) -> Iterator[tuple[OverlayReading, np.ndarray]]:
    import imageio.v3 as iio

    by_index = {item.index: item for item in readings}
    for index, frame in enumerate(iio.imiter(path)):
        if index % step:
            continue
        reading = by_index.get(index)
        if reading is not None and reading.timestamp is not None:
            yield reading, np.asarray(frame, dtype=np.uint8)


def _timestamp_flags(reading: OverlayReading) -> int:
    """QC bits recording whether this frame's capture time was manufactured."""
    bits = 0
    if reading.interpolated:
        bits |= int(QCFlag.TIMESTAMP_INTERPOLATED)
    if reading.corrected:
        bits |= int(QCFlag.TIMESTAMP_CORRECTED)
    return bits


def extract_frames_with_overlay_timestamps(
    path: str | Path,
    out_dir: str | Path,
    *,
    step: int = 1,
    resize: int | tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Extract JPEG frames named by the timestamp burned into each frame.

    Files are written as ``allsky-YYYYMMDD-HHMM.jpg`` at quality
    :data:`JPEG_QUALITY`, alongside a ``manifest.parquet`` describing them. Frame
    names carry minute resolution, matching the manifest's ``sample_id``, so a
    second frame captured within the same minute is skipped rather than
    overwriting the first.

    Parameters
    ----------
    path:
        One-day timelapse video named ``allsky-YYYYMMDD``.
    out_dir:
        Output directory, created if missing. The manifest is overwritten on
        every call, so use one directory per video.
    step:
        Keep one frame every *step* source frames.
    resize:
        Output size in pixels — ``int`` for square, ``(width, height)``
        otherwise. ``None`` keeps the native resolution. The overlay is always
        read at native resolution first.

    Returns
    -------
    pandas.DataFrame
        One row per written frame with columns :data:`MANIFEST_COLUMNS`:
        ``frame_path`` (as written), ``timestamp`` (naive local,
        ``datetime64[ns]``), ``video`` (source filename), ``index`` (source
        frame position, ``int64``) and ``qc_frame_flags`` (``int64`` bitmask of
        :class:`~allsky.data.contracts.QCFlag`, recording whether the capture
        time was interpolated or corrected).

    Raises
    ------
    OverlayTimestampError
        Propagated from :func:`read_video_timestamps` when the day's overlays
        cannot be trusted; nothing is written in that case.
    """
    import imageio.v3 as iio

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    video_name = Path(path).name
    readings = read_video_timestamps(path, step=step)

    rows: list[dict[str, object]] = []
    written: set[str] = set()
    collisions = 0
    for reading, image in _frames_with_timestamps(path, readings, step):
        stamp = reading.timestamp
        if stamp is None:
            continue
        name = f"{stamp:{FRAME_FILENAME_FORMAT}}.jpg"
        if name in written:
            collisions += 1
            continue
        frame_file = out_path / name
        iio.imwrite(
            frame_file, image if resize is None else _resized(image, resize), quality=JPEG_QUALITY
        )
        written.add(name)
        rows.append(
            {
                "frame_path": str(frame_file),
                "timestamp": pd.Timestamp(stamp),
                "video": video_name,
                "index": reading.index,
                "qc_frame_flags": _timestamp_flags(reading),
            }
        )

    if collisions:
        logger.info(
            "%s: %d frame(s) shared a minute with an earlier frame and were skipped "
            "(manifest sample_id has minute resolution)",
            video_name,
            collisions,
        )
    manifest = pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"]).astype("datetime64[ns]")
    manifest["index"] = manifest["index"].astype("int64")
    manifest["qc_frame_flags"] = manifest["qc_frame_flags"].astype("int64")
    manifest.to_parquet(out_path / MANIFEST_FILENAME, index=False)
    logger.info(
        "extracted %d frame(s) from %s to %s spanning %s..%s",
        len(manifest),
        video_name,
        out_path,
        manifest["timestamp"].min() if len(manifest) else "-",
        manifest["timestamp"].max() if len(manifest) else "-",
    )
    return manifest


def extract_frames_for(
    video: str | Path,
    out_dir: str | Path,
    video_config: VideoConfig,
    *,
    step: int = 1,
    resize: int | None = None,
) -> pd.DataFrame:
    """Extract *video*'s frames, timestamped by the clock *video_config* selects.

    The single place the ``overlay`` / ``modelled`` choice is made, so the three
    commands that extract frames cannot disagree about which clock a config
    names, nor about the warning the modelled cadence earns.

    Parameters
    ----------
    video:
        One-day timelapse to read.
    out_dir:
        Directory the JPEGs and their ``manifest.parquet`` are written to.
    video_config:
        Supplies ``timestamps`` (the clock) and, for the modelled mapping, the
        ``start_time`` / ``minutes_per_frame`` pair it places frames with.
    step:
        Keep every *step*-th frame.
    resize:
        Square edge in pixels to resize to, or None to keep the native size.

    Returns
    -------
    pandas.DataFrame
        The frame manifest, one row per written JPEG.

    Raises
    ------
    OverlayTimestampError
        Under ``timestamps="overlay"``, when the day's burned-in stamps cannot
        be read as a strictly increasing sequence.  It propagates so each caller
        keeps its own policy: ``prepare-local`` skips the day, ``extract-frames``
        exits non-zero.
    """
    if video_config.timestamps == "overlay":
        return extract_frames_with_overlay_timestamps(video, out_dir, step=step, resize=resize)

    from allsky.video import extract_frames

    logger.warning(
        "video.timestamps is 'modelled': frame N is placed at %s + N x %s min, a cadence this "
        "camera does not follow (see docs/allsky-archive.md)",
        video_config.start_time,
        video_config.minutes_per_frame,
    )
    return extract_frames(video, out_dir, video_config, step=step, resize=resize)
