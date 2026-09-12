"""The camera's auto-exposure as a photometer: read the burned-in exposure time.

The ZWO ASI678MC all-sky camera runs uncontrolled auto-exposure that sweeps more
than 1000x over a day, and records the exposure it chose nowhere but in the
frame itself, as the red ``Exposure: 12.89 ms`` line of the overlay. A pixel
value is ``DN ~ g * L * t`` (Debevec & Malik 1997, SIGGRAPH, eq. 1: the
reciprocity between scene radiance ``L`` and exposure time ``t``), so with ``t``
read back off the image the relative radiance of the sky disc follows as
``L ~ DN_lin / t``. Measured over the archive in the audit of 2026-08-28
(``exposicao-como-fotometro-2026-08-29-09h.md``): ``log2(DN_lin / t)``
correlates with the diffuse irradiance at ``r = +0.903`` and its slope is
stationary across the chronological splits, which the solar geometry is not.

This module ports the audit's reader (``extract2.py``) into the package. What
is reproduced verbatim: the overlay region, the red-text colour key, the
run-length segmentation of the number from its unit, the unit width classes,
the greedy left-to-right glyph decode with its narrow-glyph penalty, the number
parsing and the disc statistics. Three things differ, and the docstrings of
:func:`read_exposure_from_overlay_crop` and :func:`log2_relative_radiance` say
why: the audit's glyph templates were lost with its scratch directory and were
rebuilt from oracle-labelled archive glyphs, each glyph is matched over a
one-pixel horizontal shift window, and a glyph is scored by the Bernoulli
log-likelihood of its pixels under the template's ink frequencies rather than
by intersection-over-union against a binarised template.

Frames are ``(H, W, 3)`` ``uint8`` RGB arrays at the camera's native
``1920x1080`` resolution, in image (row, column) coordinates; the overlay
geometry is pinned to raw pixel columns, so nothing here accepts a resized
frame. Exposure times are seconds. An unreadable overlay yields ``None``,
never a NaN and never an imputed value.
"""

import base64
import logging
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "EXPOSURE_FEATURE_COLUMNS",
    "SHUFFLED_FEATURE_COLUMNS",
    "ExposureJoin",
    "ExposureRecord",
    "attach_exposure_features",
    "exposure_record",
    "exposure_records_for_video",
    "iter_video_frames",
    "log2_relative_radiance",
    "read_exposure_from_overlay_crop",
    "read_exposure_seconds",
    "records_frame",
    "saturated_fraction",
]

#: Native frame geometry of the ZWO ASI678MC stream the archive holds.
NATIVE_FRAME_HEIGHT = 1080
NATIVE_FRAME_WIDTH = 1920

#: Where the value of the ``Exposure:`` line sits in the native frame
#: (audit ``extract2.py``: ``ROWS`` / ``COLS``).
EXPOSURE_ROW_SLICE = slice(100, 121)
EXPOSURE_COL_SLICE = slice(120, 225)
OVERLAY_CROP_HEIGHT = EXPOSURE_ROW_SLICE.stop - EXPOSURE_ROW_SLICE.start

#: Colour key of the red overlay text (audit ``extract2.py``, ``read_exposure``).
TEXT_RED_FLOOR = 150
TEXT_RED_OVER_GREEN_FLOOR = 90
TEXT_RED_OVER_BLUE_FLOOR = 90

#: The blank run that separates the number from its unit is the widest gap in
#: the line; anything narrower than this is a gap between glyphs.
MIN_NUMBER_TO_UNIT_GAP_PX = 4

#: The unit is told by the pixel width of its text — ``us``, ``ms``, ``sec`` —
#: as ``(unit, widest pixel width, seconds per unit)`` (audit ``extract2.py``);
#: a wider run is not an exposure line.
UNIT_WIDTH_SCALES: tuple[tuple[str, int, float], ...] = (
    ("us", 25, 1e-6),
    ("ms", 31, 1e-3),
    ("sec", 36, 1.0),
)
MICROSECONDS_UNIT = "us"

#: Glyph cell the templates are matched in.
GLYPH_WINDOW_WIDTH = 14
#: A template narrower than this that leaves ink right after itself is scored
#: as having cut a glyph in two (audit ``extract2.py``). The audit subtracted
#: it from an IoU in ``[0, 1]``; here it comes off a mean log-likelihood in
#: nats per pixel, where leave-one-day-out reads are the same at 0.15, 0.3
#: and 0.5, so the audit's value is kept.
NARROW_GLYPH_PENALTY = 0.15
NARROW_GLYPH_WIDTH_CEILING = 13
#: Horizontal offsets each glyph is tried at, relative to the first ink column
#: after the previous glyph. The audit tried only ``0``; see
#: :func:`read_exposure_from_overlay_crop`.
GLYPH_MATCH_SHIFTS = (-1, 0, 1)

#: One label per packed template, in template order. The comma of ``1,485 us``
#: is labelled ``.`` like the audit's bank did, and stripped by the microsecond
#: parse; ``u`` and ``s`` are rejection classes that turn a letter read inside
#: the number into an unreadable line rather than a digit.
GLYPH_LABELS = ".0112234456677899su"
#: Per-pixel ink counts ``(E, 21, 14)`` of the archive's own glyphs at each
#: character's dominant widths (at least 15 % of the character's samples),
#: little-endian ``uint16``, zlib then base64; :data:`GLYPH_SAMPLE_COUNTS` is
#: how many glyphs each template counts over. Samples are the glyphs standing
#: alone in their ink run, labelled by the audit's per-frame readings over
#: eight days (2026-06-03, 06-15, 07-01, 07-15, 07-30, 08-07, 08-12, 08-25).
GLYPH_SAMPLE_COUNTS = (
    398,
    139,
    764,
    161,
    298,
    220,
    307,
    215,
    46,
    230,
    90,
    171,
    493,
    107,
    209,
    181,
    164,
    817,
    862,
)
PACKED_GLYPH_INK_COUNTS = (
    "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"
    "u07/AjPdmvI="
)
#: Pseudo-count added to ink and blank alike before a count becomes an ink
#: probability, so no pixel is ever certain: the Jeffreys prior for a
#: Bernoulli rate (Krichevsky & Trofimov 1981, IEEE Trans. Inf. Theory 27(2)).
GLYPH_INK_PSEUDO_COUNT = 0.5

#: ITU-R BT.709-6 (2015), item 3.3: luminance weights of linear RGB.
BT709_LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)
#: Display gamma the audit undid before weighting (``exposicao-como-fotometro``,
#: sec. 3.4); the camera's own transfer curve is not documented.
SRGB_GAMMA = 2.2
DN_FULL_SCALE = 255.0

#: The sky disc the radiance is averaged over (audit ``extract2.py``: ``SS`` and
#: ``DISC``): a 500 px circle about the frame centre on a 3-pixel subsampling
#: grid. It is a fixed central region, not the fisheye horizon of
#: :mod:`allsky.lens`, so it never touches the overlay nor the horizon mask.
DISC_SUBSAMPLE_STEP = 3
DISC_CENTER_X_PX = 960.0
DISC_CENTER_Y_PX = 540.0
DISC_RADIUS_PX = 500.0
#: A channel at or above this DN is counted as saturated (audit ``sat_frac``).
SATURATION_DN = 254

#: Columns :func:`attach_exposure_features` adds to a manifest.
EXPOSURE_FEATURE_COLUMNS = (
    "exposure_s",
    "log2_exposure_s",
    "log2_relative_radiance",
    "sat_frac",
)
#: Within-split permutations of the two log features: the control arm that
#: shows whether a network uses them.
SHUFFLED_FEATURE_COLUMNS = ("log2_exposure_s_shuffled", "log2_relative_radiance_shuffled")
_SHUFFLE_SOURCES = ("log2_exposure_s", "log2_relative_radiance")
_RECORD_COLUMNS = ("video", "frame_index", *EXPOSURE_FEATURE_COLUMNS)


def _glyph_bank() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(log p_ink, log p_blank, width)`` per template, ``(E, 21, 14)`` and ``(E,)``."""
    count = len(GLYPH_LABELS)
    if len(GLYPH_SAMPLE_COUNTS) != count:
        raise ValueError(
            f"{count} glyph labels but {len(GLYPH_SAMPLE_COUNTS)} sample counts in the bank"
        )
    ink_counts = np.frombuffer(
        zlib.decompress(base64.b64decode(PACKED_GLYPH_INK_COUNTS)), dtype="<u2"
    ).reshape(count, OVERLAY_CROP_HEIGHT, GLYPH_WINDOW_WIDTH)
    samples = np.asarray(GLYPH_SAMPLE_COUNTS, dtype=np.float64)[:, None, None]
    if (ink_counts > samples).any():
        raise ValueError("a glyph template counts more ink than samples")
    ink_probability = (ink_counts + GLYPH_INK_PSEUDO_COUNT) / (samples + 2 * GLYPH_INK_PSEUDO_COUNT)
    widths = np.array(
        [
            int(np.nonzero((template >= 0.5).any(axis=0))[0].max()) + 1
            for template in ink_probability
        ]
    )
    return np.log(ink_probability), np.log1p(-ink_probability), widths


_LOG_INK, _LOG_BLANK, _TEMPLATE_WIDTHS = _glyph_bank()
_TEMPLATE_COLUMN_MASK = np.zeros(_LOG_INK.shape, dtype=bool)
for _index, _width in enumerate(_TEMPLATE_WIDTHS):
    _TEMPLATE_COLUMN_MASK[_index, :, :_width] = True
_TEMPLATE_PIXELS = _TEMPLATE_COLUMN_MASK.sum(axis=(1, 2))


def _disc_mask() -> np.ndarray:
    rows, cols = np.mgrid[
        0:NATIVE_FRAME_HEIGHT:DISC_SUBSAMPLE_STEP, 0:NATIVE_FRAME_WIDTH:DISC_SUBSAMPLE_STEP
    ]
    inside: np.ndarray = (cols - DISC_CENTER_X_PX) ** 2 + (
        rows - DISC_CENTER_Y_PX
    ) ** 2 <= DISC_RADIUS_PX**2
    return inside


_DISC = _disc_mask()


@dataclass(frozen=True, slots=True)
class ExposureRecord:
    """What one frame yields for the exposure features.

    Attributes
    ----------
    frame_index:
        Zero-based position of the frame in its source video.
    exposure_s:
        Exposure time in seconds read off the overlay, or ``None`` when the
        line could not be read.
    log2_exposure_s:
        ``log2(exposure_s)``, ``None`` with it.
    log2_relative_radiance:
        :func:`log2_relative_radiance` of the frame, ``None`` with it.
    sat_frac:
        Fraction of the subsampled disc pixels with any channel at or above
        :data:`SATURATION_DN`, dimensionless in ``[0, 1]``; defined whether or
        not the overlay was readable.
    """

    frame_index: int
    exposure_s: float | None
    log2_exposure_s: float | None
    log2_relative_radiance: float | None
    sat_frac: float


@dataclass(frozen=True, slots=True)
class ExposureJoin:
    """A manifest with the exposure features attached, and what it cost.

    Attributes
    ----------
    manifest:
        The source rows whose overlay was readable, in source order, with
        :data:`EXPOSURE_FEATURE_COLUMNS` and :data:`SHUFFLED_FEATURE_COLUMNS`
        appended.
    removed_sample_ids:
        ``sample_id`` of every source row dropped for an unreadable overlay,
        in source order.
    seed:
        The seed the within-split permutations were drawn with.
    """

    manifest: pd.DataFrame
    removed_sample_ids: tuple[str, ...]
    seed: int


def _check_native_frame(frame: np.ndarray) -> None:
    expected = (NATIVE_FRAME_HEIGHT, NATIVE_FRAME_WIDTH, 3)
    if frame.shape != expected:
        raise ValueError(
            f"expected a native {expected} RGB frame, got shape {frame.shape}; the overlay "
            "geometry and the disc are pinned to raw pixels, so a resized frame cannot be read"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"expected a uint8 frame, got {frame.dtype}")


def _text_mask(crop: np.ndarray) -> np.ndarray:
    band = np.asarray(crop, dtype=np.int16)
    red, green, blue = band[..., 0], band[..., 1], band[..., 2]
    mask: np.ndarray = (
        (red > TEXT_RED_FLOOR)
        & (red - green > TEXT_RED_OVER_GREEN_FLOOR)
        & (red - blue > TEXT_RED_OVER_BLUE_FLOOR)
    )
    return mask


def _ink_runs(column_ink: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` column spans holding ink, left to right."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for column, ink in enumerate(column_ink):
        if ink > 0 and start is None:
            start = column
        elif ink == 0 and start is not None:
            runs.append((start, column))
            start = None
    if start is not None:
        runs.append((start, len(column_ink)))
    return runs


def _split_number_from_unit(
    runs: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if len(runs) < 2:
        return None
    gaps = [(runs[i + 1][0] - runs[i][1], i) for i in range(len(runs) - 1)]
    widest_gap, position = max(gaps)
    if widest_gap < MIN_NUMBER_TO_UNIT_GAP_PX:
        return None
    return runs[: position + 1], runs[position + 1 :]


def _unit_of_width(unit_width_px: int) -> tuple[str, float] | None:
    for unit, ceiling, scale in UNIT_WIDTH_SCALES:
        if unit_width_px <= ceiling:
            return unit, scale
    return None


def _glyph_scores(mask: np.ndarray, column_ink: np.ndarray, left: int) -> np.ndarray:
    """Mean log-likelihood per pixel of the cell at *left* under every template, penalised.

    Each template's own columns only: a pixel with ink contributes ``log
    p_ink``, a blank one ``log p_blank``, so the pixels a template is sure
    about (the closed top of an ``8``, the open one of a ``6``) carry the
    decision and the pixels its samples disagree on (the faint stroke edges
    the colour key keeps in some frames and drops in others) carry little.
    """
    window = np.zeros((OVERLAY_CROP_HEIGHT, GLYPH_WINDOW_WIDTH), dtype=bool)
    cell = mask[:, left : left + GLYPH_WINDOW_WIDTH]
    window[:, : cell.shape[1]] = cell
    log_likelihood = np.where(window[None], _LOG_INK, _LOG_BLANK) * _TEMPLATE_COLUMN_MASK
    scores = np.asarray(log_likelihood.sum(axis=(1, 2)) / _TEMPLATE_PIXELS, dtype=np.float64)
    for index, width in enumerate(_TEMPLATE_WIDTHS):
        right = left + int(width)
        if right < mask.shape[1] and column_ink[right] > 0 and width < NARROW_GLYPH_WIDTH_CEILING:
            scores[index] -= NARROW_GLYPH_PENALTY
    return scores


def _read_number_text(mask: np.ndarray, column_ink: np.ndarray, start: int, stop: int) -> str:
    """Greedy left-to-right glyph decode of the number's columns ``[start, stop)``."""
    text = ""
    left = start
    while left < stop:
        while left < stop and column_ink[left] == 0:
            left += 1
        if left >= stop:
            break
        best_score = -np.inf
        best_template = 0
        best_left = left
        for shift in GLYPH_MATCH_SHIFTS:
            candidate = left + shift
            if candidate < 0 or candidate >= stop:
                continue
            scores = _glyph_scores(mask, column_ink, candidate)
            template = int(np.argmax(scores))
            if scores[template] > best_score:
                best_score = float(scores[template])
                best_template = template
                best_left = candidate
        text += GLYPH_LABELS[best_template]
        left = best_left + int(_TEMPLATE_WIDTHS[best_template]) + 1
    return text


def _parse_exposure(text: str, unit: str, scale: float) -> float | None:
    if not text or text.count(".") > 1:
        return None
    digits = text.replace(".", "") if unit == MICROSECONDS_UNIT else text
    try:
        value = float(digits)
    except ValueError:
        return None
    return value * scale


def read_exposure_from_overlay_crop(crop: np.ndarray) -> float | None:
    """Read the exposure time from the crop of the overlay's ``Exposure:`` value.

    The red text is isolated by colour, split into ink runs by column, and the
    widest blank run separates the number from its unit. The unit is classified
    by its pixel width (:data:`UNIT_WIDTH_SCALES`), the number is decoded glyph
    by glyph against the packed template bank with the audit's narrow-glyph
    penalty, and parsed; a microsecond value drops its thousands separator,
    which the bank labels ``.``.

    Three deviations from the audit's ``extract2.py``, all forced by the loss
    of its template file (``cents2.npy``). The bank is rebuilt from the
    archive's own glyphs. Each glyph is matched at :data:`GLYPH_MATCH_SHIFTS`
    around the first ink column instead of at that column alone, absorbing the
    one-pixel width jitter the h264 anti-aliasing puts on touching glyphs,
    which otherwise leaves the next cell misaligned and cascades. And a glyph
    is scored by the mean Bernoulli log-likelihood of its pixels under the
    template's smoothed ink frequencies, not by intersection-over-union with a
    binarised template. The overlay's thin strokes are rendered faint (the
    closing stroke at the top right of an ``8`` reaches a red of 70-150 on
    the dark border the line sits on) and drop below the audit's colour key
    unevenly from frame to frame, so the mask's stroke thickness varies while
    the glyph does not; under IoU a thinned ``8`` matched the narrower ``6``
    template better than its own and was read as ``6`` in about 1 % of the
    frames of a day outside the bank, always lowering the exposure. The
    likelihood lets the pixels a template is sure of decide and discounts the
    ones its samples disagree on, which is where the thinning happens.

    Measured leave-one-day-out over the eight bank days (3 966 frames), the
    audit's readings as reference: the IoU port disagrees on 76 frames, the
    likelihood port on 43. Read by eye on the crop's red dominance (not on
    the mask, which is what loses the faint strokes), 19 of the 43 are the
    port's error, every one an ``8`` taken for ``6``; 14 are the audit's own
    (``7`` as ``2``, ``3`` as ``9``, ``6`` as ``8``); 9 are glyphs neither
    reading fits; one is unreadable. On the held-out day 2026-06-26 (565
    frames) the two disagree on 6: 4 the audit's, 1 the port's (again ``8``
    as ``6``), 1 undecidable. The residual port error is therefore about
    0.5 % of frames (0.7 % if every undecidable glyph is charged to it), an
    ``8`` whose top has thinned away read as ``6``, 0.08-0.16 stop low;
    ``tests/allsky/fixtures/exposure_1_80ms.png`` holds one such frame.

    Parameters
    ----------
    crop:
        ``(21, W, 3)`` ``uint8`` RGB crop, ``frame[EXPOSURE_ROW_SLICE,
        EXPOSURE_COL_SLICE]`` of a native frame, image (row, column)
        coordinates.

    Returns
    -------
    float or None
        Exposure time in seconds, or ``None`` when the line is not a readable
        ``<number> <unit>`` pair: no red text, no number-to-unit gap, a unit
        cut off by the crop's right edge (a ``ms`` value of 100 ms or more
        overflows the audit's region and its truncated ``m`` would otherwise
        pass for ``us``), a unit of unknown width, a letter or a second
        decimal point read inside the number.

    Raises
    ------
    ValueError
        If *crop* is not a 21-row RGB array.
    """
    if crop.ndim != 3 or crop.shape[0] != OVERLAY_CROP_HEIGHT or crop.shape[2] != 3:
        raise ValueError(
            f"expected a ({OVERLAY_CROP_HEIGHT}, W, 3) overlay crop, got shape {crop.shape}"
        )
    mask = _text_mask(crop)
    column_ink = mask.sum(axis=0)
    parts = _split_number_from_unit(_ink_runs(column_ink))
    if parts is None:
        return None
    number_runs, unit_runs = parts
    if unit_runs[-1][1] >= mask.shape[1]:
        return None
    unit = _unit_of_width(unit_runs[-1][1] - unit_runs[0][0])
    if unit is None:
        return None
    text = _read_number_text(mask, column_ink, number_runs[0][0], number_runs[-1][1])
    return _parse_exposure(text, *unit)


def read_exposure_seconds(frame: np.ndarray) -> float | None:
    """Read the exposure time the camera burned into a native frame.

    Parameters
    ----------
    frame:
        ``(1080, 1920, 3)`` ``uint8`` RGB frame at native resolution, image
        (row, column) coordinates.

    Returns
    -------
    float or None
        Exposure time in seconds, or ``None`` when the overlay line is
        unreadable (see :func:`read_exposure_from_overlay_crop`).

    Raises
    ------
    ValueError
        If *frame* is not a native uint8 RGB frame.
    """
    _check_native_frame(frame)
    return read_exposure_from_overlay_crop(frame[EXPOSURE_ROW_SLICE, EXPOSURE_COL_SLICE])


def _disc_pixels(frame: np.ndarray) -> np.ndarray:
    """``(N, 3)`` ``int16`` RGB of the subsampled disc pixels of a native frame."""
    subsampled = frame[::DISC_SUBSAMPLE_STEP, ::DISC_SUBSAMPLE_STEP]
    return np.asarray(subsampled[_DISC], dtype=np.int16)


def _log2_relative_radiance_of_means(channel_means: np.ndarray, exposure_s: float) -> float:
    linear = (channel_means / DN_FULL_SCALE) ** SRGB_GAMMA
    red_w, green_w, blue_w = BT709_LUMA_WEIGHTS
    luminance = red_w * linear[0] + green_w * linear[1] + blue_w * linear[2]
    return float(np.log2(luminance / exposure_s))


def log2_relative_radiance(frame: np.ndarray, exposure_s: float) -> float:
    """``log2`` of the sky disc's relative radiance, ``DN_lin / t``.

    Inverts the reciprocity ``DN ~ g * L * t`` (Debevec & Malik 1997, eq. 1):
    the mean of each channel over the subsampled disc is linearised by undoing
    :data:`SRGB_GAMMA`, the three are weighted by :data:`BT709_LUMA_WEIGHTS`,
    and the result is divided by the exposure time. The gain ``g`` is unknown
    and constant, hence *relative*; the log base 2 is the audit's choice
    (``exposicao-como-fotometro``, sec. 3.4), since ``t`` spans decades.

    The linearisation is applied to the per-channel MEANS, not to each pixel
    before averaging. That is what the audit computed (``join_exposicao``:
    ``logL`` from ``mean_r``, ``mean_g``, ``mean_b``), and every number it
    reports — the ``r = +0.903`` and the split-stationary slope — was measured
    on that definition, so the port keeps it rather than the per-pixel form
    the physics would suggest.

    Parameters
    ----------
    frame:
        ``(1080, 1920, 3)`` ``uint8`` RGB frame at native resolution.
    exposure_s:
        Exposure time in seconds, ``> 0``.

    Returns
    -------
    float
        ``log2(DN_lin / exposure_s)``, dimensionless (relative radiance in
        stops); ``-inf`` for an all-black disc.

    Raises
    ------
    ValueError
        If *exposure_s* is not positive or *frame* is not a native frame.
    """
    if not exposure_s > 0:
        raise ValueError(f"exposure_s must be positive, got {exposure_s!r}")
    _check_native_frame(frame)
    return _log2_relative_radiance_of_means(_disc_pixels(frame).mean(axis=0), exposure_s)


def saturated_fraction(frame: np.ndarray) -> float:
    """Fraction of the subsampled disc whose brightest channel is saturated.

    Parameters
    ----------
    frame:
        ``(1080, 1920, 3)`` ``uint8`` RGB frame at native resolution.

    Returns
    -------
    float
        Share of disc pixels with ``max(R, G, B) >= SATURATION_DN``, in
        ``[0, 1]`` (audit ``sat_frac``).

    Raises
    ------
    ValueError
        If *frame* is not a native frame.
    """
    _check_native_frame(frame)
    return _saturated_share(_disc_pixels(frame))


def _saturated_share(pixels: np.ndarray) -> float:
    return float((pixels.max(axis=1) >= SATURATION_DN).mean())


def exposure_record(frame: np.ndarray, frame_index: int) -> ExposureRecord:
    """Everything the exposure features need from one native frame.

    Parameters
    ----------
    frame:
        ``(1080, 1920, 3)`` ``uint8`` RGB frame at native resolution.
    frame_index:
        Zero-based position of the frame in its video, copied into the record.

    Returns
    -------
    ExposureRecord
        The overlay exposure with its ``log2``, the disc's
        :func:`log2_relative_radiance` and :func:`saturated_fraction`. When the
        overlay is unreadable the three exposure-derived fields are ``None``
        and only ``sat_frac`` is filled.

    Raises
    ------
    ValueError
        If *frame* is not a native frame.
    """
    _check_native_frame(frame)
    pixels = _disc_pixels(frame)
    sat_frac = _saturated_share(pixels)
    exposure_s = read_exposure_seconds(frame)
    if exposure_s is None:
        return ExposureRecord(frame_index, None, None, None, sat_frac)
    return ExposureRecord(
        frame_index=frame_index,
        exposure_s=exposure_s,
        log2_exposure_s=float(np.log2(exposure_s)),
        log2_relative_radiance=_log2_relative_radiance_of_means(pixels.mean(axis=0), exposure_s),
        sat_frac=sat_frac,
    )


def iter_video_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Stream every frame of a video with its zero-based index.

    The codec decodes sequentially, so there is no cheaper access to frame
    ``k`` than decoding the ``k`` before it; callers wanting a subset stop the
    iteration once past the last index they need.

    Parameters
    ----------
    path:
        Video file readable by imageio-ffmpeg.

    Yields
    ------
    tuple[int, numpy.ndarray]
        ``(index, frame)`` with the frame as an ``(H, W, 3)`` ``uint8`` RGB
        array in image (row, column) coordinates.
    """
    import imageio.v3 as iio

    for index, frame in enumerate(iio.imiter(path)):
        yield index, np.asarray(frame, dtype=np.uint8)


def exposure_records_for_video(
    path: str | Path, frame_indices: Iterable[int]
) -> list[ExposureRecord]:
    """Decode one video once and record the exposure features of chosen frames.

    Parameters
    ----------
    path:
        One-day timelapse video at native resolution.
    frame_indices:
        Zero-based frame positions to record; decoding stops after the
        largest.

    Returns
    -------
    list of ExposureRecord
        One record per requested index, in ascending index order.

    Raises
    ------
    ValueError
        If a requested frame is not a native ``1920x1080`` frame, or the video
        ends before the largest requested index — either means the manifest
        was built from a different file than the one at *path*.
    """
    wanted = sorted({int(index) for index in frame_indices})
    if not wanted:
        return []
    remaining = set(wanted)
    last = wanted[-1]
    records: list[ExposureRecord] = []
    for index, frame in iter_video_frames(path):
        if index in remaining:
            try:
                records.append(exposure_record(frame, index))
            except ValueError as exc:
                raise ValueError(f"{Path(path).name} frame {index}: {exc}") from exc
            remaining.discard(index)
        if index >= last:
            break
    if remaining:
        raise ValueError(
            f"{Path(path).name} ended before frame(s) {sorted(remaining)[:10]} the manifest "
            "names; the manifest was built from another copy of this video"
        )
    return records


def records_frame(video: str, records: Iterable[ExposureRecord]) -> pd.DataFrame:
    """Tabulate one video's records as the shard :func:`attach_exposure_features` joins.

    Parameters
    ----------
    video:
        Source video filename, as the manifest's ``video`` column spells it.
    records:
        The video's :class:`ExposureRecord` values.

    Returns
    -------
    pandas.DataFrame
        Columns ``video`` (string), ``frame_index`` (int64) and
        :data:`EXPOSURE_FEATURE_COLUMNS`; the three exposure-derived columns are
        nullable ``Float64`` with ``pd.NA`` where the overlay was unreadable,
        ``sat_frac`` is ``float64``.
    """
    rows = list(records)
    frame = pd.DataFrame(
        {
            "video": pd.array([video] * len(rows), dtype="string"),
            "frame_index": np.array([row.frame_index for row in rows], dtype=np.int64),
            "exposure_s": pd.array([row.exposure_s for row in rows], dtype="Float64"),
            "log2_exposure_s": pd.array([row.log2_exposure_s for row in rows], dtype="Float64"),
            "log2_relative_radiance": pd.array(
                [row.log2_relative_radiance for row in rows], dtype="Float64"
            ),
            "sat_frac": np.array([row.sat_frac for row in rows], dtype=np.float64),
        },
        columns=list(_RECORD_COLUMNS),
    )
    return frame


def attach_exposure_features(
    manifest: pd.DataFrame, records: pd.DataFrame, *, seed: int
) -> ExposureJoin:
    """Join per-frame exposure records onto a manifest, dropping unreadable rows.

    Rows are matched on ``(video, frame_index)``. A manifest row whose overlay
    was unreadable is REMOVED, never imputed, and its ``sample_id`` reported;
    a manifest row with no record at all is an error, because the records are
    computed from the manifest and a missing one means a shard from another
    build.

    The two control columns are each source column permuted WITHIN its split:
    one permutation per split, drawn from ``numpy.random.default_rng(seed)`` in
    the manifest's split order and applied to both columns alike, so every
    split keeps its own marginal distribution and the pair keeps its joint one
    while losing all connection to the frame it sits beside. Rows without a
    split label form their own group.

    Parameters
    ----------
    manifest:
        v2 manifest with ``sample_id``, ``video``, ``frame_index`` and
        ``split`` columns; dtypes are preserved so the content hash of the
        untouched columns does not move.
    records:
        Concatenated :func:`records_frame` shards covering every manifest row.
    seed:
        Seed of the within-split permutations.

    Returns
    -------
    ExposureJoin
        The augmented manifest, the removed ``sample_id`` values and the seed.

    Raises
    ------
    KeyError
        If *manifest* lacks one of the join or split columns.
    ValueError
        If a manifest row has no record, or a ``(video, frame_index)`` pair
        occurs twice in *records*.
    """
    for column in ("sample_id", "video", "frame_index", "split"):
        if column not in manifest.columns:
            raise KeyError(f"manifest lacks the {column!r} column")
    keys = ["video", "frame_index"]
    if records.duplicated(subset=keys).any():
        raise ValueError("records name the same (video, frame_index) more than once")
    joined = manifest.merge(
        records[[*keys, *EXPOSURE_FEATURE_COLUMNS]],
        on=keys,
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = joined["_merge"].to_numpy() != "both"
    if unmatched.any():
        missing = joined.loc[unmatched, "sample_id"].astype(str).tolist()
        raise ValueError(
            f"{len(missing)} manifest row(s) have no exposure record, e.g. {missing[:5]}; "
            "the shards do not cover this manifest"
        )
    joined = joined.drop(columns="_merge")
    unreadable = joined["exposure_s"].isna().to_numpy()
    removed = tuple(str(value) for value in joined.loc[unreadable, "sample_id"])
    kept = joined.loc[~unreadable].reset_index(drop=True)
    for column in EXPOSURE_FEATURE_COLUMNS:
        kept[column] = kept[column].astype("float64")

    rng = np.random.default_rng(seed)
    groups = kept["split"].astype(object).where(kept["split"].notna(), "").to_numpy()
    permuted = np.arange(len(kept))
    for group in pd.unique(groups):
        members = np.flatnonzero(groups == group)
        permuted[members] = members[rng.permutation(len(members))]
    for source, target in zip(_SHUFFLE_SOURCES, SHUFFLED_FEATURE_COLUMNS, strict=True):
        kept[target] = kept[source].to_numpy()[permuted]
    return ExposureJoin(manifest=kept, removed_sample_ids=removed, seed=seed)
