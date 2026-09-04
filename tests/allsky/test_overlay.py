"""Tests for allsky.overlay — reading the timestamp the camera burns into each frame.

The fixtures paint the stamp with the reader's own glyph bank onto a non-black
background, so the blue-high/red-low/green-low colour key is what isolates the
text, and encode it losslessly: h264's default lossy path is not part of the
contract under test.
"""

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.config import VideoConfig
from allsky.data.contracts import QCFlag
from allsky.overlay import (
    DIGIT_CELL_LEFT_EDGES,
    DIGIT_CELL_WIDTH,
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    OverlayReading,
    OverlayTimestampError,
    _correct_against_neighbours,
    extract_frames_for,
    extract_frames_with_overlay_timestamps,
    read_frame_timestamp,
    read_video_timestamps,
)
from tests.allsky import _archive_fake as fake

MINUTE_APART = ("20260810120000", "20260810120100", "20260810120200")
SIX_MINUTES = (
    "20260810120000",
    "20260810120100",
    "20260810120200",
    "20260810120300",
    "20260810120400",
    "20260810120500",
)


def _video(tmp_path: Path, stamps: Sequence[str], *, day: str = "20260810") -> Path:
    return fake.write_overlay_video(tmp_path / f"allsky-{day}.mp4", stamps)


def _naive(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text)


@pytest.mark.parametrize(
    "stamp",
    ["20260810120000", "20251231235959", "20260101000000", "19990909090909", "20260810073412"],
)
def test_a_rendered_stamp_reads_back_digit_for_digit(stamp: str):
    reading = read_frame_timestamp(fake.render_overlay_frame(stamp))
    assert reading.text == stamp
    assert reading.timestamp == _naive(f"{stamp[:8]}T{stamp[8:]}")
    assert reading.interpolated is False


@pytest.mark.parametrize("exemplar", [0, 1, 2, 5, 13])
def test_every_exemplar_in_the_glyph_bank_reads_as_its_own_digit(exemplar: int):
    stamp = "20260810120345"
    reading = read_frame_timestamp(fake.render_overlay_frame(stamp, exemplar=exemplar))
    assert reading.text == stamp


def test_the_colour_key_and_not_the_brightness_is_what_finds_the_text():
    stamp = "20260810120000"
    on_grey = fake.render_overlay_frame(stamp, background=(200, 200, 200))
    on_red = fake.render_overlay_frame(stamp, background=(255, 20, 20))
    assert read_frame_timestamp(on_grey).text == stamp
    assert read_frame_timestamp(on_red).text == stamp


@pytest.mark.parametrize("impossible", ["20261310120000", "20260230120000", "20260810250000"])
def test_an_impossible_reading_yields_no_timestamp_rather_than_an_exception(impossible: str):
    reading = read_frame_timestamp(fake.render_overlay_frame(impossible))
    assert reading.text == impossible
    assert reading.timestamp is None


def _scattered_ink_frame(pixels_per_cell: int, seed: int) -> np.ndarray:
    """A frame whose digit cells hold ink but no glyph."""
    from allsky import overlay as module

    frame = np.empty((fake.FRAME_HEIGHT, fake.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :] = fake.OVERLAY_BACKGROUND
    rows = module.OVERLAY_ROW_SLICE.stop - module.OVERLAY_ROW_SLICE.start
    rng = np.random.default_rng(seed)
    for left in module.DIGIT_CELL_LEFT_EDGES:
        cell = frame[module.OVERLAY_ROW_SLICE, left : left + module.DIGIT_CELL_WIDTH]
        chosen = rng.choice(rows * module.DIGIT_CELL_WIDTH, pixels_per_cell, replace=False)
        lit = np.zeros(rows * module.DIGIT_CELL_WIDTH, dtype=bool)
        lit[chosen] = True
        cell[lit.reshape(rows, module.DIGIT_CELL_WIDTH)] = fake.OVERLAY_TEXT
    return frame


@pytest.mark.parametrize("pixels_per_cell", [5, 20, 30, 39])
def test_a_cell_too_sparse_to_be_a_glyph_is_refused_instead_of_read_as_ones(
    pixels_per_cell: int,
):
    """The gate stood at 20 ink pixels while the sparsest exemplar in the bank
    carries 51 and the sparsest cell in 14 archive days carries 44, so a cell of
    scattered ink well below any real glyph read as `1` in 200/200 trials — a
    parseable, wrong minute instead of the honest failure the gate exists for."""
    reading = read_frame_timestamp(_scattered_ink_frame(pixels_per_cell, seed=pixels_per_cell))

    assert reading.text == ""
    assert reading.timestamp is None


def test_the_ink_floor_stays_below_every_cell_the_archive_actually_wrote():
    """Measured over 4.326 cells of 14 archive days: the sparsest carries 44.
    A floor at or above that refuses real frames, and an unreadable fraction
    past `MAX_UNREADABLE_FRACTION` refuses the whole day."""
    from allsky.overlay import MIN_CELL_INK_PIXELS

    sparsest_real_cell_measured_on_the_archive = 44
    assert sparsest_real_cell_measured_on_the_archive > MIN_CELL_INK_PIXELS


def test_every_exemplar_clears_the_ink_floor():
    """The bank is what a rendered frame is made of, so no exemplar may sit under
    the gate that decides whether a cell is a glyph at all."""
    from allsky.overlay import _BANK, MIN_CELL_INK_PIXELS

    sparsest = min(int(exemplar.sum()) for exemplars in _BANK.values() for exemplar in exemplars)

    assert sparsest >= MIN_CELL_INK_PIXELS


def test_a_frame_too_narrow_to_hold_the_overlay_is_refused():
    narrow = fake.render_overlay_frame(
        "20260810120000", width=DIGIT_CELL_LEFT_EDGES[-1] + DIGIT_CELL_WIDTH
    )
    with pytest.raises(OverlayTimestampError, match="px wide"):
        read_frame_timestamp(narrow)


def test_a_frame_without_colour_channels_is_refused():
    grey = np.zeros((60, 480), dtype=np.uint8)
    with pytest.raises(OverlayTimestampError, match="needs an RGB frame"):
        read_frame_timestamp(grey)


def test_reading_a_video_returns_one_reading_per_frame_in_capture_order(tmp_path: Path):
    readings = read_video_timestamps(_video(tmp_path, MINUTE_APART))
    assert [item.index for item in readings] == [0, 1, 2]
    assert [item.text for item in readings] == list(MINUTE_APART)
    assert [item.timestamp for item in readings] == [
        _naive("2026-08-10 12:00"),
        _naive("2026-08-10 12:01"),
        _naive("2026-08-10 12:02"),
    ]
    assert not any(item.interpolated for item in readings)


def test_a_step_keeps_only_every_nth_frame_and_still_validates_the_interval(tmp_path: Path):
    readings = read_video_timestamps(_video(tmp_path, SIX_MINUTES), step=2)
    assert [item.index for item in readings] == [0, 2, 4]
    assert [item.timestamp for item in readings] == [
        _naive("2026-08-10 12:00"),
        _naive("2026-08-10 12:02"),
        _naive("2026-08-10 12:04"),
    ]


def test_a_step_below_one_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="step must be >= 1"):
        read_video_timestamps(_video(tmp_path, MINUTE_APART), step=0)


def test_a_video_whose_stamps_run_backwards_is_refused(tmp_path: Path):
    stamps = ("20260810120000", "20260810120200", "20260810120100")
    with pytest.raises(OverlayTimestampError, match="go backwards at frame 2"):
        read_video_timestamps(_video(tmp_path, stamps))


def test_two_frames_carrying_the_same_stamp_count_as_running_backwards(tmp_path: Path):
    stamps = ("20260810120000", "20260810120000")
    with pytest.raises(OverlayTimestampError, match="go backwards"):
        read_video_timestamps(_video(tmp_path, stamps))


@pytest.mark.parametrize(
    "stamps",
    [
        ("20260810120000", "20260810120010"),
        ("20260810120000", "20260810122000"),
    ],
)
def test_an_unusual_capture_interval_is_reported_but_kept(
    tmp_path: Path, stamps: tuple[str, str], caplog: pytest.LogCaptureFixture
):
    with caplog.at_level(logging.INFO, logger="allsky.overlay"):
        readings = read_video_timestamps(_video(tmp_path, stamps))

    assert [str(item.timestamp) for item in readings] == [
        "2026-08-10 12:00:00",
        f"2026-08-10 {stamps[1][8:10]}:{stamps[1][10:12]}:{stamps[1][12:]}",
    ]
    assert "capture cadence" in caplog.text


def test_a_single_unreadable_frame_between_two_good_ones_is_interpolated(tmp_path: Path):
    stamps = (
        "20260810120000",
        "20260810120100",
        "20261310120200",
        "20260810120300",
        "20260810120400",
        "20260810120500",
    )
    readings = read_video_timestamps(_video(tmp_path, stamps))

    repaired = readings[2]
    assert repaired.interpolated is True
    assert repaired.timestamp == _naive("2026-08-10 12:02")
    assert repaired.text == "20261310120200"
    assert [item.interpolated for item in readings] == [False, False, True, False, False, False]


def test_a_stamp_landing_outside_the_videos_own_day_is_treated_as_unreadable(tmp_path: Path):
    stamps = (
        "20260810120000",
        "20260810120100",
        "20240101120200",
        "20260810120300",
        "20260810120400",
        "20260810120500",
    )
    readings = read_video_timestamps(_video(tmp_path, stamps))
    assert readings[2].interpolated is True
    assert readings[2].timestamp == _naive("2026-08-10 12:02")


def test_a_video_whose_stamps_are_mostly_unreadable_is_refused_rather_than_guessed(
    tmp_path: Path,
):
    stamps = (
        "20260810120000",
        "20260813120100",
        "20260814120200",
        "20260810120300",
        "20260810120400",
    )
    with pytest.raises(OverlayTimestampError, match="refusing to timestamp this video"):
        read_video_timestamps(_video(tmp_path, stamps))


def test_the_day_a_video_covers_can_be_supplied_when_its_name_does_not_say(tmp_path: Path):
    path = fake.write_overlay_video(tmp_path / "timelapse.mp4", MINUTE_APART)
    with pytest.raises(OverlayTimestampError, match="expected an allsky-YYYYMMDD name"):
        read_video_timestamps(path)
    readings = read_video_timestamps(path, video_date=dt.date(2026, 8, 10))
    assert readings[0].timestamp == _naive("2026-08-10 12:00")


def test_a_second_frame_landing_in_the_same_minute_is_skipped_not_overwritten(tmp_path: Path):
    stamps = (
        "20260810120010",
        "20260810120040",
        "20260810120130",
        "20260810120220",
        "20260810120310",
    )
    out_dir = tmp_path / "frames"
    manifest = extract_frames_with_overlay_timestamps(_video(tmp_path, stamps), out_dir)

    assert manifest["index"].tolist() == [0, 2, 3, 4]
    assert manifest["timestamp"].tolist() == [
        pd.Timestamp("2026-08-10 12:00:10"),
        pd.Timestamp("2026-08-10 12:01:30"),
        pd.Timestamp("2026-08-10 12:02:20"),
        pd.Timestamp("2026-08-10 12:03:10"),
    ]


def test_extraction_writes_the_documented_manifest_next_to_the_frames(tmp_path: Path):
    out_dir = tmp_path / "frames"
    video = _video(tmp_path, MINUTE_APART)
    manifest = extract_frames_with_overlay_timestamps(video, out_dir)

    assert list(manifest.columns) == list(MANIFEST_COLUMNS)
    assert len(manifest) == len(MINUTE_APART)
    assert (manifest["video"] == video.name).all()
    assert manifest["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert manifest["index"].dtype == np.dtype("int64")
    for frame_path in manifest["frame_path"]:
        assert Path(frame_path).is_file()

    persisted = pd.read_parquet(out_dir / MANIFEST_FILENAME)
    pd.testing.assert_frame_equal(persisted, manifest)


def test_extraction_creates_the_output_directory_and_honours_step_and_resize(tmp_path: Path):
    out_dir = tmp_path / "deep" / "frames"
    manifest = extract_frames_with_overlay_timestamps(
        _video(tmp_path, SIX_MINUTES), out_dir, step=2, resize=32
    )

    assert manifest["index"].tolist() == [0, 2, 4]
    assert iio.imread(manifest["frame_path"].iloc[0]).shape == (32, 32, 3)
    assert (out_dir / MANIFEST_FILENAME).is_file()


def _ambiguous_video(tmp_path: Path) -> Path:
    """Three frames where the middle one's minute cell is a 6/8 blend.

    The sequence says the middle frame is 12:08; the pixels, on their own, read
    12:06 with 8 a runner-up within the score margin. A minute apart because the
    manifest's sample_id has minute resolution and would otherwise keep one row
    of the three.
    """
    frames = np.stack(
        [
            fake.render_overlay_frame("20260810120700"),
            fake.blended_overlay_frame("20260810120800", "20260810120600"),
            fake.render_overlay_frame("20260810120900"),
        ]
    )
    return fake.write_frames_video(tmp_path / "allsky-20260810.mp4", frames)


def test_a_contested_cell_is_re_decided_from_the_pixels_of_a_real_frame(tmp_path: Path):
    """The machinery was only ever driven from a hand-built alternatives list, so
    ``_cell_scores`` and ``_cell_alternatives`` never produced a runner-up in any
    test and the path from pixels to a correction was unexercised end to end.
    """
    readings = read_video_timestamps(_ambiguous_video(tmp_path))

    assert [reading.text for reading in readings] == [
        "20260810120700",
        "20260810120600",
        "20260810120900",
    ]
    assert readings[1].timestamp == _naive("2026-08-10 12:08")
    assert [reading.corrected for reading in readings] == [False, True, False]


def test_the_corrected_bit_reaches_the_manifest_the_extraction_writes(tmp_path: Path):
    """``qc_frame_flags`` is a persisted column, and ``TIMESTAMP_CORRECTED``
    appeared in no test in the repo: the bit could have been lost in the staging
    rename without anything failing.
    """
    manifest = extract_frames_with_overlay_timestamps(
        _ambiguous_video(tmp_path), tmp_path / "frames"
    )

    flags = manifest["qc_frame_flags"].tolist()
    assert flags == [0, int(QCFlag.TIMESTAMP_CORRECTED), 0]
    assert manifest["timestamp"].tolist() == [
        pd.Timestamp("2026-08-10 12:07"),
        pd.Timestamp("2026-08-10 12:08"),
        pd.Timestamp("2026-08-10 12:09"),
    ]


@pytest.mark.parametrize("position", [0, 1, -1])
def test_an_unreadable_stamp_at_either_end_of_a_video_leaves_no_manifest_row(
    tmp_path: Path, position: int
):
    """Interpolation needs a reading on both sides, so a frame at the boundary
    is not repaired. What it does instead is documented here as it stands: the
    frame simply does not appear in the manifest — no row, no flag, no gap
    marker. Whether an unreadable boundary frame should be dropped or recorded
    as an acquisition gap is a decision about the record, not about this code.
    """
    stamps = [f"2026081012{minute:02d}00" for minute in range(10)]
    frames = [fake.render_overlay_frame(stamp) for stamp in stamps]
    # An impossible month: read digit for digit, refused as a date.
    frames[position] = fake.render_overlay_frame("20261310120000")
    video = fake.write_frames_video(tmp_path / "allsky-20260810.mp4", np.stack(frames))

    readings = read_video_timestamps(video)
    manifest = extract_frames_with_overlay_timestamps(video, tmp_path / "frames")

    unreadable = readings[position]
    interior = position not in (0, -1)
    assert (unreadable.timestamp is not None) is interior
    assert (unreadable.index in manifest["index"].tolist()) is interior
    assert len(manifest) == len(stamps) - (0 if interior else 1)


def test_a_lookalike_digit_is_re_decided_from_the_frames_around_it():
    readings = [
        OverlayReading(index=0, timestamp=_naive("2026-08-10 12:07"), text="20260810120700"),
        OverlayReading(index=1, timestamp=_naive("2026-08-10 12:06"), text="20260810120600"),
        OverlayReading(index=2, timestamp=_naive("2026-08-10 12:09"), text="20260810120900"),
    ]
    unambiguous = tuple(tuple(digit) for digit in "20260810120700")
    ambiguous = tuple(
        ("6", "8") if position == 11 else (digit,)
        for position, digit in enumerate("20260810120600")
    )
    corrected = _correct_against_neighbours(
        readings, [unambiguous, ambiguous, tuple(tuple(d) for d in "20260810120900")]
    )

    assert corrected[1].corrected is True
    assert corrected[1].timestamp == _naive("2026-08-10 12:08")
    assert corrected[1].text == "20260810120600"
    assert [item.corrected for item in corrected] == [False, True, False]


def test_a_reading_its_neighbours_agree_with_is_never_re_decided():
    readings = [
        OverlayReading(index=0, timestamp=_naive("2026-08-10 12:07"), text="20260810120700"),
        OverlayReading(index=1, timestamp=_naive("2026-08-10 12:08"), text="20260810120800"),
        OverlayReading(index=2, timestamp=_naive("2026-08-10 12:09"), text="20260810120900"),
    ]
    ambiguous = tuple(
        ("8", "6") if position == 11 else (digit,)
        for position, digit in enumerate("20260810120800")
    )
    corrected = _correct_against_neighbours(
        readings,
        [
            tuple(tuple(d) for d in "20260810120700"),
            ambiguous,
            tuple(tuple(d) for d in "20260810120900"),
        ],
    )

    assert [item.timestamp for item in corrected] == [item.timestamp for item in readings]
    assert not any(item.corrected for item in corrected)


def test_a_frame_with_no_overlay_at_all_reads_as_unknown_not_as_the_year_1111():
    blank = np.zeros((60, 480, 3), dtype=np.uint8)
    reading = read_frame_timestamp(blank)
    assert reading.timestamp is None
    assert reading.text == ""


def test_a_manufactured_timestamp_is_flagged_in_the_frame_manifest(tmp_path: Path):
    stamps = (
        "20260810120000",
        "20260810120100",
        "20261310120200",
        "20260810120300",
        "20260810120400",
        "20260810120500",
    )
    manifest = extract_frames_with_overlay_timestamps(_video(tmp_path, stamps), tmp_path / "frames")

    flags = manifest.set_index("index")["qc_frame_flags"]
    assert flags.loc[2] & int(QCFlag.TIMESTAMP_INTERPOLATED)
    assert not (flags.drop(index=2) & int(QCFlag.TIMESTAMP_INTERPOLATED)).any()


def test_the_configured_filename_date_format_decides_which_day_the_overlay_is_judged_against(
    tmp_path: Path,
):
    video = fake.write_overlay_video(tmp_path / "skycam_20260810.mp4", MINUTE_APART)
    config = VideoConfig(filename_date_format="skycam_%Y%m%d")

    manifest = extract_frames_for(video, tmp_path / "frames", config)

    assert list(manifest["timestamp"]) == [
        pd.Timestamp(_naive(f"2026-08-10 12:0{n}")) for n in "012"
    ]


def test_a_name_the_configured_format_does_not_describe_falls_back_to_the_allsky_naming(
    tmp_path: Path,
):
    config = VideoConfig(filename_date_format="skycam_%Y%m%d")

    manifest = extract_frames_for(_video(tmp_path, MINUTE_APART), tmp_path / "frames", config)

    assert list(manifest["timestamp"]) == [
        pd.Timestamp(_naive(f"2026-08-10 12:0{n}")) for n in "012"
    ]


def test_extraction_decodes_the_video_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    decodes = 0
    undecorated = iio.imiter

    def counting_imiter(*args, **kwargs):
        nonlocal decodes
        decodes += 1
        return undecorated(*args, **kwargs)

    monkeypatch.setattr(iio, "imiter", counting_imiter)

    extract_frames_with_overlay_timestamps(_video(tmp_path, MINUTE_APART), tmp_path / "frames")

    assert decodes == 1


def test_extraction_leaves_only_frames_and_the_manifest_in_the_output_directory(tmp_path: Path):
    out_dir = tmp_path / "frames"
    extract_frames_with_overlay_timestamps(_video(tmp_path, MINUTE_APART), out_dir)

    assert sorted(path.name for path in out_dir.iterdir()) == [
        "allsky-20260810-1200.jpg",
        "allsky-20260810-1201.jpg",
        "allsky-20260810-1202.jpg",
        MANIFEST_FILENAME,
    ]


def test_a_video_the_reader_will_not_vouch_for_leaves_no_frame_behind(tmp_path: Path):
    stamps = ("20260810120000", "20260810120200", "20260810120100")
    out_dir = tmp_path / "frames"
    with pytest.raises(OverlayTimestampError, match="go backwards"):
        extract_frames_with_overlay_timestamps(_video(tmp_path, stamps), out_dir)

    assert list(out_dir.rglob("*")) == []


def test_a_read_timestamp_carries_no_provenance_flag(tmp_path: Path):
    manifest = extract_frames_with_overlay_timestamps(
        _video(tmp_path, MINUTE_APART), tmp_path / "frames"
    )
    assert (manifest["qc_frame_flags"] == 0).all()


def test_an_unreadable_first_frame_is_reported_as_dropped_not_as_interpolated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """The repair interpolates only between two readable neighbours, so an edge
    frame stays unstamped and leaves the manifest; the warning used to count it
    among the interpolated ones and nothing said a frame was gone."""
    stamps = (
        "20261310120000",
        "20260810120100",
        "20260810120200",
        "20260810120300",
        "20260810120400",
        "20260810120500",
    )

    with caplog.at_level(logging.WARNING, logger="allsky.overlay"):
        readings = read_video_timestamps(_video(tmp_path, stamps))

    assert readings[0].timestamp is None
    assert "1 frame(s) at the edges" in caplog.text
    assert "were interpolated" not in caplog.text
