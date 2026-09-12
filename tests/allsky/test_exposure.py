"""Tests for allsky.exposure: the overlay exposure reader, the relative radiance
and the manifest join.

The overlay fixtures are real crops of the ``Exposure:`` line cut from archive
frames at native resolution and stored losslessly; each expected value is the
audit's own reading of the same ``(video, frame_index)`` (``exposure_*.parquet``
of 2026-08-28) unless the test name says a human read it.
"""

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.exposure import (
    BT709_LUMA_WEIGHTS,
    DN_FULL_SCALE,
    EXPOSURE_COL_SLICE,
    EXPOSURE_FEATURE_COLUMNS,
    EXPOSURE_ROW_SLICE,
    NATIVE_FRAME_HEIGHT,
    NATIVE_FRAME_WIDTH,
    SHUFFLED_FEATURE_COLUMNS,
    SRGB_GAMMA,
    ExposureRecord,
    attach_exposure_features,
    exposure_record,
    exposure_records_for_video,
    log2_relative_radiance,
    read_exposure_from_overlay_crop,
    read_exposure_seconds,
    records_frame,
    saturated_fraction,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_AUDIT_RED_FLOOR = 150


def _crop(name: str) -> np.ndarray:
    return np.asarray(iio.imread(_FIXTURES / f"{name}.png"))


def _native_frame(fill: int = 0) -> np.ndarray:
    return np.full((NATIVE_FRAME_HEIGHT, NATIVE_FRAME_WIDTH, 3), fill, dtype=np.uint8)


def _frame_with_overlay(name: str, fill: int = 0) -> np.ndarray:
    frame = _native_frame(fill)
    frame[EXPOSURE_ROW_SLICE, EXPOSURE_COL_SLICE] = _crop(name)
    return frame


@pytest.mark.parametrize(
    ("fixture", "expected_s"),
    [
        ("exposure_409us", 409e-6),
        ("exposure_1485us", 1485e-6),
        ("exposure_2_06ms", 2.06e-3),
        ("exposure_189us", 189e-6),
        pytest.param(
            "exposure_1_80ms",
            1.80e-3,
            marks=pytest.mark.xfail(
                strict=True,
                reason="the residual 8-as-6 misread the reader's docstring measures at "
                "about 0.5 % of frames: the thinned top of this 8 still reads 1.60 ms",
            ),
        ),
    ],
    ids=[
        "409 us",
        "1,485 us with a thousands comma",
        "2.06 ms",
        "189 us on a day outside the glyph bank",
        "1.80 ms on a day outside the glyph bank",
    ],
)
def test_a_real_overlay_crop_reads_the_exposure_the_audit_read(fixture: str, expected_s: float):
    assert read_exposure_from_overlay_crop(_crop(fixture)) == pytest.approx(expected_s, rel=1e-12)


def test_a_seconds_regime_crop_reads_the_1_4_sec_a_human_reads_on_it():
    assert read_exposure_from_overlay_crop(_crop("exposure_1_4sec")) == pytest.approx(1.4)


def test_the_crop_the_audit_recorded_as_4_391_s_reads_the_4_69_ms_it_shows():
    assert read_exposure_from_overlay_crop(_crop("exposure_4_69ms")) == pytest.approx(4.69e-3)


def test_a_unit_cut_off_by_the_crop_edge_is_unreadable_rather_than_a_wrong_unit():
    assert read_exposure_from_overlay_crop(_crop("exposure_unit_cut_off")) is None


def test_the_exposure_label_alone_has_no_unit_gap_and_is_unreadable():
    assert read_exposure_from_overlay_crop(_crop("exposure_label_only")) is None


def test_a_blank_crop_is_unreadable():
    assert read_exposure_from_overlay_crop(np.zeros((21, 105, 3), dtype=np.uint8)) is None


def test_text_with_red_at_or_below_150_is_no_longer_text():
    crop = _crop("exposure_409us").copy()
    crop[..., 0] = np.minimum(crop[..., 0], _AUDIT_RED_FLOOR)

    assert read_exposure_from_overlay_crop(crop) is None


def test_text_whose_red_does_not_dominate_the_other_channels_is_no_longer_text():
    crop = _crop("exposure_409us").copy()
    crop[..., 1] = crop[..., 0]

    assert read_exposure_from_overlay_crop(crop) is None


def test_a_crop_of_the_wrong_height_is_refused():
    with pytest.raises(ValueError, match="overlay crop"):
        read_exposure_from_overlay_crop(np.zeros((20, 105, 3), dtype=np.uint8))


def test_a_native_frame_is_read_through_the_pinned_overlay_region():
    assert read_exposure_seconds(_frame_with_overlay("exposure_409us")) == pytest.approx(409e-6)


def test_the_overlay_region_is_pinned_to_raw_pixels_so_a_shifted_line_is_not_found():
    frame = _native_frame()
    frame[EXPOSURE_ROW_SLICE.start + 30 : EXPOSURE_ROW_SLICE.stop + 30, EXPOSURE_COL_SLICE] = _crop(
        "exposure_409us"
    )

    assert read_exposure_seconds(frame) is None


@pytest.mark.parametrize("shape", [(540, 960, 3), (1080, 1920), (1080, 1920, 4)])
def test_a_frame_that_is_not_native_rgb_is_refused(shape: tuple[int, ...]):
    with pytest.raises(ValueError, match="native"):
        read_exposure_seconds(np.zeros(shape, dtype=np.uint8))


def test_a_float_frame_is_refused():
    with pytest.raises(ValueError, match="uint8"):
        read_exposure_seconds(np.zeros((NATIVE_FRAME_HEIGHT, NATIVE_FRAME_WIDTH, 3)))


def test_a_uniform_grey_disc_gives_the_closed_form_relative_radiance():
    grey = 128
    exposure_s = 0.01
    linear = (grey / DN_FULL_SCALE) ** SRGB_GAMMA
    expected = np.log2(sum(BT709_LUMA_WEIGHTS) * linear / exposure_s)

    assert log2_relative_radiance(_native_frame(grey), exposure_s) == pytest.approx(expected)


def test_doubling_the_exposure_time_lowers_the_relative_radiance_by_one_stop():
    frame = _native_frame(100)

    assert log2_relative_radiance(frame, 0.002) == pytest.approx(
        log2_relative_radiance(frame, 0.001) - 1.0
    )


def test_the_overlay_pixels_are_outside_the_disc_and_do_not_move_the_radiance():
    plain = _native_frame(60)
    stamped = _frame_with_overlay("exposure_409us", fill=60)

    assert log2_relative_radiance(stamped, 0.001) == pytest.approx(
        log2_relative_radiance(plain, 0.001), abs=0.0
    )


@pytest.mark.parametrize("exposure_s", [0.0, -0.001])
def test_a_non_positive_exposure_time_is_refused(exposure_s: float):
    with pytest.raises(ValueError, match="positive"):
        log2_relative_radiance(_native_frame(100), exposure_s)


@pytest.mark.parametrize(
    ("fill", "expected"), [(0, 0.0), (253, 0.0), (254, 1.0), (255, 1.0)], ids=str
)
def test_saturation_counts_a_disc_pixel_from_dn_254_up(fill: int, expected: float):
    assert saturated_fraction(_native_frame(fill)) == pytest.approx(expected, abs=0.0)


def test_a_record_carries_the_exposure_its_log_the_radiance_and_the_saturation():
    frame = _frame_with_overlay("exposure_409us", fill=128)

    record = exposure_record(frame, 7)

    assert record.frame_index == 7
    assert record.exposure_s == pytest.approx(409e-6)
    assert record.log2_exposure_s == pytest.approx(np.log2(409e-6))
    assert record.log2_relative_radiance == pytest.approx(log2_relative_radiance(frame, 409e-6))
    assert record.sat_frac == pytest.approx(0.0, abs=0.0)


def test_an_unreadable_frame_yields_none_for_every_exposure_field_but_keeps_saturation():
    record = exposure_record(_native_frame(255), 3)

    assert (record.exposure_s, record.log2_exposure_s, record.log2_relative_radiance) == (
        None,
        None,
        None,
    )
    assert record.sat_frac == pytest.approx(1.0, abs=0.0)


def _write_native_video(path: Path, n_frames: int) -> Path:
    frames = np.zeros((n_frames, NATIVE_FRAME_HEIGHT, NATIVE_FRAME_WIDTH, 3), dtype=np.uint8)
    iio.imwrite(path, frames, fps=1, macro_block_size=1)
    return path


def test_records_come_back_in_index_order_for_exactly_the_requested_frames(tmp_path: Path):
    video = _write_native_video(tmp_path / "allsky-20260101.mp4", 4)

    records = exposure_records_for_video(video, [3, 0])

    assert [record.frame_index for record in records] == [0, 3]
    assert all(record.exposure_s is None for record in records)


def test_a_video_that_ends_before_a_requested_frame_is_refused(tmp_path: Path):
    video = _write_native_video(tmp_path / "allsky-20260101.mp4", 2)

    with pytest.raises(ValueError, match="ended before frame"):
        exposure_records_for_video(video, [0, 5])


def test_a_video_at_another_resolution_is_refused_naming_the_frame(synthetic_video: Path):
    with pytest.raises(ValueError, match="frame 0: expected a native"):
        exposure_records_for_video(synthetic_video, [0])


def test_no_requested_frames_decodes_nothing(tmp_path: Path):
    assert exposure_records_for_video(tmp_path / "absent.mp4", []) == []


def test_records_frame_keeps_an_unreadable_frame_as_an_explicit_missing_value():
    frame = records_frame(
        "allsky-20260101.mp4",
        [ExposureRecord(0, 1e-3, -9.9658, 8.5, 0.0), ExposureRecord(1, None, None, None, 0.2)],
    )

    assert list(frame.columns) == ["video", "frame_index", *EXPOSURE_FEATURE_COLUMNS]
    assert frame["exposure_s"].isna().tolist() == [False, True]
    assert str(frame["exposure_s"].dtype) == "Float64"
    assert frame["sat_frac"].tolist() == pytest.approx([0.0, 0.2])


def _manifest(splits: Sequence[str | None]) -> pd.DataFrame:
    n = len(splits)
    return pd.DataFrame(
        {
            "sample_id": pd.array([f"allsky-20260101-{i:04d}" for i in range(n)], dtype="string"),
            "video": pd.array(["allsky-20260101.mp4"] * n, dtype="string"),
            "frame_index": np.arange(n, dtype=np.int64),
            "split": pd.array(list(splits), dtype="string"),
            "target_dhi": np.linspace(50.0, 300.0, n),
        }
    )


def _records(n: int, unreadable: tuple[int, ...] = (), offset: float = 0.0) -> pd.DataFrame:
    rows = [
        ExposureRecord(i, None, None, None, 0.1)
        if i in unreadable
        else ExposureRecord(i, 1e-3 * (i + 1), float(i) + offset, 100.0 + i + offset, 0.1)
        for i in range(n)
    ]
    return records_frame("allsky-20260101.mp4", rows)


def test_unreadable_rows_are_removed_and_named_never_imputed():
    manifest = _manifest(["train"] * 6)

    join = attach_exposure_features(manifest, _records(6, unreadable=(1, 4)), seed=0)

    assert join.removed_sample_ids == ("allsky-20260101-0001", "allsky-20260101-0004")
    assert join.manifest["frame_index"].tolist() == [0, 2, 3, 5]
    assert not join.manifest[list(EXPOSURE_FEATURE_COLUMNS)].isna().any().any()


def test_a_manifest_row_without_a_record_is_an_error_not_a_missing_value():
    manifest = _manifest(["train"] * 4)

    with pytest.raises(ValueError, match="no exposure record"):
        attach_exposure_features(manifest, _records(3), seed=0)


def test_a_frame_recorded_twice_is_refused():
    manifest = _manifest(["train"] * 2)
    doubled = pd.concat([_records(2), _records(2)], ignore_index=True)

    with pytest.raises(ValueError, match="more than once"):
        attach_exposure_features(manifest, doubled, seed=0)


def test_the_source_columns_and_their_dtypes_survive_the_join():
    manifest = _manifest(["train", "val"])

    join = attach_exposure_features(manifest, _records(2), seed=0)

    assert list(join.manifest.columns) == [
        *manifest.columns,
        *EXPOSURE_FEATURE_COLUMNS,
        *SHUFFLED_FEATURE_COLUMNS,
    ]
    assert str(join.manifest["sample_id"].dtype) == "string"
    assert str(join.manifest["frame_index"].dtype) == "int64"


def _split_manifest() -> pd.DataFrame:
    return _manifest(["train"] * 8 + ["val"] * 8 + ["test"] * 8)


def test_the_shuffled_control_permutes_each_split_among_its_own_rows():
    manifest = _split_manifest()

    join = attach_exposure_features(manifest, _records(24), seed=0)

    for split in ("train", "val", "test"):
        rows = join.manifest[join.manifest["split"] == split]
        assert sorted(rows["log2_exposure_s_shuffled"]) == sorted(rows["log2_exposure_s"])
        assert sorted(rows["log2_relative_radiance_shuffled"]) == sorted(
            rows["log2_relative_radiance"]
        )


def test_the_shuffled_control_is_not_the_original_order():
    manifest = _split_manifest()

    join = attach_exposure_features(manifest, _records(24), seed=0)

    for split in ("train", "val", "test"):
        rows = join.manifest[join.manifest["split"] == split]
        assert rows["log2_exposure_s_shuffled"].tolist() != rows["log2_exposure_s"].tolist()


def test_both_shuffled_columns_move_together_so_their_pairs_stay_real():
    manifest = _split_manifest()

    join = attach_exposure_features(manifest, _records(24), seed=0)

    original_pairs = set(
        zip(join.manifest["log2_exposure_s"], join.manifest["log2_relative_radiance"], strict=True)
    )
    shuffled_pairs = set(
        zip(
            join.manifest["log2_exposure_s_shuffled"],
            join.manifest["log2_relative_radiance_shuffled"],
            strict=True,
        )
    )
    assert shuffled_pairs == original_pairs


def test_rows_without_a_split_label_are_shuffled_only_among_themselves():
    manifest = _manifest(["train"] * 8 + [None] * 8)

    join = attach_exposure_features(manifest, _records(16), seed=0)

    unlabelled = join.manifest[join.manifest["split"].isna()]
    assert sorted(unlabelled["log2_exposure_s_shuffled"]) == sorted(unlabelled["log2_exposure_s"])


def test_the_same_seed_reproduces_the_control_and_another_seed_changes_it():
    manifest = _split_manifest()

    first = attach_exposure_features(manifest, _records(24), seed=3).manifest
    again = attach_exposure_features(manifest, _records(24), seed=3).manifest
    other = attach_exposure_features(manifest, _records(24), seed=4).manifest

    assert first["log2_exposure_s_shuffled"].tolist() == again["log2_exposure_s_shuffled"].tolist()
    assert first["log2_exposure_s_shuffled"].tolist() != other["log2_exposure_s_shuffled"].tolist()


def test_the_join_reports_the_seed_it_shuffled_with():
    join = attach_exposure_features(_manifest(["train"] * 3), _records(3), seed=11)

    assert join.seed == 11


@pytest.mark.parametrize("column", ["sample_id", "video", "frame_index", "split"])
def test_a_manifest_lacking_a_join_or_split_column_is_refused(column: str):
    manifest = _manifest(["train"] * 2).drop(columns=column)

    with pytest.raises(KeyError, match=column):
        attach_exposure_features(manifest, _records(2), seed=0)
