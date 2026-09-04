"""Tests for allsky.preprocessing — mask, crop, resize and visual QC.

Pure numpy/PIL: no torch, no network, synthetic arrays only.
"""

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from allsky.config import CropConfig, PadConfig, PrepareConfig
from allsky.data.contracts import QCFlag
from allsky.preprocessing import (
    _needs_preprocessing,
    apply_static_mask,
    center_crop,
    estimate_circular_mask,
    load_mask,
    pad_frame,
    process_frame,
    resize_image,
    resolve_mask,
    visual_qc,
)


def _rgb(height: int = 32, width: int = 48, fill: int = 128) -> np.ndarray:
    """Constant-fill ``(H, W, 3)`` uint8 RGB image."""
    return np.full((height, width, 3), fill, dtype=np.uint8)


class TestResizeImage:
    def test_square_int_size(self):
        out = resize_image(_rgb(32, 48), 16)
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_tuple_size_is_width_height(self):
        # PIL resize takes (width, height); the array is (height, width, 3).
        out = resize_image(_rgb(32, 48), (40, 20))
        assert out.shape == (20, 40, 3)
        assert out.dtype == np.uint8


class TestCenterCrop:
    def test_disabled_is_identity(self):
        image = _rgb(32, 48)
        out = center_crop(image, CropConfig(enabled=False, height=8, width=8))
        assert np.array_equal(out, image)

    def test_centered_box_shape_and_dtype(self):
        out = center_crop(_rgb(32, 48), CropConfig(enabled=True, height=16, width=16))
        assert out.shape == (16, 16, 3)
        assert out.dtype == np.uint8

    def test_offsets_shift_and_clip(self):
        # A huge top offset clips to keep the box inside the frame.
        out = center_crop(_rgb(32, 48), CropConfig(enabled=True, height=16, width=16, top=100))
        assert out.shape == (16, 16, 3)

    def test_none_dims_fall_back_to_full_extent(self):
        out = center_crop(_rgb(32, 48), CropConfig(enabled=True))
        assert out.shape == (32, 48, 3)


class TestEstimateCircularMask:
    def test_center_kept_corners_dropped(self):
        mask = estimate_circular_mask((32, 32))
        assert mask.shape == (32, 32)
        assert mask.dtype == np.bool_
        assert mask[16, 16]  # centre kept
        assert not mask[0, 0]  # corner outside the inscribed disc

    def test_radius_fraction_shrinks_disc(self):
        big = estimate_circular_mask((40, 40), radius_fraction=1.0)
        small = estimate_circular_mask((40, 40), radius_fraction=0.5)
        assert small.sum() < big.sum()


class TestApplyStaticMask:
    def test_auto_circular_zeros_corners(self):
        image = _rgb(32, 32, fill=200)
        out = apply_static_mask(image, None)
        assert out.shape == image.shape
        assert out.dtype == np.uint8
        assert (out[0, 0] == 0).all()  # corner masked out
        assert (out[16, 16] == 200).all()  # centre preserved

    def test_boolean_array_mask(self):
        image = _rgb(4, 4, fill=255)
        keep = np.zeros((4, 4), dtype=bool)
        keep[1:3, 1:3] = True
        out = apply_static_mask(image, keep)
        assert (out[0, 0] == 0).all()
        assert (out[1, 1] == 255).all()

    def test_png_path_mask_roundtrip(self, tmp_path: Path):
        from PIL import Image

        keep = np.zeros((8, 8), dtype=np.uint8)
        keep[2:6, 2:6] = 255
        mask_path = tmp_path / "mask.png"
        Image.fromarray(keep, mode="L").save(mask_path)

        loaded = load_mask(mask_path)
        assert loaded.dtype == np.bool_
        assert loaded[3, 3]
        assert not loaded[0, 0]

        out = apply_static_mask(_rgb(8, 8, fill=100), mask_path)
        assert (out[0, 0] == 0).all()
        assert (out[3, 3] == 100).all()

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="does not match"):
            apply_static_mask(_rgb(8, 8), np.ones((4, 4), dtype=bool))

    def test_non_rgb_raises(self):
        with pytest.raises(ValueError, match="RGB image"):
            apply_static_mask(np.zeros((8, 8), dtype=np.uint8), None)

    def test_non_uint8_raises(self):
        with pytest.raises(TypeError, match="uint8"):
            apply_static_mask(np.zeros((8, 8, 3), dtype=np.float32), None)


class TestVisualQC:
    def test_dark_frame_flagged(self):
        flags = visual_qc(_rgb(16, 16, fill=3))
        assert flags == {QCFlag.FRAME_DARK}

    def test_saturated_frame_flagged(self):
        flags = visual_qc(_rgb(16, 16, fill=255))
        assert flags == {QCFlag.FRAME_SATURATED}

    def test_normal_frame_has_no_flags(self):
        assert visual_qc(_rgb(16, 16, fill=128)) == set()

    def test_partial_saturation_below_threshold(self):
        image = _rgb(10, 10, fill=128)
        image[0, :] = 255  # 10% of pixels saturated < 20% default threshold
        assert QCFlag.FRAME_SATURATED not in visual_qc(image)

    @pytest.mark.parametrize("shape", [(0, 0, 3), (0, 4, 3), (4, 0, 3)])
    def test_pixelless_frame_is_flagged_unreadable_not_clean(self, shape: tuple[int, int, int]):
        flags = visual_qc(np.zeros(shape, dtype=np.uint8))
        assert flags == {QCFlag.FRAME_UNREADABLE}

    @pytest.mark.parametrize("saturated_level", [0, 1, 128, 200, 254, 255])
    @pytest.mark.parametrize("saturated_rows", [0, 1, 2, 3, 5, 10])
    def test_saturated_fraction_matches_the_channel_axis_reduction(
        self, saturated_level: int, saturated_rows: int
    ):
        # The counted form must agree with the (arr >= level).all(axis=2).mean()
        # reduction it replaced, exactly, at every threshold — including rows
        # straddling the 0.2 default fraction.
        image = _rgb(10, 10, fill=128)
        image[:saturated_rows] = 255
        expected = float((image >= saturated_level).all(axis=2).mean())
        for threshold in (0.0, 0.1, expected, 0.2, 0.5, 1.0):
            flagged = QCFlag.FRAME_SATURATED in visual_qc(
                image,
                saturated_level=saturated_level,
                saturated_fraction_threshold=threshold,
            )
            assert flagged == (expected > threshold)


def test_dark_flag_needs_no_float64_copy_of_the_frame():
    frame = _rgb(1024, 1024, fill=200)

    tracemalloc.start()
    visual_qc(frame)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak_bytes < 2 * frame.nbytes


@pytest.mark.parametrize("fill", [0, 3, 9, 10, 11, 128, 255])
def test_dark_flag_matches_the_float64_luminance_of_the_whole_frame(fill: int):
    rng = np.random.default_rng(0)
    frame = _rgb(24, 32, fill=fill)
    frame[0] = rng.integers(0, 256, (32, 3), dtype=np.uint8)
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float64)
    expected = float((frame.astype(np.float64) @ weights).mean())

    for threshold in (expected - 1e-6, expected + 1e-6):
        flagged = QCFlag.FRAME_DARK in visual_qc(frame, dark_threshold=threshold)
        assert flagged == (expected < threshold)


class TestProcessFrame:
    def test_default_config_is_identity(self):
        image = _rgb(24, 24, fill=77)
        out = process_frame(image, PrepareConfig())
        assert np.array_equal(out, image)

    def test_resize_only(self):
        cfg = PrepareConfig.model_validate({"resize": 16})
        out = process_frame(_rgb(32, 32), cfg)
        assert out.shape == (16, 16, 3)

    def test_crop_then_resize(self):
        cfg = PrepareConfig.model_validate(
            {"crop": {"enabled": True, "height": 20, "width": 20}, "resize": 8}
        )
        out = process_frame(_rgb(32, 48), cfg)
        assert out.shape == (8, 8, 3)

    def test_png_mask_applied(self, tmp_path: Path):
        from PIL import Image

        keep = np.full((16, 16), 255, dtype=np.uint8)
        keep[0, 0] = 0
        mask_path = tmp_path / "m.png"
        Image.fromarray(keep, mode="L").save(mask_path)
        cfg = PrepareConfig.model_validate({"mask": {"path": str(mask_path)}})
        out = process_frame(_rgb(16, 16, fill=210), cfg)
        assert (out[0, 0] == 0).all()
        assert (out[8, 8] == 210).all()


class TestResolveMask:
    @staticmethod
    def _mask_config(tmp_path: Path) -> PrepareConfig:
        from PIL import Image

        keep = np.full((16, 16), 255, dtype=np.uint8)
        keep[0, 0] = 0
        mask_path = tmp_path / "m.png"
        Image.fromarray(keep, mode="L").save(mask_path)
        return PrepareConfig.model_validate({"mask": {"path": str(mask_path)}})

    def test_no_mask_path_resolves_to_none(self):
        assert resolve_mask(PrepareConfig()) is None

    def test_resolved_mask_is_read_only(self, tmp_path: Path):
        keep = resolve_mask(self._mask_config(tmp_path))
        assert keep is not None
        # The same buffer is shared by every frame of the run.
        with pytest.raises(ValueError, match="read-only"):
            keep[0, 0] = True

    def test_preresolved_mask_is_byte_identical_to_the_path(self, tmp_path: Path):
        cfg = self._mask_config(tmp_path)
        image = _rgb(16, 16, fill=210)
        assert np.array_equal(
            process_frame(image, cfg, mask=resolve_mask(cfg)), process_frame(image, cfg)
        )

    def test_mask_none_never_falls_back_to_the_circular_estimate(self):
        # process_frame must not silently zero pixels when no mask is configured.
        image = _rgb(24, 24, fill=77)
        assert np.array_equal(process_frame(image, PrepareConfig(), mask=None), image)

    def test_extract_step_decodes_the_mask_once_per_video(self, tmp_path: Path, monkeypatch):
        from allsky import preprocessing

        cfg = self._mask_config(tmp_path)
        decodes = 0
        original = preprocessing.load_mask

        def counting_load_mask(path, **kwargs):
            nonlocal decodes
            decodes += 1
            return original(path, **kwargs)

        monkeypatch.setattr(preprocessing, "load_mask", counting_load_mask)
        mask = preprocessing.resolve_mask(cfg)
        for _ in range(5):
            process_frame(_rgb(16, 16, fill=210), cfg, mask=mask)
        assert decodes == 1


class TestPadFrame:
    def test_a_disabled_pad_hands_the_frame_straight_back(self) -> None:
        frame = np.zeros((4, 6, 3), dtype=np.uint8)

        assert pad_frame(frame, PadConfig()) is frame

    def test_each_side_grows_by_its_own_amount(self) -> None:
        """The four sides are independent because the sky disc is not concentric
        with the sensor: centring it in the output takes unequal padding."""
        frame = np.full((10, 20, 3), 7, dtype=np.uint8)

        padded = pad_frame(frame, PadConfig(enabled=True, top=3, bottom=5, left=1, right=2))

        assert padded.shape == (18, 23, 3)
        assert padded[3, 1, 0] == 7
        assert padded[0, 0, 0] == 0

    def test_the_fill_level_is_written_not_the_edge_pixel(self) -> None:
        """Padded rows are sky the camera does not image, so they must not
        replicate a measured pixel that a reader could take for one."""
        frame = np.full((4, 4, 3), 200, dtype=np.uint8)

        padded = pad_frame(frame, PadConfig(enabled=True, top=2, fill=13))

        assert padded[0, 0, 0] == 13
        assert padded[1, 3, 2] == 13


def test_a_pad_only_config_rewrites_the_frame():
    """``prepare-local`` asks this gate before rewriting a JPEG; a config whose only
    pixel stage is ``pad`` extracted unpadded frames without a word."""
    cfg = PrepareConfig.model_validate({"pad": {"enabled": True, "top": 4}})

    assert _needs_preprocessing(cfg) is True
    assert process_frame(_rgb(8, 8), cfg).shape == (12, 8, 3)
