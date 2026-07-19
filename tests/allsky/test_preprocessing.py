"""Tests for allsky.preprocessing — mask, crop, resize and visual QC.

Pure numpy/PIL: no torch, no network, synthetic arrays only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from allsky.config import CropConfig, PrepareConfig
from allsky.data.contracts import QCFlag
from allsky.preprocessing import (
    apply_static_mask,
    center_crop,
    estimate_circular_mask,
    load_mask,
    process_frame,
    resize_image,
    visual_qc,
)

if TYPE_CHECKING:
    from pathlib import Path


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

    def test_thresholds_are_overridable(self):
        image = _rgb(10, 10, fill=128)
        image[:2, :] = 255  # 20% saturated
        assert QCFlag.FRAME_SATURATED in visual_qc(image, saturated_fraction_threshold=0.1)


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
