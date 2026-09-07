"""Tests for the rotation about the zenith in allsky.augmentation.

The rotation is legal only because the solar-geometry planes turn with the
pixels, so what is pinned here is that every plane of the stack moves by the
same angle, that the angle is the one drawn, and that a pipeline without
rotation still produces the frames it produced before the transform existed.
"""

import hashlib

import numpy as np
import pytest

from allsky.augmentation import (
    AugmentationPipeline,
    rotate_about_zenith,
    rotate_frame,
    translate,
)

SIDE = 9


@pytest.fixture
def stack() -> np.ndarray:
    """RGB planes with one marked pixel at the top right, plus a geometry ramp."""
    frame = np.zeros((4, SIDE, SIDE), dtype=np.float32)
    frame[0, 1, SIDE - 2] = 1.0
    frame[3] = np.arange(SIDE * SIDE, dtype=np.float32).reshape(SIDE, SIDE) / (SIDE * SIDE)
    return frame


class TestRotateFrame:
    def test_a_quarter_turn_moves_the_marked_pixel_from_the_top_right_to_the_top_left(
        self, stack: np.ndarray
    ):
        turned = rotate_frame(stack, 90.0)

        np.testing.assert_array_equal(np.argwhere(turned[0] > 0.5), [[1, 1]])

    def test_a_quarter_turn_turns_the_geometry_plane_with_the_pixels(self, stack: np.ndarray):
        turned = rotate_frame(stack, 90.0)

        np.testing.assert_allclose(turned, np.rot90(stack, 1, axes=(1, 2)), atol=1e-5)

    def test_zero_degrees_is_the_identity_bit_for_bit(self, stack: np.ndarray):
        assert rotate_frame(stack, 0.0) is stack

    def test_the_uncovered_corners_take_the_per_plane_fill(self):
        frame = np.ones((2, SIDE, SIDE), dtype=np.float32)
        fill = np.array([-2.0, 0.5], dtype=np.float32).reshape(2, 1, 1)

        turned = rotate_frame(frame, 45.0, fill=fill)

        assert turned[0, 0, 0] == pytest.approx(-2.0)
        assert turned[1, 0, 0] == pytest.approx(0.5)
        assert turned[0, SIDE // 2, SIDE // 2] == pytest.approx(1.0)

    def test_a_batch_is_rotated_plane_by_plane_like_a_frame(self, stack: np.ndarray):
        batch = np.stack([stack, stack[::-1]])

        turned = rotate_frame(batch, 90.0)

        np.testing.assert_array_equal(turned[0], rotate_frame(stack, 90.0))
        np.testing.assert_array_equal(turned[1], rotate_frame(stack[::-1], 90.0))


class TestRotateAboutZenith:
    def test_the_angle_applied_is_the_one_drawn(self, stack: np.ndarray):
        angle = float(np.random.default_rng(5).uniform(-180.0, 180.0))

        turned = rotate_about_zenith(stack, np.random.default_rng(5), max_deg=180.0)

        assert not np.array_equal(turned, stack)
        np.testing.assert_array_equal(turned, rotate_frame(stack, angle))

    def test_a_zero_half_width_is_the_identity_bit_for_bit(self, stack: np.ndarray):
        assert rotate_about_zenith(stack, np.random.default_rng(5), max_deg=0.0) is stack

    def test_every_draw_stays_inside_the_half_width(self, stack: np.ndarray):
        turned = [
            rotate_about_zenith(stack, np.random.default_rng(seed), max_deg=10.0)
            for seed in range(6)
        ]
        marker_rows = [int(np.argwhere(frame[0] == frame[0].max())[0, 0]) for frame in turned]

        assert all(row <= 2 for row in marker_rows), marker_rows


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.default_rng(0).random((3, 48, 48), dtype=np.float32)


@pytest.fixture
def valid() -> np.ndarray:
    rows, cols = np.mgrid[:48, :48]
    mask: np.ndarray = np.hypot(rows - 23.5, cols - 23.5) <= 23.5
    return mask


#: md5 of the pipeline's output on the two fixtures above, recorded before
#: ``p_rotate`` existed. A pipeline that leaves rotation off must still draw
#: and produce exactly these bytes.
_FRAMES_BEFORE_ROTATION_EXISTED = {
    ("all", 7): "bd548701a0397c5582fed390802eb798",
    ("all", 8): "56f283868a05319b5590a82bfd61b9ce",
    ("half", 7): "4c4b56702af3a91b28689b4d3447bdb7",
    ("half", 8): "a036f4d005877327d1548794a46732c1",
}
_PIPELINES = {
    "all": AugmentationPipeline(
        p_exposure=1.0,
        exposure_log2=0.6,
        p_noise=1.0,
        noise_sigma=0.02,
        p_translate=1.0,
        translate_px=3,
        p_erase=1.0,
    ),
    "half": AugmentationPipeline(p_exposure=0.5, p_noise=0.5, p_translate=0.5, p_erase=0.5),
}


class TestAugmentationPipelineWithGeometry:
    @pytest.mark.parametrize(("label", "seed"), sorted(_FRAMES_BEFORE_ROTATION_EXISTED))
    def test_without_rotation_the_pipeline_produces_the_frames_it_produced_before(
        self, label: str, seed: int, frame: np.ndarray, valid: np.ndarray
    ):
        out = _PIPELINES[label](frame, np.random.default_rng(seed), valid)

        digest = hashlib.md5(np.ascontiguousarray(out).tobytes()).hexdigest()  # noqa: S324 — an oracle of bytes, not a credential
        assert digest == _FRAMES_BEFORE_ROTATION_EXISTED[label, seed]

    def test_the_rotation_turns_the_geometry_planes_with_the_frame(self, frame: np.ndarray):
        geometry = np.random.default_rng(1).random((2, 48, 48), dtype=np.float32)
        pipeline = AugmentationPipeline(p_rotate=1.0, rotate_max_deg=90.0)
        replay = np.random.default_rng(4)
        replay.random()
        angle = float(replay.uniform(-90.0, 90.0))

        out = pipeline(frame, np.random.default_rng(4), geometry=geometry)

        assert out.shape == (5, 48, 48)
        np.testing.assert_array_equal(out, rotate_frame(np.concatenate([frame, geometry]), angle))

    def test_the_translation_shifts_the_geometry_planes_with_the_frame(self, frame: np.ndarray):
        geometry = np.random.default_rng(1).random((1, 48, 48), dtype=np.float32)
        pipeline = AugmentationPipeline(p_translate=1.0, translate_px=3)
        replay = np.random.default_rng(4)
        replay.random()

        out = pipeline(frame, np.random.default_rng(4), geometry=geometry)

        np.testing.assert_array_equal(
            out, translate(np.concatenate([frame, geometry]), replay, max_shift=3)
        )

    def test_the_photometric_transforms_leave_the_geometry_planes_untouched(
        self, frame: np.ndarray
    ):
        geometry = np.random.default_rng(1).random((1, 48, 48), dtype=np.float32)
        pipeline = AugmentationPipeline(p_exposure=1.0, p_noise=1.0, p_erase=1.0)

        out = pipeline(frame, np.random.default_rng(4), geometry=geometry)

        assert not np.array_equal(out[:3], frame)
        np.testing.assert_array_equal(out[3:], geometry)

    def test_rotation_alone_enables_the_pipeline(self):
        assert AugmentationPipeline(p_rotate=0.5).enabled
        assert not AugmentationPipeline().enabled
