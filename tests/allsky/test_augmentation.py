"""Tests for allsky.augmentation.

The physical invariants are the point: an augmentation that quietly destroys the
red/blue ratio, moves the sun, or invents pixel values would corrupt the signal
it is supposed to leave alone.
"""

import numpy as np
import pytest

from allsky.augmentation import (
    AugmentationPipeline,
    exposure_jitter,
    polar_unwrap,
    random_erasing,
    sensor_noise,
    translate,
)


@pytest.fixture
def frame() -> np.ndarray:
    """A deterministic CHW float32 frame in [0, 1]."""
    return np.random.default_rng(0).random((3, 48, 48), dtype=np.float32)


class TestExposureJitter:
    def test_the_red_blue_ratio_is_invariant(self, frame: np.ndarray):
        """The R/B ratio is the classical cloud discriminator, so a gain that
        changed it would delete signal rather than add invariance. A
        channel-common gain in linear space leaves it untouched by construction.

        Equality is exact in real arithmetic; float32's gamma round trip costs
        ~3e-3, and clipping breaks it wherever the gain saturates a channel.
        """
        jittered = exposure_jitter(frame, np.random.default_rng(1))

        before = frame[0] / np.maximum(frame[2], 1e-6)
        after = jittered[0] / np.maximum(jittered[2], 1e-6)
        np.testing.assert_allclose(after, before, rtol=0.05, atol=5e-3)

    def test_it_actually_changes_the_brightness(self, frame: np.ndarray):
        jittered = exposure_jitter(frame, np.random.default_rng(1))

        assert not np.allclose(jittered, frame)

    def test_a_zero_range_is_the_identity(self, frame: np.ndarray):
        jittered = exposure_jitter(frame, np.random.default_rng(1), log2_range=0.0)

        np.testing.assert_allclose(jittered, frame, rtol=1e-5, atol=1e-6)

    def test_the_output_stays_in_the_unit_range(self, frame: np.ndarray):
        for seed in range(8):
            out = exposure_jitter(frame, np.random.default_rng(seed), log2_range=2.0)
            assert out.min() >= 0.0
            assert out.max() <= 1.0


class TestSensorNoise:
    def test_the_added_noise_has_zero_mean(self, frame: np.ndarray):
        sigma = 0.02
        four_standard_errors = 4.0 * sigma / np.sqrt(frame.size)

        noisy = sensor_noise(frame, np.random.default_rng(2), sigma=sigma)

        assert abs(float((noisy - frame).mean())) < four_standard_errors

    def test_the_noised_frame_stays_in_the_unit_range(self, frame: np.ndarray):
        noisy = sensor_noise(frame, np.random.default_rng(2), sigma=0.02)

        assert noisy.min() >= 0.0
        assert noisy.max() <= 1.0


class TestRandomErasing:
    def test_it_erases_a_region(self, frame: np.ndarray):
        erased = random_erasing(frame, np.random.default_rng(3), area_range=(0.05, 0.06))

        assert not np.array_equal(erased, frame)
        assert erased.shape == frame.shape

    def test_the_solar_disc_is_never_erased(self, frame: np.ndarray):
        """Occluding the sun is not a nuisance transform: it changes the physics
        the model is being asked about."""
        sun = (24, 24)
        for seed in range(40):
            erased = random_erasing(
                frame,
                np.random.default_rng(seed),
                area_range=(0.05, 0.06),
                keep_solar_disc=sun,
                disc_radius=6,
            )
            patch = slice(sun[0] - 6, sun[0] + 7), slice(sun[1] - 6, sun[1] + 7)
            np.testing.assert_array_equal(
                erased[:, patch[0], patch[1]], frame[:, patch[0], patch[1]]
            )


class TestTranslate:
    def test_it_replicates_the_edge_and_never_wraps(self):
        """Wrapping would paste the horizon from one side of the dome onto the
        other — the same physically impossible content that rules flips out."""
        marked = np.zeros((3, 16, 16), dtype=np.float32)
        marked[:, -1, :] = 1.0

        for seed in range(40):
            shifted = translate(marked, np.random.default_rng(seed), max_shift=3)
            assert shifted[:, :2, :].max() == 0.0, "bottom row reappeared at the top"

    def test_it_resamples_nothing(self, frame: np.ndarray):
        """A rigid shift with edge replication introduces no interpolated value."""
        shifted = translate(frame, np.random.default_rng(4), max_shift=3)

        assert set(np.unique(shifted).tolist()) <= set(np.unique(frame).tolist())

    def test_zero_max_shift_is_the_identity(self, frame: np.ndarray):
        assert np.array_equal(translate(frame, np.random.default_rng(4), max_shift=0), frame)


class TestPolarUnwrap:
    def test_it_invents_no_pixel_values(self, frame: np.ndarray):
        """Nearest-neighbour sampling only reuses values the sensor produced;
        an interpolating resampler would synthesise colours it never saw."""
        unwrapped = polar_unwrap(frame, sun_row=20.0, sun_col=30.0)

        assert set(np.unique(unwrapped).tolist()) <= set(np.unique(frame).tolist())

    def test_the_requested_output_shape_is_honoured(self, frame: np.ndarray):
        unwrapped = polar_unwrap(frame, sun_row=24.0, sun_col=24.0, out_shape=(32, 16))

        assert unwrapped.shape == (3, 32, 16)
        assert unwrapped.dtype == np.float32

    def test_the_angular_coordinate_advances_along_the_output_rows(self):
        """Modulo the single wrap of the seam, which is why two steps may not."""
        size = 64
        centre = size // 2
        rows, cols = np.mgrid[0:size, 0:size]
        angle = np.arctan2(cols - centre, rows - centre)
        base = ((angle + np.pi) / (2 * np.pi)).astype(np.float32)
        chw = np.stack([base] * 3)

        unwrapped = polar_unwrap(chw, sun_row=centre, sun_col=centre, out_shape=(64, 32))
        column = unwrapped[0, :, 10]

        steps = np.diff(column)
        assert (steps > 0).sum() >= len(steps) - 2

    def test_a_quarter_turn_about_the_sun_becomes_a_quarter_roll_of_the_output_rows(self):
        """The whole point of SPIN: rotational invariance around the sun turns
        into a translation the network already handles. An odd frame size puts
        the sun on a pixel, so a 90-degree turn is an exact lattice symmetry and
        the two unwraps must agree pixel for pixel, not merely closely."""
        size = 65
        centre = float(size // 2)
        n_theta = 64
        scene = np.random.default_rng(0).random((3, size, size), dtype=np.float32)
        turned = np.ascontiguousarray(np.rot90(scene, k=1, axes=(1, 2)))

        unwrapped = polar_unwrap(scene, sun_row=centre, sun_col=centre, out_shape=(n_theta, 32))
        turned_unwrapped = polar_unwrap(
            turned, sun_row=centre, sun_col=centre, out_shape=(n_theta, 32)
        )

        np.testing.assert_array_equal(turned_unwrapped, np.roll(unwrapped, n_theta // 4, axis=1))


class TestAugmentationPipeline:
    def test_the_default_pipeline_is_a_no_op(self, frame: np.ndarray):
        """Every probability defaults to zero so an existing experiment keeps
        its numbers until augmentation is asked for explicitly."""
        pipeline = AugmentationPipeline()

        assert not pipeline.enabled
        assert np.array_equal(pipeline(frame, np.random.default_rng(0)), frame)

    def test_the_same_seed_gives_the_same_frame(self, frame: np.ndarray):
        pipeline = AugmentationPipeline(p_exposure=1.0, p_noise=1.0, p_translate=1.0, p_erase=1.0)

        first = pipeline(frame, np.random.default_rng(11))
        second = pipeline(frame, np.random.default_rng(11))

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_frames(self, frame: np.ndarray):
        pipeline = AugmentationPipeline(p_exposure=1.0, p_noise=1.0)

        assert not np.array_equal(
            pipeline(frame, np.random.default_rng(11)),
            pipeline(frame, np.random.default_rng(12)),
        )

    def test_the_output_contract_survives_every_transform(self, frame: np.ndarray):
        pipeline = AugmentationPipeline(p_exposure=1.0, p_noise=1.0, p_translate=1.0, p_erase=1.0)

        out = pipeline(frame, np.random.default_rng(3))

        assert out.shape == frame.shape
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
        assert 0.0 <= out.min() <= out.max() <= 1.0


class TestAbsentPixelsAreNotSkySignal:
    """The ROI mask and the isotropic pad write exact zeros over regions the
    camera never imaged. Treated as dark readings, the noise turns "nothing was
    measured here" into a faint signal and the erasing fill is dragged toward
    black — an erasure darker than any sky."""

    @staticmethod
    def _frame_and_mask() -> tuple[np.ndarray, np.ndarray]:
        chw = np.full((3, 16, 16), 0.6, dtype=np.float32)
        valid = np.ones((16, 16), dtype=bool)
        valid[:, :8] = False  # the absent half
        chw[:, :, :8] = 0.0
        return chw, valid

    def test_noise_leaves_the_absent_pixels_exactly_zero(self):
        chw, valid = self._frame_and_mask()

        noisy = sensor_noise(chw, np.random.default_rng(0), sigma=0.05, valid=valid)

        assert np.array_equal(noisy[:, :, :8], np.zeros_like(noisy[:, :, :8]))
        assert not np.array_equal(noisy[:, :, 8:], chw[:, :, 8:])

    def test_without_the_mask_the_absent_pixels_are_noised(self):
        """Which is the behaviour the mask exists to stop."""
        chw, _valid = self._frame_and_mask()

        noisy = sensor_noise(chw, np.random.default_rng(0), sigma=0.05)

        assert not np.array_equal(noisy[:, :, :8], np.zeros_like(noisy[:, :, :8]))

    def test_the_erasing_fill_averages_only_the_imaged_pixels(self):
        chw, valid = self._frame_and_mask()

        erased = random_erasing(chw, np.random.default_rng(3), area_range=(0.5, 0.6), valid=valid)

        # The imaged half is a flat 0.6, so the fill must be 0.6 — not the 0.3
        # the whole-frame mean would give.
        written = erased[:, :, 8:][erased[:, :, 8:] != 0.6]
        assert written.size > 0, "the patch must land inside the imaged half"
        assert float(written.max()) == pytest.approx(0.6, abs=1e-6)
