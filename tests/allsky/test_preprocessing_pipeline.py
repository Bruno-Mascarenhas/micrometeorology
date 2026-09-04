"""Tests for the deterministic preprocessing pipeline.

The contract that matters is symmetry: preprocessing must be applied identically
at training and inference, so the settings travel in the checkpoint's config.
"""

import numpy as np
import pytest

from allsky.preprocessing import (
    IMAGENET_MEAN,
    TIMESTAMP_BAND_FRACTION,
    OverlayPolicy,
    PreprocessingPipeline,
    imagenet_standardize,
    remove_timestamp_band,
)


@pytest.fixture
def frame() -> np.ndarray:
    """A deterministic CHW float32 frame in [0, 1]."""
    return np.random.default_rng(0).random((3, 224, 224), dtype=np.float32)


class TestRemoveTimestampBand:
    def test_keep_returns_the_frame_untouched(self, frame: np.ndarray):
        assert remove_timestamp_band(frame, policy="keep") is frame

    @pytest.mark.parametrize("policy", ["fill", "inpaint"])
    def test_the_sky_below_the_band_is_never_touched(
        self, frame: np.ndarray, policy: OverlayPolicy
    ):
        """Only the overlay band may change: the rest is the signal."""
        out = remove_timestamp_band(frame, policy=policy)

        band = round(TIMESTAMP_BAND_FRACTION * frame.shape[1])
        np.testing.assert_array_equal(out[:, band:], frame[:, band:])
        assert not np.array_equal(out[:, :band], frame[:, :band])

    def test_fill_writes_the_normalisation_mean_so_the_band_vanishes(self, frame: np.ndarray):
        """The fill value is chosen so that after ImageNet standardisation the
        band is exactly zero — the cheapest possible token for the backbone."""
        band = round(TIMESTAMP_BAND_FRACTION * frame.shape[1])
        filled = remove_timestamp_band(frame, policy="fill")

        for channel, mean in enumerate(IMAGENET_MEAN):
            np.testing.assert_allclose(filled[channel, :band], mean, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            np.abs(imagenet_standardize(filled)[:, :band]).max(), 0.0, atol=1e-5
        )

    def test_inpaint_is_smoother_at_the_seam_but_that_is_not_the_criterion(self, frame: np.ndarray):
        """Mirroring does leave a smaller discontinuity than a constant fill —
        and is still not the default. A diffuse-irradiance regressor consumes
        cloud texture, and mirroring fabricates plausible sky texture across
        ~15% of the frame: a deterministic hallucination the model reads as real
        cloud. The constant band is uglier and honest."""
        band = round(TIMESTAMP_BAND_FRACTION * frame.shape[1])
        filled = remove_timestamp_band(frame, policy="fill")
        painted = remove_timestamp_band(frame, policy="inpaint")

        def seam_jump(a: np.ndarray) -> float:
            return float(np.abs(a[:, band] - a[:, band - 1]).mean())

        assert seam_jump(painted) < seam_jump(filled)

    def test_crop_shifts_the_scene_and_says_so(self, frame: np.ndarray):
        out = remove_timestamp_band(frame, policy="crop")

        assert out.shape == frame.shape
        np.testing.assert_array_equal(out[:, 0], frame[:, round(TIMESTAMP_BAND_FRACTION * 224)])

    def test_a_zero_band_is_refused_rather_than_painting_one_row(self, frame: np.ndarray):
        """`max(1, ...)` turned "no band" into one painted row, so an ablation that
        zeroes the fraction to switch the band off got an altered frame instead."""
        with pytest.raises(ValueError, match="covers no row"):
            remove_timestamp_band(frame, policy="fill", band_fraction=0.0)

    def test_a_band_deeper_than_half_the_frame_repeats_the_row_just_below_it(
        self, frame: np.ndarray
    ):
        """With fewer rows below the band than the band is tall there is nothing
        left to mirror, so the first row below it is repeated instead."""
        band_fraction = 0.6
        band = round(band_fraction * frame.shape[1])

        painted = remove_timestamp_band(frame, policy="inpaint", band_fraction=band_fraction)

        np.testing.assert_array_equal(
            painted[:, :band], np.repeat(frame[:, band : band + 1, :], band, axis=1)
        )

    def test_a_band_that_covers_the_whole_frame_is_refused(self, frame: np.ndarray):
        with pytest.raises(ValueError, match="covers the whole frame"):
            remove_timestamp_band(frame, policy="fill", band_fraction=1.0)

    def test_an_unknown_policy_is_rejected(self, frame: np.ndarray):
        with pytest.raises(ValueError, match="unknown overlay policy"):
            remove_timestamp_band(frame, policy="blur")  # type: ignore[arg-type]


class TestPreprocessingPipeline:
    def test_the_default_is_a_no_op(self, frame: np.ndarray):
        pipeline = PreprocessingPipeline()

        assert not pipeline.enabled
        assert np.array_equal(pipeline(frame), frame)

    def test_it_is_deterministic(self, frame: np.ndarray):
        """No generator argument at all: the same frame in must give the same
        frame out, wherever and whenever it runs."""
        pipeline = PreprocessingPipeline(overlay="inpaint", roi_radius_fraction=0.98)

        np.testing.assert_array_equal(pipeline(frame), pipeline(frame))

    def test_the_roi_zeroes_the_corners_and_keeps_the_dome(self, frame: np.ndarray):
        out = PreprocessingPipeline(roi_radius_fraction=0.95)(frame)

        assert out[0, 0, 0] == 0.0
        assert out[0, 112, 112] == pytest.approx(frame[0, 112, 112])

    def test_the_output_contract_holds(self, frame: np.ndarray):
        out = PreprocessingPipeline(overlay="inpaint", roi_radius_fraction=0.98)(frame)

        assert out.shape == frame.shape
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]


class TestUint8Route:
    """The fast route the dataloader takes must be the slow route's twin."""

    @pytest.mark.parametrize("overlay", ["keep", "fill", "inpaint", "crop"])
    @pytest.mark.parametrize("roi", [None, 0.5, 0.98])
    @pytest.mark.parametrize("band_fraction", [0.05, 0.16, 0.6])
    def test_it_produces_the_same_pixels_as_the_float_route(
        self, overlay: OverlayPolicy, roi: float | None, band_fraction: float
    ) -> None:
        """Every stage is a constant write, a data move, or a multiply by exactly
        0 or 1, so the two routes agree on pixels rather than merely on looks."""
        hwc = np.random.default_rng(5).integers(0, 256, (96, 64, 3), dtype=np.uint8)
        pipeline = PreprocessingPipeline(
            overlay=overlay, roi_radius_fraction=roi, band_fraction=band_fraction
        )

        native = hwc.astype(np.float32) / 255.0
        through_float = (
            (pipeline(native.transpose(2, 0, 1)).transpose(1, 2, 0) * 255.0)
            .round()
            .astype(np.uint8)
        )

        np.testing.assert_array_equal(pipeline.apply_uint8_hwc(hwc), through_float)

    def test_it_never_mutates_the_frame_it_was_given(self) -> None:
        hwc = np.random.default_rng(6).integers(0, 256, (48, 48, 3), dtype=np.uint8)
        before = hwc.copy()

        PreprocessingPipeline(overlay="fill", roi_radius_fraction=0.9).apply_uint8_hwc(hwc)

        np.testing.assert_array_equal(hwc, before)

    def test_a_disabled_pipeline_hands_the_frame_straight_back(self) -> None:
        hwc = np.zeros((8, 8, 3), dtype=np.uint8)

        assert PreprocessingPipeline().apply_uint8_hwc(hwc) is hwc


class TestInPlaceStandardisation:
    def test_it_matches_the_copying_form_bit_for_bit(self) -> None:
        frame = np.random.default_rng(7).random((3, 32, 32), dtype=np.float32)

        np.testing.assert_array_equal(
            imagenet_standardize(frame.copy()),
            imagenet_standardize(frame.copy(), copy=False),
        )

    def test_it_writes_into_the_buffer_it_was_given(self) -> None:
        frame = np.random.default_rng(8).random((3, 8, 8), dtype=np.float32)

        assert imagenet_standardize(frame, copy=False) is frame

    def test_it_refuses_a_dtype_it_cannot_standardize_in_place(self) -> None:
        with pytest.raises(TypeError, match="float32"):
            imagenet_standardize(np.zeros((3, 4, 4), dtype=np.float64), copy=False)
