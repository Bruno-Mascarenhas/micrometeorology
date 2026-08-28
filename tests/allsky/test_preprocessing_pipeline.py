"""Tests for the deterministic preprocessing pipeline.

The contract that matters is symmetry: preprocessing must be applied identically
at training and inference, and its identity must change whenever the pixels do.
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

    def test_the_identity_changes_with_every_setting(self):
        identities = {
            PreprocessingPipeline().identity,
            PreprocessingPipeline(overlay="inpaint").identity,
            PreprocessingPipeline(overlay="fill").identity,
            PreprocessingPipeline(overlay="inpaint", band_fraction=0.2).identity,
            PreprocessingPipeline(overlay="inpaint", roi_radius_fraction=0.98).identity,
        }

        assert len(identities) == 5

    def test_the_identity_is_stable_across_instances(self):
        """It is written into the run and compared at load, so it must not
        depend on object identity or on dict ordering."""
        assert (
            PreprocessingPipeline(overlay="inpaint").identity
            == PreprocessingPipeline(overlay="inpaint").identity
        )

    def test_the_roi_zeroes_the_corners_and_keeps_the_dome(self, frame: np.ndarray):
        out = PreprocessingPipeline(roi_radius_fraction=0.95)(frame)

        assert out[0, 0, 0] == 0.0
        assert out[0, 112, 112] == pytest.approx(frame[0, 112, 112])

    def test_the_output_contract_holds(self, frame: np.ndarray):
        out = PreprocessingPipeline(overlay="inpaint", roi_radius_fraction=0.98)(frame)

        assert out.shape == frame.shape
        assert out.dtype == np.float32
        assert out.flags["C_CONTIGUOUS"]
