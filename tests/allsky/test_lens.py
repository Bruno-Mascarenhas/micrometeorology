"""Tests for allsky.lens.

The point of the type is that one description of the optics answers both
questions asked of it — where a direction lands, and which pixels are sky. A
mask and a projection that disagreed about the same lens would be the drift the
module exists to prevent.
"""

import numpy as np
import pytest

from allsky.lens import PLANETARIO_NATIVE, LensCalibration, isotropic_calibration


@pytest.fixture
def lente() -> LensCalibration:
    return LensCalibration(centre_row=112.0, centre_col=112.0, radius_px=100.0)


class TestPixelOf:
    def test_the_zenith_maps_to_the_optical_centre(self, lente: LensCalibration):
        assert lente.pixel_of(0.0, 0.0) == pytest.approx((112.0, 112.0))

    def test_the_horizon_maps_to_the_edge_of_the_dome(self, lente: LensCalibration):
        assert lente.pixel_of(np.pi / 2, 0.0) == pytest.approx((12.0, 112.0))

    def test_east_falls_to_the_left_because_the_camera_looks_up(self, lente: LensCalibration):
        """A dome seen from below is mirrored against a map seen from above.
        Fitting the Planetario sun track unmirrored puts the optical centre
        outside the frame and quintuples the residual."""
        assert lente.pixel_of(np.pi / 2, np.pi / 2) == pytest.approx((112.0, 12.0))

    def test_the_mount_rotation_is_a_separate_parameter_from_the_lens(self, lente: LensCalibration):
        turned = LensCalibration(
            centre_row=112.0, centre_col=112.0, radius_px=100.0, azimuth_offset_rad=np.pi / 2
        )

        assert turned.pixel_of(np.pi / 2, 0.0) == pytest.approx(
            lente.pixel_of(np.pi / 2, np.pi / 2)
        )

    def test_the_two_projection_laws_differ_away_from_the_centre(self):
        common = {"centre_row": 100.0, "centre_col": 100.0, "radius_px": 90.0}
        equidistant = LensCalibration(**common, equidistant=True)
        equisolid = LensCalibration(**common, equidistant=False)

        one_pixel = 1.0
        off_axis_equidistant = equidistant.pixel_of(np.pi / 4, 0.0)
        off_axis_equisolid = equisolid.pixel_of(np.pi / 4, 0.0)

        assert abs(off_axis_equidistant[0] - off_axis_equisolid[0]) > one_pixel
        assert equidistant.pixel_of(0.0, 0.0) == pytest.approx(equisolid.pixel_of(0.0, 0.0))


class TestKeepMask:
    def test_the_horizon_the_projection_reports_is_the_edge_of_the_mask(self):
        """The two answers come from one description, so they cannot disagree:
        the pixel the projection puts at 90 degrees is the last one kept."""
        lente = LensCalibration(centre_row=64.0, centre_col=64.0, radius_px=40.0)
        keep = lente.keep_mask((128, 128))

        row, col = lente.pixel_of(np.pi / 2, 0.0)
        assert keep[round(row), round(col)]
        assert not keep[round(row) - 1, round(col)]

    def test_a_decentred_axis_gives_a_decentred_mask(self):
        lente = LensCalibration(centre_row=30.0, centre_col=70.0, radius_px=20.0)
        keep = lente.keep_mask((100, 100))
        rows, cols = np.where(keep)

        assert rows.mean() == pytest.approx(30.0, abs=0.5)
        assert cols.mean() == pytest.approx(70.0, abs=0.5)

    def test_the_heuristic_fills_the_shorter_dimension(self):
        keep = LensCalibration.centred_in((40, 80)).keep_mask((40, 80))

        assert keep[20, 40]
        assert not keep[0, 0]
        assert keep.sum() < 40 * 80


class TestDirectionOf:
    def test_it_inverts_pixel_of_everywhere_inside_the_dome(self, lente: LensCalibration):
        directions = lente.direction_of((224, 224))
        keep = lente.keep_mask((224, 224))
        rows, cols = np.nonzero(keep)

        worst = 0.0
        for row, col in zip(rows[::53], cols[::53], strict=True):
            x, y, z = directions[:, row, col]
            zenith = float(np.arccos(np.clip(z, -1.0, 1.0)))
            azimuth = float(np.arctan2(y, x))
            back_row, back_col = lente.pixel_of(zenith, azimuth)
            worst = max(worst, float(np.hypot(back_row - row, back_col - col)))

        assert worst < 1e-3

    def test_the_optical_centre_points_at_the_zenith(self, lente: LensCalibration):
        directions = lente.direction_of((224, 224))

        assert directions[:, 112, 112] == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)

    def test_a_pixel_left_of_centre_looks_east_because_the_camera_looks_up(
        self, lente: LensCalibration
    ):
        directions = lente.direction_of((224, 224))

        east = directions[1, 112, 62]
        assert east > 0.0
        assert directions[1, 112, 162] < 0.0


class TestIsotropicCalibration:
    def test_the_disc_is_concentric_and_inscribed_at_any_frame_size(self):
        for size in (224, 512):
            calibration = isotropic_calibration(size)

            assert calibration.centre_row == pytest.approx(size / 2, abs=0.2)
            assert calibration.centre_col == pytest.approx(size / 2, abs=0.2)
            assert calibration.radius_px == pytest.approx(size / 2, abs=0.05)

    def test_it_keeps_the_mount_rotation_the_native_fit_measured(self):
        assert isotropic_calibration(224).azimuth_offset_rad == pytest.approx(
            PLANETARIO_NATIVE.azimuth_offset_rad
        )
