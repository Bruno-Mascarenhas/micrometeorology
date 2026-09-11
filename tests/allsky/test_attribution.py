"""Occlusion sensitivity and the geometry that maps it back onto the camera frame."""

import numpy as np
import pandas as pd
import pytest

from allsky.attribution import (
    OVERLAY_TEXT_BAND_RAW,
    Box,
    OcclusionMap,
    frame_geometry_of,
    neutralise_region,
    occlusion_map,
    region_masks,
    render_attribution_rgba,
)
from allsky.config import PrepareConfig, SiteConfig
from allsky.snapshot import load_served_model
from tests.allsky._block_probe import stub_image_backbone
from tests.allsky._frame_probe import train_frame_probes

SERVED_GEOMETRY = PrepareConfig.model_validate(
    {
        "crop": {"enabled": True, "top": 0, "left": 12, "height": 1080, "width": 1711},
        "pad": {"enabled": True, "top": 254, "bottom": 377, "left": 0, "right": 0, "fill": 0},
        "resize": 512,
    }
)
NOON = pd.Timestamp("2025-03-20T12:00:00")


def test_the_served_geometry_places_the_crop_at_the_measured_offset():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=512)

    assert (frame.crop.top, frame.crop.left, frame.crop.height, frame.crop.width) == (
        0,
        116,
        1080,
        1711,
    )
    assert (frame.padded_height, frame.padded_width) == (1711, 1711)


def test_the_content_box_is_the_crop_inside_the_padded_square_scaled_to_the_input():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=512)

    box = frame.content_box

    assert box.top == pytest.approx(254 * 512 / 1711)
    assert box.height == pytest.approx(1080 * 512 / 1711)
    assert box.left == 0.0
    assert box.width == pytest.approx(512.0)


def test_input_to_raw_inverts_raw_to_input_inside_the_field_of_view():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=512)
    raw_box = Box(100.0, 400.0, 200.0, 300.0)

    back = frame.input_to_raw(frame.raw_to_input(raw_box))

    assert (back.top, back.left, back.height, back.width) == pytest.approx(
        (100.0, 400.0, 200.0, 300.0)
    )


def test_the_overlay_band_is_clipped_to_the_input_and_sits_in_the_top_left():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=512)
    top, left, height, width = OVERLAY_TEXT_BAND_RAW

    band = frame.raw_to_input(Box(top, left, height, width))

    assert band.left == 0.0
    assert band.top == pytest.approx(254 * 512 / 1711)
    assert 0 < band.width < 512 / 4
    assert 0 < band.height < 512 / 8


def test_a_checkpoint_without_geometry_maps_the_whole_frame_onto_the_input():
    frame = frame_geometry_of(None, raw_height=9, raw_width=16, input_size=8)

    assert (frame.crop.height, frame.crop.width) == (9, 16)
    assert (frame.content_box.height, frame.content_box.width) == pytest.approx((8.0, 8.0))
    assert frame.pad_top == 0


def test_region_masses_are_disjoint_and_sum_to_one_on_a_uniform_map():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=64)
    masks = region_masks(frame)

    mass = masks.mass_of(np.ones((64, 64), dtype=np.float32))

    assert masks.overlay_band.any()
    assert 0.0 < mass["disc"] < 1.0
    assert not (masks.disc & masks.pad).any()
    assert sum(mass.values()) == pytest.approx(1.0)


def test_rasterised_magnitude_reproduces_non_overlapping_cells():
    grid = np.array([[2.0, -1.0], [0.0, 4.0]], dtype=np.float32)
    positions = np.array([(0, 0), (0, 4), (4, 0), (4, 4)])
    occlusion = OcclusionMap("kindex", 4, 4, grid, base_value=0.5, positions=positions)

    heat = occlusion.rasterised(8)

    assert heat[0, 0] == pytest.approx(0.5)
    assert heat[0, 7] == pytest.approx(0.25)
    assert heat[7, 7] == pytest.approx(1.0)
    assert occlusion.peak == (1, 1)


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    return train_frame_probes(tmp_path_factory.mktemp("probes"))


@pytest.fixture(scope="module")
def served(probes):
    return load_served_model(
        probes["probe_s0"], trust_checkpoint=True, image_backbone_builder=stub_image_backbone
    )


def _first_frame(probes):
    return min((probes["dataset"] / "frames").glob("*.jpg"))


def test_occlusion_grid_covers_the_input_at_the_requested_stride(served, probes):
    site = SiteConfig()
    features = served.scalar_features(NOON, site=site)
    planes = served.image_planes(_first_frame(probes), NOON, site=site)

    occlusion = occlusion_map(
        served,
        planes,
        features,
        timestamp=NOON,
        site=site,
        target="kindex",
        window_px=4,
        stride_px=4,
    )

    assert occlusion.grid.shape == (2, 2)
    assert np.isfinite(occlusion.grid).all()
    assert np.isfinite(occlusion.base_value)


def test_occlusion_refuses_a_window_larger_than_the_input(served, probes):
    site = SiteConfig()
    features = served.scalar_features(NOON, site=site)
    planes = served.image_planes(_first_frame(probes), NOON, site=site)

    with pytest.raises(ValueError, match="do not fit"):
        occlusion_map(
            served, planes, features, timestamp=NOON, site=site, target="kindex", window_px=16
        )


def test_occlusion_refuses_a_head_the_model_lacks(served, probes):
    site = SiteConfig()
    features = served.scalar_features(NOON, site=site)
    planes = served.image_planes(_first_frame(probes), NOON, site=site)

    with pytest.raises(ValueError, match="cloud_fraction"):
        occlusion_map(
            served,
            planes,
            features,
            timestamp=NOON,
            site=site,
            target="cloud_fraction",
            window_px=4,
        )


def test_neutralising_a_region_returns_a_physical_prediction(served, probes):
    site = SiteConfig()
    features = served.scalar_features(NOON, site=site)
    planes = served.image_planes(_first_frame(probes), NOON, site=site)

    record = neutralise_region(
        served, planes, features, box=Box(0, 0, 4, 4), timestamp=NOON, site=site
    )

    assert set(record) >= {"dhi", "kindex", "sky_class"}


def test_rendered_map_matches_the_published_image_size_and_alpha_range():
    grid = np.array([[1.0, 0.0], [0.0, 0.5]], dtype=np.float32)
    positions = np.array([(0, 0), (0, 4), (4, 0), (4, 4)])
    occlusion = OcclusionMap("kindex", 4, 4, grid, base_value=0.5, positions=positions)
    frame = frame_geometry_of(None, raw_height=9, raw_width=16, input_size=8)

    rgba = render_attribution_rgba(occlusion, frame, out_width=32, out_height=18)

    assert rgba.shape == (18, 32, 4)
    assert rgba.dtype == np.uint8
    assert rgba[..., 3].max() <= round(0.85 * 255)


def test_the_rendered_map_is_transparent_outside_the_field_of_view():
    frame = frame_geometry_of(SERVED_GEOMETRY, raw_height=1080, raw_width=1920, input_size=512)
    grid = np.ones((1, 1), dtype=np.float32)
    occlusion = OcclusionMap("kindex", 512, 512, grid, base_value=0.5, positions=np.array([(0, 0)]))

    rgba = render_attribution_rgba(occlusion, frame, out_width=1280, out_height=720)

    left_edge = round(116 * 1280 / 1920)
    right_edge = round((116 + 1711) * 1280 / 1920)
    assert rgba[:, : left_edge - 1, 3].max() == 0
    assert rgba[:, right_edge + 1 :, 3].max() == 0
    assert rgba[:, left_edge + 5 : right_edge - 5, 3].min() > 0


def test_the_horizon_follows_the_frame_when_no_geometry_was_recorded():
    frame = frame_geometry_of(None, raw_height=1080, raw_width=1920, input_size=64)

    masks = region_masks(frame)

    centre_row = round(601.7 * 64 / 1080)
    centre_col = round(971.7 * 64 / 1920)
    assert masks.disc[centre_row, centre_col]
    assert not masks.disc[0, 0]
    assert not masks.disc[0, 63]
    cols = np.nonzero(masks.disc.any(axis=0))[0]
    assert (cols.max() - cols.min()) == pytest.approx(2 * 855.5 * 64 / 1920, abs=3)
