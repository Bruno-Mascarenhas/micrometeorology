"""Tests for solar geometry rendered as image channels.

The whole point of the feature is that the extra channels REACH the network.
An arm whose channels are computed, standardised, batched and then dropped by a
frozen projection returns the control's number and reads as a null result about
the physics — which is what happened to the tabular exposure features. So most
of what is asserted here is reachability: the channels exist, they are wired to
trainable weights, and those weights sit in the optimizer group that can move
them.
"""

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from allsky.augmentation import AugmentationPipeline
from allsky.config import ExperimentConfig, geometry_channels_of
from allsky.data.datasets import MultimodalImageDataset
from allsky.data.manifest import build_manifest
from allsky.features import resolve_feature_set
from allsky.geometry import (
    GEOMETRY_CHANNEL_NAMES,
    resolve_geometry_channels,
    solar_geometry_maps,
)
from allsky.lens import LensCalibration, isotropic_calibration
from allsky.modeling.geometry_adapter import (
    GeometryPatchProjection,
    PatchProjectionNotFoundError,
    attach_extra_input_channels,
)
from allsky.modeling.registry import build_model
from allsky.modeling.visual_encoder import ImageEncoder
from labmim_core import solar
from labmim_core.site import SiteConfig

FRAME_PX = 32
PATCH_PX = 8


@pytest.fixture
def calibration() -> LensCalibration:
    return isotropic_calibration(FRAME_PX)


class TestSolarGeometryMaps:
    def test_the_stack_matches_the_declared_channel_names(self, calibration: LensCalibration):
        maps = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=0.4, sun_azimuth_rad=1.2
        )

        assert maps.shape == (len(GEOMETRY_CHANNEL_NAMES), FRAME_PX, FRAME_PX)
        assert maps.dtype == np.float32

    def test_the_sun_angle_peaks_on_the_pixel_the_lens_puts_the_sun_at(
        self, calibration: LensCalibration
    ):
        zenith, azimuth = 0.5, 2.0
        maps = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=zenith, sun_azimuth_rad=azimuth
        )
        cos_sun_angle = maps[GEOMETRY_CHANNEL_NAMES.index("cos_sun_angle")]

        peak = np.unravel_index(int(cos_sun_angle.argmax()), cos_sun_angle.shape)
        expected = calibration.pixel_of(zenith, azimuth)
        assert np.hypot(peak[0] - expected[0], peak[1] - expected[1]) <= 1.0

    def test_the_disc_is_brightest_at_the_sun_and_decays_away_from_it(
        self, calibration: LensCalibration
    ):
        maps = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=0.3, sun_azimuth_rad=0.0
        )
        disc = maps[GEOMETRY_CHANNEL_NAMES.index("solar_disc")]
        row, col = calibration.pixel_of(0.3, 0.0)

        assert disc.max() == pytest.approx(disc[round(row), round(col)], abs=1e-3)
        assert disc.min() < disc.max()

    def test_the_zenith_channel_does_not_depend_on_where_the_sun_is(
        self, calibration: LensCalibration
    ):
        """It is fixed for a fixed camera, so it is a spatial prior and not a
        per-sample signal. Stating that here keeps a later reading of a null
        result honest."""
        index = GEOMETRY_CHANNEL_NAMES.index("cos_pixel_zenith")
        morning = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=1.2, sun_azimuth_rad=1.5
        )
        afternoon = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=0.2, sun_azimuth_rad=4.5
        )

        assert np.array_equal(morning[index], afternoon[index])

    def test_the_zenith_channel_is_one_at_the_optical_centre(self, calibration: LensCalibration):
        maps = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=0.1, sun_azimuth_rad=0.0
        )
        zenith = maps[GEOMETRY_CHANNEL_NAMES.index("cos_pixel_zenith")]
        centre = (round(calibration.centre_row), round(calibration.centre_col))

        assert zenith[centre] == pytest.approx(1.0, abs=2e-3)


class TestChannelSelection:
    def test_true_means_every_map_and_false_means_none(self):
        assert resolve_geometry_channels(True) == GEOMETRY_CHANNEL_NAMES
        assert resolve_geometry_channels(False) == ()
        assert resolve_geometry_channels(None) == ()

    def test_a_subset_comes_back_in_the_canonical_order_however_it_was_asked_for(self):
        """The plane a trained weight belongs to must not depend on the order
        someone typed the names in, or a checkpoint would reload onto a
        different channel than it was fitted on."""
        assert resolve_geometry_channels(["solar_disc", "cos_sun_angle"]) == (
            "cos_sun_angle",
            "solar_disc",
        )

    def test_an_unknown_channel_name_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown geometry channel"):
            resolve_geometry_channels(["cos_sun_angle", "cos_moon_angle"])

    def test_an_empty_list_is_refused_rather_than_read_as_disabled(self):
        with pytest.raises(ValueError, match="empty list"):
            resolve_geometry_channels([])

    def test_a_repeated_channel_is_refused(self):
        with pytest.raises(ValueError, match="repeats a channel"):
            resolve_geometry_channels(["cos_sun_angle", "cos_sun_angle"])

    def test_a_subset_builds_only_the_maps_it_names(self, calibration: LensCalibration):
        full = solar_geometry_maps(
            calibration, (FRAME_PX, FRAME_PX), sun_zenith_rad=0.4, sun_azimuth_rad=1.2
        )
        one = solar_geometry_maps(
            calibration,
            (FRAME_PX, FRAME_PX),
            sun_zenith_rad=0.4,
            sun_azimuth_rad=1.2,
            channels=("cos_sun_angle",),
        )

        assert one.shape == (1, FRAME_PX, FRAME_PX)
        assert np.array_equal(one[0], full[GEOMETRY_CHANNEL_NAMES.index("cos_sun_angle")])


class TestGeometryPatchProjection:
    def test_it_reproduces_the_wrapped_projection_before_any_training(self):
        pretrained = nn.Conv2d(3, 6, kernel_size=PATCH_PX, stride=PATCH_PX)
        adapter = GeometryPatchProjection(pretrained, 3)
        frame = torch.randn(2, 6, FRAME_PX, FRAME_PX)

        assert torch.equal(adapter(frame), pretrained(frame[:, :3]))

    def test_it_rejects_a_frame_with_the_wrong_number_of_planes(self):
        adapter = GeometryPatchProjection(nn.Conv2d(3, 6, kernel_size=PATCH_PX, stride=PATCH_PX), 3)

        with pytest.raises(ValueError, match="expected 6 input channels"):
            adapter(torch.randn(2, 4, FRAME_PX, FRAME_PX))

    def test_the_extra_branch_starts_at_zero_and_carries_gradient(self):
        adapter = GeometryPatchProjection(nn.Conv2d(3, 6, kernel_size=PATCH_PX, stride=PATCH_PX), 3)

        assert float(adapter.extra_proj.weight.abs().sum()) == 0.0

        adapter(torch.randn(2, 6, FRAME_PX, FRAME_PX)).sum().backward()

        gradient = adapter.extra_proj.weight.grad
        assert gradient is not None
        assert float(gradient.abs().sum()) > 0.0

    def test_a_backbone_without_a_patch_convolution_fails_loudly(self):
        with pytest.raises(PatchProjectionNotFoundError, match=r"no patch_embed\.proj"):
            attach_extra_input_channels(nn.Linear(3, 4), 3)

    def test_a_zero_width_adapter_is_refused_rather_than_built_inert(self):
        with pytest.raises(ValueError, match="extra_channels must be positive"):
            GeometryPatchProjection(nn.Conv2d(3, 6, kernel_size=PATCH_PX, stride=PATCH_PX), 0)


class _StubViT(nn.Module):
    """Backbone shaped like DINOv2: a ``patch_embed.proj`` OUTSIDE ``blocks``.

    The shape is the subject of the test, not an implementation detail — the
    projection sitting outside ``blocks`` is exactly why ``unfreeze_last_n``
    never reaches it.
    """

    def __init__(self, dim: int = 8, n_blocks: int = 2) -> None:
        super().__init__()
        self.dim = dim
        # Typed Any for the same reason allsky.modeling.visual_encoder types its
        # hub module that way: nn.Module.__getattr__ returns Tensor | Module.
        self.patch_embed: Any = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, kernel_size=PATCH_PX, stride=PATCH_PX)
        self.blocks = nn.ModuleList(nn.Linear(dim, dim) for _ in range(n_blocks))
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frame: Any) -> Any:
        hidden = self.pool(self.patch_embed.proj(frame)).flatten(1)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class TestImageEncoderWiring:
    def test_the_extra_branch_trains_even_when_the_backbone_is_frozen(self):
        """The failure this guards is silent: a zero-initialised projection that
        the freeze sweep also owns leaves the channels inert, and the arm then
        reports the control's error as evidence against the physics."""
        encoder = ImageEncoder(_StubViT(), frozen=True, unfreeze_last_n=1, extra_input_channels=3)
        adapter = encoder.extra_channel_projection

        assert adapter is not None
        assert all(p.requires_grad for p in adapter.extra_proj.parameters())
        assert not any(p.requires_grad for p in adapter.pretrained.parameters())

    def test_the_extra_branch_is_not_priced_at_the_pretrained_learning_rate(self):
        encoder = ImageEncoder(_StubViT(), unfreeze_last_n=1, extra_input_channels=3)
        adapter = encoder.extra_channel_projection
        assert adapter is not None
        extra_ids = {id(p) for p in adapter.extra_proj.parameters()}

        groups = encoder.param_groups(1e-5)
        backbone_group = next(g for g in groups if "lr" in g)
        default_group = next(g for g in groups if "lr" not in g)

        assert not any(id(p) in extra_ids for p in backbone_group["params"])
        assert extra_ids <= {id(p) for p in default_group["params"]}

    def test_the_encoder_is_unchanged_by_the_extra_channels_at_initialisation(self):
        torch.manual_seed(0)
        encoder = ImageEncoder(_StubViT(), extra_input_channels=3)
        frame = torch.randn(2, 6, FRAME_PX, FRAME_PX)

        backbone: Any = encoder.backbone
        with torch.no_grad():
            widened = encoder({"image": frame})
            encoder.extra_channel_projection = None
            backbone.patch_embed.proj = backbone.patch_embed.proj.pretrained
            plain = encoder({"image": frame[:, :3]})

        assert torch.equal(widened, plain)

    def test_no_adapter_is_installed_when_no_extra_channels_are_asked_for(self):
        assert ImageEncoder(_StubViT()).extra_channel_projection is None


def _manifest(tmp_path: Path, n: int = 4) -> tuple[pd.DataFrame, Path]:
    site = SiteConfig()
    times = pd.date_range("2025-03-21 09:00", periods=n, freq="30min")
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    for i, ts in enumerate(times):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        iio.imwrite(path, rng.integers(0, 256, (FRAME_PX, FRAME_PX, 3)).astype(np.uint8))
        rows.append({"frame_path": str(path), "timestamp": ts, "video": "v.mp4", "index": i})
    sensor_index = pd.date_range("2025-03-21 06:00", "2025-03-21 18:00", freq="5min")
    e0h = solar.extraterrestrial_ghi(sensor_index, site)
    sensor = pd.DataFrame(
        {
            "AirT1_C_Avg": 25.0,
            "DP1_C_Avg": 15.0,
            "RH1": 70.0,
            "BP1_mbar_Avg": 1010.0,
            "WS_ms": 2.0,
            "WindDir": 90.0,
            "CM3Up_Wm2_Avg": 0.7 * e0h,
            "PSP_Wm2_Avg": 0.2 * e0h,
        },
        index=sensor_index,
    )
    manifest, _ = build_manifest(pd.DataFrame(rows), sensor, site=site, data_root=tmp_path)
    return manifest, tmp_path


class TestImageDatasetChannels:
    def test_the_frame_carries_rgb_plus_one_plane_per_geometry_channel(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = MultimodalImageDataset(
            manifest,
            resolve_feature_set("bare"),
            data_root=root,
            image_size=FRAME_PX,
            train=True,
            geometry_channels=GEOMETRY_CHANNEL_NAMES,
        )

        image = dataset[0]["image"]

        assert image.shape == (3 + len(GEOMETRY_CHANNEL_NAMES), FRAME_PX, FRAME_PX)
        assert bool(torch.isfinite(image).all())

    def test_the_geometry_planes_change_with_the_sun_and_the_rgb_planes_do_not_move(
        self, tmp_path: Path
    ):
        manifest, root = _manifest(tmp_path)
        plain = MultimodalImageDataset(
            manifest, resolve_feature_set("bare"), data_root=root, image_size=FRAME_PX, train=True
        )
        with_geometry = MultimodalImageDataset(
            manifest,
            resolve_feature_set("bare"),
            data_root=root,
            image_size=FRAME_PX,
            train=True,
            geometry_channels=GEOMETRY_CHANNEL_NAMES,
        )

        assert torch.equal(with_geometry[0]["image"][:3], plain[0]["image"])
        assert not torch.equal(with_geometry[0]["image"][3:], with_geometry[-1]["image"][3:])

    def test_a_translating_augmentation_is_refused_instead_of_silently_misaligning(
        self, tmp_path: Path
    ):
        manifest, root = _manifest(tmp_path)

        with pytest.raises(ValueError, match="incompatible with p_translate"):
            MultimodalImageDataset(
                manifest,
                resolve_feature_set("bare"),
                data_root=root,
                image_size=FRAME_PX,
                train=True,
                augment=AugmentationPipeline(p_translate=1.0),
                geometry_channels=GEOMETRY_CHANNEL_NAMES,
            )

    def test_a_subset_narrows_the_frame_to_rgb_plus_those_planes(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)
        dataset = MultimodalImageDataset(
            manifest,
            resolve_feature_set("bare"),
            data_root=root,
            image_size=FRAME_PX,
            train=True,
            geometry_channels=("cos_sun_angle",),
        )

        assert dataset[0]["image"].shape == (4, FRAME_PX, FRAME_PX)

    def test_a_manifest_without_solar_angles_is_refused(self, tmp_path: Path):
        manifest, root = _manifest(tmp_path)

        with pytest.raises(ValueError, match="solar_zenith"):
            MultimodalImageDataset(
                manifest.drop(columns=["solar_zenith"]),
                [c for c in resolve_feature_set("bare") if c != "solar_zenith"],
                data_root=root,
                image_size=FRAME_PX,
                train=True,
                geometry_channels=GEOMETRY_CHANNEL_NAMES,
            )


class TestConfigFlow:
    def test_the_flag_reaches_the_model_the_same_way_the_frame_size_does(self):
        cfg = ExperimentConfig.model_validate(
            {
                "features": {"set": "bare"},
                "targets": {"dhi": {"enabled": True}},
                "model": {"name": "image_only", "geometry_channels": True},
                "data": {"input_mode": "image"},
            }
        )

        assert geometry_channels_of(cfg) == GEOMETRY_CHANNEL_NAMES

        model: Any = build_model(cfg, 9, embedding_dim=None, image_backbone=_StubViT())
        adapter = model.visual_encoder.extra_channel_projection

        assert adapter is not None
        assert adapter.in_channels == 3 + len(GEOMETRY_CHANNEL_NAMES)

    def test_a_named_subset_widens_the_projection_by_exactly_that_many_planes(self):
        cfg = ExperimentConfig.model_validate(
            {
                "features": {"set": "bare"},
                "targets": {"dhi": {"enabled": True}},
                "model": {"name": "image_only", "geometry_channels": ["cos_sun_angle"]},
                "data": {"input_mode": "image"},
            }
        )

        assert geometry_channels_of(cfg) == ("cos_sun_angle",)

        model: Any = build_model(cfg, 9, embedding_dim=None, image_backbone=_StubViT())

        assert model.visual_encoder.extra_channel_projection.in_channels == 4

    def test_the_flag_is_a_recognised_knob_and_does_not_warn_as_a_typo(self, caplog: Any):
        import logging

        cfg = ExperimentConfig.model_validate(
            {
                "features": {"set": "bare"},
                "targets": {"dhi": {"enabled": True}},
                "model": {"name": "image_only", "geometry_channels": False},
                "data": {"input_mode": "image"},
            }
        )

        with caplog.at_level(logging.WARNING, logger="allsky.modeling.registry"):
            build_model(cfg, 9, embedding_dim=None, image_backbone=_StubViT())

        assert not any("unknown hyper-parameter" in r.getMessage() for r in caplog.records)
