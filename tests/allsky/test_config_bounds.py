"""Load-time validation of the allsky config tree.

``epochs``/``patience``/``min_delta`` are the settings whose out-of-range values
fail silently at run time (an exit-0 run with no checkpoint, a one-epoch
"converged" run, early stopping that never fires), and ``alignment.strategy``
was a bare ``str`` whose typos surfaced only deep inside dataset construction —
or, in image mode, never. All are rejected at load time instead. Torch-free:
pydantic validation only.
"""

import pytest
from pydantic import ValidationError

from allsky.config import AlignmentConfig, ExperimentConfig, PrepareConfig
from labmim_core.site import SiteConfig


def _config(train: dict) -> ExperimentConfig:
    return ExperimentConfig.model_validate({"train": train})


class TestTrainBounds:
    def test_zero_epochs_is_rejected_because_no_checkpoint_would_ever_be_written(self):
        with pytest.raises(ValidationError, match="epochs"):
            _config({"epochs": 0})

    def test_negative_epochs_rejected(self):
        with pytest.raises(ValidationError, match="epochs"):
            _config({"epochs": -1})

    def test_one_epoch_accepted(self):
        assert _config({"epochs": 1}).train.epochs == 1

    def test_zero_grad_accum_steps_is_rejected_rather_than_silently_trained_as_one(self):
        with pytest.raises(ValidationError, match="grad_accum_steps"):
            _config({"grad_accum_steps": 0})


class TestEarlyStoppingBounds:
    def test_zero_patience_is_rejected_because_it_stops_after_one_improving_epoch(self):
        with pytest.raises(ValidationError, match="patience"):
            _config({"early_stopping": {"patience": 0}})

    def test_negative_min_delta_rejected(self):
        # A negative min_delta makes a worsening metric count as an improvement.
        with pytest.raises(ValidationError, match="min_delta"):
            _config({"early_stopping": {"min_delta": -0.01}})

    def test_defaults_and_boundaries_accepted(self):
        early = _config({"early_stopping": {"patience": 1, "min_delta": 0.0}}).train.early_stopping
        assert (early.patience, early.min_delta) == (1, 0.0)


class TestWeightAverageBounds:
    @pytest.mark.parametrize("decay", [0.0, 1.0, 1.5, -0.1])
    def test_a_decay_outside_the_open_unit_interval_is_rejected(self, decay: float):
        with pytest.raises(ValidationError, match="decay"):
            _config({"weight_average": {"enabled": True, "decay": decay}})

    def test_a_start_epoch_of_zero_is_rejected(self):
        with pytest.raises(ValidationError, match="start_epoch"):
            _config({"weight_average": {"enabled": True, "start_epoch": 0}})

    def test_a_start_past_the_epoch_budget_is_rejected_because_no_ema_would_be_written(self):
        with pytest.raises(ValidationError, match=r"no ema\.ckpt"):
            _config({"epochs": 3, "weight_average": {"enabled": True, "start_epoch": 4}})

    def test_a_start_past_the_budget_is_harmless_while_the_average_is_off(self):
        cfg = _config({"epochs": 3, "weight_average": {"enabled": False, "start_epoch": 4}})
        assert cfg.train.weight_average.enabled is False

    def test_the_default_is_off_with_the_documented_decay(self):
        average = _config({}).train.weight_average
        assert (average.enabled, average.decay, average.start_epoch) == (False, 0.999, 1)


class TestLayerDecayBounds:
    @pytest.mark.parametrize("layer_decay", [0.0, 1.5, -0.5])
    def test_a_decay_outside_the_half_open_unit_interval_is_rejected(self, layer_decay: float):
        with pytest.raises(ValidationError, match="layer_decay"):
            _config({"backbone_lr": 1e-5, "layer_decay": layer_decay})

    def test_a_decay_of_one_is_accepted(self):
        assert _config({"backbone_lr": 1e-5, "layer_decay": 1.0}).train.layer_decay == 1.0

    def test_a_decay_without_a_backbone_rate_is_rejected_because_it_would_be_inert(self):
        with pytest.raises(ValidationError, match="backbone_lr is unset"):
            _config({"layer_decay": 0.75})

    def test_the_default_is_none(self):
        assert _config({}).train.layer_decay is None


class TestAlignmentStrategy:
    def test_every_strategy_the_config_names_resolves_a_window(self):
        """One name set, owned here; the dataset dispatches on it and must know every member."""
        from typing import get_args

        import pandas as pd

        from allsky.config import AlignmentStrategyName
        from allsky.data.datasets import _windows_for

        manifest = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(["2026-03-21T15:00:00Z", "2026-03-21T15:01:00Z"]),
                "day_id": ["2026-03-21", "2026-03-21"],
            }
        )
        for name in get_args(AlignmentStrategyName):
            alignment = AlignmentConfig(strategy=name)
            windows = _windows_for(alignment.strategy, manifest, 5.0, -3.0, max_frames=5)
            assert (windows == []) == (name == "center_frame")

    def test_typo_is_rejected_at_load_time(self):
        with pytest.raises(ValidationError):
            AlignmentConfig.model_validate({"strategy": "centre_frame"})

    @pytest.mark.parametrize(
        "payload", [{"window_minutes": -10.0}, {"window_minutes": 0.0}, {"max_frames": 0}]
    )
    def test_a_window_that_can_select_no_frame_is_refused_at_load_time(self, payload: dict):
        """max_frames < 1 was refused only at dataset construction and a negative
        window_minutes reached resolve_time_windows unchecked."""
        with pytest.raises(ValidationError):
            AlignmentConfig.model_validate(payload)

    def test_typo_rejected_through_both_config_roots(self):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate({"data": {"alignment": {"strategy": "attention"}}})
        with pytest.raises(ValidationError):
            PrepareConfig.model_validate({"alignment": {"strategy": "attention"}})


class TestOnlyTheLearnedPoolerNeedsEmbeddingMode:
    def test_the_learned_pooler_is_rejected_in_image_mode(self):
        with pytest.raises(ValidationError, match="learned pooler"):
            ExperimentConfig.model_validate(
                {
                    "data": {
                        "input_mode": "image",
                        "alignment": {"strategy": "attention_pooling"},
                    }
                }
            )

    def test_the_mean_pooled_window_is_accepted_in_image_mode(self):
        cfg = ExperimentConfig.model_validate(
            {"data": {"input_mode": "image", "alignment": {"strategy": "mean_embedding"}}}
        )

        assert cfg.data.alignment.strategy == "mean_embedding"

    def test_windowed_strategy_with_embedding_mode_is_accepted(self):
        cfg = ExperimentConfig.model_validate(
            {"data": {"input_mode": "embedding", "alignment": {"strategy": "mean_embedding"}}}
        )
        assert cfg.data.alignment.strategy == "mean_embedding"

    def test_center_frame_with_image_mode_stays_the_default(self):
        cfg = ExperimentConfig.model_validate({"data": {"input_mode": "image"}})
        assert cfg.data.alignment.strategy == "center_frame"


@pytest.mark.parametrize(
    "payload",
    [
        {"val_fraction": 1.2},
        {"val_fraction": -0.2},
        {"test_fraction": 1.0},
        {"gap_days": -3},
        {"val_fraction": 0.6, "test_fraction": 0.5},
    ],
)
def test_a_split_that_cannot_partition_the_days_is_refused_at_load_time(payload: dict):
    """create_day_splits refuses every one of these, but only at split time."""
    with pytest.raises(ValidationError):
        PrepareConfig.model_validate({"splits": payload})


def test_a_misspelt_site_key_is_rejected_instead_of_keeping_the_station_clock():
    """``SiteConfig`` ignored unknown keys, so ``utc_offset_hour: -8`` in a YAML
    ``site:`` block computed a UTC-8 station on Salvador's clock without failing."""
    with pytest.raises(ValidationError, match="utc_offset_hour"):
        SiteConfig.model_validate({"utc_offset_hour": -8.0})


def test_a_config_that_enables_no_head_is_refused_instead_of_dying_in_autograd():
    """Heads returns an empty pack, MultitaskLoss returns a constant with no
    grad_fn, and backward() fails naming none of the four switches — after the
    seed, the manifest, the normalizers, the model and the loaders."""
    with pytest.raises(ValidationError, match="enables no head"):
        ExperimentConfig.model_validate({"targets": {"dhi": {"enabled": False}}})


def test_geometry_channels_are_refused_in_embedding_mode():
    """``build_model`` returned a PrecomputedEmbedding with no extra-channel
    projection and no warning for a config that asked for the planes."""
    with pytest.raises(ValidationError, match="geometry_channels"):
        ExperimentConfig.model_validate(
            {
                "data": {"input_mode": "embedding"},
                "model": {"name": "image_only", "geometry_channels": True},
            }
        )


class TestSiteBounds:
    """A ``site:`` block is one YAML edit away from a sign typo, and every
    number it carries flows into geometry that clips rather than raises:
    ``cos_zenith`` clamps to [-1, 1] and ``solar_azimuth_deg`` clamps its
    arccos ratio, so an impossible latitude produced a finite, confident
    elevation with nothing failing anywhere in the chain."""

    def test_latitude_beyond_the_pole_is_rejected(self):
        with pytest.raises(ValidationError, match="latitude"):
            SiteConfig.model_validate({"latitude": -130.0})

    def test_longitude_beyond_the_antimeridian_is_rejected(self):
        with pytest.raises(ValidationError, match="longitude"):
            SiteConfig.model_validate({"longitude": 190.0})

    def test_utc_offset_outside_the_real_range_is_rejected(self):
        with pytest.raises(ValidationError, match="utc_offset_hours"):
            SiteConfig.model_validate({"utc_offset_hours": -25.0})

    def test_the_station_defaults_still_validate(self):
        site = SiteConfig()
        assert (site.latitude, site.longitude, site.utc_offset_hours) == (
            -13.0055,
            -38.5089,
            -3.0,
        )


class TestPixelSectionBounds:
    """Every one of these is a fraction or a probability, and unbounded each one
    accepted a value that changes what the model is fed with no error: a band
    taller than the frame, an ROI larger than it, a probability above 1."""

    @pytest.mark.parametrize(
        ("section", "payload"),
        [
            ("preprocessing", {"band_fraction": 1.5}),
            ("preprocessing", {"band_fraction": -0.1}),
            ("preprocessing", {"roi_radius_fraction": 1.5}),
            ("preprocessing", {"roi_radius_fraction": 0.0}),
            ("augmentation", {"p_exposure": 1.5}),
            ("augmentation", {"p_noise": -0.1}),
            ("augmentation", {"p_translate": 1.5}),
            ("augmentation", {"p_erase": 1.5}),
        ],
    )
    def test_a_fraction_outside_its_range_is_rejected(self, section: str, payload: dict):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate({section: payload})

    @pytest.mark.parametrize("payload", [{"batch_size": 0}, {"num_workers": -1}, {"lr": 0.0}])
    def test_a_train_knob_outside_its_range_is_rejected(self, payload: dict):
        with pytest.raises(ValidationError):
            _config(payload)


class TestPixelSectionsNeedImageMode:
    """Both sections describe transforms over a decoded frame, and embedding mode
    decodes none: the run accepted the section, never applied it, and reported
    metrics as though it had."""

    @pytest.mark.parametrize(
        "payload",
        [
            {"preprocessing": {"overlay": "fill"}},
            {"preprocessing": {"roi_radius_fraction": 0.9}},
            {"augmentation": {"p_noise": 0.3}},
        ],
    )
    def test_a_pixel_section_is_refused_in_embedding_mode(self, payload: dict):
        with pytest.raises(ValidationError, match="decodes none"):
            ExperimentConfig.model_validate({"data": {"input_mode": "embedding"}, **payload})

    def test_the_same_section_is_accepted_in_image_mode(self):
        cfg = ExperimentConfig.model_validate(
            {"data": {"input_mode": "image"}, "augmentation": {"p_noise": 0.3}}
        )
        assert cfg.augmentation.p_noise == pytest.approx(0.3)

    def test_a_rotation_alone_is_refused_in_embedding_mode(self):
        with pytest.raises(ValidationError, match="decodes none"):
            ExperimentConfig.model_validate(
                {"data": {"input_mode": "embedding"}, "augmentation": {"p_rotate": 0.5}}
            )


class TestRotationBounds:
    @pytest.mark.parametrize(
        "payload", [{"p_rotate": 1.5}, {"p_rotate": -0.1}, {"rotate_max_deg": 181.0}]
    )
    def test_a_rotation_knob_outside_its_range_is_rejected(self, payload: dict):
        with pytest.raises(ValidationError):
            ExperimentConfig.model_validate({"augmentation": payload})

    def test_the_default_is_off_over_the_whole_circle(self):
        augmentation = ExperimentConfig().augmentation
        assert augmentation.p_rotate == 0.0
        assert augmentation.rotate_max_deg == pytest.approx(180.0)


_REGRESSION_HEADS = {"dhi": {"enabled": True}, "kindex": {"enabled": True}}


class TestCMixupNeedsSingleFramesAndTheKindexHead:
    def test_it_loads_on_the_single_frame_image_path_with_the_kindex_head(self):
        cfg = ExperimentConfig.model_validate(
            {
                "data": {"input_mode": "image"},
                "targets": _REGRESSION_HEADS,
                "train": {"cmixup": {"enabled": True, "alpha": 0.4, "bandwidth": 0.1, "p": 0.5}},
            }
        )
        assert cfg.train.cmixup.bandwidth == pytest.approx(0.1)

    def test_embedding_mode_is_refused(self):
        with pytest.raises(ValidationError, match="input_mode"):
            ExperimentConfig.model_validate(
                {
                    "data": {"input_mode": "embedding"},
                    "targets": _REGRESSION_HEADS,
                    "train": {"cmixup": {"enabled": True}},
                }
            )

    def test_a_window_of_frames_is_refused(self):
        with pytest.raises(ValidationError, match="window"):
            ExperimentConfig.model_validate(
                {
                    "data": {"input_mode": "image", "alignment": {"strategy": "mean_embedding"}},
                    "targets": _REGRESSION_HEADS,
                    "train": {"cmixup": {"enabled": True}},
                }
            )

    def test_a_run_without_the_kindex_head_is_refused(self):
        with pytest.raises(ValidationError, match="kindex"):
            ExperimentConfig.model_validate(
                {
                    "data": {"input_mode": "image"},
                    "targets": {"dhi": {"enabled": True}},
                    "train": {"cmixup": {"enabled": True}},
                }
            )

    @pytest.mark.parametrize(
        "payload", [{"alpha": 0.0}, {"bandwidth": 0.0}, {"p": 1.5}, {"p": -0.1}]
    )
    def test_a_knob_outside_its_range_is_rejected(self, payload: dict):
        with pytest.raises(ValidationError):
            _config({"cmixup": payload})

    def test_the_documented_defaults_hold_while_off(self):
        cmixup = ExperimentConfig().train.cmixup
        assert cmixup.enabled is False
        assert cmixup.alpha == pytest.approx(1.0)
        assert cmixup.bandwidth == pytest.approx(0.05)
        assert cmixup.p == pytest.approx(1.0)
