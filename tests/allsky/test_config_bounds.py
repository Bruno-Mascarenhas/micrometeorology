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
    def test_zero_epochs_rejected(self):
        # epochs: 0 used to exit 0 while advertising checkpoints never written.
        with pytest.raises(ValidationError, match="epochs"):
            _config({"epochs": 0})

    def test_negative_epochs_rejected(self):
        with pytest.raises(ValidationError, match="epochs"):
            _config({"epochs": -1})

    def test_one_epoch_accepted(self):
        assert _config({"epochs": 1}).train.epochs == 1

    def test_zero_grad_accum_steps_rejected(self):
        # grad_accum_steps: 0 used to train with 1 while reporting the config as honoured.
        with pytest.raises(ValidationError, match="grad_accum_steps"):
            _config({"grad_accum_steps": 0})


class TestEarlyStoppingBounds:
    def test_zero_patience_rejected(self):
        # patience: 0 stopped after a single epoch even when it improved.
        with pytest.raises(ValidationError, match="patience"):
            _config({"early_stopping": {"patience": 0}})

    def test_negative_min_delta_rejected(self):
        # A negative min_delta makes a worsening metric count as an improvement.
        with pytest.raises(ValidationError, match="min_delta"):
            _config({"early_stopping": {"min_delta": -0.01}})

    def test_defaults_and_boundaries_accepted(self):
        early = _config({"early_stopping": {"patience": 1, "min_delta": 0.0}}).train.early_stopping
        assert (early.patience, early.min_delta) == (1, 0.0)


class TestAlignmentStrategy:
    def test_every_window_mode_the_dataset_implements_is_accepted(self):
        """One name set, owned here and read by the dataset that implements it."""
        from allsky.data.datasets import _WINDOW_MODES

        for name in _WINDOW_MODES:
            assert AlignmentConfig(strategy=name).strategy == name

    def test_typo_is_rejected_at_load_time(self):
        with pytest.raises(ValidationError):
            AlignmentConfig.model_validate({"strategy": "centre_frame"})

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


def test_a_misspelt_site_key_is_rejected_instead_of_keeping_the_station_clock():
    """``SiteConfig`` ignored unknown keys, so ``utc_offset_hour: -8`` in a YAML
    ``site:`` block computed a UTC-8 station on Salvador's clock without failing."""
    with pytest.raises(ValidationError, match="utc_offset_hour"):
        SiteConfig.model_validate({"utc_offset_hour": -8.0})
