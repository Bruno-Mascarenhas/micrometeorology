"""Load-time validation of the allsky config tree.

``epochs``/``patience``/``min_delta`` are the settings whose out-of-range values
fail silently at run time (an exit-0 run with no checkpoint, a one-epoch
"converged" run, early stopping that never fires), and ``alignment.strategy``
was a bare ``str`` whose typos surfaced only deep inside dataset construction —
or, in image mode, never. All are rejected at load time instead. Torch-free:
pydantic validation only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from allsky.config import AlignmentConfig, ExperimentConfig, PrepareConfig


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


class TestWindowedPoolingNeedsEmbeddingMode:
    """Image mode has no windowing, so asking for it must not be silent.

    ``MultimodalImageDataset`` never windows and ``build_visual_encoder`` ignores
    ``temporal_pooling`` in image mode, so ``mean_embedding`` + ``input_mode:
    image`` used to train a plain centre-frame model while the config — and the
    manifest meta derived from it — reported a windowed experiment.
    """

    @pytest.mark.parametrize("strategy", ["mean_embedding", "attention_pooling"])
    def test_windowed_strategy_with_image_mode_is_rejected(self, strategy):
        with pytest.raises(ValidationError, match="input_mode 'embedding' only"):
            ExperimentConfig.model_validate(
                {"data": {"input_mode": "image", "alignment": {"strategy": strategy}}}
            )

    def test_windowed_strategy_with_embedding_mode_is_accepted(self):
        cfg = ExperimentConfig.model_validate(
            {"data": {"input_mode": "embedding", "alignment": {"strategy": "mean_embedding"}}}
        )
        assert cfg.data.alignment.strategy == "mean_embedding"

    def test_center_frame_with_image_mode_stays_the_default(self):
        cfg = ExperimentConfig.model_validate({"data": {"input_mode": "image"}})
        assert cfg.data.alignment.strategy == "center_frame"
