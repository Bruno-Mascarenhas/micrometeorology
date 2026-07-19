"""Compatibility wrapper for running solrad_correction experiments."""

from __future__ import annotations

from solrad_correction.evaluation.policy import align_test_frame, prediction_index
from solrad_correction.experiments.pipeline import run_pipeline

# Internal back-compat aliases: kept as module attributes for any private
# consumer that historically reached for them, but intentionally NOT advertised
# in ``__all__`` — ``__all__`` is the public export surface and must not publish
# underscore-private names.
_prediction_index_for_policy = prediction_index
_test_frame_for_policy = align_test_frame


def run_experiment(config):
    """Execute a complete experiment from config.

    This public wrapper preserves the historical import path while the
    implementation lives in composable pipeline stages.
    """
    return run_pipeline(config)


__all__ = ["run_experiment"]
