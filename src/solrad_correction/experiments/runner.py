"""Compatibility wrapper for running solrad_correction experiments."""

from __future__ import annotations

from solrad_correction.experiments.pipeline import (
    _prediction_index_for_policy,
    _test_frame_for_policy,
    run_pipeline,
)


def run_experiment(config):
    """Execute a complete experiment from config.

    This public wrapper preserves the historical import path while the
    implementation lives in composable pipeline stages.
    """
    return run_pipeline(config)


__all__ = ["_prediction_index_for_policy", "_test_frame_for_policy", "run_experiment"]
