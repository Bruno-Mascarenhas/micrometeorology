"""Tests for allsky.evaluation.metrics (regression + classification helpers).

Torch-free: numpy + scikit-learn only.  Covers nmae/nrmse normalization, NaN and
empty safety, and degenerate single-class classification input.
"""

from __future__ import annotations

import math

import numpy as np

from allsky.evaluation.metrics import (
    CLASSIFICATION_METRIC_KEYS,
    REGRESSION_METRIC_KEYS,
    classification_metrics,
    regression_metrics,
)


class TestRegressionMetrics:
    def test_perfect_prediction(self):
        obs = np.array([100.0, 200.0, 300.0, 400.0])
        metrics = regression_metrics(obs, obs.copy())
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["bias"] == 0.0
        assert metrics["nmae"] == 0.0
        assert metrics["nrmse"] == 0.0
        assert metrics["n"] == 4.0

    def test_bias_alias_matches_mbe(self):
        obs = np.array([10.0, 20.0, 30.0])
        pred = np.array([12.0, 24.0, 33.0])  # over-predicts -> positive bias
        metrics = regression_metrics(obs, pred)
        assert metrics["bias"] == metrics["mbe"]
        assert metrics["bias"] > 0

    def test_nmae_nrmse_normalized_by_mean_obs(self):
        obs = np.array([100.0, 100.0, 100.0, 100.0])  # mean 100
        pred = np.array([110.0, 90.0, 110.0, 90.0])  # abs error 10 each
        metrics = regression_metrics(obs, pred)
        assert metrics["mae"] == 10.0
        assert math.isclose(metrics["nmae"], 0.1)
        assert math.isclose(metrics["nrmse"], metrics["rmse"] / 100.0)

    def test_nan_pairs_are_dropped(self):
        obs = np.array([100.0, np.nan, 300.0, 400.0])
        pred = np.array([100.0, 200.0, np.nan, 400.0])
        metrics = regression_metrics(obs, pred)
        # only rows 0 and 3 survive (both finite) -> perfect
        assert metrics["n"] == 2.0
        assert metrics["mae"] == 0.0

    def test_empty_is_nan_filled_with_count(self):
        metrics = regression_metrics(np.array([]), np.array([]))
        assert set(metrics) == set(REGRESSION_METRIC_KEYS)
        assert metrics["n"] == 0.0
        assert math.isnan(metrics["rmse"])
        assert math.isnan(metrics["nmae"])

    def test_all_nan_is_safe(self):
        obs = np.array([np.nan, np.nan])
        metrics = regression_metrics(obs, obs)
        assert metrics["n"] == 0.0
        assert math.isnan(metrics["mae"])

    def test_zero_mean_obs_gives_nan_normalized(self):
        obs = np.array([-10.0, 10.0, -20.0, 20.0])  # mean 0
        pred = np.array([-9.0, 11.0, -19.0, 21.0])
        metrics = regression_metrics(obs, pred)
        assert math.isnan(metrics["nmae"])
        assert math.isnan(metrics["nrmse"])
        assert not math.isnan(metrics["mae"])


class TestClassificationMetrics:
    def test_perfect_three_class(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        metrics = classification_metrics(y, y.copy(), n_classes=3)
        assert metrics["accuracy"] == 1.0
        assert metrics["balanced_accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["n"] == 6
        assert metrics["confusion"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_missing_labels_dropped(self):
        y_true = np.array([0, 1, -1, 2, 1])  # one missing sentinel
        y_pred = np.array([0, 1, 2, 2, 1])
        metrics = classification_metrics(y_true, y_pred, n_classes=3)
        assert metrics["n"] == 4  # the -1 row is excluded
        assert metrics["accuracy"] == 1.0

    def test_confusion_is_fixed_shape(self):
        metrics = classification_metrics(np.array([0, 0]), np.array([0, 1]), n_classes=3)
        assert len(metrics["confusion"]) == 3
        assert all(len(row) == 3 for row in metrics["confusion"])

    def test_degenerate_single_class_is_safe(self):
        # Every row is class 0 and predicted 0 (a common early-training collapse).
        y = np.zeros(5, dtype=np.int64)
        metrics = classification_metrics(y, y, n_classes=3)
        assert set(CLASSIFICATION_METRIC_KEYS).issubset(metrics)
        assert metrics["accuracy"] == 1.0
        assert 0.0 <= metrics["macro_f1"] <= 1.0
        assert not math.isnan(metrics["balanced_accuracy"])

    def test_empty_is_nan_filled(self):
        metrics = classification_metrics(np.array([]), np.array([]), n_classes=3)
        assert metrics["n"] == 0
        assert math.isnan(metrics["accuracy"])
        assert metrics["confusion"] == [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
