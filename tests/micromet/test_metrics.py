"""Tests for statistical metrics."""

from __future__ import annotations

import numpy as np
import pytest

from micrometeorology.stats.metrics import (
    compute_all,
    correlation,
    d_index,
    ioa,
    mae,
    mbe,
    r_squared,
    rmse,
)


class TestRMSE:
    def test_identical(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rmse(obs, obs) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.5, 2.5, 3.5])
        # RMSE = sqrt(mean([0.25, 0.25, 0.25])) = sqrt(0.25) = 0.5
        assert rmse(obs, pred) == pytest.approx(0.5)

    def test_with_nans(self):
        obs = np.array([1.0, np.nan, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert rmse(obs, pred) == pytest.approx(0.0, abs=1e-10)


class TestMAE:
    def test_identical(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert mae(obs, obs) == pytest.approx(0.0, abs=1e-10)

    def test_known_value(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 3.0, 4.0])
        assert mae(obs, pred) == pytest.approx(1.0)


class TestMBE:
    def test_no_bias(self):
        obs = np.array([1.0, 2.0, 3.0])
        assert mbe(obs, obs) == pytest.approx(0.0, abs=1e-10)

    def test_positive_bias(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([2.0, 3.0, 4.0])
        assert mbe(obs, pred) == pytest.approx(1.0)

    def test_negative_bias(self):
        obs = np.array([2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert mbe(obs, pred) == pytest.approx(-1.0)


class TestRSquared:
    def test_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r_squared(obs, obs) == pytest.approx(1.0)

    def test_range(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.1, 1.9, 3.2, 3.8])
        r2 = r_squared(obs, pred)
        assert 0 < r2 <= 1


class TestDIndex:
    def test_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert d_index(obs, obs) == pytest.approx(1.0)

    def test_range(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5])
        d = d_index(obs, pred)
        assert 0 <= d <= 1


class TestCorrelation:
    def test_perfect_positive(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        assert correlation(obs, obs) == pytest.approx(1.0)

    def test_perfect_negative(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([4.0, 3.0, 2.0, 1.0])
        assert correlation(obs, pred) == pytest.approx(-1.0)


class TestIOA:
    def test_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ioa(obs, obs) == pytest.approx(1.0)


class TestComputeAll:
    def test_returns_all_metrics(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = np.array([1.1, 2.2, 2.8, 4.1, 5.3])
        result = compute_all(obs, pred)
        assert "RMSE" in result
        assert "MAE" in result
        assert "MBE" in result
        assert "R²" in result
        assert "r" in result
        assert "d" in result
        assert "IOA" in result
        assert "NRMSE" in result

    def test_insufficient_data(self):
        obs = np.array([1.0, np.nan])
        pred = np.array([np.nan, 2.0])
        result = compute_all(obs, pred)
        assert all(np.isnan(v) for v in result.values())
