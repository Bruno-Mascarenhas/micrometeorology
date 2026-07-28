"""Tests for statistical metrics."""

from __future__ import annotations

import numpy as np
import pytest

from micrometeorology.stats.metrics import (
    compute_all,
    correlation,
    d_index,
    ia,
    ioa,
    mae,
    mbe,
    nrmse,
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

    def test_negative_when_worse_than_mean(self):
        """R² is 1 - ss_res/ss_tot, not r², so it goes negative below the mean."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([3.0, 2.0, 1.0])
        # ss_res = 8, ss_tot = 2 -> 1 - 4 = -3
        assert r_squared(obs, pred) == pytest.approx(-3.0)
        assert correlation(obs, pred) == pytest.approx(-1.0)

    def test_negative_when_biased_but_perfectly_correlated(self):
        """A constant +10 bias keeps r == 1 while R² collapses."""
        obs = np.array([1.0, 2.0, 3.0])
        pred = obs + 10.0
        # ss_res = 300, ss_tot = 2 -> 1 - 150 = -149
        assert r_squared(obs, pred) == pytest.approx(-149.0)
        assert correlation(obs, pred) == pytest.approx(1.0)


class TestNRMSE:
    def test_normalised_by_observed_range(self):
        obs = np.array([100.0, 110.0])
        pred = np.array([101.0, 109.0])
        # RMSE 1.0 / range 10.0 = 0.1; normalising by mean(obs) would give 0.0095
        assert nrmse(obs, pred) == pytest.approx(0.1)

    def test_nan_when_observed_range_is_zero(self):
        obs = np.array([5.0, 5.0])
        pred = np.array([4.0, 6.0])
        assert np.isnan(nrmse(obs, pred))


class TestDIndex:
    def test_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert d_index(obs, obs) == pytest.approx(1.0)

    def test_range(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([1.5, 2.5, 3.5, 4.5])
        d = d_index(obs, pred)
        assert 0 <= d <= 1

    def test_ia_delegates_to_d_index(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([10.0, -5.0, 20.0, -8.0])
        assert ia(obs, pred) == d_index(obs, pred)


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

    def test_refined_ioa_uses_reciprocal_branch(self):
        """Above ratio 1 the refined form flips to 1/ratio - 1, keeping IOA in [-1, 1].

        A naive ``1 - ratio`` would report -4.625 here, breaching the documented
        lower bound.
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        pred = np.array([10.0, -5.0, 20.0, -8.0])
        # numerator 45, denominator 8 -> ratio 5.625 -> 1/5.625 - 1
        assert ioa(obs, pred) == pytest.approx(1.0 / 5.625 - 1.0)
        assert -1.0 <= ioa(obs, pred) <= 1.0


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

    def test_known_values(self):
        """Pin every ALL_METRICS entry to a value, not just its key.

        Key-presence plus the all-NaN case below is satisfied by a ``compute_all``
        that returns NaN unconditionally; these exact values are what rule that out
        and what pins each name to the right function.
        """
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = obs + 1.0
        result = compute_all(obs, pred)
        assert result["RMSE"] == pytest.approx(1.0)
        assert result["MAE"] == pytest.approx(1.0)
        assert result["MBE"] == pytest.approx(1.0)
        assert result["R²"] == pytest.approx(0.5)
        assert result["r"] == pytest.approx(1.0)
        assert result["d"] == pytest.approx(8 / 9)
        assert result["IOA"] == pytest.approx(7 / 12)
        assert result["NRMSE"] == pytest.approx(0.25)

    def test_insufficient_data(self):
        obs = np.array([1.0, np.nan])
        pred = np.array([np.nan, 2.0])
        result = compute_all(obs, pred)
        assert all(np.isnan(v) for v in result.values())
