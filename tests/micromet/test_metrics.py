"""Tests for statistical metrics."""

import numpy as np
import pytest

from micrometeorology.stats.metrics import (
    ALL_METRICS,
    compute_all,
    correlation,
    d_index,
    ioa,
    is_circular_column,
    mbe,
    nrmse,
    r_squared,
    rmse,
)


class TestRMSE:
    def test_with_nans(self):
        obs = np.array([1.0, np.nan, 3.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert rmse(obs, pred) == pytest.approx(0.0, abs=1e-10)


class TestMBE:
    def test_negative_bias(self):
        obs = np.array([2.0, 3.0, 4.0])
        pred = np.array([1.0, 2.0, 3.0])
        assert mbe(obs, pred) == pytest.approx(-1.0)


class TestRSquared:
    def test_perfect(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r_squared(obs, obs) == pytest.approx(1.0)

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
    def test_known_values(self):
        """Pin every ALL_METRICS entry to a value, not just its key.

        Key-presence plus the all-NaN case below is satisfied by a ``compute_all``
        returning NaN unconditionally; these exact values rule that out and bind
        each name to the right function.
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


class TestNonFinitePairsAreDropped:
    """One diverged prediction must not destroy the whole metrics record.

    ``np.isnan`` lets +-inf through, so a single infinite value would make
    RMSE/MAE/MBE/NRMSE inf, R2 -inf, r and d NaN, and IOA a plausible-looking
    but wrong -1.0.
    """

    @staticmethod
    def _pairs() -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        obs = rng.normal(200.0, 50.0, 500)
        pred = obs + rng.normal(0.0, 5.0, 500)
        return obs, pred

    def test_every_metric_matches_the_finite_subset(self):
        obs, pred = self._pairs()
        contaminated_obs = obs.copy()
        contaminated_pred = pred.copy()
        contaminated_pred[17] = np.inf
        contaminated_obs[3] = -np.inf
        contaminated_pred[42] = -np.inf
        contaminated_obs[99] = np.inf

        dropped = [3, 17, 42, 99]
        expected = compute_all(np.delete(obs, dropped), np.delete(pred, dropped))
        result = compute_all(contaminated_obs, contaminated_pred)

        assert set(result) == set(expected), "compute_all's key set is a cron-parsed contract"
        for name, value in expected.items():
            assert result[name] == pytest.approx(value), name
            assert np.isfinite(result[name]), name

    def test_ioa_is_not_the_misleading_minus_one(self):
        """IOA is the one metric whose inf breakage looks like a real number."""
        obs, pred = self._pairs()
        contaminated_pred = pred.copy()
        contaminated_pred[17] = np.inf
        assert ioa(obs, contaminated_pred) != pytest.approx(-1.0)
        assert ioa(obs, contaminated_pred) == pytest.approx(
            ioa(np.delete(obs, 17), np.delete(pred, 17))
        )

    def test_dropping_pairs_is_logged(self, caplog):
        """Trading a screaming inf for a normal-looking number must stay visible."""
        obs = np.array([1.0, 2.0, 3.0, np.inf])
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        with caplog.at_level("WARNING", logger="micrometeorology.stats.metrics"):
            rmse(obs, pred)
        assert "1" in caplog.text
        assert "4" in caplog.text

    def test_all_finite_input_is_untouched(self):
        obs, pred = self._pairs()
        result = compute_all(obs, pred)
        assert result["RMSE"] == pytest.approx(
            float(np.sqrt(np.mean(np.square(obs - pred)))), rel=0, abs=0
        )


class TestCircularColumnDetection:
    """Pinned to the column names the real station archive actually carries.

    A false negative publishes a bearing scored on the line; a false positive
    throws away five valid metrics of an ordinary linear column.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "WD",
            "WindDir",
            "WD_WXT_Avg",
            "WindDir_GMX",
            "WindDir1_GMX",
            "WindDir2_GMX",
            "WindDir_D1_WVT",
            "Wd_MeanUnitVector",
            "wind_direction",
            "dir_10m",
        ],
    )
    def test_bearings_are_circular(self, name):
        assert is_circular_column(name)

    @pytest.mark.parametrize(
        "name",
        [
            # Dispersions of direction are ordinary scalar spreads, not bearings.
            "WindDir_SD",
            "Wd_StdDev",
            "WindDir_SD1_WVT",
            "WS",
            "WS_ms_S_WVT",
            "T",
            "AirT_C_Avg",
            "pressure",
            "precip",
            "Sw_dw",
            # The separator in the "dir_" prefix is what keeps this one linear.
            "Direct_Wm2",
        ],
    )
    def test_linear_columns_are_not_circular(self, name):
        assert not is_circular_column(name)


class TestCircularMetrics:
    """A bearing scored on the line inflates RMSE and can flip the bias sign.

    Reproduced on the real paired series (station archive vs. the operational
    WRF run, n=15803 hours): linear RMSE 104.5 / MBE +4.6 against circular
    RMSE 64.5 / MBE -19.5, with 1888 pairs separated by more than 180 deg.
    """

    def test_the_short_way_round_is_taken(self):
        obs = np.array([350.0, 350.0, 10.0, 10.0])
        pred = np.array([10.0, 10.0, 350.0, 350.0])
        result = compute_all(obs, pred, circular=True)
        assert result["MAE"] == pytest.approx(20.0)
        assert result["RMSE"] == pytest.approx(20.0)
        # Two pairs +20 deg and two -20 deg: the bias cancels, it is not -340.
        assert result["MBE"] == pytest.approx(0.0)

    def test_the_bias_keeps_its_sign_convention(self):
        """Positive MBE = model clockwise of the observation, across 0 deg too."""
        obs = np.array([350.0, 350.0, 180.0])
        pred = np.array([10.0, 10.0, 200.0])
        assert compute_all(obs, pred, circular=True)["MBE"] == pytest.approx(20.0)

    def test_the_linear_reading_is_what_it_replaces(self):
        obs = np.array([350.0, 350.0, 180.0])
        pred = np.array([10.0, 10.0, 200.0])
        assert compute_all(obs, pred, circular=False)["MBE"] == pytest.approx(-220.0)

    def test_mean_normalised_metrics_are_suppressed(self):
        obs = np.array([350.0, 10.0, 180.0, 90.0])
        pred = np.array([10.0, 350.0, 200.0, 80.0])
        result = compute_all(obs, pred, circular=True)
        assert set(result) == set(ALL_METRICS), "the key set is parsed downstream"
        for name in ("RMSE", "MAE", "MBE"):
            assert np.isfinite(result[name]), name
        for name in ("R²", "r", "d", "IOA", "NRMSE"):
            assert np.isnan(result[name]), name

    def test_a_linear_column_is_untouched_by_the_flag_default(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = obs + 1.0
        assert compute_all(obs, pred) == compute_all(obs, pred, circular=False)
        assert np.isfinite(compute_all(obs, pred)["R²"])

    def test_wrapping_never_changes_a_within_range_residual(self):
        """Below 180 deg of separation the wrap is a no-op, so nothing shifts."""
        obs = np.array([90.0, 100.0, 110.0, 120.0])
        pred = np.array([95.0, 90.0, 130.0, 115.0])
        circular = compute_all(obs, pred, circular=True)
        linear = compute_all(obs, pred, circular=False)
        for name in ("RMSE", "MAE", "MBE"):
            assert circular[name] == pytest.approx(linear[name]), name
