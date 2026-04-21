"""Tests for model interface consistency."""

from __future__ import annotations

import numpy as np
import pytest

from solrad_correction.datasets.tabular import TabularDataset
from solrad_correction.models.base import BaseRegressorModel
from solrad_correction.models.svm import SVMRegressor


class TestBaseInterface:
    def test_is_abstract(self):
        """BaseRegressorModel cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseRegressorModel()  # type: ignore

    def test_svm_has_required_methods(self):
        model = SVMRegressor()
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "evaluate")
        assert hasattr(model, "save")
        assert hasattr(model, "name")


class TestSVMInterface:
    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        features = rng.normal(0, 1, (100, 5)).astype(np.float32)
        targets = (
            features[:, 0] * 2 + features[:, 1] - 0.5 * features[:, 2] + rng.normal(0, 0.1, 100)
        ).astype(np.float32)
        return TabularDataset(X=features, y=targets, feature_names=[f"f{i}" for i in range(5)])

    def test_fit_predict(self, synthetic_data):
        model = SVMRegressor(kernel="rbf", C=10.0)
        model.fit(synthetic_data)
        preds = model.predict(synthetic_data)
        assert preds.shape == (100,)
        assert preds.dtype == np.float32

    def test_evaluate(self, synthetic_data):
        model = SVMRegressor(kernel="rbf", C=10.0)
        model.fit(synthetic_data)
        metrics = model.evaluate(synthetic_data)
        assert "RMSE" in metrics
        assert "MAE" in metrics
        assert "R²" in metrics

    def test_save_load(self, synthetic_data, tmp_path):
        model = SVMRegressor(kernel="rbf", C=10.0)
        model.fit(synthetic_data)
        preds_before = model.predict(synthetic_data)

        path = tmp_path / "svm_test.joblib"
        model.save(path)

        loaded = SVMRegressor.load(path)
        preds_after = loaded.predict(synthetic_data)
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    def test_name(self):
        model = SVMRegressor(kernel="rbf")
        assert "SVM" in model.name
