"""Torch-gated tests for allsky.training.losses.MultitaskLoss.

Covers each loss kind, target normalization, per-head masking (including an
all-missing batch that must contribute an exact, finite zero), the
heteroscedastic NLL sanity property and the fixed per-head weighting.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from allsky.config import TargetsConfig  # noqa: E402
from allsky.features.normalization import TargetNormalizer  # noqa: E402
from allsky.training.losses import MultitaskLoss  # noqa: E402

# Non-trivial normalizers so "normalization was applied" is observable.
NORMS = {
    "dhi": TargetNormalizer(mean=100.0, std=50.0),
    "kindex": TargetNormalizer(mean=0.5, std=0.2),
}


def _targets(**heads: dict) -> TargetsConfig:
    """Build a TargetsConfig with every head disabled unless named in *heads*."""
    base = {
        "dhi": {"enabled": False},
        "kindex": {"enabled": False},
        "sky": {"enabled": False},
        "cloud_fraction": {"enabled": False},
    }
    base.update(heads)
    return TargetsConfig.model_validate(base)


class TestRegressionKinds:
    @pytest.mark.parametrize("kind", ["mse", "mae", "huber"])
    def test_dhi_regression_kinds_finite_and_grad(self, kind: str):
        targets = _targets(dhi={"enabled": True, "loss": kind, "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.zeros(4, requires_grad=True)
        batch = _batch(dhi=torch.tensor([100.0, 150.0, 50.0, 100.0]))
        out = loss_fn({"dhi": pred}, batch)
        assert set(out) == {"loss", "loss_dhi"}
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert pred.grad is not None

    def test_mse_uses_normalized_targets(self):
        targets = _targets(dhi={"enabled": True, "loss": "mse", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.tensor([0.0, 1.0])
        raw = torch.tensor([100.0, 150.0])  # -> normalized (0.0, 1.0) with mean 100 std 50
        out = loss_fn({"dhi": pred}, _batch(dhi=raw))
        # Perfect prediction in normalized space -> zero.
        assert float(out["loss_dhi"]) == pytest.approx(0.0, abs=1e-6)

    def test_mse_value_matches_manual_normalization(self):
        targets = _targets(kindex={"enabled": True, "loss": "mse", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.tensor([0.0, 0.0])
        raw = torch.tensor([0.7, 0.9])  # normalized: (0.2/0.2, 0.4/0.2) = (1.0, 2.0)
        out = loss_fn({"kindex": pred}, _batch(kindex=raw))
        expected = ((1.0 - 0.0) ** 2 + (2.0 - 0.0) ** 2) / 2
        assert float(out["loss_kindex"]) == pytest.approx(expected, rel=1e-6)

    def test_cloud_fraction_not_normalized(self):
        targets = _targets(cloud_fraction={"enabled": True, "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.tensor([0.3, 0.6])
        target = torch.tensor([0.3, 0.6])  # already in [0, 1]; compared raw
        out = loss_fn({"cloud_fraction": pred}, _batch(cloud_fraction=target))
        assert float(out["loss_cloud_fraction"]) == pytest.approx(0.0, abs=1e-6)


class TestSky:
    def test_cross_entropy_and_grad(self):
        targets = _targets(sky={"enabled": True, "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        logits = torch.randn(4, 3, requires_grad=True)
        batch = _batch(sky_class=torch.tensor([0, 1, 2, 0]))
        out = loss_fn({"sky_logits": logits}, batch)
        assert set(out) == {"loss", "loss_sky"}
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert logits.grad is not None


class TestMasking:
    def test_missing_rows_ignored(self):
        targets = _targets(dhi={"enabled": True, "loss": "mae", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.tensor([0.0, 0.0, 0.0])
        # Middle row missing (NaN) -> ignored; valid rows both normalize to 0.
        raw = torch.tensor([100.0, float("nan"), 100.0])
        out = loss_fn({"dhi": pred}, _batch(dhi=raw))
        assert float(out["loss_dhi"]) == pytest.approx(0.0, abs=1e-6)

    def test_sky_missing_class_ignored(self):
        targets = _targets(sky={"enabled": True, "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        logits = torch.zeros(3, 3)
        # -1 is masked; the two valid rows both target class 0 with equal logits.
        out = loss_fn({"sky_logits": logits}, _batch(sky_class=torch.tensor([0, -1, 0])))
        assert float(out["loss_sky"]) == pytest.approx(-torch.log(torch.tensor(1 / 3)), rel=1e-5)

    def test_all_missing_batch_contributes_finite_zero(self):
        targets = _targets(
            dhi={"enabled": True, "loss": "huber", "weight": 2.0},
            sky={"enabled": True, "weight": 1.0},
        )
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.zeros(3, requires_grad=True)
        logits = torch.zeros(3, 3, requires_grad=True)
        batch = _batch(
            dhi=torch.tensor([float("nan")] * 3),
            sky_class=torch.tensor([-1, -1, -1]),
        )
        out = loss_fn({"dhi": pred, "sky_logits": logits}, batch)
        assert float(out["loss_dhi"]) == 0.0
        assert float(out["loss_sky"]) == 0.0
        assert torch.isfinite(out["loss"])
        assert float(out["loss"]) == 0.0
        # Still grad-safe (backward must not raise even with zero contributions).
        out["loss"].backward()

    def test_heteroscedastic_all_missing_is_zero(self):
        targets = _targets(dhi={"enabled": True, "loss": "heteroscedastic", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.zeros(2, requires_grad=True)
        log_var = torch.zeros(2, requires_grad=True)
        out = loss_fn(
            {"dhi": pred, "dhi_log_var": log_var},
            _batch(dhi=torch.tensor([float("nan"), float("nan")])),
        )
        assert float(out["loss_dhi"]) == 0.0
        out["loss"].backward()


class TestHeteroscedastic:
    def test_higher_log_var_lowers_loss_for_large_errors(self):
        # Identity normalizer so the raw residual is the normalized residual.
        norms = {"dhi": TargetNormalizer(mean=0.0, std=1.0)}
        targets = _targets(dhi={"enabled": True, "loss": "heteroscedastic", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, norms)
        pred = torch.zeros(2)
        target = torch.full((2,), 5.0)  # large error
        low = loss_fn({"dhi": pred, "dhi_log_var": torch.zeros(2)}, _batch(dhi=target))["loss_dhi"]
        high = loss_fn({"dhi": pred, "dhi_log_var": torch.full((2,), 3.0)}, _batch(dhi=target))[
            "loss_dhi"
        ]
        assert float(high) < float(low)

    def test_higher_log_var_raises_loss_for_small_errors(self):
        norms = {"dhi": TargetNormalizer(mean=0.0, std=1.0)}
        targets = _targets(dhi={"enabled": True, "loss": "heteroscedastic", "weight": 1.0})
        loss_fn = MultitaskLoss(targets, norms)
        pred = torch.zeros(2)
        target = torch.zeros(2)  # zero error: inflating variance only adds the log-var penalty
        low = loss_fn({"dhi": pred, "dhi_log_var": torch.zeros(2)}, _batch(dhi=target))["loss_dhi"]
        high = loss_fn({"dhi": pred, "dhi_log_var": torch.full((2,), 3.0)}, _batch(dhi=target))[
            "loss_dhi"
        ]
        assert float(high) > float(low)


class TestWeighting:
    def test_single_head_total_is_weighted_component(self):
        targets = _targets(dhi={"enabled": True, "loss": "mse", "weight": 3.0})
        loss_fn = MultitaskLoss(targets, NORMS)
        pred = torch.tensor([0.0, 0.0])
        out = loss_fn({"dhi": pred}, _batch(dhi=torch.tensor([200.0, 0.0])))
        assert float(out["loss"]) == pytest.approx(3.0 * float(out["loss_dhi"]), rel=1e-6)

    def test_total_is_weighted_sum_of_components(self):
        targets = _targets(
            dhi={"enabled": True, "loss": "mse", "weight": 2.0},
            sky={"enabled": True, "weight": 0.5},
        )
        loss_fn = MultitaskLoss(targets, NORMS)
        out = loss_fn(
            {"dhi": torch.tensor([0.0, 1.0]), "sky_logits": torch.randn(2, 3)},
            _batch(dhi=torch.tensor([120.0, 180.0]), sky_class=torch.tensor([0, 2])),
        )
        expected = 2.0 * float(out["loss_dhi"]) + 0.5 * float(out["loss_sky"])
        assert float(out["loss"]) == pytest.approx(expected, rel=1e-6)

    def test_learned_uncertainty_stub_raises(self):
        targets = _targets(dhi={"enabled": True, "loss": "mse"})
        with pytest.raises(NotImplementedError, match="learned uncertainty"):
            MultitaskLoss(targets, NORMS, learned_uncertainty=True)


def _batch(**overrides: Any) -> dict[str, Any]:
    """A batch dict with sensible defaults for every target key."""
    size_source = next(iter(overrides.values()))
    size = int(size_source.shape[0])
    batch: dict[str, Any] = {
        "features": torch.zeros(size, 3),
        "dhi": torch.full((size,), float("nan")),
        "kindex": torch.full((size,), float("nan")),
        "sky_class": torch.full((size,), -1, dtype=torch.long),
        "cloud_fraction": torch.full((size,), float("nan")),
    }
    batch.update(overrides)
    return batch
