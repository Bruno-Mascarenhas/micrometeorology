"""Torch-gated tests for C-Mixup: the pairing, the blend, and the engine wiring.

The pairing is what makes the method C-Mixup rather than mixup — a partner
drawn by label distance — so the tests pin that a narrow kernel picks the
label neighbour, that the blend is the convex combination the plan names, that
a row with a missing target is left alone, and that with the mixer off (or
gated shut) the engine trains bit for bit as it trained without it.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from allsky.config import ExperimentConfig, TargetsConfig
from allsky.features.normalization import TargetNormalizer
from allsky.training.cmixup import CMixup, MixPlan
from allsky.training.engine import run_experiment
from allsky.training.losses import MultitaskLoss
from labmim_core.sky import SKY_CLASS_COUNT
from tests.allsky import _synthetic as synthetic


def _batch(
    kindex: list[float],
    *,
    dhi: list[float] | None = None,
    sky: list[int] | None = None,
    side: int = 4,
) -> dict[str, torch.Tensor]:
    n_rows = len(kindex)
    return {
        "image": torch.arange(n_rows * 3 * side * side, dtype=torch.float32).reshape(
            n_rows, 3, side, side
        ),
        "features": torch.arange(n_rows * 2, dtype=torch.float32).reshape(n_rows, 2),
        "kindex": torch.tensor(kindex),
        "dhi": torch.tensor(dhi if dhi is not None else [0.5] * n_rows),
        "dhi_scale": torch.ones(n_rows),
        "sky_class": torch.tensor(sky if sky is not None else [0] * n_rows),
        "cloud_fraction": torch.full((n_rows,), float("nan")),
    }


def _mixer(bandwidth: float = 0.05, *, p: float = 1.0, sky: bool = True) -> CMixup:
    return CMixup(
        alpha=1.0, bandwidth=bandwidth, p=p, regression_targets=("dhi", "kindex"), sky=sky
    )


class TestPairing:
    def test_a_narrow_kernel_always_pairs_a_row_with_its_label_neighbour(self):
        batch = _batch([0.10, 0.11, 0.30, 0.31])
        mixer = _mixer(bandwidth=0.01)

        partners = [mixer.plan(batch, np.random.default_rng(seed)) for seed in range(40)]

        assert all(plan is not None for plan in partners)
        assert all(plan.partner.tolist() == [1, 0, 3, 2] for plan in partners if plan is not None)

    def test_a_wide_kernel_reaches_the_far_rows_too(self):
        batch = _batch([0.10, 0.11, 0.30, 0.31])
        mixer = _mixer(bandwidth=1.0)

        partners = {
            int(plan.partner[0])
            for seed in range(40)
            if (plan := mixer.plan(batch, np.random.default_rng(seed))) is not None
        }

        assert partners == {1, 2, 3}

    def test_a_row_never_pairs_with_itself(self):
        batch = _batch([0.10, 0.50, 0.90])
        mixer = _mixer(bandwidth=0.05)

        for seed in range(20):
            plan = mixer.plan(batch, np.random.default_rng(seed))
            assert plan is not None
            assert (plan.partner != torch.arange(3)).all()

    def test_a_row_with_a_missing_target_is_neither_mixed_nor_chosen(self):
        batch = _batch([0.10, 0.11, float("nan"), 0.12], sky=[0, 1, 1, -1])
        mixer = _mixer(bandwidth=0.05)

        for seed in range(20):
            plan = mixer.plan(batch, np.random.default_rng(seed))
            assert plan is not None
            assert plan.mixed.tolist() == [True, True, False, False]
            assert plan.lam[2:].tolist() == [1.0, 1.0]
            assert plan.partner[2:].tolist() == [2, 3]
            assert set(plan.partner[:2].tolist()) <= {0, 1}

    def test_fewer_than_two_labelled_rows_mix_nothing(self):
        batch = _batch([0.10, float("nan"), float("nan")])

        assert _mixer().plan(batch, np.random.default_rng(0)) is None

    def test_the_gate_skips_the_fraction_of_batches_p_names(self):
        batch = _batch([0.10, 0.11, 0.12, 0.13])

        mixed = sum(
            _mixer(p=0.0).plan(batch, np.random.default_rng(seed)) is not None for seed in range(20)
        )

        assert mixed == 0

    def test_the_weight_comes_from_the_beta_the_plan_was_drawn_with(self):
        batch = _batch([0.10, 0.11, 0.12, 0.13])
        replay = np.random.default_rng(3)
        replay.random()
        replay.random(4)
        expected = replay.beta(1.0, 1.0, size=4).astype(np.float32)

        plan = _mixer().plan(batch, np.random.default_rng(3))

        assert plan is not None
        np.testing.assert_array_equal(plan.lam.numpy(), expected)


class TestBlend:
    @staticmethod
    def _plan(lam: list[float], partner: list[int], mixed: list[bool]) -> MixPlan:
        return MixPlan(
            partner=torch.tensor(partner),
            lam=torch.tensor(lam, dtype=torch.float32),
            mixed=torch.tensor(mixed),
        )

    def test_inputs_and_regression_targets_are_the_convex_combination_of_the_pair(self):
        batch = _batch([0.2, 0.4, 0.6], dhi=[10.0, 20.0, 30.0])
        plan = self._plan([0.25, 0.75, 1.0], [1, 2, 2], [True, True, False])

        out = _mixer().apply(batch, plan)

        torch.testing.assert_close(out["kindex"], torch.tensor([0.35, 0.45, 0.6]))
        torch.testing.assert_close(out["dhi"], torch.tensor([17.5, 22.5, 30.0]))
        torch.testing.assert_close(
            out["image"][0], 0.25 * batch["image"][0] + 0.75 * batch["image"][1]
        )
        torch.testing.assert_close(
            out["features"][1], 0.75 * batch["features"][1] + 0.25 * batch["features"][2]
        )

    def test_the_sky_class_becomes_a_two_point_distribution(self):
        batch = _batch([0.2, 0.4], sky=[0, 3])
        plan = self._plan([0.3, 0.6], [1, 0], [True, True])

        out = _mixer().apply(batch, plan)

        expected = torch.zeros(2, SKY_CLASS_COUNT)
        expected[0, 0], expected[0, 3] = 0.3, 0.7
        expected[1, 3], expected[1, 0] = 0.6, 0.4
        torch.testing.assert_close(out["sky_distribution"], expected)
        assert torch.equal(out["sky_class"], batch["sky_class"])

    def test_an_unmixed_row_keeps_its_own_sample_and_a_one_hot_target(self):
        batch = _batch([0.2, float("nan")], sky=[2, 1])
        plan = self._plan([1.0, 1.0], [0, 1], [False, False])

        out = _mixer().apply(batch, plan)

        assert torch.equal(out["image"], batch["image"])
        assert torch.isnan(out["kindex"][1])
        assert out["kindex"][0] == 0.2
        assert out["sky_distribution"][0].tolist() == [0.0, 0.0, 1.0, 0.0]

    def test_a_window_batch_is_refused(self):
        batch = _batch([0.2, 0.4])
        batch["image_seq"] = batch.pop("image").unsqueeze(1)

        with pytest.raises(ValueError, match="image"):
            _mixer().apply(batch, self._plan([0.5, 0.5], [1, 0], [True, True]))

    def test_the_kernel_target_must_be_among_the_mixed_ones(self):
        with pytest.raises(ValueError, match="kindex"):
            CMixup(alpha=1.0, bandwidth=0.05, p=1.0, regression_targets=("dhi",), sky=False)


NORMS = {"dhi": TargetNormalizer(mean=0.0, std=1.0), "kindex": TargetNormalizer(mean=0.0, std=1.0)}


def _sky_loss(sky: dict[str, Any]) -> MultitaskLoss:
    targets = TargetsConfig.model_validate(
        {"dhi": {"enabled": False}, "kindex": {"enabled": False}, "sky": {"enabled": True, **sky}}
    )
    return MultitaskLoss(targets, NORMS)


def _sky_batch(labels: list[int], distribution: torch.Tensor | None = None) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "features": torch.zeros(len(labels), 2),
        "dhi": torch.full((len(labels),), float("nan")),
        "kindex": torch.full((len(labels),), float("nan")),
        "cloud_fraction": torch.full((len(labels),), float("nan")),
        "sky_class": torch.tensor(labels),
    }
    if distribution is not None:
        batch["sky_distribution"] = distribution
    return batch


class TestSkyLossWithADistribution:
    LOGITS = torch.tensor([[2.0, 0.5, -1.0, 0.0], [0.1, 0.2, 1.5, -0.3]])

    def test_a_mixed_row_costs_the_convex_combination_of_the_two_hard_losses(self):
        loss_fn = _sky_loss({})
        distribution = torch.zeros(2, SKY_CLASS_COUNT)
        distribution[0, 0], distribution[0, 2] = 0.3, 0.7
        distribution[1, 3] = 1.0

        mixed = loss_fn({"sky_logits": self.LOGITS}, _sky_batch([0, 3], distribution))["loss_sky"]

        rows = torch.nn.functional.cross_entropy(
            self.LOGITS, torch.tensor([0, 3]), reduction="none"
        )
        alternative = torch.nn.functional.cross_entropy(
            self.LOGITS[:1], torch.tensor([2]), reduction="none"
        )
        expected = ((0.3 * rows[0] + 0.7 * alternative[0]) + rows[1]) / 2
        assert float(mixed) == pytest.approx(float(expected), rel=1e-5)

    @pytest.mark.parametrize(
        "sky",
        [
            {},
            {"label_smoothing": 0.1},
            {"ordinal_tau": 1.0},
            {"class_weights": (1.0, 2.0, 3.0, 4.0)},
            {"class_weights": (1.0, 2.0, 3.0, 4.0), "label_smoothing": 0.1},
            {"class_weights": (1.0, 2.0, 3.0, 4.0), "ordinal_tau": 1.0},
        ],
        ids=["plain", "smoothing", "ordinal", "weights", "weights+smoothing", "weights+ordinal"],
    )
    def test_a_one_hot_distribution_costs_what_the_hard_label_costs(self, sky: dict[str, Any]):
        loss_fn = _sky_loss(sky)
        labels = [0, 2]
        one_hot = torch.nn.functional.one_hot(torch.tensor(labels), SKY_CLASS_COUNT).float()

        hard = loss_fn({"sky_logits": self.LOGITS}, _sky_batch(labels))["loss_sky"]
        soft = loss_fn({"sky_logits": self.LOGITS}, _sky_batch(labels, one_hot))["loss_sky"]

        assert float(soft) == pytest.approx(float(hard), rel=1e-5)

    def test_a_row_without_a_class_is_masked_whatever_its_distribution_says(self):
        loss_fn = _sky_loss({})
        distribution = torch.full((2, SKY_CLASS_COUNT), 0.25)

        out = loss_fn({"sky_logits": self.LOGITS}, _sky_batch([1, -1], distribution))["loss_sky"]

        expected = -(0.25 * torch.log_softmax(self.LOGITS[0], dim=-1)).sum()
        assert float(out) == pytest.approx(float(expected), rel=1e-5)


class _StubBackbone(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.dim = dim
        self.model: Any = nn.Module()
        self.model.patch_embed = nn.Module()
        self.model.patch_embed.proj = nn.Conv2d(3, 3, kernel_size=1)
        self.proj = nn.Linear(3, dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.proj(self.model.patch_embed.proj(image).mean(dim=(2, 3)))
        return out


def _image_cfg(root: Path, **overrides: Any) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "experiment": True,
        "seed": 0,
        "output_dir": str(root / "out"),
        "data": {"data_root": str(root), "input_mode": "image"},
        "features": {"set": "safe"},
        "targets": {
            "dhi": {"enabled": True, "loss": "mae"},
            "kindex": {"enabled": True, "kind": "kstar", "loss": "mae"},
            "sky": {"enabled": True},
        },
        "model": {"name": "image_only", "image_size": 8, "backbone_frozen": False},
        "train": {
            "epochs": 2,
            "batch_size": 8,
            "num_workers": 0,
            "device": "cpu",
            "early_stopping": {"monitor": "val/kindex_mae", "patience": 100},
        },
    }
    for section, values in overrides.items():
        payload[section] = {**payload[section], **values}
    return ExperimentConfig.model_validate(payload)


def _run(tmp_path: Path, name: str, cfg: ExperimentConfig, root: Path) -> pd.DataFrame:
    run_dir = tmp_path / name
    run_experiment(
        cfg, data_root=root, output_dir=run_dir, image_backbone_builder=lambda: _StubBackbone()
    )
    return pd.read_csv(run_dir / "metrics.csv")


class TestEngineWiring:
    def test_a_gate_that_never_opens_trains_bit_for_bit_like_no_mixer(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)

        plain = _run(tmp_path, "plain", _image_cfg(root), root)
        gated = _run(
            tmp_path, "gated", _image_cfg(root, train={"cmixup": {"enabled": True, "p": 0.0}}), root
        )

        pd.testing.assert_frame_equal(plain, gated[plain.columns])

    def test_a_gate_that_never_opens_counts_no_mixed_row_and_is_warned_about(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)

        with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
            gated = _run(
                tmp_path,
                "gated",
                _image_cfg(root, train={"cmixup": {"enabled": True, "p": 0.0}}),
                root,
            )

        assert (gated["train_cmixup_mixed_rows"] == 0).all()
        assert "no row was mixed" in caplog.text

    def test_a_mixed_run_counts_the_rows_it_blended_per_epoch(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)

        mixed = _run(
            tmp_path,
            "mixed",
            _image_cfg(root, train={"cmixup": {"enabled": True, "bandwidth": 0.5}}),
            root,
        )

        assert (mixed["train_cmixup_mixed_rows"] > 0).all()

    def test_a_run_without_the_mixer_logs_no_mixed_row_column(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)

        plain = _run(tmp_path, "plain", _image_cfg(root), root)

        assert "train_cmixup_mixed_rows" not in plain.columns

    def test_a_mixed_run_trains_to_finite_metrics_on_a_different_trajectory(self, tmp_path: Path):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)

        plain = _run(tmp_path, "plain", _image_cfg(root), root)
        mixed = _run(
            tmp_path,
            "mixed",
            _image_cfg(root, train={"cmixup": {"enabled": True, "bandwidth": 0.5}}),
            root,
        )

        assert np.isfinite(mixed["train_loss"]).all()
        assert not np.allclose(mixed["train_loss"], plain["train_loss"])

    def test_the_default_is_off(self):
        assert ExperimentConfig().train.cmixup.enabled is False


class TestRotationWarning:
    def test_rotation_without_a_solar_channel_is_warned_about_at_start(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)
        cfg = _image_cfg(root, train={"epochs": 1})
        cfg = cfg.model_copy(
            update={"augmentation": cfg.augmentation.model_copy(update={"p_rotate": 1.0})}
        )

        with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
            _run(tmp_path, "run", cfg, root)

        assert any("geometry_channels" in record.message for record in caplog.records)

    def test_rotation_with_the_solar_channel_is_not_warned_about(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, _, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6, write_images=True)
        cfg = _image_cfg(root, train={"epochs": 1}, model={"geometry_channels": ["cos_sun_angle"]})
        cfg = cfg.model_copy(
            update={"augmentation": cfg.augmentation.model_copy(update={"p_rotate": 1.0})}
        )

        with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
            _run(tmp_path, "run", cfg, root)

        assert not any("geometry_channels" in record.message for record in caplog.records)
