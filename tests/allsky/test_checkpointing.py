"""Torch-gated tests for allsky.training.checkpointing.

Covers payload completeness against the executor spec, atomic-write safety on an
injected failure, a full state round-trip and the ``_orig_mod.`` compile-prefix
strip on load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from allsky.training.checkpointing import (  # noqa: E402
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)


def _tiny_state() -> tuple[nn.Module, Any, Any]:
    """A tiny model + optimizer (stepped once) + a cosine scheduler."""
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(4, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    return model, optimizer, scheduler


def _save(path: Path, model: nn.Module, optimizer: Any, scheduler: Any, **overrides: Any) -> Path:
    kwargs: dict[str, Any] = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": None,
        "epoch": 2,
        "global_step": 8,
        "best_metric": {"name": "loss", "value": 0.5, "epoch": 1},
        "config": {"name": "exp", "seed": 0},
        "normalizers": {
            "feature_normalizer": {"columns": ["a"], "mean": [0.0], "std": [1.0]},
            "target_normalizers": {"dhi": {"mean": 100.0, "std": 50.0}},
        },
        "feature_columns": ["a", "b", "c"],
        "feature_groups": {"solar": ["a"], "temperature": ["b"]},
        "dataset_version": "2",
        "split_id": "deadbeef",
        "manifest_sha256": "cafef00d",
        "backbone_info": {
            "name": "fake",
            "revision": "r0",
            "pooling": "cls",
            "dim": 8,
            "frozen": True,
        },
    }
    kwargs.update(overrides)
    return Path(save_checkpoint(path, **kwargs))


class TestPayloadCompleteness:
    def test_every_spec_key_present(self, tmp_path: Path):
        model, optimizer, scheduler = _tiny_state()
        _save(tmp_path / "last.ckpt", model, optimizer, scheduler)
        ckpt = load_checkpoint(tmp_path / "last.ckpt")

        for key in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "epoch",
            "global_step",
            "epochs_no_improve",
            "best_metric",
            "config",
            "normalizers",
            "feature_columns",
            "feature_groups",
            "dataset_version",
            "split_id",
            "manifest_sha256",
            "backbone",
            "code_version",
            "rng_state",
        ):
            assert key in ckpt, f"missing checkpoint key {key!r}"

        assert set(ckpt["best_metric"]) >= {"name", "value"}
        assert set(ckpt["normalizers"]) == {"feature_normalizer", "target_normalizers"}
        assert ckpt["scheduler_state"] is not None
        assert ckpt["scaler_state"] is None
        assert set(ckpt["rng_state"]) >= {"python", "numpy", "torch"}
        assert ckpt["backbone"]["dim"] == 8
        assert (
            ckpt["code_version"]["package_version"] is not None
            or "git_commit" in ckpt["code_version"]
        )
        assert ckpt["feature_columns"] == ["a", "b", "c"]
        # New optional field: None when not supplied (old checkpoints omit it).
        assert ckpt["epochs_no_improve"] is None

    def test_epochs_no_improve_round_trips(self, tmp_path: Path):
        model, optimizer, scheduler = _tiny_state()
        _save(tmp_path / "last.ckpt", model, optimizer, scheduler, epochs_no_improve=3)
        ckpt = load_checkpoint(tmp_path / "last.ckpt")
        assert ckpt["epochs_no_improve"] == 3


class TestAtomicWrite:
    def test_injected_failure_leaves_no_partial_file(self, tmp_path: Path, monkeypatch):
        model, optimizer, scheduler = _tiny_state()
        target = tmp_path / "last.ckpt"

        import allsky.training.checkpointing as checkpointing

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(checkpointing.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            _save(target, model, optimizer, scheduler)

        assert not target.exists()
        # No leftover temp file in the directory either.
        assert list(tmp_path.glob(".last.ckpt.tmp-*")) == []


class TestRoundTrip:
    def test_restores_model_and_optimizer_state(self, tmp_path: Path):
        model, optimizer, scheduler = _tiny_state()
        _save(tmp_path / "last.ckpt", model, optimizer, scheduler)

        # Fresh, differently-initialized model/optimizer.
        torch.manual_seed(999)
        restored = nn.Linear(3, 2)
        assert not torch.allclose(next(restored.parameters()), next(model.parameters()))
        restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_sched = torch.optim.lr_scheduler.CosineAnnealingLR(restored_opt, T_max=5)

        ckpt = load_checkpoint(tmp_path / "last.ckpt")
        restored.load_state_dict(ckpt["model_state"])
        restored_opt.load_state_dict(ckpt["optimizer_state"])
        restored_sched.load_state_dict(ckpt["scheduler_state"])

        for a, b in zip(model.parameters(), restored.parameters(), strict=True):
            assert torch.allclose(a, b)
        # Optimizer moment buffers survived.
        assert restored_opt.state_dict()["state"], "optimizer state was not restored"
        assert ckpt["epoch"] == 2
        assert ckpt["global_step"] == 8

    def test_strips_orig_mod_prefix_on_load(self, tmp_path: Path):
        model, optimizer, scheduler = _tiny_state()
        path = tmp_path / "compiled.ckpt"
        _save(path, model, optimizer, scheduler)

        # Rewrite the payload with compile-prefixed model_state keys.
        raw = torch.load(path, weights_only=False)
        raw["model_state"] = {f"_orig_mod.{k}": v for k, v in raw["model_state"].items()}
        torch.save(raw, path)

        ckpt = load_checkpoint(path)
        assert all(not k.startswith("_orig_mod.") for k in ckpt["model_state"])
        # And it loads back into a plain module.
        nn.Linear(3, 2).load_state_dict(ckpt["model_state"])


class TestRngState:
    def test_capture_restore_round_trip(self):
        state = capture_rng_state()
        first = torch.randn(5)
        restore_rng_state(state)
        second = torch.randn(5)
        assert torch.equal(first, second)
