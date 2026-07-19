"""Torch-gated engine tests for review findings F6, F7 and F10.

All offline and CPU-only. Reuses the shared synthetic builders in
``tests/allsky/_synthetic.py`` (extended with tiny JPEG writing and a real
on-disk embedding store) so the *production* code paths — the default image
backbone builder and the default preloading embedding reader — are exercised
without hand-injecting anything into ``build_model``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from allsky.config import ExperimentConfig  # noqa: E402
from allsky.training.engine import run_experiment  # noqa: E402
from tests.allsky import _synthetic as synthetic  # noqa: E402


class TinyConvBackbone(nn.Module):
    """Minimal conv image backbone exposing ``.dim`` (stands in for DINOv2)."""

    def __init__(self, dim: int = 12) -> None:
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(3, dim, kernel_size=3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: Any) -> Any:
        return self.pool(self.conv(x)).flatten(1)


def _image_cfg(root: Path, *, model: str = "film", epochs: int = 1) -> ExperimentConfig:
    """A v6-shaped image-mode config (FiLM, unfrozen backbone, separate backbone LR)."""
    return ExperimentConfig.model_validate(
        {
            "experiment": True,
            "seed": 0,
            "output_dir": str(root / "out"),
            "data": {
                "manifest": "manifest.parquet",
                "data_root": str(root),
                "split_artifact": "splits.json",
                "input_mode": "image",
            },
            "features": {"set": "safe"},
            "targets": {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
            "model": {
                "name": model,
                "image_size": 8,
                "backbone_frozen": False,
                "unfreeze_last_n": 2,
            },
            "train": {
                "epochs": epochs,
                "batch_size": 8,
                "num_workers": 0,
                "device": "cpu",
                "backbone_lr": 1e-5,
                "early_stopping": {"monitor": "val_loss", "patience": 100},
            },
        }
    )


class TestImageBackboneProduction:
    """Finding F6: image mode must build a backbone in production (no build_model injection)."""

    def test_image_mode_runs_via_run_experiment_hook(self, tmp_path: Path):
        # The documented run_experiment hook (not a hand-built model) supplies the
        # backbone; the production image dataset + training path runs end to end.
        root, _manifest, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=8
        )
        run_dir = tmp_path / "run"
        summary = run_experiment(
            _image_cfg(root, model="film", epochs=1),
            data_root=root,
            output_dir=run_dir,
            image_backbone_builder=lambda: TinyConvBackbone(dim=12),
        )
        assert summary["epochs_ran"] == 1
        assert (run_dir / "last.ckpt").exists()
        assert (run_dir / "best.ckpt").exists()
        assert len(pd.read_csv(run_dir / "metrics.csv")) == 1

    def test_default_builder_calls_build_backbone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # With NO injected builder, the engine must construct the config-named
        # backbone itself via allsky.embeddings.backbone.build_backbone (finding
        # F6). Monkeypatching build_backbone to a tiny nn.Module stub both proves
        # the default path is reached and keeps the run offline (no DINOv2).
        import allsky.embeddings.backbone as backbone_mod

        root, _manifest, _ = synthetic.make_dataset(
            tmp_path, n_days=3, per_day=6, write_images=True, image_px=8
        )
        calls: list[tuple[str, str, str]] = []

        def fake_build_backbone(
            name: str, *, pooling: str = "cls", device: str = "auto", **_: Any
        ) -> nn.Module:
            calls.append((name, pooling, device))
            return TinyConvBackbone(dim=12)

        monkeypatch.setattr(backbone_mod, "build_backbone", fake_build_backbone)

        run_dir = tmp_path / "run"
        summary = run_experiment(
            _image_cfg(root, model="film", epochs=1),
            data_root=root,
            output_dir=run_dir,
            # image_backbone_builder intentionally omitted -> default path.
        )
        assert calls, "the default builder never called build_backbone"
        assert calls[0][0] == "dinov2_vits14"  # config default backbone name
        assert calls[0][1] == "cls"  # config default pooling
        assert calls[0][2] == "cpu"  # resolved run device passed through
        assert summary["epochs_ran"] == 1
        assert (run_dir / "last.ckpt").exists()


class TestEmbeddingPreload:
    """Finding F7: the engine's default embedding reader preloads all shards."""

    def test_embedding_run_preloads_by_default(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, manifest, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6)
        synthetic.make_embeddings_store(root, manifest, dim=8, shard_size=4, subdir="emb")
        cfg = synthetic.make_config(root, model="concat", epochs=1)

        run_dir = tmp_path / "run"
        with caplog.at_level(logging.INFO, logger="allsky.embeddings.storage"):
            summary = run_experiment(
                cfg,
                data_root=root,
                output_dir=run_dir,
                # embedding_reader intentionally omitted -> default reader used.
            )
        assert summary["epochs_ran"] == 1
        assert any("preloaded" in record.getMessage() for record in caplog.records), (
            "the default reader did not preload the embedding shards"
        )

    def test_preload_escape_hatch_disables_it(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, manifest, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6)
        synthetic.make_embeddings_store(root, manifest, dim=8, shard_size=4, subdir="emb")
        cfg = synthetic.make_config(root, model="concat", epochs=1)
        cfg.data.embeddings_preload = False

        run_dir = tmp_path / "run"
        with caplog.at_level(logging.INFO, logger="allsky.embeddings.storage"):
            summary = run_experiment(cfg, data_root=root, output_dir=run_dir)
        assert summary["epochs_ran"] == 1
        assert not any("preloaded" in record.getMessage() for record in caplog.records)


class TestFreshRunStaleMetrics:
    """Finding F10: a fresh run into a reused dir must not append to stale metrics."""

    def test_stale_metrics_rotated_on_fresh_start(self, tmp_path: Path):
        root, manifest, _ = synthetic.make_dataset(tmp_path, n_days=3, per_day=6)
        reader = synthetic.reader_for(manifest)
        run_dir = tmp_path / "run"

        # First run: 3 epochs -> metrics.csv has epochs [1, 2, 3].
        run_experiment(
            synthetic.make_config(root, model="sensor_only", epochs=3),
            data_root=root,
            output_dir=run_dir,
            embedding_reader=reader,
        )
        assert list(pd.read_csv(run_dir / "metrics.csv")["epoch"]) == [1, 2, 3]

        # Fresh (non-resume) run for 2 epochs into the SAME dir: stale rows are
        # rotated aside, so the file holds exactly this run's epochs [1, 2].
        run_experiment(
            synthetic.make_config(root, model="sensor_only", epochs=2),
            data_root=root,
            output_dir=run_dir,
            embedding_reader=reader,
        )
        rows = pd.read_csv(run_dir / "metrics.csv")
        assert list(rows["epoch"]) == [1, 2]
        assert (run_dir / "metrics.csv.stale").exists()
        assert (run_dir / "metrics.json.stale").exists()
        assert len(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))) == 2
