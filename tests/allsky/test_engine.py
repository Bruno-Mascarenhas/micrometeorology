"""Torch-gated integration tests for allsky.training.engine.run_experiment.

All offline and CPU-only: a tiny synthetic v2 manifest (3 days x 20 samples) is
built with the real manifest builder, embeddings are served by a deterministic
dict-backed fake reader, and models train for 1-3 epochs.  Covers a basic run,
full resume equivalence, early stopping, gradient accumulation, the climatology
baseline and the clear cuda-unavailable error.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from allsky import solar  # noqa: E402
from allsky.config import ExperimentConfig, SiteConfig  # noqa: E402
from allsky.data.manifest import build_manifest, write_manifest_parquet  # noqa: E402
from allsky.data.splits import create_day_splits, save_split_artifact  # noqa: E402
from allsky.evaluation.evaluator import evaluate_checkpoint  # noqa: E402
from allsky.training.engine import run_experiment  # noqa: E402

_MET = {
    "AirT1_C_Avg": (20.0, 30.0),
    "DP1_C_Avg": (10.0, 20.0),
    "RH1": (50.0, 90.0),
    "BP1_mbar_Avg": (1005.0, 1015.0),
    "WS_ms": (0.0, 8.0),
    "WindDir": (0.0, 360.0),
}


class DictEmbeddingReader:
    """Deterministic dict-backed reader implementing the EmbeddingReader protocol."""

    def __init__(self, sample_ids: list[str], dim: int = 8) -> None:
        rng = np.random.default_rng(0)
        self._data = {str(s): rng.standard_normal(dim).astype(np.float32) for s in sample_ids}
        self.dim = dim

    def __call__(self, sample_id: str) -> np.ndarray:
        return self._data[str(sample_id)]

    def sample_ids(self) -> list[str]:
        return list(self._data)


def _sensor(site: SiteConfig, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    index = pd.date_range(first + pd.Timedelta(hours=8), last + pd.Timedelta(hours=19), freq="5min")
    rng = np.random.default_rng(0)
    e0h = solar.extraterrestrial_ghi(index, site)
    data = {k: rng.uniform(lo, hi, len(index)) for k, (lo, hi) in _MET.items()}
    data["CM3Up_Wm2_Avg"] = np.clip(0.7 * e0h, 0.0, None)
    data["PSP_Wm2_Avg"] = np.clip(0.2 * e0h, 0.0, None)
    return pd.DataFrame(data, index=index)


def _make_dataset(
    tmp_path: Path, *, n_days: int = 3, per_day: int = 20
) -> tuple[Path, pd.DataFrame, Any]:
    """Build a tiny manifest + split under ``tmp_path/data`` (no image files needed)."""
    site = SiteConfig()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2025-03-20", periods=n_days, freq="D")
    rows = []
    idx = 0
    for day in days:
        for ts in pd.date_range(day + pd.Timedelta(hours=9), periods=per_day, freq="30min"):
            rows.append(
                {
                    "frame_path": f"frames/allsky-{ts:%Y%m%d-%H%M}.jpg",
                    "timestamp": ts,
                    "video": "v.mp4",
                    "index": idx,
                }
            )
            idx += 1
    frames = pd.DataFrame(rows)
    manifest, meta = build_manifest(
        frames, _sensor(site, days[0], days[-1]), site=site, data_root=root
    )
    write_manifest_parquet(manifest, meta, root / "manifest.parquet")
    split = create_day_splits(
        manifest["day_id"].tolist(), val_fraction=0.34, test_fraction=0.0, seed=0
    )
    save_split_artifact(split, root / "splits.json")
    return root, manifest, split


def _cfg(
    root: Path,
    *,
    model: str = "sensor_only",
    epochs: int = 2,
    batch_size: int = 8,
    grad_accum_steps: int = 1,
    monitor: str = "val_loss",
    patience: int = 100,
    targets: dict | None = None,
    scheduler: str = "none",
    num_workers: int = 0,
    strategy: str = "center_frame",
    window_minutes: float = 10.0,
) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "experiment": True,
            "seed": 0,
            "output_dir": str(root / "out"),
            "data": {
                "manifest": "manifest.parquet",
                "data_root": str(root),
                "split_artifact": "splits.json",
                "embeddings_dir": "emb",
                "input_mode": "embedding",
                "alignment": {"strategy": strategy, "window_minutes": window_minutes},
            },
            "features": {"set": "safe"},
            "targets": targets
            or {"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
            "model": {"name": model},
            "train": {
                "epochs": epochs,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "device": "cpu",
                "grad_accum_steps": grad_accum_steps,
                "scheduler": {"name": scheduler},
                "early_stopping": {"monitor": monitor, "patience": patience},
            },
        }
    )


def _reader(manifest: pd.DataFrame) -> DictEmbeddingReader:
    return DictEmbeddingReader([str(s) for s in manifest["sample_id"]])


class TestBasicRun:
    def test_sensor_only_two_epochs_writes_artifacts(self, tmp_path: Path):
        root, manifest, _ = _make_dataset(tmp_path)
        cfg = _cfg(root, epochs=2)
        run_dir = tmp_path / "run"
        summary = run_experiment(
            cfg, data_root=root, output_dir=run_dir, embedding_reader=_reader(manifest)
        )

        assert summary["epochs_ran"] == 2
        assert summary["epoch"] == 2
        rows = pd.read_csv(run_dir / "metrics.csv")
        assert len(rows) == 2
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "last.ckpt").exists()
        assert (run_dir / "best.ckpt").exists()
        assert (run_dir / "runs").is_dir()  # TensorBoard event dir
        # Physical-unit quick metrics were computed.
        assert "val_dhi_mae" in rows.columns
        assert "final_val_metrics" in summary
        assert "loss" in summary["final_val_metrics"]


class TestResumeEquivalence:
    def test_resume_matches_uninterrupted_run(self, tmp_path: Path):
        root, manifest, _ = _make_dataset(tmp_path)
        reader = _reader(manifest)

        # Run A: 3 epochs uninterrupted.
        summary_a = run_experiment(
            _cfg(root, epochs=3),
            data_root=root,
            output_dir=tmp_path / "runA",
            embedding_reader=reader,
        )

        # Run B: 2 epochs, then resume 'auto' targeting 3 epochs (+1 more).
        run_experiment(
            _cfg(root, epochs=2),
            data_root=root,
            output_dir=tmp_path / "runB",
            embedding_reader=reader,
        )
        summary_b = run_experiment(
            _cfg(root, epochs=3),
            data_root=root,
            output_dir=tmp_path / "runB",
            resume="auto",
            embedding_reader=reader,
        )

        assert summary_b["global_step"] == summary_a["global_step"]
        assert summary_b["epoch"] == 3
        assert summary_b["epochs_ran"] == 1  # only the resumed epoch ran this call
        for key, value in summary_a["final_val_metrics"].items():
            assert summary_b["final_val_metrics"][key] == pytest.approx(value, abs=1e-5, rel=1e-4)
        # metrics.csv accumulated all 3 epochs across the interrupted run.
        assert len(pd.read_csv(tmp_path / "runB" / "metrics.csv")) == 3


class TestResumeEquivalenceMultiWorker:
    def test_resume_matches_uninterrupted_with_workers(self, tmp_path: Path):
        # The original divergence mode (Wave D finding 1): num_workers > 0 with
        # persistent_workers, where a global-RNG-dependent shuffle made the resumed
        # epoch order drift from the uninterrupted one. The dedicated per-epoch
        # generator makes the order a pure function of (seed, epoch), so 2+resume+1
        # must equal 3 uninterrupted, byte-for-byte in the final val metrics.
        #
        # Kept deliberately tiny (60 rows, 8-dim fake embeddings, a small MLP) so it
        # runs in a few seconds even paying the worker-process startup cost. There
        # is no pytest-timeout plugin in the locked env, so no @pytest.mark.timeout
        # marker is applied; the tiny fixture is the runtime guard instead.
        root, manifest, _ = _make_dataset(tmp_path)
        reader = _reader(manifest)

        summary_a = run_experiment(
            _cfg(root, epochs=3, num_workers=2),
            data_root=root,
            output_dir=tmp_path / "runA",
            embedding_reader=reader,
        )
        run_experiment(
            _cfg(root, epochs=2, num_workers=2),
            data_root=root,
            output_dir=tmp_path / "runB",
            embedding_reader=reader,
        )
        summary_b = run_experiment(
            _cfg(root, epochs=3, num_workers=2),
            data_root=root,
            output_dir=tmp_path / "runB",
            resume="auto",
            embedding_reader=reader,
        )

        assert summary_b["global_step"] == summary_a["global_step"]
        assert summary_b["epoch"] == 3
        assert summary_b["epochs_ran"] == 1
        assert summary_a["final_val_metrics"]  # non-empty
        for key, value in summary_a["final_val_metrics"].items():
            assert summary_b["final_val_metrics"][key] == pytest.approx(value, abs=1e-5, rel=1e-4)


class TestMetricsResumeTruncation:
    def test_stale_row_past_checkpoint_is_dropped_on_resume(self, tmp_path: Path):
        # Reproduce the crash-window: metrics.csv/json are flushed before last.ckpt
        # each epoch, so a crash in that gap can leave a row for an epoch the
        # checkpoint never completed. Resume must drop it, not duplicate it.
        root, manifest, _ = _make_dataset(tmp_path)
        reader = _reader(manifest)
        run_dir = tmp_path / "run"

        run_experiment(
            _cfg(root, epochs=2), data_root=root, output_dir=run_dir, embedding_reader=reader
        )
        assert list(pd.read_csv(run_dir / "metrics.csv")["epoch"]) == [1, 2]

        # Inject a stale epoch-3 row into both metrics files while last.ckpt still
        # records epoch 2 (i.e. the checkpoint never advanced to epoch 3).
        history = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        stale = {**history[-1], "epoch": 3}
        (run_dir / "metrics.json").write_text(json.dumps([*history, stale]), encoding="utf-8")
        staged = pd.concat(
            [pd.read_csv(run_dir / "metrics.csv"), pd.DataFrame([stale])], ignore_index=True
        )
        staged.to_csv(run_dir / "metrics.csv", index=False)
        assert list(pd.read_csv(run_dir / "metrics.csv")["epoch"]) == [1, 2, 3]

        # Resume for a genuine epoch 3: truncation drops the stale row first, then
        # the loop writes epoch 3 exactly once -> no duplicate.
        run_experiment(
            _cfg(root, epochs=3),
            data_root=root,
            output_dir=run_dir,
            resume="auto",
            embedding_reader=reader,
        )
        rows = pd.read_csv(run_dir / "metrics.csv")
        assert list(rows["epoch"]) == [1, 2, 3]
        assert len(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))) == 3


class TestWindowStrategies:
    @pytest.mark.parametrize("strategy", ["mean_embedding", "attention_pooling"])
    def test_windowed_run_trains_checkpoints_evaluates(self, tmp_path: Path, strategy: str):
        # End-to-end wiring of the dataset-level windowed strategies (finding 4):
        # the engine builds the embedding dataset with window=strategy and, for
        # attention_pooling, builds/evaluates the model with the learned attention
        # temporal pooler. window_minutes=70 spans the 30-min frame cadence so each
        # interior row's window actually contains co-frames (not just itself). Uses
        # the 'concat' model because it (unlike sensor_only) consumes the embedding.
        root, manifest, _ = _make_dataset(tmp_path)
        reader = _reader(manifest)
        cfg = _cfg(
            root,
            model="concat",
            epochs=2,
            strategy=strategy,
            window_minutes=70.0,
            targets={"dhi": {"enabled": True, "loss": "huber"}, "sky": {"enabled": True}},
        )
        run_dir = tmp_path / "run"
        summary = run_experiment(cfg, data_root=root, output_dir=run_dir, embedding_reader=reader)

        assert summary["epochs_ran"] == 2
        assert (run_dir / "last.ckpt").exists()
        assert (run_dir / "best.ckpt").exists()
        assert len(pd.read_csv(run_dir / "metrics.csv")) == 2

        # Evaluate the trained checkpoint on val: the evaluator must rebuild the
        # dataset/model with the same window + temporal pooler (an attention-pooled
        # checkpoint carries extra weights that a mean-pooled rebuild would reject).
        result = evaluate_checkpoint(
            run_dir / "best.ckpt", split="val", data_root=root, embedding_reader=reader
        )
        assert result.n_samples > 0
        assert result.enabled_targets == ["dhi", "sky"]
        assert len(result.predictions) == result.n_samples


class TestEarlyStopping:
    def test_climatology_plateau_triggers_early_stop(self, tmp_path: Path):
        root, manifest, _ = _make_dataset(tmp_path)
        cfg = _cfg(
            root,
            model="climatology",
            epochs=8,
            patience=1,
            monitor="val_loss",
            targets={"dhi": {"enabled": True, "loss": "mse"}},
        )
        summary = run_experiment(
            cfg, data_root=root, output_dir=tmp_path / "run", embedding_reader=_reader(manifest)
        )
        # Constant predictions -> val loss never improves after epoch 1.
        assert summary["epochs_ran"] == 2
        assert summary["epochs_ran"] < cfg.train.epochs
        assert summary["best_metric"]["epoch"] == 1


class TestGradAccumulation:
    def test_global_step_counts_optimizer_steps(self, tmp_path: Path):
        root, manifest, split = _make_dataset(tmp_path)
        batch_size, k, epochs = 8, 3, 2
        cfg = _cfg(root, epochs=epochs, batch_size=batch_size, grad_accum_steps=k)
        summary = run_experiment(
            cfg, data_root=root, output_dir=tmp_path / "run", embedding_reader=_reader(manifest)
        )

        train_days = set(split.days_for("train"))
        n_train = int(manifest["day_id"].astype(str).isin(train_days).sum())
        n_batches = math.ceil(n_train / batch_size)
        expected = math.ceil(n_batches / k) * epochs
        assert summary["global_step"] == expected


class TestClimatology:
    def test_runs_end_to_end(self, tmp_path: Path):
        root, manifest, _ = _make_dataset(tmp_path)
        cfg = _cfg(
            root,
            model="climatology",
            epochs=2,
            targets={"dhi": {"enabled": True, "loss": "mse"}, "sky": {"enabled": True}},
        )
        summary = run_experiment(
            cfg, data_root=root, output_dir=tmp_path / "run", embedding_reader=_reader(manifest)
        )
        assert summary["epochs_ran"] == 2
        assert (tmp_path / "run" / "best.ckpt").exists()


class TestDeviceErrors:
    def test_cuda_requested_without_cuda_raises(self, tmp_path: Path):
        if torch.cuda.is_available():
            pytest.skip("cuda is available; the unavailable-path error cannot be exercised")
        root, manifest, _ = _make_dataset(tmp_path)
        cfg = _cfg(root, epochs=1)
        with pytest.raises(RuntimeError, match="cuda"):
            run_experiment(
                cfg,
                data_root=root,
                output_dir=tmp_path / "run",
                device="cuda",
                embedding_reader=_reader(manifest),
            )
