"""Tests for the ``allsky train`` experiment engine dispatch.

An ``experiment: true`` config routes to the multimodal engine (exercised end to
end with on-disk safetensors embeddings). Also checks ``--resume auto``
acceptance and bad-resume-path rejection. Non-experiment configs are rejected by
the command; that torch-free behaviour is covered in ``test_cli.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from allsky import solar  # noqa: E402
from allsky.cli import app  # noqa: E402
from allsky.config import SiteConfig  # noqa: E402
from allsky.data.manifest import build_manifest, write_manifest_parquet  # noqa: E402
from allsky.data.splits import create_day_splits, save_split_artifact  # noqa: E402
from allsky.embeddings.storage import save_shard, shard_path, write_index, write_meta  # noqa: E402

runner = CliRunner()

_MET = {
    "AirT1_C_Avg": (20.0, 30.0),
    "DP1_C_Avg": (10.0, 20.0),
    "RH1": (50.0, 90.0),
    "BP1_mbar_Avg": (1005.0, 1015.0),
    "WS_ms": (0.0, 8.0),
    "WindDir": (0.0, 360.0),
}

_EXPERIMENT_YAML = """\
experiment: true
seed: 0
output_dir: {out}
data:
  manifest: manifest.parquet
  data_root: {root}
  split_artifact: splits.json
  embeddings_dir: emb
  input_mode: embedding
features:
  set: safe
targets:
  dhi:
    enabled: true
    loss: huber
model:
  name: sensor_only
train:
  epochs: 2
  batch_size: 8
  num_workers: 0
  device: cpu
  early_stopping:
    monitor: val_loss
    patience: 100
"""


def _build_experiment(tmp_path: Path, dim: int = 8) -> tuple[Path, Path]:
    """Build a manifest + split + on-disk embeddings; return (root, config_path)."""
    site = SiteConfig()
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    days = pd.date_range("2025-03-20", periods=3, freq="D")
    rows = []
    idx = 0
    for day in days:
        for ts in pd.date_range(day + pd.Timedelta(hours=9), periods=20, freq="30min"):
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

    index = pd.date_range(
        days[0] + pd.Timedelta(hours=8), days[-1] + pd.Timedelta(hours=19), freq="5min"
    )
    rng = np.random.default_rng(0)
    e0h = solar.extraterrestrial_ghi(index, site)
    sensor_data = {k: rng.uniform(lo, hi, len(index)) for k, (lo, hi) in _MET.items()}
    sensor_data["CM3Up_Wm2_Avg"] = np.clip(0.7 * e0h, 0.0, None)
    sensor_data["PSP_Wm2_Avg"] = np.clip(0.2 * e0h, 0.0, None)
    sensor = pd.DataFrame(sensor_data, index=index)

    manifest, meta = build_manifest(frames, sensor, site=site, data_root=root)
    write_manifest_parquet(manifest, meta, root / "manifest.parquet")
    split = create_day_splits(
        manifest["day_id"].tolist(), val_fraction=0.34, test_fraction=0.0, seed=0
    )
    save_split_artifact(split, root / "splits.json")

    # On-disk embeddings covering every sample_id (the CLI builds a real reader).
    sample_ids = [str(s) for s in manifest["sample_id"]]
    emb_dir = root / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.random.default_rng(1).standard_normal((len(sample_ids), dim)).astype(np.float32)
    save_shard(shard_path(emb_dir, 0), embeddings)
    write_index(
        emb_dir, pd.DataFrame({"sample_id": sample_ids, "shard": 0, "row": range(len(sample_ids))})
    )
    write_meta(
        emb_dir,
        {
            "backbone": "fake",
            "revision": "r0",
            "pooling": "cls",
            "dim": dim,
            "count": len(sample_ids),
        },
    )

    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(_EXPERIMENT_YAML.format(out=root / "out", root=root), encoding="utf-8")
    return root, config_path


class TestExperimentDispatch:
    def test_experiment_config_routes_to_engine(self, tmp_path: Path):
        root, config_path = _build_experiment(tmp_path)
        run_dir = tmp_path / "run"
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_path),
                "--epochs",
                "1",
                "--data-root",
                str(root),
                "--out-dir",
                str(run_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (run_dir / "last.ckpt").exists()
        assert (run_dir / "metrics.json").exists()

    def test_resume_auto_accepted(self, tmp_path: Path):
        root, config_path = _build_experiment(tmp_path)
        run_dir = tmp_path / "run"
        # 'auto' with no existing checkpoint must be accepted and start fresh.
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_path),
                "--epochs",
                "1",
                "--data-root",
                str(root),
                "--out-dir",
                str(run_dir),
                "--resume",
                "auto",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (run_dir / "last.ckpt").exists()

    def test_bad_resume_path_errors(self, tmp_path: Path):
        root, config_path = _build_experiment(tmp_path)
        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_path),
                "--data-root",
                str(root),
                "--out-dir",
                str(tmp_path / "run"),
                "--resume",
                str(tmp_path / "nope" / "last.ckpt"),
            ],
        )
        assert result.exit_code != 0


class TestNonExperimentRejected:
    def test_non_experiment_config_is_rejected_before_the_engine(self, tmp_path: Path):
        # A config without 'experiment: true' is rejected with a clear pointer to
        # the experiment configs — it never reaches the training engine.
        config_path = tmp_path / "legacy.yaml"
        config_path.write_text("train:\n  epochs: 1\n", encoding="utf-8")
        result = runner.invoke(
            app, ["train", "--config", str(config_path), "--out-dir", str(tmp_path / "out")]
        )
        assert result.exit_code != 0
        assert "experiment" in result.output
