"""CLI-level end-to-end smoke test for a shipped experiment config (Wave C5).

Drives the real ``allsky`` CLI over tiny synthetic data (from
``tests.allsky._synthetic``) against a tmp experiment YAML that ``extends:`` the
repository's ``configs/allsky/experiments/v1_sensor_only.yaml`` and overrides
only paths / epochs / batch / device / amp:

1. ``allsky train`` (2 epochs) -> exit 0, ``last.ckpt`` + ``best.ckpt`` +
   ``metrics.csv`` written;
2. ``allsky train --resume auto --epochs 3`` -> exit 0, one more epoch runs;
3. ``allsky evaluate --checkpoint best.ckpt --split val`` -> exit 0, ``report.md``
   written.

Offline, CPU-only, no DINOv2 download (real on-disk safetensors embeddings feed
the embedding-mode dataset; the sensor-only model ignores them but the loader
still reads them). Kept well under a minute: 3 days x 20 rows, a tiny MLP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from allsky.cli import app  # noqa: E402
from allsky.embeddings.storage import (  # noqa: E402
    save_shard,
    shard_path,
    write_index,
    write_meta,
)
from tests.allsky import _synthetic as synthetic  # noqa: E402

runner = CliRunner()

#: The repository experiment the tmp config inherits from (extends resolves it,
#: and its own extends chain -> _base.yaml + models/sensor_only.yaml).
_REPO_V1 = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "allsky"
    / "experiments"
    / "v1_sensor_only.yaml"
)

_TMP_CONFIG = """\
extends: ["{repo_v1}"]
name: e2e_v1
output_dir: {out}
data:
  data_root: {root}
  manifest: manifest.parquet
  split_artifact: splits.json
  embeddings_dir: emb
train:
  epochs: 2
  batch_size: 8
  num_workers: 0
  device: cpu
  amp:
    enabled: false
"""


def _write_embeddings(root: Path, manifest: pd.DataFrame, dim: int = 8) -> None:
    """Write real on-disk safetensors embeddings covering every sample_id."""
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


def _make_config(tmp_path: Path, root: Path, run_dir: Path) -> Path:
    """Write the tmp experiment YAML extending the repo's v1_sensor_only.yaml."""
    config_path = tmp_path / "e2e.yaml"
    config_path.write_text(
        _TMP_CONFIG.format(repo_v1=_REPO_V1, out=run_dir, root=root),
        encoding="utf-8",
    )
    return config_path


def test_train_resume_evaluate_end_to_end(tmp_path: Path) -> None:
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    _write_embeddings(root, manifest)
    run_dir = tmp_path / "run"
    config_path = _make_config(tmp_path, root, run_dir)

    # 1) Train two epochs.
    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--data-root",
            str(root),
            "--out-dir",
            str(run_dir),
            "--no-amp",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "last.ckpt").exists()
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "metrics.csv").exists()

    # 2) Resume for one more epoch (auto-discovers last.ckpt in the run dir).
    resumed = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_path),
            "--data-root",
            str(root),
            "--out-dir",
            str(run_dir),
            "--epochs",
            "3",
            "--resume",
            "auto",
            "--no-amp",
        ],
    )
    assert resumed.exit_code == 0, resumed.output
    assert (run_dir / "last.ckpt").exists()

    # 3) Evaluate the best checkpoint on the validation split.
    report_dir = tmp_path / "eval"
    evaluated = runner.invoke(
        app,
        [
            "evaluate",
            "--checkpoint",
            str(run_dir / "best.ckpt"),
            "--split",
            "val",
            "--data-root",
            str(root),
            "--report-dir",
            str(report_dir),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert (report_dir / "report.md").exists()
    assert (report_dir / "metrics.json").exists()
