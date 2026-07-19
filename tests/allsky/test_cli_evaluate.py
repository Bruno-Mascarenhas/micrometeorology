"""CliRunner end-to-end tests for the ``allsky evaluate`` command (Wave C4b).

A tiny sensor_only experiment is trained with the C4a engine over real on-disk
safetensors embeddings, then evaluated through the CLI (the CLI builds a real
:class:`SafetensorsEmbeddingReader`).  Covers a clean run (report dir created,
exit 0), a missing checkpoint (non-zero exit) and cross-model comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")

from allsky.cli import app  # noqa: E402
from allsky.embeddings.storage import (  # noqa: E402
    SafetensorsEmbeddingReader,
    save_shard,
    shard_path,
    write_index,
    write_meta,
)
from allsky.evaluation.reports import compare_experiments  # noqa: E402
from allsky.training.engine import run_experiment  # noqa: E402
from tests.allsky import _synthetic as synthetic  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _write_embeddings(root: Path, manifest: pd.DataFrame, dim: int = 8) -> Path:
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
    return emb_dir


def _train(tmp_path: Path) -> tuple[Path, Path]:
    """Build data + on-disk embeddings, train a checkpoint; return (root, run_dir)."""
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    emb_dir = _write_embeddings(root, manifest)
    reader = SafetensorsEmbeddingReader(emb_dir)
    cfg = synthetic.make_config(root, epochs=2)
    run_dir = tmp_path / "run"
    run_experiment(cfg, data_root=root, output_dir=run_dir, embedding_reader=reader)
    return root, run_dir


class TestEvaluateCommand:
    def test_end_to_end_writes_report(self, tmp_path: Path):
        root, run_dir = _train(tmp_path)
        report_dir = tmp_path / "eval-out"
        result = runner.invoke(
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
        assert result.exit_code == 0, result.output
        assert (report_dir / "metrics.json").exists()
        assert (report_dir / "report.md").exists()
        assert (report_dir / "stratified.csv").exists()
        assert (report_dir / "predictions.parquet").exists()

    def test_default_report_dir_under_checkpoint(self, tmp_path: Path):
        root, run_dir = _train(tmp_path)
        result = runner.invoke(
            app,
            ["evaluate", "--checkpoint", str(run_dir / "best.ckpt"), "--data-root", str(root)],
        )
        assert result.exit_code == 0, result.output
        assert (run_dir / "eval-val" / "metrics.json").exists()

    def test_no_predictions_flag(self, tmp_path: Path):
        root, run_dir = _train(tmp_path)
        report_dir = tmp_path / "eval-out"
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--checkpoint",
                str(run_dir / "best.ckpt"),
                "--data-root",
                str(root),
                "--report-dir",
                str(report_dir),
                "--no-predictions",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (report_dir / "metrics.json").exists()
        assert not (report_dir / "predictions.parquet").exists()

    def test_missing_checkpoint_exits_nonzero(self, tmp_path: Path):
        result = runner.invoke(
            app,
            ["evaluate", "--checkpoint", str(tmp_path / "nope.ckpt"), "--data-root", str(tmp_path)],
        )
        assert result.exit_code != 0


class TestCompareExperiments:
    def test_compare_two_runs_produces_table(self, tmp_path: Path):
        root, run_dir = _train(tmp_path)
        report_dirs = []
        for name, ckpt in (("best", "best.ckpt"), ("last", "last.ckpt")):
            report_dir = tmp_path / f"eval-{name}"
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--checkpoint",
                    str(run_dir / ckpt),
                    "--data-root",
                    str(root),
                    "--report-dir",
                    str(report_dir),
                ],
            )
            assert result.exit_code == 0, result.output
            report_dirs.append(report_dir)

        table = compare_experiments(report_dirs, out_dir=tmp_path / "cmp")
        assert len(table) == 2
        assert {"experiment", "model", "split", "n_samples"} <= set(table.columns)
        assert any(col.startswith("dhi_") for col in table.columns)
        assert (tmp_path / "cmp" / "comparison.csv").exists()
        assert (tmp_path / "cmp" / "comparison.md").exists()
