"""CliRunner end-to-end tests for the ``allsky evaluate`` command (Wave C4b).

A tiny sensor_only experiment is trained with the C4a engine over real on-disk
safetensors embeddings, then evaluated through the CLI (the CLI builds a real
:class:`SafetensorsEmbeddingReader`).  Covers a clean run (report dir created,
exit 0), a missing checkpoint (non-zero exit) and cross-model comparison.
"""

from pathlib import Path

from typer.testing import CliRunner

from allsky.cli import app
from allsky.embeddings.storage import (
    SafetensorsEmbeddingReader,
)
from allsky.evaluation.reports import compare_experiments
from allsky.training.engine import run_experiment
from tests.allsky import _synthetic as synthetic

runner = CliRunner()


def _train(tmp_path: Path) -> tuple[Path, Path]:
    """Build data + on-disk embeddings, train a checkpoint; return (root, run_dir)."""
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    emb_dir = synthetic.make_embeddings_store(root, manifest, revision="r0", pooling="cls")
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
        assert (report_dir / "eval_metrics.json").exists()
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
        assert (run_dir / "eval-val" / "eval_metrics.json").exists()

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
        assert (report_dir / "eval_metrics.json").exists()
        assert not (report_dir / "predictions.parquet").exists()

    def test_the_config_supplies_the_data_root_when_no_flag_does(self, tmp_path: Path):
        """The checkpoint's own baked root is valid here, so a --config that was
        ignored would exit 0; only a --config that is read reaches the empty
        directory and fails."""
        _, run_dir = _train(tmp_path)
        empty_root = tmp_path / "no-dataset"
        empty_root.mkdir()
        config_path = synthetic.write_config_yaml(
            tmp_path / "from-config.yaml", synthetic.make_config(empty_root)
        )

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--checkpoint",
                str(run_dir / "best.ckpt"),
                "--config",
                str(config_path),
            ],
        )

        assert result.exit_code == 1, result.output

    def test_the_data_root_flag_wins_over_the_config(self, tmp_path: Path):
        """Inverted, the command would score the checkpoint against another dataset
        and write a plausible eval_metrics.json with no error at all."""
        root, run_dir = _train(tmp_path)
        empty_root = tmp_path / "no-dataset"
        empty_root.mkdir()
        config_path = synthetic.write_config_yaml(
            tmp_path / "from-config.yaml", synthetic.make_config(empty_root)
        )
        report_dir = tmp_path / "eval-out"

        result = runner.invoke(
            app,
            [
                "evaluate",
                "--checkpoint",
                str(run_dir / "best.ckpt"),
                "--config",
                str(config_path),
                "--data-root",
                str(root),
                "--report-dir",
                str(report_dir),
            ],
        )

        assert result.exit_code == 0, result.output
        assert (report_dir / "eval_metrics.json").exists()

    def test_missing_checkpoint_exits_nonzero(self, tmp_path: Path):
        result = runner.invoke(
            app,
            ["evaluate", "--checkpoint", str(tmp_path / "nope.ckpt"), "--data-root", str(tmp_path)],
        )
        assert result.exit_code != 0


class TestTrustCheckpointFlag:
    """``load_checkpoint``'s escape hatch must be reachable from the CLI.

    Without it, a checkpoint the restricted reader refuses can only be loaded by
    editing code — so the safe default would be un-overridable in the field.
    """

    def test_the_flag_reaches_load_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        seen: list[dict[str, object]] = []

        def record_trust(_path, *, map_location="cpu", trust_pickle=False):
            seen.append({"map_location": map_location, "trust_pickle": trust_pickle})
            raise RuntimeError("stop after the load call")

        monkeypatch.setattr("allsky.training.checkpointing.load_checkpoint", record_trust)
        checkpoint = tmp_path / "last.ckpt"
        checkpoint.write_bytes(b"not really a checkpoint")

        for extra_flags, expected in (([], False), (["--trust-checkpoint"], True)):
            seen.clear()
            result = runner.invoke(
                app,
                [
                    "evaluate",
                    "--checkpoint",
                    str(checkpoint),
                    "--data-root",
                    str(tmp_path),
                    *extra_flags,
                ],
            )
            assert result.exit_code == 1, result.output
            assert [call["trust_pickle"] for call in seen] == [expected]


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
