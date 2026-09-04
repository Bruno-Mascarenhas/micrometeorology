"""Tests for the ``allsky train`` experiment engine dispatch.

An ``experiment: true`` config routes to the multimodal engine (exercised end to
end with on-disk safetensors embeddings). Also checks ``--resume auto``
acceptance and bad-resume-path rejection. Non-experiment configs are rejected by
the command; that torch-free behaviour is covered in ``test_cli.py``.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from allsky.cli import app
from allsky.training.checkpointing import load_checkpoint as _load_checkpoint
from allsky.training.errors import TrainingError
from tests.allsky import _synthetic as synthetic

runner = CliRunner()


def _build_experiment(tmp_path: Path, dim: int = 8) -> tuple[Path, Path]:
    """Build a manifest + split + on-disk embeddings; return (root, config_path).

    The dataset, the store and the config all come from ``_synthetic``.
    """
    root, manifest, _split = synthetic.make_dataset(tmp_path)
    synthetic.make_embeddings_store(root, manifest, dim=dim)
    config = synthetic.make_config(
        root, epochs=2, targets={"dhi": {"enabled": True, "loss": "huber"}}
    )
    return root, synthetic.write_config_yaml(tmp_path / "experiment.yaml", config)


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

    def test_a_run_the_engine_refuses_reports_the_reason_instead_of_a_traceback(
        self, tmp_path: Path, monkeypatch
    ):
        root, config_path = _build_experiment(tmp_path)

        def _refuse(*_args, **_kwargs):
            raise TrainingError("lower train.lr, enable train.grad_clip_norm or turn AMP off")

        monkeypatch.setattr("allsky.training.engine.run_experiment", _refuse)
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
            ],
        )

        assert result.exit_code == 1
        assert "lower train.lr" in result.output
        assert "Traceback" not in result.output

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

    def test_the_batch_size_and_device_overrides_reach_the_engine(
        self, tmp_path: Path, monkeypatch
    ):
        """Both flags mutate the loaded config in place and ExperimentTrainConfig has
        no validate_assignment, so the assignment skips every pydantic check."""
        seen: list[tuple[int, str]] = []

        def record_the_config(cfg, **_kwargs):
            seen.append((cfg.train.batch_size, cfg.train.device))
            return {"ok": True}

        monkeypatch.setattr("allsky.training.run_experiment", record_the_config)
        root, config_path = _build_experiment(tmp_path)

        result = runner.invoke(
            app,
            [
                "train",
                "--config",
                str(config_path),
                "--data-root",
                str(root),
                "--batch-size",
                "3",
                "--device",
                "mps",
            ],
        )

        assert result.exit_code == 0, result.output
        assert seen == [(3, "mps")]

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
        assert result.exit_code == 2, result.output
        assert "resume checkpoint does not exist" in result.output


class TestTrustCheckpointFlag:
    @pytest.mark.parametrize(
        ("extra_flags", "expected"), [([], False), (["--trust-checkpoint"], True)]
    )
    def test_the_flag_reaches_the_resume_load(
        self, tmp_path: Path, monkeypatch, extra_flags: list[str], expected: bool
    ) -> None:
        """The restricted-unpickler default must be overridable from the CLI.

        Without a flag, a checkpoint the allowlist refuses could only be resumed
        by editing code — so the safe default would be un-overridable in the field.
        """
        trust_values: list[bool] = []

        def record_trust(_cfg, **kwargs):
            trust_values.append(bool(kwargs["trust_checkpoint"]))
            return {"ok": True}

        monkeypatch.setattr("allsky.training.run_experiment", record_trust)
        root, config_path = _build_experiment(tmp_path)

        result = runner.invoke(
            app,
            ["train", "--config", str(config_path), "--data-root", str(root), *extra_flags],
        )

        assert result.exit_code == 0, result.output
        assert trust_values == [expected]

    def test_a_resumed_checkpoint_is_read_under_the_restricted_reader_by_default(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        seen: list[bool] = []
        real_load = _load_checkpoint

        def record_trust(path, *, map_location="cpu", trust_pickle=False):
            seen.append(trust_pickle)
            return real_load(path, map_location=map_location, trust_pickle=trust_pickle)

        root, config_path = _build_experiment(tmp_path)
        run_dir = tmp_path / "run"
        base = [
            "train",
            "--config",
            str(config_path),
            "--epochs",
            "1",
            "--data-root",
            str(root),
            "--out-dir",
            str(run_dir),
        ]
        assert runner.invoke(app, base).exit_code == 0

        monkeypatch.setattr("allsky.training.engine.load_checkpoint", record_trust)
        assert runner.invoke(app, [*base, "--resume", "auto"]).exit_code == 0
        assert seen == [False]

        seen.clear()
        assert runner.invoke(app, [*base, "--resume", "auto", "--trust-checkpoint"]).exit_code == 0
        assert seen == [True]
