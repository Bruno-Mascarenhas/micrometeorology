"""Torch-gated tests for the exponential moving average of the weights.

The shadow's arithmetic is pinned in closed form on a two-parameter model; the
engine tests train the synthetic sensor_only experiment on CPU and check the
contract around ``ema.ckpt``: absent when the average is off, written in the
format of ``last.ckpt`` when it is on, evaluable by the evaluator unchanged,
and continued rather than restarted across a resume.
"""

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from torch import nn

from allsky.config import ExperimentConfig
from allsky.evaluation.evaluator import evaluate_checkpoint
from allsky.training.checkpointing import load_checkpoint
from allsky.training.engine import run_experiment
from allsky.training.errors import TrainingError
from allsky.training.weight_average import ExponentialMovingAverage
from tests.allsky import _synthetic as synthetic

DECAY = 0.75


class _FrozenAndBuffered(nn.Module):
    """Two scalars — one trainable, one frozen — and a buffer, to tell the tiers apart."""

    def __init__(self) -> None:
        super().__init__()
        self.trained = nn.Parameter(torch.tensor(1.0))
        self.frozen = nn.Parameter(torch.tensor(5.0), requires_grad=False)
        self.register_buffer("running", torch.tensor(0.0))


def _set(module: nn.Module, name: str, value: float) -> None:
    with torch.no_grad():
        getattr(module, name).fill_(value)


def _config(root: Path, *, epochs: int = 2, **train: Any) -> ExperimentConfig:
    payload = synthetic.make_config(root, epochs=epochs).model_dump()
    payload["train"].update(train)
    return ExperimentConfig.model_validate(payload)


def _run(tmp_path: Path, run_dir: Path, *, epochs: int = 2, **train: Any) -> dict[str, Any]:
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    return run_experiment(
        _config(root, epochs=epochs, **train),
        data_root=root,
        output_dir=run_dir,
        embedding_reader=synthetic.reader_for(manifest),
    )


class TestClosedForm:
    def test_two_updates_weight_the_iterates_by_decay_powers(self):
        model = _FrozenAndBuffered()
        average = ExponentialMovingAverage(model, decay=DECAY)

        _set(model, "trained", 3.0)
        average.update(model)
        _set(model, "trained", 7.0)
        average.update(model)

        expected = DECAY**2 * 1.0 + DECAY * (1 - DECAY) * 3.0 + (1 - DECAY) * 7.0
        assert average.state_dict()["trained"].item() == pytest.approx(expected)

    def test_a_frozen_parameter_keeps_the_value_the_average_started_from(self):
        model = _FrozenAndBuffered()
        average = ExponentialMovingAverage(model, decay=DECAY)

        _set(model, "frozen", 9.0)
        average.update(model)

        assert average.state_dict()["frozen"].item() == pytest.approx(5.0)

    def test_a_buffer_is_copied_rather_than_averaged(self):
        model = _FrozenAndBuffered()
        average = ExponentialMovingAverage(model, decay=DECAY)

        _set(model, "running", 4.0)
        average.update(model)

        assert average.state_dict()["running"].item() == pytest.approx(4.0)

    def test_the_shadow_does_not_alias_the_live_weights(self):
        model = _FrozenAndBuffered()
        average = ExponentialMovingAverage(model, decay=DECAY)

        _set(model, "trained", 3.0)

        assert average.state_dict()["trained"].item() == pytest.approx(1.0)

    def test_applying_the_average_restores_the_live_weights_afterwards(self):
        model = _FrozenAndBuffered()
        average = ExponentialMovingAverage(model, decay=DECAY)
        _set(model, "trained", 3.0)
        average.update(model)

        with average.applied_to(model):
            swapped = model.trained.item()

        assert swapped == pytest.approx(DECAY * 1.0 + (1 - DECAY) * 3.0)
        assert model.trained.item() == pytest.approx(3.0)

    def test_a_state_with_foreign_keys_is_refused(self):
        average = ExponentialMovingAverage(_FrozenAndBuffered(), decay=DECAY)

        with pytest.raises(KeyError, match="unexpected"):
            average.load_state_dict({**average.state_dict(), "stray": torch.tensor(0.0)})

    @pytest.mark.parametrize("decay", [0.0, 1.0])
    def test_a_degenerate_decay_is_refused(self, decay: float):
        with pytest.raises(ValueError, match="strictly inside"):
            ExponentialMovingAverage(_FrozenAndBuffered(), decay=decay)


class TestEngineContract:
    def test_the_average_off_writes_no_ema_checkpoint_and_no_ema_columns(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        summary = _run(tmp_path, run_dir)

        assert not (run_dir / "ema.ckpt").exists()
        assert summary["checkpoint_ema"] is None
        assert not [c for c in pd.read_csv(run_dir / "metrics.csv").columns if "ema" in c]

    def test_the_averaged_weights_are_not_the_last_weights(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        _run(tmp_path, run_dir, weight_average={"enabled": True, "decay": 0.9})

        ema = load_checkpoint(run_dir / "ema.ckpt")["model_state"]
        last = load_checkpoint(run_dir / "last.ckpt")["model_state"]
        assert any(not torch.equal(ema[key], last[key]) for key in ema)

    def test_a_decay_near_zero_tracks_the_last_weights_step_by_step(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        _run(tmp_path, run_dir, weight_average={"enabled": True, "decay": 1e-3})

        ema = load_checkpoint(run_dir / "ema.ckpt")["model_state"]
        last = load_checkpoint(run_dir / "last.ckpt")["model_state"]
        for key, tensor in last.items():
            assert torch.allclose(ema[key], tensor, atol=1e-5), key

    def test_the_ema_checkpoint_evaluates_to_finite_metrics(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        run_experiment(
            _config(root, weight_average={"enabled": True}),
            data_root=root,
            output_dir=run_dir,
            embedding_reader=reader,
        )

        result = evaluate_checkpoint(
            run_dir / "ema.ckpt", split="val", data_root=root, embedding_reader=reader
        )

        assert math.isfinite(result.global_metrics["dhi"]["mae"])

    def test_val_ema_columns_are_logged_and_finite(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        _run(tmp_path, run_dir, weight_average={"enabled": True})

        rows = pd.read_csv(run_dir / "metrics.csv")
        assert "val_ema_loss" in rows.columns
        assert "val_ema_dhi_mae" in rows.columns
        assert rows["val_ema_dhi_mae"].map(math.isfinite).all()

    def test_the_average_starts_at_start_epoch_and_the_columns_are_blank_before(
        self, tmp_path: Path
    ):
        run_dir = tmp_path / "run"
        _run(tmp_path, run_dir, epochs=3, weight_average={"enabled": True, "start_epoch": 2})

        rows = pd.read_csv(run_dir / "metrics.csv")
        assert rows["val_ema_loss"].isna().tolist() == [True, False, False]
        assert load_checkpoint(run_dir / "ema.ckpt")["epoch"] == 3

    def test_a_resumed_run_continues_the_average_it_left(self, tmp_path: Path):
        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        average = {"enabled": True, "decay": 0.9}

        run_experiment(
            _config(root, epochs=3, weight_average=average),
            data_root=root,
            output_dir=tmp_path / "runA",
            embedding_reader=reader,
        )
        run_experiment(
            _config(root, epochs=2, weight_average=average),
            data_root=root,
            output_dir=tmp_path / "runB",
            embedding_reader=reader,
        )
        run_experiment(
            _config(root, epochs=3, weight_average=average),
            data_root=root,
            output_dir=tmp_path / "runB",
            resume="auto",
            embedding_reader=reader,
        )

        uninterrupted = load_checkpoint(tmp_path / "runA" / "ema.ckpt")["model_state"]
        resumed = load_checkpoint(tmp_path / "runB" / "ema.ckpt")["model_state"]
        for key, tensor in uninterrupted.items():
            assert torch.allclose(resumed[key], tensor, atol=1e-6), key

    def test_a_resume_without_the_ema_file_restarts_the_average_and_says_so(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        run_dir = tmp_path / "run"
        run_experiment(
            _config(root, epochs=2, weight_average={"enabled": True}),
            data_root=root,
            output_dir=run_dir,
            embedding_reader=reader,
        )
        (run_dir / "ema.ckpt").unlink()

        with caplog.at_level("WARNING", logger="allsky.training.engine"):
            run_experiment(
                _config(root, epochs=3, weight_average={"enabled": True}),
                data_root=root,
                output_dir=run_dir,
                resume="auto",
                embedding_reader=reader,
            )

        assert "restarts from the restored weights" in caplog.text
        assert load_checkpoint(run_dir / "ema.ckpt")["epoch"] == 3

    def test_a_fresh_run_rotates_a_previous_ema_checkpoint_aside(self, tmp_path: Path):
        run_dir = tmp_path / "run"
        root, manifest, _ = synthetic.make_dataset(tmp_path)
        reader = synthetic.reader_for(manifest)
        run_experiment(
            _config(root, weight_average={"enabled": True}),
            data_root=root,
            output_dir=run_dir,
            embedding_reader=reader,
        )

        run_experiment(_config(root), data_root=root, output_dir=run_dir, embedding_reader=reader)

        assert not (run_dir / "ema.ckpt").exists()
        assert (run_dir / "ema.ckpt.stale").exists()

    def test_an_ema_column_cannot_be_the_early_stopping_monitor(self, tmp_path: Path):
        with pytest.raises(TrainingError, match="val_ema_loss"):
            _run(
                tmp_path,
                tmp_path / "run",
                weight_average={"enabled": True},
                early_stopping={"monitor": "val_ema_loss", "patience": 100},
            )


@pytest.fixture(scope="module")
def averaged_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    tmp_path = tmp_path_factory.mktemp("averaged")
    run_dir = tmp_path / "run"
    summary = _run(tmp_path, run_dir, weight_average={"enabled": True, "decay": 0.9})
    return run_dir, summary


class TestEmaCheckpointFormat:
    def test_the_summary_points_at_the_ema_checkpoint(
        self, averaged_run: tuple[Path, dict[str, Any]]
    ):
        run_dir, summary = averaged_run

        assert summary["checkpoint_ema"] == str(run_dir / "ema.ckpt")

    def test_the_payload_carries_the_keys_of_last(self, averaged_run: tuple[Path, dict[str, Any]]):
        run_dir, _ = averaged_run

        assert set(load_checkpoint(run_dir / "ema.ckpt")) == set(
            load_checkpoint(run_dir / "last.ckpt")
        )

    def test_the_model_state_carries_the_keys_of_last(
        self, averaged_run: tuple[Path, dict[str, Any]]
    ):
        run_dir, _ = averaged_run

        assert set(load_checkpoint(run_dir / "ema.ckpt")["model_state"]) == set(
            load_checkpoint(run_dir / "last.ckpt")["model_state"]
        )

    def test_the_ema_checkpoint_is_written_at_the_last_epoch(
        self, averaged_run: tuple[Path, dict[str, Any]]
    ):
        run_dir, _ = averaged_run

        assert load_checkpoint(run_dir / "ema.ckpt")["epoch"] == 2

    @pytest.mark.parametrize("field", ["epoch", "config", "feature_columns", "normalizers"])
    def test_a_provenance_field_is_identical_to_last(
        self, averaged_run: tuple[Path, dict[str, Any]], field: str
    ):
        run_dir, _ = averaged_run

        assert (
            load_checkpoint(run_dir / "ema.ckpt")[field]
            == load_checkpoint(run_dir / "last.ckpt")[field]
        )


@pytest.fixture(scope="module")
def resumed_without_average(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    tmp_path = tmp_path_factory.mktemp("resumed")
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    reader = synthetic.reader_for(manifest)
    run_dir = tmp_path / "run"
    run_experiment(
        _config(root, epochs=2, weight_average={"enabled": True}),
        data_root=root,
        output_dir=run_dir,
        embedding_reader=reader,
    )
    summary = run_experiment(
        _config(root, epochs=3),
        data_root=root,
        output_dir=run_dir,
        resume="auto",
        embedding_reader=reader,
    )
    return run_dir, summary


class TestResumeWithoutTheAverage:
    def test_the_previous_ema_checkpoint_is_rotated_aside(
        self, resumed_without_average: tuple[Path, dict[str, Any]]
    ):
        run_dir, _ = resumed_without_average

        assert not (run_dir / "ema.ckpt").exists()
        assert (run_dir / "ema.ckpt.stale").exists()

    def test_the_summary_reports_no_ema_checkpoint(
        self, resumed_without_average: tuple[Path, dict[str, Any]]
    ):
        _, summary = resumed_without_average

        assert summary["checkpoint_ema"] is None

    def test_the_ema_columns_leave_the_continued_history(
        self, resumed_without_average: tuple[Path, dict[str, Any]]
    ):
        run_dir, _ = resumed_without_average

        rows = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        assert [row["epoch"] for row in rows] == [1, 2, 3]
        assert not [key for row in rows for key in row if key.startswith("val_ema_")]
