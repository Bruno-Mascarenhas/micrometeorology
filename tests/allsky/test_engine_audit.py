"""Torch-gated engine tests for the all-sky correctness audit.

All offline and CPU-only: the unit-level guards run on hand-built tensors, and
the tests that drive whole runs reuse the tiny synthetic manifest builders from
``tests/allsky/test_engine.py``.
"""

import logging
import math
from pathlib import Path

import pytest
import torch

from allsky.config import ExperimentConfig
from allsky.training.checkpointing import load_checkpoint
from allsky.training.engine import (
    _build_amp,
    _build_optimizer,
    _improved,
    _MetricAccumulator,
    run_experiment,
)
from tests.allsky.test_engine import _cfg, _make_dataset, _reader


@pytest.mark.parametrize("mode", ["min", "max"])
def test_a_diverged_epoch_is_never_the_first_best(mode: str):
    assert _improved(math.nan, None, mode, 0.0) is False


@pytest.mark.parametrize(
    ("mode", "current"),
    [("min", 0.87), ("max", 0.87)],
)
def test_a_finite_epoch_improves_on_a_best_poisoned_by_an_earlier_divergence(
    mode: str, current: float
):
    assert _improved(current, math.nan, mode, 0.0) is True


def test_resume_warns_that_an_edited_learning_rate_loses_to_the_checkpoint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    root, manifest, _ = _make_dataset(tmp_path)
    reader = _reader(manifest)
    run_dir = tmp_path / "run"
    run_experiment(
        _cfg(root, epochs=2), data_root=root, output_dir=run_dir, embedding_reader=reader
    )
    edited = _cfg(root, epochs=3)
    edited.train.lr = 3e-5

    with caplog.at_level(logging.WARNING, logger="allsky.training.engine"):
        run_experiment(
            edited,
            data_root=root,
            output_dir=run_dir,
            resume="auto",
            embedding_reader=reader,
        )

    assert any("train.lr" in record.getMessage() for record in caplog.records)


def test_an_optimizer_the_engine_cannot_build_is_refused_instead_of_run_as_adamw():
    cfg = ExperimentConfig.model_validate({"train": {"optimizer": "sgd", "device": "cpu"}})
    model = torch.nn.Linear(3, 1)

    with pytest.raises(ValueError, match="sgd"):
        _build_optimizer(model, cfg)


def test_bf16_amp_autocasts_on_the_run_device_and_not_on_the_cpu():
    autocast_device, _dtype, _scaler = _build_amp(True, "bf16", "mps")

    assert autocast_device == "mps"


def test_amp_on_a_device_torch_cannot_autocast_is_refused():
    with pytest.raises(RuntimeError, match="meta"):
        _build_amp(True, "bf16", "meta")


class _HostReadCountingTensor(torch.Tensor):
    """Tensor counting every conversion to a Python scalar (one device sync each)."""

    host_reads = 0

    def __float__(self) -> float:
        type(self).host_reads += 1
        return super().__float__()

    def __int__(self) -> int:
        type(self).host_reads += 1
        return super().__int__()

    def __bool__(self) -> bool:
        type(self).host_reads += 1
        return super().__bool__()


def _counted(values: torch.Tensor) -> torch.Tensor:
    return values.as_subclass(_HostReadCountingTensor)


def test_folding_a_batch_into_the_epoch_metrics_reads_no_scalar_back_to_the_host():
    batch = {
        "features": _counted(torch.zeros(4, 3)),
        "dhi": _counted(torch.tensor([1.0, 2.0, math.nan, 4.0])),
        "kindex": _counted(torch.full((4,), math.nan)),
        "sky_class": _counted(torch.tensor([0, -1, 2, 1])),
        "cloud_fraction": _counted(torch.full((4,), math.nan)),
    }
    outputs = {
        "dhi": _counted(torch.zeros(4)),
        "kindex": _counted(torch.zeros(4)),
        "sky_logits": _counted(torch.zeros(4, 3)),
    }
    losses = {
        "loss": _counted(torch.tensor(1.5)),
        "loss_dhi": _counted(torch.tensor(0.5)),
        "loss_kindex": _counted(torch.tensor(0.0)),
        "loss_sky": _counted(torch.tensor(1.0)),
    }
    accumulator = _MetricAccumulator(
        (0.0, 1.0, 0.0, 1.0), {"loss_dhi": 1.0, "loss_kindex": 1.0, "loss_sky": 1.0}
    )
    _HostReadCountingTensor.host_reads = 0

    accumulator.update(outputs, batch, losses)

    assert _HostReadCountingTensor.host_reads == 0


@pytest.mark.parametrize("name", ["best.ckpt", "last.ckpt"])
def test_a_fresh_run_into_a_reused_dir_rotates_the_previous_checkpoint_aside(
    tmp_path: Path, name: str
):
    root, manifest, _ = _make_dataset(tmp_path)
    reader = _reader(manifest)
    run_dir = tmp_path / "run"
    run_experiment(
        _cfg(root, epochs=2), data_root=root, output_dir=run_dir, embedding_reader=reader
    )
    preserved_epoch = load_checkpoint(run_dir / name)["epoch"]

    run_experiment(
        _cfg(root, epochs=1), data_root=root, output_dir=run_dir, embedding_reader=reader
    )

    assert load_checkpoint(run_dir / f"{name}.stale")["epoch"] == preserved_epoch


@pytest.mark.parametrize("poisoned", [math.inf, math.nan])
def test_a_labelless_batch_of_a_head_cannot_poison_that_head_s_epoch_mean(poisoned: float):
    accumulator = _MetricAccumulator((0.0, 1.0, 0.0, 1.0), {"loss_dhi": 1.0})
    no_rows = {"features": torch.zeros(4, 3), "dhi": torch.full((4,), math.nan)}
    with_rows = {"features": torch.zeros(4, 3), "dhi": torch.tensor([1.0, 2.0, 3.0, 4.0])}

    accumulator.update({}, no_rows, {"loss_dhi": torch.tensor(poisoned)})
    accumulator.update({}, with_rows, {"loss_dhi": torch.tensor(0.5)})

    assert accumulator.result() == {"loss": pytest.approx(0.5), "loss_dhi": pytest.approx(0.5)}


def test_a_non_finite_loss_on_a_head_that_has_rows_still_reaches_the_epoch_metrics():
    accumulator = _MetricAccumulator((0.0, 1.0, 0.0, 1.0), {"loss_dhi": 1.0})
    batch = {"features": torch.zeros(4, 3), "dhi": torch.tensor([1.0, 2.0, 3.0, 4.0])}

    accumulator.update({}, batch, {"loss_dhi": torch.tensor(math.nan)})

    assert math.isnan(accumulator.result()["loss_dhi"])


def _poison_first_weight(path: Path) -> None:
    """Write a NaN into the first floating-point parameter of the checkpoint at *path*."""
    payload = load_checkpoint(path)
    tensor = next(
        value
        for value in payload["model_state"].values()
        if isinstance(value, torch.Tensor) and value.is_floating_point() and value.numel()
    )
    tensor.reshape(-1)[0] = math.nan
    torch.save(payload, path)


def test_resuming_from_a_checkpoint_whose_weights_diverged_is_refused(tmp_path: Path):
    root, manifest, _ = _make_dataset(tmp_path)
    reader = _reader(manifest)
    run_dir = tmp_path / "run"
    run_experiment(
        _cfg(root, epochs=1), data_root=root, output_dir=run_dir, embedding_reader=reader
    )
    _poison_first_weight(run_dir / "last.ckpt")

    with pytest.raises(RuntimeError, match=r"last\.ckpt") as excinfo:
        run_experiment(
            _cfg(root, epochs=2),
            data_root=root,
            output_dir=run_dir,
            resume="auto",
            embedding_reader=reader,
        )

    assert "best.ckpt" in str(excinfo.value)


def test_a_checkpoint_with_finite_weights_still_resumes(tmp_path: Path):
    root, manifest, _ = _make_dataset(tmp_path)
    reader = _reader(manifest)
    run_dir = tmp_path / "run"
    run_experiment(
        _cfg(root, epochs=1), data_root=root, output_dir=run_dir, embedding_reader=reader
    )

    summary = run_experiment(
        _cfg(root, epochs=2),
        data_root=root,
        output_dir=run_dir,
        resume="auto",
        embedding_reader=reader,
    )

    assert summary["epoch"] == 2
