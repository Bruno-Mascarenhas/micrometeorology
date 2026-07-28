"""Integrity verification for unpickled model artifacts (serialization.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from solrad_correction.experiments.artifacts import ArtifactLayout, write_manifest
from solrad_correction.utils.serialization import (
    ModelIntegrityError,
    load_sklearn_model,
    save_sklearn_model,
)


def test_manifest_covered_artifact_verifies_and_loads(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    model_path = root / "models" / "model.joblib"
    save_sklearn_model({"weights": [1.0, 2.0, 3.0]}, model_path)
    write_manifest(ArtifactLayout.from_experiment_dir(root))

    # sha256 matches the manifest → loads without raising.
    assert load_sklearn_model(model_path) == {"weights": [1.0, 2.0, 3.0]}


def test_tampered_artifact_with_manifest_raises(tmp_path: Path) -> None:
    root = tmp_path / "experiment"
    model_path = root / "models" / "model.joblib"
    save_sklearn_model({"weights": [1.0, 2.0, 3.0]}, model_path)
    write_manifest(ArtifactLayout.from_experiment_dir(root))

    # Overwrite the artifact after the manifest was checksummed.
    save_sklearn_model({"weights": [9.9, 9.9]}, model_path)

    with pytest.raises(ModelIntegrityError, match="Integrity check failed"):
        load_sklearn_model(model_path)


def test_missing_manifest_warns_and_loads(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    model_path = tmp_path / "loose" / "model.joblib"
    save_sklearn_model({"weights": [0.5]}, model_path)  # no manifest.json anywhere

    with caplog.at_level(logging.WARNING, logger="solrad_correction.utils.serialization"):
        loaded = load_sklearn_model(model_path)

    assert loaded == {"weights": [0.5]}
    assert any("unverified pickle" in record.message for record in caplog.records)


def test_manifest_present_but_artifact_uncovered_warns_and_loads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "experiment"
    covered = root / "models" / "model.joblib"
    save_sklearn_model({"weights": [1.0]}, covered)
    write_manifest(ArtifactLayout.from_experiment_dir(root))

    # A new artifact added after the manifest is not covered by it.
    uncovered = root / "models" / "extra.joblib"
    save_sklearn_model({"weights": [2.0]}, uncovered)

    with caplog.at_level(logging.WARNING, logger="solrad_correction.utils.serialization"):
        loaded = load_sklearn_model(uncovered)

    assert loaded == {"weights": [2.0]}
    assert any("unverified pickle" in record.message for record in caplog.records)


def test_truncated_manifest_is_reported_as_unusable_not_missing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A run killed mid-write leaves truncated JSON; the warning must say so."""
    root = tmp_path / "experiment"
    model_path = root / "models" / "model.joblib"
    save_sklearn_model({"weights": [1.0]}, model_path)
    write_manifest(ArtifactLayout.from_experiment_dir(root))
    manifest = root / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8")[:40], encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="solrad_correction.utils.serialization"):
        loaded = load_sklearn_model(model_path)

    assert loaded == {"weights": [1.0]}
    assert any("present but unusable" in record.message for record in caplog.records)
    assert not any("no manifest.json found" in record.message for record in caplog.records)


def test_unusable_manifest_does_not_mask_the_experiment_manifest(tmp_path: Path) -> None:
    """A nearer unusable manifest must not stop the walk and disable verification."""
    root = tmp_path / "experiment"
    model_path = root / "models" / "model.joblib"
    save_sklearn_model({"weights": [3.0]}, model_path)
    write_manifest(ArtifactLayout.from_experiment_dir(root))
    (root / "models" / "manifest.json").write_text("{not json", encoding="utf-8")

    # Tampered after the experiment manifest recorded its checksum.
    save_sklearn_model({"weights": [4.4]}, model_path)

    with pytest.raises(ModelIntegrityError, match="Integrity check failed"):
        load_sklearn_model(model_path)


def _seed_every_global_stream(seed: int) -> None:
    """Seed the three global RNGs a resumed run has to continue."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - the legacy global stream is what resumes
    torch.manual_seed(seed)


def _draw_from_every_global_stream() -> tuple[float, float, float]:
    """One draw from each global RNG, as a fingerprint of the stream position."""
    import random

    import numpy as np
    import torch

    return (
        random.random(),  # noqa: S311 - reproducibility fingerprint, not crypto
        float(np.random.rand()),  # noqa: NPY002 - the legacy global stream is what resumes
        float(torch.rand(1)),
    )


class TestRngStateRoundTrip:
    """A resume must continue the random stream, not replay it from the seed.

    ``load_torch_checkpoint`` reads under ``weights_only=True``, so the snapshot
    has to be made of restricted types — allsky's ``np.random.get_state()``
    tuple cannot be reused verbatim here, it would make every checkpoint
    unloadable.
    """

    def test_the_snapshot_survives_a_weights_only_checkpoint_round_trip(
        self, tmp_path: Path
    ) -> None:
        pytest.importorskip("torch")
        from solrad_correction.utils.serialization import (
            capture_rng_state,
            load_torch_checkpoint,
            restore_rng_state,
            save_torch_checkpoint,
        )

        _seed_every_global_stream(17)
        snapshot = capture_rng_state()
        uninterrupted = _draw_from_every_global_stream()

        path = tmp_path / "last.pt"
        save_torch_checkpoint(
            model_state={},
            optimizer_state=None,
            config=None,
            epoch=3,
            path=path,
            metadata={"rng_state": snapshot},
        )
        _draw_from_every_global_stream()  # drift away from the snapshot

        restore_rng_state(load_torch_checkpoint(path)["metadata"]["rng_state"])

        assert _draw_from_every_global_stream() == uninterrupted

    def test_a_checkpoint_without_rng_state_still_resumes(self) -> None:
        """Older ``last.pt`` files predate the snapshot and must not crash."""
        pytest.importorskip("torch")
        from solrad_correction.utils.serialization import restore_rng_state

        restore_rng_state({})

    def test_the_checkpoint_manager_persists_the_snapshot(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")

        from solrad_correction.training.checkpoints import CheckpointManager
        from solrad_correction.utils.serialization import load_torch_checkpoint

        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        CheckpointManager(directory=tmp_path).save_last(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            metric=0.5,
            dataloader_settings=None,
        )

        rng_state = load_torch_checkpoint(tmp_path / "last.pt")["metadata"]["rng_state"]
        assert {"python_keys", "numpy_keys", "torch"} <= set(rng_state)

    def test_the_trainer_carries_and_applies_a_snapshot(self) -> None:
        """The restore point is Trainer.train, after the module/optimizer exist."""
        torch = pytest.importorskip("torch")

        from solrad_correction.training.trainer import Trainer
        from solrad_correction.utils.serialization import capture_rng_state, restore_rng_state

        _seed_every_global_stream(5)
        snapshot = capture_rng_state()
        expected = _draw_from_every_global_stream()
        _draw_from_every_global_stream()  # drift away from the snapshot

        trainer = Trainer(model=torch.nn.Linear(2, 1), device="cpu", rng_state=snapshot)
        assert trainer._resume_rng_state is snapshot

        restore_rng_state(trainer._resume_rng_state)
        assert _draw_from_every_global_stream() == expected
