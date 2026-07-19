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
