"""Prediction path of ``allsky snapshot --checkpoint``.

Nothing exercised :func:`allsky.snapshot.predict_snapshot` before, and two
defects rode along in it:

- a capture with no ``--sensor-csv`` — the documented fallback, where every
  sensor-derived feature is meant to be imputed at its training mean — died on
  ``KeyError: sensor frame is missing required feature columns``, because
  ``build_feature_frame`` refuses an absent source column and the empty frame
  supplied every one of them absent;
- the embedding branch handed the ``(3, S, S)`` float array of the image branch
  straight to ``backbone.encode``, skipping the ``transform`` its contract
  requires. Against DINOv2 that raised ``AttributeError: 'numpy.ndarray' object
  has no attribute 'to'``, so every embedding-mode checkpoint (V0-V5, V7) could
  not predict at all.

Offline: the deterministic ``fake`` backbone and the synthetic dataset fixture.
Note the fake backbone casts whatever layout it is handed, so it cannot tell the
two array shapes apart — :func:`test_the_backbone_is_fed_the_layout_its_transform_documents`
pins the helper's contract instead.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from allsky.cli import app
from allsky.snapshot import _image_as_hwc, predict_snapshot
from tests.allsky import _synthetic as synthetic
from tests.allsky.test_e2e_experiment import _REPO_V1, _write_embeddings

runner = CliRunner()

_EMBEDDING_EXPERIMENT = """\
extends: ["{repo_v1}"]
name: snapshot_probe
output_dir: {out}
model:
  name: image_only
  backbone: fake
  backbone_pooling: cls
data:
  data_root: {root}
  manifest: manifest.parquet
  split_artifact: splits.json
  embeddings_dir: emb
train:
  epochs: 1
  batch_size: 8
  num_workers: 0
  device: cpu
  amp:
    enabled: false
"""


@pytest.fixture
def embedding_checkpoint(tmp_path: Path) -> Path:
    """Train one epoch of an embedding-mode model on the fake backbone's width."""
    root, manifest, _ = synthetic.make_dataset(tmp_path)
    _write_embeddings(root, manifest, dim=32)
    run_dir = tmp_path / "run"
    config = tmp_path / "probe.yaml"
    config.write_text(
        _EMBEDDING_EXPERIMENT.format(repo_v1=_REPO_V1, out=run_dir, root=root),
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        ["train", "--config", str(config), "--data-root", str(root), "--out-dir", str(run_dir)],
    )
    assert result.exit_code == 0, result.output
    return run_dir / "best.ckpt"


@pytest.fixture
def sky_image(tmp_path: Path) -> Path:
    """A 64x64 RGB frame standing in for a captured sky image."""
    from PIL import Image

    path = tmp_path / "frame.jpg"
    pixels = np.random.default_rng(7).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path, quality=92)
    return path


def test_a_capture_without_a_sensor_export_imputes_instead_of_raising(
    embedding_checkpoint: Path, sky_image: Path
) -> None:
    prediction = predict_snapshot(
        sky_image,
        embedding_checkpoint,
        timestamp=pd.Timestamp("2026-01-01 12:00:00"),
        sensor_csv=None,
        trust_checkpoint=True,
    )

    assert np.isfinite(prediction["predictions"]["dhi"])
    imputed = prediction["features"]["imputed"]
    assert "pressure_mbar" in imputed
    assert "wind_speed_ms" in imputed
    # Solar geometry comes from the timestamp, so it is never imputed.
    assert "solar_elevation" not in imputed


def test_the_backbone_is_fed_the_layout_its_transform_documents(sky_image: Path) -> None:
    frame = _image_as_hwc(sky_image)

    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)
