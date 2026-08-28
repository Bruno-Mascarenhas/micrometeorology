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

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from allsky.cli import app
from allsky.config import SITE_UTC_OFFSET_HOURS, ExperimentConfig, SiteConfig
from allsky.embeddings import backbone as backbone_module
from allsky.embeddings.storage import META_FILENAME, read_meta, write_meta
from allsky.snapshot import (
    DEFAULT_SENSOR_TOLERANCE,
    _feature_vector,
    _image_as_hwc,
    _sensor_row_near,
    predict_snapshot,
)
from allsky.solar import solar_elevation_deg
from allsky.training.checkpointing import load_checkpoint
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
    store = root / "emb"
    write_meta(store, {**read_meta(store), "revision": "fake-v1", "dtype": "fp32"})
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
    assert "wind_speed_ms" in imputed
    assert "wind_dir_sin" in imputed
    # Solar geometry comes from the timestamp, so it is never imputed.
    assert "solar_elevation" not in imputed


def test_the_backbone_is_fed_the_layout_its_transform_documents(sky_image: Path) -> None:
    frame = _image_as_hwc(sky_image)

    assert frame.dtype == np.uint8
    assert frame.shape == (64, 64, 3)


@pytest.mark.parametrize(
    ("column", "railed_value", "feature", "feature_set"),
    [
        ("BP1_mbar_Avg", 2.62, "pressure_mbar", "minimal"),
        ("AirT1_C_Avg", 1000.0, "air_temp_c", "safe"),
    ],
)
def test_a_railed_logger_channel_is_imputed_instead_of_fed_as_a_measurement(
    tmp_path: Path, column: str, railed_value: float, feature: str, feature_set: str
) -> None:
    sensor_csv = tmp_path / "station.csv"
    sensor_csv.write_text(
        f"timestamp,{column},WS_ms,WindDir\n2026-08-14 12:00:00,{railed_value},3.1,90.0\n",
        encoding="utf-8",
    )

    values, imputed = _feature_vector(
        pd.Timestamp("2026-08-14 12:00:00"),
        feature_columns=[feature],
        feature_set=feature_set,
        site=SiteConfig(),
        sensor_csv=sensor_csv,
        tolerance=DEFAULT_SENSOR_TOLERANCE,
        training_means=np.array([1014.93], dtype=np.float32),
    )

    assert imputed == [feature]
    assert float(values[0]) == pytest.approx(1014.93)


@pytest.mark.parametrize(
    ("column", "exported_value", "feature"),
    [
        # An hour in which the Gill railed at 1000 degC for most of its samples
        # averages to a finite number no sentinel literal matches.
        ("AirT1_C_Avg", 525.0, "air_temp_c"),
        # The logger's 7999 rail on the anemometer spelling the feature policy
        # reads; SENTINEL_VALUES names only WS_ms_S_WVT.
        ("WS_ms", 7999.0, "wind_speed_ms"),
    ],
)
def test_an_implausible_exported_value_is_refused_instead_of_served_as_a_measurement(
    tmp_path: Path, column: str, exported_value: float, feature: str
) -> None:
    sensor_csv = tmp_path / "station-hourly.csv"
    sensor_csv.write_text(
        f"timestamp,{column}\n2026-08-14 12:00:00,{exported_value}\n", encoding="utf-8"
    )

    values, imputed = _feature_vector(
        pd.Timestamp("2026-08-14 12:00:00"),
        feature_columns=[feature],
        feature_set="safe",
        site=SiteConfig(),
        sensor_csv=sensor_csv,
        tolerance=DEFAULT_SENSOR_TOLERANCE,
        training_means=np.array([3.7], dtype=np.float32),
    )

    assert imputed == [feature]
    assert float(values[0]) == pytest.approx(3.7)


def test_a_plausible_exported_value_is_served_as_measured(tmp_path: Path) -> None:
    sensor_csv = tmp_path / "station-hourly.csv"
    sensor_csv.write_text("timestamp,AirT1_C_Avg\n2026-08-14 12:00:00,27.4\n", encoding="utf-8")

    values, imputed = _feature_vector(
        pd.Timestamp("2026-08-14 12:00:00"),
        feature_columns=["air_temp_c"],
        feature_set="safe",
        site=SiteConfig(),
        sensor_csv=sensor_csv,
        tolerance=DEFAULT_SENSOR_TOLERANCE,
        training_means=np.array([25.0], dtype=np.float32),
    )

    assert imputed == []
    assert float(values[0]) == pytest.approx(27.4)


def test_a_configuration_declaring_no_plausibility_gate_refuses_to_serve_a_sensor_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sensor_csv = tmp_path / "station-hourly.csv"
    sensor_csv.write_text("timestamp,AirT1_C_Avg\n2026-08-14 12:00:00,27.4\n", encoding="utf-8")
    monkeypatch.setattr(
        "micrometeorology.common.config.get_settings",
        lambda: SimpleNamespace(sensor_limits=[]),
    )

    with pytest.raises(ValueError, match="sensor_limits"):
        _sensor_row_near(sensor_csv, pd.Timestamp("2026-08-14 12:00:00"), DEFAULT_SENSOR_TOLERANCE)


def test_an_offset_carrying_sensor_export_is_matched_on_site_local_wall_clock(
    tmp_path: Path,
) -> None:
    sensor_csv = tmp_path / "station.csv"
    sensor_csv.write_text(
        "timestamp,AirT1_C_Avg\n2026-08-14 12:00:00+00:00,21.4\n2026-08-14 15:00:00+00:00,29.0\n",
        encoding="utf-8",
    )

    row = _sensor_row_near(
        sensor_csv, pd.Timestamp("2026-08-14 12:00:00"), DEFAULT_SENSOR_TOLERANCE
    )

    assert float(row["AirT1_C_Avg"].iloc[0]) == pytest.approx(29.0)


def test_solar_geometry_is_built_on_the_pinned_site_offset_the_manifest_trains_with() -> None:
    site = SiteConfig(latitude=-13.0, longitude=-60.0)
    timestamp = pd.Timestamp("2026-01-15 09:00:00")

    values, _ = _feature_vector(
        timestamp,
        feature_columns=["solar_elevation"],
        feature_set="minimal",
        site=site,
        sensor_csv=None,
        tolerance=DEFAULT_SENSOR_TOLERANCE,
        training_means=np.zeros(1, dtype=np.float32),
    )

    trained_on = solar_elevation_deg(
        pd.DatetimeIndex([timestamp]), site, float(SITE_UTC_OFFSET_HOURS)
    )
    assert float(values[0]) == pytest.approx(float(trained_on[0]), abs=1e-3)


def _forget_the_recorded_recipe(checkpoint_path: Path) -> None:
    """Rewrite the checkpoint as a release predating the portable encoding recipe wrote it."""
    import torch

    payload = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    payload["backbone"] = None
    torch.save(payload, checkpoint_path)


def _embedding_store_of(checkpoint_path: Path) -> Path:
    payload = load_checkpoint(checkpoint_path, map_location="cpu", trust_pickle=True)
    cfg = ExperimentConfig.model_validate(payload["config"])
    assert cfg.data.embeddings_dir is not None
    return Path(cfg.data.data_root) / cfg.data.embeddings_dir


def test_the_live_frame_is_embedded_with_the_pooling_its_training_store_recorded(
    embedding_checkpoint: Path, sky_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    write_meta(store, {**read_meta(store), "pooling": "mean"})
    seen: dict[str, Any] = {}
    real_build_backbone = backbone_module.build_backbone

    def recording_build_backbone(name: str, **kwargs: Any) -> Any:
        seen.update({"name": name, **kwargs})
        return real_build_backbone(name, **kwargs)

    monkeypatch.setattr(backbone_module, "build_backbone", recording_build_backbone)
    predict_snapshot(
        sky_image,
        embedding_checkpoint,
        timestamp=pd.Timestamp("2026-01-01 12:00:00"),
        trust_checkpoint=True,
    )

    assert seen["pooling"] == "mean"


def test_a_checkpoint_recording_no_recipe_refuses_to_guess_when_its_store_is_unreachable(
    embedding_checkpoint: Path, sky_image: Path
) -> None:
    (_embedding_store_of(embedding_checkpoint) / META_FILENAME).unlink()
    _forget_the_recorded_recipe(embedding_checkpoint)

    with pytest.raises(ValueError, match=META_FILENAME):
        predict_snapshot(
            sky_image,
            embedding_checkpoint,
            timestamp=pd.Timestamp("2026-01-01 12:00:00"),
            trust_checkpoint=True,
        )


def test_a_checkpoint_carrying_its_own_recipe_predicts_with_the_store_gone(
    embedding_checkpoint: Path, sky_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    recorded = read_meta(store)
    shutil.rmtree(store)
    seen: dict[str, Any] = {}
    real_build_backbone = backbone_module.build_backbone

    def recording_build_backbone(name: str, **kwargs: Any) -> Any:
        seen.update({"name": name, **kwargs})
        return real_build_backbone(name, **kwargs)

    monkeypatch.setattr(backbone_module, "build_backbone", recording_build_backbone)
    prediction = predict_snapshot(
        sky_image,
        embedding_checkpoint,
        timestamp=pd.Timestamp("2026-01-01 12:00:00"),
        trust_checkpoint=True,
    )

    assert prediction["predictions"]
    assert (seen["name"], seen["pooling"], seen["dtype"]) == (
        recorded["backbone"],
        recorded["pooling"],
        recorded["dtype"],
    )


def test_the_live_frame_is_embedded_at_the_precision_its_training_store_recorded(
    embedding_checkpoint: Path, sky_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    write_meta(store, {**read_meta(store), "dtype": "fp32"})
    seen: dict[str, Any] = {}
    real_build_backbone = backbone_module.build_backbone

    def recording_build_backbone(name: str, **kwargs: Any) -> Any:
        seen.update({"name": name, **kwargs})
        return real_build_backbone(name, **kwargs)

    monkeypatch.setattr(backbone_module, "build_backbone", recording_build_backbone)
    predict_snapshot(
        sky_image,
        embedding_checkpoint,
        timestamp=pd.Timestamp("2026-01-01 12:00:00"),
        trust_checkpoint=True,
    )

    assert seen.get("dtype") == "fp32"


def test_a_store_encoded_under_a_superseded_backbone_revision_refuses_to_predict(
    embedding_checkpoint: Path, sky_image: Path
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    write_meta(store, {**read_meta(store), "revision": "fake-v0"})

    with pytest.raises(ValueError, match="revision"):
        predict_snapshot(
            sky_image,
            embedding_checkpoint,
            timestamp=pd.Timestamp("2026-01-01 12:00:00"),
            trust_checkpoint=True,
        )


@pytest.mark.parametrize("absent", ["revision", "dim", "dtype"])
def test_a_store_sidecar_missing_part_of_the_encoding_recipe_refuses_to_predict(
    embedding_checkpoint: Path, sky_image: Path, absent: str
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    meta = read_meta(store)
    del meta[absent]
    write_meta(store, meta)

    with pytest.raises(ValueError, match=absent):
        predict_snapshot(
            sky_image,
            embedding_checkpoint,
            timestamp=pd.Timestamp("2026-01-01 12:00:00"),
            trust_checkpoint=True,
        )


def test_a_moved_checkpoint_predicts_against_the_store_it_is_pointed_at(
    tmp_path: Path, embedding_checkpoint: Path, sky_image: Path
) -> None:
    store = _embedding_store_of(embedding_checkpoint)
    shipped_beside_checkpoint = tmp_path / "shipped" / "emb"
    shutil.copytree(store, shipped_beside_checkpoint)
    shutil.rmtree(store)

    prediction = predict_snapshot(
        sky_image,
        embedding_checkpoint,
        timestamp=pd.Timestamp("2026-01-01 12:00:00"),
        trust_checkpoint=True,
        embeddings_dir=shipped_beside_checkpoint,
    )

    assert np.isfinite(prediction["predictions"]["dhi"])
