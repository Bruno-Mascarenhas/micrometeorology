"""Tests for the ``precompute-embeddings`` CLI (dry-run, fake run, resume, errors)."""

import hashlib
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from allsky.cli import app
from allsky.cli.embeddings import _config_sha256
from allsky.config import PrepareConfig, load_prepare_config

runner = CliRunner()


def _build_dataset(dataset_dir: Path, n: int = 4, size: int = 16) -> None:
    """Write tiny JPEGs + a manifest.parquet under *dataset_dir*."""
    frames = dataset_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    rows = []
    for i in range(n):
        name = f"allsky-20250321-09{i * 10:02d}.jpg"
        iio.imwrite(
            frames / name, rng.integers(0, 256, (size, size, 3), dtype=np.uint8), quality=90
        )
        rows.append({"sample_id": name.removesuffix(".jpg"), "image_path": f"frames/{name}"})
    pd.DataFrame(rows).to_parquet(dataset_dir / "manifest.parquet", index=False)


def _write_config(tmp_path: Path, dataset_dir: Path, backbone: str = "fake") -> Path:
    config = tmp_path / "prepare.yaml"
    config.write_text(
        "output:\n"
        f"  dataset_dir: {dataset_dir}\n"
        "embeddings:\n"
        f"  backbone: {backbone}\n"
        "  batch_size: 2\n"
        "  shard_size: 3\n",
        encoding="utf-8",
    )
    return config


def _summary(output: str) -> dict:
    """Extract the JSON summary the command prints to stdout."""
    start = output.index("{")
    end = output.rindex("}") + 1
    parsed: dict = json.loads(output[start:end])
    return parsed


def test_dry_run_writes_nothing(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=4)
    config = _write_config(tmp_path, dataset_dir)

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not (dataset_dir / "embeddings").exists()


def test_fake_backbone_run(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=4)
    config = _write_config(tmp_path, dataset_dir)

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    assert result.exit_code == 0, result.output
    summary = _summary(result.output)
    assert summary["encoded"] == 4
    assert summary["backbone"] == "fake"

    emb_dir = dataset_dir / "embeddings"
    assert (emb_dir / "index.parquet").exists()
    assert (emb_dir / "embeddings.meta.json").exists()
    assert list(emb_dir.glob("embeddings-*.safetensors"))


def test_out_override(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=3)
    config = _write_config(tmp_path, dataset_dir)
    out = tmp_path / "custom_emb"

    result = runner.invoke(
        app, ["precompute-embeddings", "--config", str(config), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert (out / "index.parquet").exists()


def test_resume_encodes_zero_on_second_run(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=4)
    config = _write_config(tmp_path, dataset_dir)

    first = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    assert first.exit_code == 0, first.output
    assert _summary(first.output)["encoded"] == 4

    second = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    assert second.exit_code == 0, second.output
    summary = _summary(second.output)
    assert summary["encoded"] == 0
    assert summary["skipped"] == 4


def test_no_resume_reprocesses(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=3)
    config = _write_config(tmp_path, dataset_dir)

    runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config), "--no-resume"])
    assert result.exit_code == 0, result.output
    assert _summary(result.output)["encoded"] == 3


def test_unknown_backbone_errors(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=2)
    config = _write_config(tmp_path, dataset_dir, backbone="not-a-real-backbone")

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    assert result.exit_code == 1
    assert "not-a-real-backbone" in result.output
    assert "dinov2_vits14" in result.output
    assert "fake" in result.output


def test_missing_manifest_errors(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)  # no manifest.parquet
    config = _write_config(tmp_path, dataset_dir)

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])
    assert result.exit_code == 1
    assert "manifest not found" in result.output


@pytest.mark.parametrize(
    "patch",
    [
        {"mask": {"path": "masks/horizon.png"}},
        {"crop": {"enabled": True, "top": 40}},
        {"resize": 224},
        {"video": {"timestamps": "modelled"}},
    ],
)
def test_a_config_edit_that_changes_the_encoded_pixels_changes_the_resume_hash(patch: dict):
    base = _config_sha256(PrepareConfig())

    assert _config_sha256(PrepareConfig.model_validate(patch)) != base


def test_widening_the_video_pattern_leaves_the_resume_hash_alone():
    base = _config_sha256(PrepareConfig())
    grown = PrepareConfig.model_validate({"video": {"pattern": "data/all-sky/allsky-2026*.mp4"}})

    assert _config_sha256(grown) == base


def _write_mask(path: Path, *, blacked_rows: int, size: int = 16) -> Path:
    keep = np.full((size, size), 255, dtype=np.uint8)
    keep[:blacked_rows] = 0
    iio.imwrite(path, keep)
    return path


def test_rewriting_the_mask_png_in_place_changes_the_resume_hash(tmp_path: Path):
    mask = _write_mask(tmp_path / "horizon.png", blacked_rows=0)
    cfg = PrepareConfig.model_validate({"mask": {"path": str(mask)}})
    before = _config_sha256(cfg)

    _write_mask(mask, blacked_rows=4)

    assert _config_sha256(cfg) != before


def test_a_splits_edit_leaves_the_resume_hash_alone(tmp_path: Path):
    mask = _write_mask(tmp_path / "horizon.png", blacked_rows=0)
    base = _config_sha256(PrepareConfig.model_validate({"mask": {"path": str(mask)}}))

    redrawn = PrepareConfig.model_validate({"mask": {"path": str(mask)}, "splits": {"seed": 7}})

    assert _config_sha256(redrawn) == base


def _pre_migration_config_sha256(cfg: PrepareConfig) -> str:
    """The digest formula every embedding store on disk was stamped with."""
    canonical = json.dumps(cfg.embeddings.model_dump(), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _meta_path(dataset_dir: Path) -> Path:
    return dataset_dir / "embeddings" / "embeddings.meta.json"


def _restamp(dataset_dir: Path, config_sha256: str) -> None:
    meta = json.loads(_meta_path(dataset_dir).read_text(encoding="utf-8"))
    meta["config_sha256"] = config_sha256
    _meta_path(dataset_dir).write_text(json.dumps(meta), encoding="utf-8")


def test_a_store_stamped_by_the_old_formula_resumes_and_is_restamped(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=4)
    config = _write_config(tmp_path, dataset_dir)
    assert runner.invoke(app, ["precompute-embeddings", "--config", str(config)]).exit_code == 0
    cfg = load_prepare_config(config)
    _restamp(dataset_dir, _pre_migration_config_sha256(cfg))

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert _summary(result.output)["encoded"] == 0
    assert json.loads(_meta_path(dataset_dir).read_text(encoding="utf-8"))[
        "config_sha256"
    ] == _config_sha256(cfg)


def test_a_store_stamped_with_an_unrelated_digest_still_refuses_to_resume(tmp_path: Path):
    pytest.importorskip("torch")
    dataset_dir = tmp_path / "dataset"
    _build_dataset(dataset_dir, n=4)
    config = _write_config(tmp_path, dataset_dir)
    assert runner.invoke(app, ["precompute-embeddings", "--config", str(config)]).exit_code == 0
    _restamp(dataset_dir, "0" * 64)

    result = runner.invoke(app, ["precompute-embeddings", "--config", str(config)])

    assert result.exit_code == 1
    assert "cannot resume" in result.output
