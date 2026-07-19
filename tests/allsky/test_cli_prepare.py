"""Tests for the Wave C2b prepare CLI (validate-dataset, prepare-local, export).

Offline end-to-end: a tiny synthetic mp4 + a synthetic TOA5 ``.dat`` drive
``prepare-local`` into a real manifest; ``validate-dataset`` is exercised on a
good and a broken manifest; ``export-colab-bundle`` produces a bundle that
:func:`allsky.bundle.validate_bundle` accepts. CliRunner only, tmp_path only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from allsky import solar
from allsky.bundle import validate_bundle
from allsky.cli import app
from allsky.config import SiteConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

_SAFE_COLUMNS = ("AirT1_C_Avg", "DP1_C_Avg", "RH1", "BP1_mbar_Avg", "WS_ms", "WindDir")


@pytest.fixture(scope="module")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny 8-frame 64x64 mp4 named like a real all-sky file (2026-01-01)."""
    videos = tmp_path_factory.mktemp("videos")
    path = videos / "allsky-20260101.mp4"
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(8, 64, 64, 3)).astype(np.uint8)
    iio.imwrite(path, frames, fps=25)
    return path


@pytest.fixture(scope="module")
def synthetic_dat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minimal Campbell TOA5 .dat covering 2026-01-01 06:00-06:10."""
    path = tmp_path_factory.mktemp("sensors") / "synthetic.dat"
    columns = ["TIMESTAMP", *_SAFE_COLUMNS, "CM3Up_Wm2_Avg", "PSP_Wm2_Avg"]
    header = ",".join(f'"{c}"' for c in columns)
    units = ",".join('"unit"' for _ in columns)
    process = ",".join('"Avg"' for _ in columns)
    values = {
        "AirT1_C_Avg": 25.0,
        "DP1_C_Avg": 15.0,
        "RH1": 70.0,
        "BP1_mbar_Avg": 1010.0,
        "WS_ms": 3.0,
        "WindDir": 180.0,
        "CM3Up_Wm2_Avg": 120.0,
        "PSP_Wm2_Avg": 30.0,
    }
    lines = [
        '"TOA5","LBM","CR5000","0","std","prog","sig","table"',
        header,
        units,
        process,
    ]
    for ts in pd.date_range("2026-01-01 06:00", "2026-01-01 06:10", freq="1min"):
        row = [f'"{ts:%Y-%m-%d %H:%M:%S}"', *(str(values[c]) for c in columns[1:])]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_config(
    path: Path,
    *,
    dataset_dir: Path,
    video_pattern: str,
    dat_path: Path,
    seed: int = 42,
) -> Path:
    """Write a PrepareConfig YAML for the CLI."""
    path.write_text(
        "video:\n"
        f"  pattern: '{video_pattern}'\n"
        "sensor:\n"
        f"  paths: ['{dat_path}']\n"
        "output:\n"
        f"  dataset_dir: '{dataset_dir}'\n"
        "splits:\n"
        f"  seed: {seed}\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# manifest builders for validate-dataset / splits
# ---------------------------------------------------------------------------


def _sensor_frame(site: SiteConfig, index: pd.DatetimeIndex) -> pd.DataFrame:
    e0h = solar.extraterrestrial_ghi(index, site)
    n = len(index)
    return pd.DataFrame(
        {
            "AirT1_C_Avg": [25.0] * n,
            "DP1_C_Avg": [15.0] * n,
            "RH1": [70.0] * n,
            "BP1_mbar_Avg": [1010.0] * n,
            "WS_ms": [3.0] * n,
            "WindDir": [180.0] * n,
            "CM3Up_Wm2_Avg": 0.7 * e0h,
            "PSP_Wm2_Avg": 0.2 * e0h,
        },
        index=index,
    )


def _frames(data_root: Path, times: list[str], *, create_files: bool) -> pd.DataFrame:
    frames_dir = data_root / "frames"
    if create_files:
        frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, when in enumerate(times):
        ts = pd.Timestamp(when)
        frame_path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        if create_files:
            frame_path.write_bytes(b"jpeg")
        rows.append({"frame_path": str(frame_path), "timestamp": ts, "video": "v.mp4", "index": i})
    return pd.DataFrame(rows)


def _write_manifest(dataset_dir: Path, times: list[str], *, create_files: bool) -> Path:
    site = SiteConfig()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    index = pd.date_range("2025-03-21 06:00", "2025-03-23 18:00", freq="1h")
    manifest, meta = build_manifest(
        _frames(dataset_dir, times, create_files=create_files),
        _sensor_frame(site, index),
        site=site,
        data_root=dataset_dir,
    )
    write_manifest_parquet(manifest, meta, dataset_dir / "manifest.parquet")
    return dataset_dir / "manifest.parquet"


# ---------------------------------------------------------------------------
# validate-dataset
# ---------------------------------------------------------------------------


class TestValidateDataset:
    def test_good_manifest_exit_zero(self, tmp_path: Path):
        dataset_dir = tmp_path / "good"
        _write_manifest(dataset_dir, ["2025-03-21 12:00"], create_files=True)
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        result = runner.invoke(app, ["validate-dataset", "--config", str(config)])
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

    def test_broken_manifest_exit_one(self, tmp_path: Path):
        dataset_dir = tmp_path / "broken"
        # Frames not written to disk -> image files missing -> validation error.
        _write_manifest(dataset_dir, ["2025-03-21 12:00"], create_files=False)
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        result = runner.invoke(app, ["validate-dataset", "--config", str(config)])
        assert result.exit_code == 1
        assert "ERROR" in result.output

    def test_missing_manifest_exit_one(self, tmp_path: Path):
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=tmp_path / "absent",
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        result = runner.invoke(app, ["validate-dataset", "--config", str(config)])
        assert result.exit_code == 1
        assert "manifest not found" in result.output


# ---------------------------------------------------------------------------
# prepare-local
# ---------------------------------------------------------------------------


class TestPrepareLocal:
    def test_dry_run_writes_nothing(
        self, tmp_path: Path, synthetic_video: Path, synthetic_dat: Path
    ):
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{synthetic_video.parent}/allsky-*.mp4",
            dat_path=synthetic_dat,
        )
        result = runner.invoke(app, ["prepare-local", "--config", str(config), "--dry-run"])
        assert result.exit_code == 0, result.output
        assert "DRY RUN" in result.output
        assert "videos found:   1" in result.output
        assert not dataset_dir.exists()

    def test_full_run_builds_manifest(
        self, tmp_path: Path, synthetic_video: Path, synthetic_dat: Path
    ):
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{synthetic_video.parent}/allsky-*.mp4",
            dat_path=synthetic_dat,
        )
        result = runner.invoke(
            app,
            ["prepare-local", "--config", str(config), "--steps", "extract-frames,build-manifest"],
        )
        assert result.exit_code == 0, result.output

        manifest_path = dataset_dir / "manifest.parquet"
        assert manifest_path.exists()
        assert (manifest_path.with_name("manifest.parquet.meta.json")).exists()
        manifest = pd.read_parquet(manifest_path)
        assert len(manifest) > 0
        assert "sample_id" in manifest.columns
        assert manifest["sample_id"].iloc[0].startswith("allsky-20260101-")
        # frames were extracted as JPEGs under a per-video directory.
        jpegs = list((dataset_dir / "frames" / "allsky-20260101").glob("*.jpg"))
        assert len(jpegs) == 8

    def test_resume_does_not_reextract(
        self, tmp_path: Path, synthetic_video: Path, synthetic_dat: Path
    ):
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{synthetic_video.parent}/allsky-*.mp4",
            dat_path=synthetic_dat,
        )
        steps = [
            "prepare-local",
            "--config",
            str(config),
            "--steps",
            "extract-frames,build-manifest",
        ]
        first = runner.invoke(app, steps)
        assert first.exit_code == 0, first.output

        jpeg_dir = dataset_dir / "frames" / "allsky-20260101"
        before = {p: p.stat().st_mtime_ns for p in jpeg_dir.glob("*.jpg")}
        manifest_mtime = (dataset_dir / "manifest.parquet").stat().st_mtime_ns

        second = runner.invoke(app, steps)
        assert second.exit_code == 0, second.output
        assert "resume: skipping extraction" in second.output
        assert "resume: manifest up to date" in second.output

        after = {p: p.stat().st_mtime_ns for p in jpeg_dir.glob("*.jpg")}
        assert before == after  # no JPEG re-extracted
        assert (dataset_dir / "manifest.parquet").stat().st_mtime_ns == manifest_mtime

    def test_unknown_step_exits_one(self, tmp_path: Path):
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=tmp_path / "d",
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        result = runner.invoke(app, ["prepare-local", "--config", str(config), "--steps", "bogus"])
        assert result.exit_code == 1
        assert "unknown step" in result.output


class TestSplitsGuard:
    def test_splits_guard_and_force(self, tmp_path: Path):
        dataset_dir = tmp_path / "dataset"
        # Multi-day manifest so a day split is feasible.
        _write_manifest(
            dataset_dir,
            ["2025-03-21 12:00", "2025-03-22 12:00", "2025-03-23 12:00"],
            create_files=True,
        )
        config = _write_config(
            tmp_path / "c1.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
            seed=42,
        )
        first = runner.invoke(app, ["prepare-local", "--config", str(config), "--steps", "splits"])
        assert first.exit_code == 0, first.output
        assert (dataset_dir / "splits.json").exists()

        # A different seed -> different split_id -> guarded.
        config2 = _write_config(
            tmp_path / "c2.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
            seed=99,
        )
        guarded = runner.invoke(
            app, ["prepare-local", "--config", str(config2), "--steps", "splits"]
        )
        assert guarded.exit_code == 1
        assert "different split already exists" in guarded.output

        forced = runner.invoke(
            app, ["prepare-local", "--config", str(config2), "--steps", "splits", "--force"]
        )
        assert forced.exit_code == 0, forced.output


# ---------------------------------------------------------------------------
# export-colab-bundle
# ---------------------------------------------------------------------------


class TestExportColabBundle:
    def test_export_produces_valid_bundle(self, tmp_path: Path):
        dataset_dir = tmp_path / "dataset"
        _write_manifest(dataset_dir, ["2025-03-21 12:00"], create_files=True)
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        out = tmp_path / "bundle.tar.gz"
        result = runner.invoke(
            app,
            ["export-colab-bundle", "--config", str(config), "--out", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()
        report = validate_bundle(out)
        assert report["manifest_sha256_ok"] is True
        assert "allsky_bundle/config/c.yaml" in report["members"]
