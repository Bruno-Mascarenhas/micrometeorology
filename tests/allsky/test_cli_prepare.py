"""Tests for the Wave C2b prepare CLI (validate-dataset, prepare-local, export).

Offline end-to-end: a tiny synthetic mp4 + a synthetic TOA5 ``.dat`` drive
``prepare-local`` into a real manifest; ``validate-dataset`` is exercised on a
good and a broken manifest; ``export-colab-bundle`` produces a bundle that
:func:`allsky.bundle.validate_bundle` accepts. CliRunner only, tmp_path only.
"""

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from allsky import solar
from allsky.bundle import validate_bundle
from allsky.cli import app
from allsky.config import SiteConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet
from tests.allsky._archive_fake import write_overlay_video

runner = CliRunner()


def _write_config(
    path: Path,
    *,
    dataset_dir: Path,
    video_pattern: str,
    dat_path: Path,
    seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
) -> Path:
    """Write a PrepareConfig YAML for the CLI.

    The synthetic fixture video is 64 px wide and carries no burned-in stamp, so
    these runs ask for the modelled frame->time mapping; the overlay default is
    exercised by ``test_prepare_local_timestamps_frames_from_the_overlay``.
    """
    path.write_text(
        "video:\n"
        f"  pattern: '{video_pattern}'\n"
        "  timestamps: 'modelled'\n"
        "sensor:\n"
        f"  paths: ['{dat_path}']\n"
        "output:\n"
        f"  dataset_dir: '{dataset_dir}'\n"
        "splits:\n"
        f"  seed: {seed}\n"
        f"  val_fraction: {val_fraction}\n"
        f"  test_fraction: {test_fraction}\n"
        "  gap_days: 0\n",
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

    def test_skip_image_check_accepts_a_frameless_dataset(self, tmp_path: Path):
        # The shape of an embedding-mode Colab bundle: manifest + meta, no JPEGs.
        dataset_dir = tmp_path / "frameless"
        _write_manifest(dataset_dir, ["2025-03-21 12:00"], create_files=False)
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )
        result = runner.invoke(
            app, ["validate-dataset", "--config", str(config), "--skip-image-check"]
        )
        assert result.exit_code == 0, result.output
        assert "OK" in result.output

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

    def test_prepare_local_timestamps_frames_from_the_overlay(
        self, tmp_path: Path, synthetic_dat: Path
    ):
        videos = tmp_path / "videos"
        videos.mkdir()
        stamps = [f"2026010106{minute:02d}30" for minute in range(4)]
        write_overlay_video(videos / "allsky-20260101.mp4", stamps)
        dataset_dir = tmp_path / "dataset"
        config = tmp_path / "overlay.yaml"
        config.write_text(
            "video:\n"
            f"  pattern: '{videos}/allsky-*.mp4'\n"
            "sensor:\n"
            f"  paths: ['{synthetic_dat}']\n"
            "output:\n"
            f"  dataset_dir: '{dataset_dir}'\n",
            encoding="utf-8",
        )
        result = runner.invoke(
            app,
            ["prepare-local", "--config", str(config), "--steps", "extract-frames,build-manifest"],
        )
        assert result.exit_code == 0, result.output

        frames = pd.read_parquet(dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet")
        read_back = sorted(str(value) for value in frames["timestamp"])
        assert read_back == [
            "2026-01-01 06:00:30",
            "2026-01-01 06:01:30",
            "2026-01-01 06:02:30",
            "2026-01-01 06:03:30",
        ]

    def test_a_video_whose_clock_steps_backwards_is_skipped_not_fatal(
        self, tmp_path: Path, synthetic_dat: Path
    ):
        videos = tmp_path / "videos"
        videos.mkdir()
        write_overlay_video(
            videos / "allsky-20260101.mp4",
            [f"2026010106{minute:02d}30" for minute in range(4)],
        )
        # 06:02:30 lands before 06:03:30, exactly as the 2026-06-04 archive video
        # steps its clock back 7 s partway through the night.
        write_overlay_video(
            videos / "allsky-20260102.mp4",
            ["20260102060030", "20260102060130", "20260102060330", "20260102060230"],
        )
        dataset_dir = tmp_path / "dataset"
        config = tmp_path / "overlay.yaml"
        config.write_text(
            "video:\n"
            f"  pattern: '{videos}/allsky-*.mp4'\n"
            "sensor:\n"
            f"  paths: ['{synthetic_dat}']\n"
            "output:\n"
            f"  dataset_dir: '{dataset_dir}'\n",
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            ["prepare-local", "--config", str(config), "--steps", "extract-frames"],
        )

        assert result.exit_code == 0, result.output
        assert "skipping allsky-20260102" in result.output
        assert "go backwards" in result.output
        assert "1 video(s) could not be timestamped" in result.output
        assert (dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet").exists()
        assert not (dataset_dir / "frames" / "allsky-20260102" / "manifest.parquet").exists()

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
            [f"2025-03-{day} 12:00" for day in range(21, 27)],
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

        # Different fractions -> different assignment -> guarded. The seed cannot
        # serve here: a chronological split ignores it by construction.
        config2 = _write_config(
            tmp_path / "c2.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
            val_fraction=0.4,
            test_fraction=0.2,
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
        assert not any(name.endswith(".jpg") for name in report["members"])

    def test_include_frames_packs_the_manifest_images(self, tmp_path: Path):
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
            [
                "export-colab-bundle",
                "--config",
                str(config),
                "--out",
                str(out),
                "--include-frames",
            ],
        )
        assert result.exit_code == 0, result.output
        members = validate_bundle(out)["members"]
        assert "allsky_bundle/frames/allsky-20250321-1200.jpg" in members
