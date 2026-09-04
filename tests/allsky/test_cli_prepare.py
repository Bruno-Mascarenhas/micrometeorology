"""Tests for the Wave C2b prepare CLI (validate-dataset, prepare-local, export).

Offline end-to-end: a tiny synthetic mp4 + a synthetic TOA5 ``.dat`` drive
``prepare-local`` into a real manifest; ``validate-dataset`` is exercised on a
good and a broken manifest; ``export-colab-bundle`` produces a bundle that
:func:`allsky.bundle.validate_bundle` accepts. CliRunner only, tmp_path only.
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from allsky.bundle import validate_bundle
from allsky.cli import app
from allsky.cli.prepare import _check_sensor_coverage, _load_sensor_df
from allsky.config import PrepareConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet
from labmim_core import solar
from labmim_core.site import SiteConfig
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


class TestTheMergedSensorExport:
    """``sensor.paths`` is a list because the logger's tables change over time.

    Every other test in the suite passes a one-element list of one constant
    file, so the per-column merge and the sentinel mask both ran on input that
    could not tell a working implementation from a broken one.
    """

    @staticmethod
    def _config(paths: tuple[Path, Path]) -> PrepareConfig:
        return PrepareConfig.model_validate({"sensor": {"paths": [str(path) for path in paths]}})

    def test_the_later_file_supplies_the_column_the_earlier_one_lacks(
        self, two_dat_files: tuple[Path, Path]
    ):
        merged = _load_sensor_df(self._config(two_dat_files))

        shared = pd.Timestamp("2026-01-01 06:02:00")
        assert "PSP_Wm2_Avg" in merged.columns
        assert merged.loc[shared, "PSP_Wm2_Avg"] == pytest.approx(30.0)

    def test_the_earlier_file_wins_a_column_both_files_carry(
        self, two_dat_files: tuple[Path, Path]
    ):
        """Chronological order decides a conflict, so a re-exported later file
        cannot rewrite a value the archive already published.
        """
        merged = _load_sensor_df(self._config(two_dat_files))

        shared = pd.Timestamp("2026-01-01 06:02:00")
        assert merged.loc[shared, "RH1"] == pytest.approx(70.2)
        assert merged.loc[shared, "CM3Up_Wm2_Avg"] == pytest.approx(122.0)

    def test_a_missing_sample_and_a_railed_channel_both_arrive_as_gaps(
        self, two_dat_files: tuple[Path, Path]
    ):
        """The bare ``NAN`` token and the railed sentinel reach the manifest
        builder the same way — as absences — or the normaliser fits its mean and
        std over an instrument fault.
        """
        merged = _load_sensor_df(self._config(two_dat_files))

        assert pd.isna(merged.loc[pd.Timestamp("2026-01-01 06:01:00"), "RH1"])
        # 1000 degC is finite, so the reader's own -900 floor passes it through
        # and only mask_sentinels stands between it and the normaliser.
        assert pd.isna(merged.loc[pd.Timestamp("2026-01-01 06:02:00"), "AirT1_C_Avg"])

    def test_a_rail_below_the_readers_floor_is_a_gap_the_later_file_may_fill(
        self, two_dat_files: tuple[Path, Path]
    ):
        """-7999 never reaches the merge as a value: the reader has already made
        it a gap, so the per-column rule takes the later file's reading there
        rather than keeping the earlier file's rail.
        """
        merged = _load_sensor_df(self._config(two_dat_files))

        assert merged.loc[pd.Timestamp("2026-01-01 06:03:00"), "AirT1_C_Avg"] == pytest.approx(25.3)

    def test_the_merged_index_is_the_union_in_chronological_order(
        self, two_dat_files: tuple[Path, Path]
    ):
        merged = _load_sensor_df(self._config(two_dat_files))

        assert list(merged.index) == list(
            pd.date_range("2026-01-01 06:00", "2026-01-01 06:03", freq="1min")
        )


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

    def test_a_split_artifact_beside_the_manifest_is_read_and_named(self, tmp_path: Path):
        """The leakage cross-check only runs when an artifact is found, and no test
        ever put one there — the whole wiring was unexercised."""
        from allsky.data.splits import create_day_splits, save_split_artifact

        days = ["2025-03-21 12:00", "2025-03-22 12:00", "2025-03-23 12:00"]
        dataset_dir = tmp_path / "with-split"
        _write_manifest(dataset_dir, days, create_files=True)
        save_split_artifact(
            create_day_splits(
                [day[:10] for day in days],
                seed=1,
                val_fraction=0.34,
                test_fraction=0.0,
                gap_days=0,
            ),
            dataset_dir / "splits.json",
        )
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )

        result = runner.invoke(app, ["validate-dataset", "--config", str(config)])

        assert result.exit_code == 0, result.output
        assert "Split artifact:" in result.output

    @pytest.mark.parametrize(("strict", "expected_exit"), [(False, 0), (True, 1)])
    def test_a_manifest_whose_sidecar_is_gone_warns_and_fails_only_under_strict(
        self, tmp_path: Path, strict: bool, expected_exit: int
    ):
        """The warning half of validate-dataset — the echo, the sidecar warning and
        `--strict`'s promotion — was never executed by any test."""
        dataset_dir = tmp_path / "no-sidecar"
        manifest_path = _write_manifest(dataset_dir, ["2025-03-21 12:00"], create_files=True)
        manifest_path.with_name("manifest.parquet.meta.json").unlink()
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern="none-*.mp4",
            dat_path=tmp_path / "x.dat",
        )

        result = runner.invoke(
            app,
            ["validate-dataset", "--config", str(config), *(["--strict"] if strict else [])],
        )

        assert result.exit_code == expected_exit, result.output
        assert "WARNING:" in result.output
        assert "dataset_version" in result.output

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

    def test_a_video_that_decodes_to_no_frame_is_skipped_not_resumed_forever(
        self, tmp_path: Path, synthetic_video: Path, synthetic_dat: Path, monkeypatch
    ):
        """An empty extraction still carried the ``qc_frame_flags`` column, so the
        resume gate read the 0-row manifest as complete and skipped the day on every
        later run — the day left the dataset with no warning after the first line."""
        import imageio.v3 as iio

        def decode_nothing(*_args, **_kwargs):
            return iter(())

        monkeypatch.setattr(iio, "imiter", decode_nothing)
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{synthetic_video.parent}/allsky-*.mp4",
            dat_path=synthetic_dat,
        )

        result = runner.invoke(
            app, ["prepare-local", "--config", str(config), "--steps", "extract-frames"]
        )

        assert result.exit_code == 0, result.output
        assert "decoded to no frame" in result.output
        assert "1 video(s) could not be timestamped" in result.output
        assert not (dataset_dir / "frames" / synthetic_video.stem / "manifest.parquet").exists()

    def test_a_day_whose_video_was_pruned_stays_in_the_dataset(
        self, tmp_path: Path, synthetic_video: Path, synthetic_dat: Path
    ):
        """`sync-archive --prune-uploaded` deletes the mp4 once Drive holds it, and
        the production configs point `video.pattern` at that same directory. The
        video list came from the glob alone, so the day silently left the manifest
        even though its extracted frames were still on disk."""
        import shutil

        # The fixture video is module-scoped; this test prunes its own copy.
        videos = tmp_path / "videos"
        videos.mkdir()
        video = videos / synthetic_video.name
        shutil.copy(synthetic_video, video)
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{videos}/allsky-*.mp4",
            dat_path=synthetic_dat,
        )
        steps = [
            "prepare-local",
            "--config",
            str(config),
            "--steps",
            "extract-frames,build-manifest",
        ]
        assert runner.invoke(app, steps).exit_code == 0
        before = len(pd.read_parquet(dataset_dir / "manifest.parquet"))

        video.unlink()
        result = runner.invoke(app, steps)

        assert result.exit_code == 0, result.output
        assert "already-extracted day(s)" in result.output
        assert len(pd.read_parquet(dataset_dir / "manifest.parquet")) == before

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


def test_importing_the_prepare_command_module_does_not_pull_pandas():
    """The module docstring promises a torch-free, pandas-free ``allsky --help``."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, allsky.cli.prepare;"
                "print(sorted(m for m in sys.modules if m.split('.')[0] == 'pandas'))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == "[]"


class TestSensorCoverageGate:
    """The gate runs before extraction because extraction costs minutes per video
    and the pairing step then discards every frame; a coverage gap has to surface
    here, not as an empty manifest an hour later.
    """

    @staticmethod
    def _dat(path: Path, index: pd.DatetimeIndex) -> Path:
        frame = _sensor_frame(SiteConfig(), index)
        columns = ["TIMESTAMP", *frame.columns]
        lines = [
            '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","TEST","1","LBM"',
            ",".join(f'"{name}"' for name in columns),
            ",".join('"unit"' for _ in columns),
            ",".join('"Avg"' for _ in columns),
        ]
        for stamp, row in frame.iterrows():
            values = ",".join(f"{value:.4f}" for value in row)
            lines.append(f'"{stamp:%Y-%m-%d %H:%M:%S}",{values}')
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _cfg(self, tmp_path: Path, index: pd.DatetimeIndex) -> PrepareConfig:
        dat = self._dat(tmp_path / "sensor.dat", index)
        return PrepareConfig.model_validate({"sensor": {"paths": [str(dat)]}})

    def test_a_video_day_entirely_before_the_logger_is_not_covered(self, tmp_path: Path):
        """`day + 1 day >= sensor_start` counted the day before the logger's first
        record as covered, so a whole video day outside the record passed the gate
        and contributed nothing to the manifest."""
        cfg = self._cfg(tmp_path, pd.date_range("2026-01-10 00:00", periods=48, freq="h"))

        with pytest.raises(typer.Exit) as raised:
            _check_sensor_coverage(cfg, [str(tmp_path / "allsky-20260109.mp4")])

        assert raised.value.exit_code == 1

    def test_a_day_inside_the_record_is_covered(self, tmp_path: Path):
        cfg = self._cfg(tmp_path, pd.date_range("2026-01-10 00:00", periods=48, freq="h"))

        _check_sensor_coverage(cfg, [str(tmp_path / "allsky-20260110.mp4")])

    def test_a_day_after_the_record_is_reported_as_partial_coverage(self, tmp_path: Path, capsys):
        cfg = self._cfg(tmp_path, pd.date_range("2026-01-10 00:00", periods=48, freq="h"))

        _check_sensor_coverage(
            cfg,
            [str(tmp_path / "allsky-20260110.mp4"), str(tmp_path / "allsky-20260201.mp4")],
        )

        assert "only 1 of 2 video days" in capsys.readouterr().out

    def test_a_sensor_export_that_is_not_there_is_named_not_a_traceback(self, tmp_path: Path):
        cfg = PrepareConfig.model_validate({"sensor": {"paths": [str(tmp_path / "gone.dat")]}})

        with pytest.raises(typer.Exit) as raised:
            _check_sensor_coverage(cfg, [str(tmp_path / "allsky-20260110.mp4")])

        assert raised.value.exit_code == 1


def test_two_sensor_files_merge_per_column_not_by_dropping_the_whole_row(tmp_path: Path):
    """`concat().sort_index()` then `duplicated(keep="first")` sorted BEFORE
    deduplicating, so "first" was whichever row the sort left first rather than
    the first file's, and it dropped the whole row — losing a column the later
    file adds at a stamp the earlier one also carries."""
    header = '"TOA5","CR5000","CR5000","2754","CR5000.Std.06","TEST","1","LBM"'

    def write(path: Path, columns: list[str], values: list[float]) -> Path:
        names = ["TIMESTAMP", *columns]
        path.write_text(
            "\n".join(
                [
                    header,
                    ",".join(f'"{n}"' for n in names),
                    ",".join('"unit"' for _ in names),
                    ",".join('"Avg"' for _ in names),
                    '"2026-01-10 00:00:00",' + ",".join(f"{v}" for v in values),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    earlier = write(tmp_path / "a.dat", ["AirT1_C_Avg"], [21.0])
    later = write(tmp_path / "b.dat", ["AirT1_C_Avg", "RH1"], [99.0, 70.0])
    cfg = PrepareConfig.model_validate({"sensor": {"paths": [str(earlier), str(later)]}})

    merged = _load_sensor_df(cfg)

    assert len(merged) == 1
    assert float(merged["AirT1_C_Avg"].iloc[0]) == pytest.approx(21.0), "earlier file wins"
    assert float(merged["RH1"].iloc[0]) == pytest.approx(70.0), "later file's column survives"


def test_a_failed_swap_puts_the_previous_extraction_back(
    tmp_path: Path, synthetic_video: Path, synthetic_dat: Path, monkeypatch
):
    """Between the two renames the day has no directory at all: failing there
    without restoring would strand the previous extraction under `.superseded`,
    which nothing reads, and the day would leave the dataset in silence."""
    import shutil

    from allsky.cli.prepare import _extract_replacing_frames

    videos = tmp_path / "videos"
    videos.mkdir()
    video = videos / synthetic_video.name
    shutil.copy(synthetic_video, video)
    cfg = PrepareConfig.model_validate(
        {
            "video": {"pattern": f"{videos}/allsky-*.mp4", "timestamps": "modelled"},
            "sensor": {"paths": [str(synthetic_dat)]},
            "output": {"dataset_dir": str(tmp_path / "dataset")},
        }
    )
    video_dir = tmp_path / "dataset" / "frames" / video.stem
    _extract_replacing_frames(str(video), video_dir, cfg)
    before = sorted(p.name for p in video_dir.iterdir())

    real_rename = Path.rename
    calls = {"n": 0}

    def fail_on_the_swap(self: Path, target):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("no space left on device")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", fail_on_the_swap)

    with pytest.raises(OSError, match="no space left"):
        _extract_replacing_frames(str(video), video_dir, cfg)

    monkeypatch.undo()
    assert video_dir.is_dir(), "the day must still have a directory"
    assert sorted(p.name for p in video_dir.iterdir()) == before
