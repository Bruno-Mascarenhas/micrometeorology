"""``prepare-local``'s build-manifest resume must key on the manifest's inputs.

The old key was ``sha256(cfg.model_dump())`` alone, which is inverted in both
directions: blind to the frame set and the sensor files the manifest is actually
built from, and sensitive to sections that reach no manifest row. So

- a newly extracted video day was silently dropped ("resume: manifest up to
  date"), the dataset quietly stopped growing, and
- lowering ``embeddings.batch_size`` after an OOM forced a full rebuild that
  reset every ``split`` label to null.

Offline: the shared ``synthetic_video`` fixture from :mod:`tests.allsky.conftest`
plus a multi-day TOA5 ``.dat`` built here.
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from allsky.cli import app
from allsky.cli.prepare import _manifest_inputs_sha256
from allsky.config import PrepareConfig
from tests.allsky.conftest import _SAFE_COLUMNS
from tests.allsky.test_cli_prepare import _write_config

runner = CliRunner()


@pytest.fixture(scope="module")
def multi_day_dat(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A TOA5 .dat covering 06:00-06:10 on 2026-01-01..2026-01-05.

    The shared ``synthetic_dat`` fixture covers one day only, so a second video
    day would produce no manifest rows at all and hide the very staleness these
    tests are about.
    """
    path = tmp_path_factory.mktemp("multi-day-sensors") / "synthetic.dat"
    columns = ["TIMESTAMP", *_SAFE_COLUMNS, "CM3Up_Wm2_Avg", "PSP_Wm2_Avg"]
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
        ",".join(f'"{c}"' for c in columns),
        ",".join('"unit"' for _ in columns),
        ",".join('"Avg"' for _ in columns),
    ]
    readings = ",".join(str(values[c]) for c in columns[1:])
    for day in pd.date_range("2026-01-01", periods=5, freq="D"):
        lines.extend(
            f'"{ts:%Y-%m-%d %H:%M:%S}",{readings}'
            for ts in pd.date_range(day + pd.Timedelta(hours=6), periods=11, freq="1min")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _prepare(config: Path, *extra: str) -> Result:
    return runner.invoke(
        app,
        [
            "prepare-local",
            "--config",
            str(config),
            "--steps",
            "extract-frames,build-manifest",
            *extra,
        ],
    )


def _days(dataset_dir: Path) -> list[str]:
    return sorted(pd.read_parquet(dataset_dir / "manifest.parquet")["day_id"].astype(str).unique())


@pytest.fixture
def two_day_videos(tmp_path: Path, synthetic_video: Path) -> Path:
    """A private video directory seeded with day 1 only (day 2 added by the test)."""
    videos = tmp_path / "videos"
    videos.mkdir()
    shutil.copy(synthetic_video, videos / "allsky-20260101.mp4")
    return videos


class TestNewVideoDayIsPickedUp:
    def test_added_day_triggers_a_rebuild(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{two_day_videos}/allsky-*.mp4",
            dat_path=multi_day_dat,
        )
        first = _prepare(config)
        assert first.exit_code == 0, first.output
        assert _days(dataset_dir) == ["2026-01-01"]

        # Day 2's timelapse lands, exactly as the daily cron would see it.
        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / "allsky-20260102.mp4")
        second = _prepare(config)
        assert second.exit_code == 0, second.output
        assert "resume: manifest up to date" not in second.output
        assert "inputs changed" in second.output
        assert _days(dataset_dir) == ["2026-01-01", "2026-01-02"]

    def test_unchanged_inputs_still_resume(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{two_day_videos}/allsky-*.mp4",
            dat_path=multi_day_dat,
        )
        assert _prepare(config).exit_code == 0
        mtime = (dataset_dir / "manifest.parquet").stat().st_mtime_ns

        second = _prepare(config)
        assert second.exit_code == 0, second.output
        assert "resume: manifest up to date" in second.output
        assert (dataset_dir / "manifest.parquet").stat().st_mtime_ns == mtime


class TestIrrelevantConfigEditsDoNotRebuild:
    def test_embeddings_batch_size_change_is_not_a_rebuild(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        dataset_dir = tmp_path / "dataset"
        base = (
            "video:\n"
            f"  pattern: '{two_day_videos}/allsky-*.mp4'\n"
            "sensor:\n"
            f"  paths: ['{multi_day_dat}']\n"
            "output:\n"
            f"  dataset_dir: '{dataset_dir}'\n"
        )
        config = tmp_path / "c.yaml"
        config.write_text(base + "embeddings:\n  batch_size: 32\n", encoding="utf-8")
        assert _prepare(config).exit_code == 0
        mtime = (dataset_dir / "manifest.parquet").stat().st_mtime_ns

        # An OOM-driven embeddings knob: zero influence on any manifest row.
        config.write_text(base + "embeddings:\n  batch_size: 8\n", encoding="utf-8")
        second = _prepare(config)
        assert second.exit_code == 0, second.output
        assert "resume: manifest up to date" in second.output
        assert (dataset_dir / "manifest.parquet").stat().st_mtime_ns == mtime


class TestLegacySidecarFallback:
    def test_sidecar_without_inputs_sha256_uses_the_config_key(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        # Every dataset prepared before this change: the upgrade must not rebuild
        # (and re-split) it behind the operator's back.
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{two_day_videos}/allsky-*.mp4",
            dat_path=multi_day_dat,
        )
        assert _prepare(config).exit_code == 0

        meta_path = dataset_dir / "manifest.parquet.meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["inputs_sha256"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        mtime = (dataset_dir / "manifest.parquet").stat().st_mtime_ns

        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / "allsky-20260102.mp4")
        second = _prepare(config)
        assert second.exit_code == 0, second.output
        assert "resume: manifest up to date" in second.output
        assert (dataset_dir / "manifest.parquet").stat().st_mtime_ns == mtime


class TestSplitLabelsSurviveARebuild:
    def test_rebuild_carries_existing_split_labels_forward(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        # Four days so create_day_splits is feasible, then a fifth arrives: the
        # splits step must abort loudly, but never leave the manifest label-less.
        for day in ("20260102", "20260103", "20260104"):
            shutil.copy(
                two_day_videos / "allsky-20260101.mp4", two_day_videos / f"allsky-{day}.mp4"
            )
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{two_day_videos}/allsky-*.mp4",
            dat_path=multi_day_dat,
        )
        first = runner.invoke(app, ["prepare-local", "--config", str(config)])
        assert first.exit_code == 0, first.output
        labelled_before = pd.read_parquet(dataset_dir / "manifest.parquet")["split"].notna().sum()
        assert labelled_before > 0

        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / "allsky-20260105.mp4")
        second = runner.invoke(app, ["prepare-local", "--config", str(config)])
        assert second.exit_code == 1, second.output
        assert "different split already exists" in second.output
        assert "--force" in second.output

        # The manifest grew AND kept every label the old artifact assigned.
        after = pd.read_parquet(dataset_dir / "manifest.parquet")
        assert max(after["day_id"].astype(str).unique()) == "2026-01-05"
        assert after["split"].notna().sum() >= labelled_before

        forced = runner.invoke(app, ["prepare-local", "--config", str(config), "--force"])
        assert forced.exit_code == 0, forced.output


class TestInputsHashContents:
    @staticmethod
    def _frames(paths: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"frame_path": paths})

    def test_frame_set_is_part_of_the_key(self):
        cfg = PrepareConfig()
        one = _manifest_inputs_sha256(cfg, [self._frames(["a.jpg"])])
        two = _manifest_inputs_sha256(cfg, [self._frames(["a.jpg", "b.jpg"])])
        assert one != two

    def test_frame_order_is_not_part_of_the_key(self):
        cfg = PrepareConfig()
        assert _manifest_inputs_sha256(cfg, [self._frames(["a.jpg", "b.jpg"])]) == (
            _manifest_inputs_sha256(cfg, [self._frames(["b.jpg", "a.jpg"])])
        )

    def test_sensor_content_is_part_of_the_key_but_mtime_is_not(self, tmp_path: Path):
        dat = tmp_path / "s.dat"
        dat.write_text("one", encoding="utf-8")
        cfg = PrepareConfig.model_validate({"sensor": {"paths": [str(dat)]}})
        frames = [self._frames(["a.jpg"])]
        before = _manifest_inputs_sha256(cfg, frames)

        # A copy that bumps mtime without changing content must not force a rebuild.
        dat.touch()
        assert _manifest_inputs_sha256(cfg, frames) == before

        # A same-length in-place edit must.
        dat.write_text("two", encoding="utf-8")
        assert _manifest_inputs_sha256(cfg, frames) != before

    def test_embeddings_and_splits_sections_are_excluded(self):
        frames = [self._frames(["a.jpg"])]
        base = _manifest_inputs_sha256(PrepareConfig(), frames)
        for section, patch in (
            ("embeddings", {"embeddings": {"batch_size": 8}}),
            ("splits", {"splits": {"seed": 7}}),
        ):
            other = PrepareConfig.model_validate(patch)
            assert _manifest_inputs_sha256(other, frames) == base, section

    def test_manifest_relevant_sections_are_included(self):
        frames = [self._frames(["a.jpg"])]
        base = _manifest_inputs_sha256(PrepareConfig(), frames)
        for patch in (
            {"night_filter": {"min_solar_elevation_deg": 10.0}},
            {"targets": {"kindex_kind": "kt"}},
            {"site": {"latitude": 0.0}},
            {"features": {"set": "extended"}},
            {"alignment": {"window_minutes": 20.0}},
            {"resize": 224},
        ):
            assert _manifest_inputs_sha256(PrepareConfig.model_validate(patch), frames) != base, (
                patch
            )
