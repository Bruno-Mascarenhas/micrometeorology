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
from allsky.cli.prepare import (
    _EXCLUDED_FROM_MANIFEST_HASH,
    _MANIFEST_CONFIG_SECTIONS,
    _manifest_inputs_sha256,
)
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
            "  timestamps: 'modelled'\n"
            "sensor:\n"
            f"  paths: ['{multi_day_dat}']\n"
            "output:\n"
            f"  dataset_dir: '{dataset_dir}'\n"
        )
        config = tmp_path / "c.yaml"
        config.write_text(base + "embeddings:\n  batch_size: 32\n", encoding="utf-8")
        assert _prepare(config).exit_code == 0
        mtime = (dataset_dir / "manifest.parquet").stat().st_mtime_ns

        config.write_text(base + "embeddings:\n  batch_size: 8\n", encoding="utf-8")
        second = _prepare(config)
        assert second.exit_code == 0, second.output
        assert "resume: manifest up to date" in second.output
        assert (dataset_dir / "manifest.parquet").stat().st_mtime_ns == mtime


class TestBuildManifestDropsUnreadableDays:
    def test_an_unreadable_per_video_manifest_drops_only_its_own_day(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        """`--steps build-manifest` can re-extract nothing, so a truncated per-video
        parquet drops that day from the manifest about to be published: the warning
        is the only trace, and a build that quietly loses a whole day is what the
        inputs hash exists to end."""
        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / "allsky-20260102.mp4")
        dataset_dir = tmp_path / "dataset"
        config = _write_config(
            tmp_path / "c.yaml",
            dataset_dir=dataset_dir,
            video_pattern=f"{two_day_videos}/allsky-*.mp4",
            dat_path=multi_day_dat,
        )
        assert _prepare(config).exit_code == 0
        assert _days(dataset_dir) == ["2026-01-01", "2026-01-02"]

        vman = dataset_dir / "frames" / "allsky-20260102" / "manifest.parquet"
        vman.write_bytes(vman.read_bytes()[: len(vman.read_bytes()) // 2])

        result = runner.invoke(
            app, ["prepare-local", "--config", str(config), "--steps", "build-manifest"]
        )

        assert result.exit_code == 0, result.output
        assert "allsky-20260102" in result.output
        assert "is unreadable" in result.output
        assert _days(dataset_dir) == ["2026-01-01"]


class TestLegacySidecarFallback:
    def test_sidecar_without_inputs_sha256_uses_the_config_key(
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
    def test_a_split_artifact_that_cannot_be_read_rebuilds_without_labels(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
        """A hand-edited or truncated splits.json must degrade to an unlabelled
        rebuild plus a warning: a build that dies here leaves no manifest at all."""
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
        assert runner.invoke(app, ["prepare-local", "--config", str(config)]).exit_code == 0
        (dataset_dir / "splits.json").write_text("{ not json", encoding="utf-8")

        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / "allsky-20260105.mp4")
        result = _prepare(config)

        assert result.exit_code == 0, result.output
        assert "cannot reuse split labels" in result.output
        assert pd.read_parquet(dataset_dir / "manifest.parquet")["split"].isna().all()

    def test_rebuild_carries_existing_split_labels_forward(
        self,
        tmp_path: Path,
        two_day_videos: Path,
        multi_day_dat: Path,
    ):
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

        after = pd.read_parquet(dataset_dir / "manifest.parquet")
        assert max(after["day_id"].astype(str).unique()) == "2026-01-05"
        assert after["split"].notna().sum() >= labelled_before

        forced = runner.invoke(app, ["prepare-local", "--config", str(config), "--force"])
        assert forced.exit_code == 0, forced.output


def test_the_splits_step_alone_ignores_the_frame_provenance(
    tmp_path: Path,
    two_day_videos: Path,
    multi_day_dat: Path,
):
    for day in ("20260102", "20260103", "20260104"):
        shutil.copy(two_day_videos / "allsky-20260101.mp4", two_day_videos / f"allsky-{day}.mp4")
    dataset_dir = tmp_path / "dataset"
    config = _write_config(
        tmp_path / "c.yaml",
        dataset_dir=dataset_dir,
        video_pattern=f"{two_day_videos}/allsky-*.mp4",
        dat_path=multi_day_dat,
    )
    assert _prepare(config).exit_code == 0
    config.write_text(config.read_text() + "resize: 32\n", encoding="utf-8")

    result = runner.invoke(app, ["prepare-local", "--config", str(config), "--steps", "splits"])

    assert result.exit_code == 0, result.output
    assert (dataset_dir / "splits.json").exists()


class TestInputsHashContents:
    @staticmethod
    def _frames(paths: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"frame_path": paths})

    def test_every_config_section_is_hashed_unless_excluded_by_name(self):
        """``pad`` was in neither list, so a padding change never invalidated the
        manifest; a section is now hashed by construction unless named as excluded."""
        assert set(_MANIFEST_CONFIG_SECTIONS) | set(_EXCLUDED_FROM_MANIFEST_HASH) == set(
            PrepareConfig.model_fields
        )
        assert "pad" in _MANIFEST_CONFIG_SECTIONS

    def test_the_pad_section_is_part_of_the_key(self):
        padded = PrepareConfig.model_validate({"pad": {"enabled": True, "top": 254}})
        frames = [self._frames(["a.jpg"])]

        assert _manifest_inputs_sha256(padded, frames) != _manifest_inputs_sha256(
            PrepareConfig(), frames
        )

    def test_the_frame_qc_flags_are_part_of_the_key(self):
        """Re-extracting to populate ``qc_frame_flags`` changes no frame path, so the
        rebuild the missing-flags warning asks for would resume on the flagless
        manifest and FRAME_DARK would never reach it."""
        cfg = PrepareConfig()
        bare = _manifest_inputs_sha256(cfg, [self._frames(["a.jpg"])])
        flagged = _manifest_inputs_sha256(
            cfg, [pd.DataFrame({"frame_path": ["a.jpg"], "qc_frame_flags": [16]})]
        )

        assert bare != flagged

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

    def test_a_touch_that_changes_only_mtime_leaves_the_key_where_it_was(self, tmp_path: Path):
        dat = tmp_path / "s.dat"
        dat.write_text("one", encoding="utf-8")
        cfg = PrepareConfig.model_validate({"sensor": {"paths": [str(dat)]}})
        frames = [self._frames(["a.jpg"])]
        before = _manifest_inputs_sha256(cfg, frames)

        dat.touch()

        assert _manifest_inputs_sha256(cfg, frames) == before

    def test_a_same_length_in_place_sensor_edit_moves_the_key(self, tmp_path: Path):
        dat = tmp_path / "s.dat"
        dat.write_text("one", encoding="utf-8")
        cfg = PrepareConfig.model_validate({"sensor": {"paths": [str(dat)]}})
        frames = [self._frames(["a.jpg"])]
        before = _manifest_inputs_sha256(cfg, frames)

        dat.write_text("two", encoding="utf-8")

        assert _manifest_inputs_sha256(cfg, frames) != before

    @pytest.mark.parametrize("patch", [{"embeddings": {"batch_size": 8}}, {"splits": {"seed": 7}}])
    def test_embeddings_and_splits_sections_are_excluded(self, patch: dict[str, object]):
        frames = [self._frames(["a.jpg"])]
        base = _manifest_inputs_sha256(PrepareConfig(), frames)

        other = PrepareConfig.model_validate(patch)

        assert _manifest_inputs_sha256(other, frames) == base

    @pytest.mark.parametrize(
        "patch",
        [
            {"night_filter": {"min_solar_elevation_deg": 10.0}},
            {"targets": {"kindex_kind": "kt"}},
            {"site": {"latitude": 0.0}},
            {"features": {"set": "extended"}},
            {"alignment": {"window_minutes": 20.0}},
            {"resize": 224},
        ],
    )
    def test_manifest_relevant_sections_are_included(self, patch: dict[str, object]):
        frames = [self._frames(["a.jpg"])]
        base = _manifest_inputs_sha256(PrepareConfig(), frames)

        assert _manifest_inputs_sha256(PrepareConfig.model_validate(patch), frames) != base
