"""Resume robustness of ``prepare-local``'s extract step.

``allsky.video.extract_frames`` publishes ``frames/<stem>/manifest.parquet``
*before* the visual-QC pass runs, so a per-video manifest on disk is not proof
that extraction finished. These tests pin the three states the old
existence-only resume mishandled:

- a truncated parquet (killed write) used to wedge every later run in
  ``pd.read_parquet``;
- a QC-less parquet (killed QC pass) used to be accepted, silently dropping
  every FRAME_DARK / FRAME_SATURATED bit;
- a mixed concat of QC'd and QC-less videos used to die in ``astype("int64")``
  with a bare ``IntCastingNaNError``.

Offline: a tiny all-dark synthetic mp4 (which does trip ``visual_qc``) plus the
shared ``synthetic_dat`` TOA5 fixture from :mod:`tests.allsky.conftest`.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner, Result

from allsky.cli import app
from allsky.cli.prepare import _apply_frame_qc, _frames_inputs_sha256, _read_frames_key
from allsky.config import VIDEO_TIME_FIELDS, PrepareConfig
from allsky.data.contracts import QCFlag
from allsky.provenance import config_subset_sha256
from tests.allsky._archive_fake import write_overlay_video
from tests.allsky.test_cli_prepare import _write_config

runner = CliRunner()

_STEPS = ("extract-frames", "build-manifest")


@pytest.fixture(scope="module")
def dark_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An 8-frame 64x64 mp4 dark enough for ``visual_qc`` to flag FRAME_DARK."""
    videos = tmp_path_factory.mktemp("dark-videos")
    path = videos / "allsky-20260101.mp4"
    iio.imwrite(path, np.full((8, 64, 64, 3), 3, dtype=np.uint8), fps=25)
    return path


def _prepare(config: Path) -> Result:
    return runner.invoke(
        app, ["prepare-local", "--config", str(config), "--steps", ",".join(_STEPS)]
    )


def _config_for(tmp_path: Path, video: Path, dat: Path) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    config = _write_config(
        tmp_path / "c.yaml",
        dataset_dir=dataset_dir,
        video_pattern=f"{video.parent}/allsky-*.mp4",
        dat_path=dat,
    )
    return config, dataset_dir


class TestResumeCompleteness:
    def test_truncated_frame_manifest_is_re_extracted(
        self,
        tmp_path: Path,
        dark_video: Path,
        synthetic_dat: Path,
    ):
        config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
        assert _prepare(config).exit_code == 0

        vman = dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet"
        vman.write_bytes(vman.read_bytes()[: len(vman.read_bytes()) // 2])

        result = _prepare(config)
        assert result.exit_code == 0, result.output
        assert "re-extracting" in result.output
        assert "qc_frame_flags" in pd.read_parquet(vman).columns

    def test_qc_less_frame_manifest_is_re_extracted_and_flags_survive(
        self,
        tmp_path: Path,
        dark_video: Path,
        synthetic_dat: Path,
    ):
        config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
        assert _prepare(config).exit_code == 0

        vman = dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet"
        frames = pd.read_parquet(vman).drop(columns=["qc_frame_flags"])
        frames.to_parquet(vman, index=False)
        (dataset_dir / "manifest.parquet").unlink()

        result = _prepare(config)
        assert result.exit_code == 0, result.output
        assert "re-extracting" in result.output

        manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
        assert (manifest["qc_flags"] & int(QCFlag.FRAME_DARK)).all()

    def test_frames_deleted_from_disk_are_re_extracted_not_resumed_onto(
        self,
        tmp_path: Path,
        dark_video: Path,
        synthetic_dat: Path,
    ):
        """The gate read the parquet and the sidecar and never the disk, so a day
        whose JPEGs were cleaned away still printed "frames already present" and the
        manifest was rebuilt with image_path values naming files that were gone."""
        config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
        assert _prepare(config).exit_code == 0
        jpeg_dir = dataset_dir / "frames" / "allsky-20260101"
        removed = sorted(path.name for path in jpeg_dir.glob("*.jpg"))
        for jpeg in jpeg_dir.glob("*.jpg"):
            jpeg.unlink()

        result = _prepare(config)

        assert result.exit_code == 0, result.output
        assert "re-extracting" in result.output
        assert sorted(path.name for path in jpeg_dir.glob("*.jpg")) == removed

    def test_complete_frame_manifest_still_resumes(
        self,
        tmp_path: Path,
        dark_video: Path,
        synthetic_dat: Path,
    ):
        config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
        assert _prepare(config).exit_code == 0
        jpeg_dir = dataset_dir / "frames" / "allsky-20260101"
        before = {p: p.stat().st_mtime_ns for p in jpeg_dir.glob("*.jpg")}

        result = _prepare(config)
        assert result.exit_code == 0, result.output
        assert "resume: skipping extraction" in result.output
        assert before == {p: p.stat().st_mtime_ns for p in jpeg_dir.glob("*.jpg")}


class TestBuildManifestWithoutExtract:
    def test_qc_less_manifest_warns_but_still_builds(
        self,
        tmp_path: Path,
        dark_video: Path,
        synthetic_dat: Path,
    ):
        config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
        assert _prepare(config).exit_code == 0

        vman = dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet"
        pd.read_parquet(vman).drop(columns=["qc_frame_flags"]).to_parquet(vman, index=False)
        (dataset_dir / "manifest.parquet").unlink()

        result = runner.invoke(
            app, ["prepare-local", "--config", str(config), "--steps", "build-manifest"]
        )
        assert result.exit_code == 0, result.output
        assert "no qc_frame_flags" in result.output
        assert len(pd.read_parquet(dataset_dir / "manifest.parquet")) > 0


class TestApplyFrameQC:
    """``_apply_frame_qc`` over the three frame-manifest schemas it can be handed."""

    @staticmethod
    def _manifest() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["allsky-20260101-0600", "allsky-20260102-0600"],
                "qc_flags": [int(QCFlag.LOW_SUN), int(QCFlag.LOW_SUN)],
            }
        )

    @staticmethod
    def _frames(*, qc_bits: list[int | None] | None) -> pd.DataFrame:
        frames = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2026-01-01 06:00"), pd.Timestamp("2026-01-02 06:00")],
                "video": ["allsky-20260101.mp4", "allsky-20260102.mp4"],
            }
        )
        if qc_bits is not None:
            frames["qc_frame_flags"] = qc_bits
        return frames

    def test_present_column_ors_the_frame_bits_in(self, capsys: pytest.CaptureFixture[str]):
        out = _apply_frame_qc(
            self._manifest(), self._frames(qc_bits=[int(QCFlag.FRAME_SATURATED), 0])
        )
        assert out["qc_flags"].tolist() == [
            int(QCFlag.LOW_SUN) | int(QCFlag.FRAME_SATURATED),
            int(QCFlag.LOW_SUN),
        ]
        assert "WARNING" not in capsys.readouterr().out

    def test_absent_column_warns_and_leaves_qc_flags_alone(
        self, capsys: pytest.CaptureFixture[str]
    ):
        out = _apply_frame_qc(self._manifest(), self._frames(qc_bits=None))
        assert out["qc_flags"].tolist() == [int(QCFlag.LOW_SUN)] * 2
        assert "no qc_frame_flags" in capsys.readouterr().out

    def test_mixed_concat_warns_instead_of_raising(self, capsys: pytest.CaptureFixture[str]):
        mixed = pd.concat(
            [
                self._frames(qc_bits=[int(QCFlag.FRAME_SATURATED), 0]).iloc[[0]],
                self._frames(qc_bits=None).iloc[[1]],
            ],
            ignore_index=True,
        )
        out = _apply_frame_qc(self._manifest(), mixed)
        assert out["qc_flags"].tolist() == [
            int(QCFlag.LOW_SUN) | int(QCFlag.FRAME_SATURATED),
            int(QCFlag.LOW_SUN),
        ]
        assert "no visual QC for video(s) ['allsky-20260102.mp4']" in capsys.readouterr().out


def _extract_only(config: Path) -> Result:
    return runner.invoke(
        app, ["prepare-local", "--config", str(config), "--steps", "extract-frames"]
    )


def _config_with_timestamps(
    path: Path, *, videos: Path, dat: Path, dataset_dir: Path, source: str
) -> Path:
    path.write_text(
        "video:\n"
        f"  pattern: '{videos}/allsky-*.mp4'\n"
        f"  timestamps: '{source}'\n"
        "sensor:\n"
        f"  paths: ['{dat}']\n"
        "output:\n"
        f"  dataset_dir: '{dataset_dir}'\n",
        encoding="utf-8",
    )
    return path


def test_a_changed_preprocessing_section_re_extracts_instead_of_resuming(
    tmp_path: Path, dark_video: Path, synthetic_dat: Path
):
    config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
    assert _prepare(config).exit_code == 0
    config.write_text(config.read_text() + "resize: 32\n", encoding="utf-8")

    result = _prepare(config)

    assert result.exit_code == 0, result.output
    shipped = {
        iio.imread(jpeg).shape[:2]
        for jpeg in (dataset_dir / "frames" / "allsky-20260101").glob("*.jpg")
    }
    assert shipped == {(32, 32)}


def test_switching_the_timestamp_source_re_extracts_instead_of_resuming(
    tmp_path: Path, synthetic_dat: Path
):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_overlay_video(
        videos / "allsky-20260101.mp4", [f"2026010112{minute:02d}30" for minute in range(3)]
    )
    dataset_dir = tmp_path / "dataset"
    config = _config_with_timestamps(
        tmp_path / "c.yaml",
        videos=videos,
        dat=synthetic_dat,
        dataset_dir=dataset_dir,
        source="overlay",
    )
    assert _extract_only(config).exit_code == 0

    _config_with_timestamps(
        config, videos=videos, dat=synthetic_dat, dataset_dir=dataset_dir, source="modelled"
    )
    result = _extract_only(config)

    assert result.exit_code == 0, result.output
    frames = pd.read_parquet(dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet")
    assert sorted(str(value) for value in frames["timestamp"]) == [
        "2026-01-01 06:00:00",
        "2026-01-01 06:01:00",
        "2026-01-01 06:02:00",
    ]


def test_re_extraction_replaces_the_previous_clocks_jpegs_on_disk(
    tmp_path: Path, synthetic_dat: Path
):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_overlay_video(
        videos / "allsky-20260101.mp4", [f"2026010112{minute:02d}30" for minute in range(3)]
    )
    dataset_dir = tmp_path / "dataset"
    config = _config_with_timestamps(
        tmp_path / "c.yaml",
        videos=videos,
        dat=synthetic_dat,
        dataset_dir=dataset_dir,
        source="overlay",
    )
    assert _extract_only(config).exit_code == 0

    _config_with_timestamps(
        config, videos=videos, dat=synthetic_dat, dataset_dir=dataset_dir, source="modelled"
    )
    result = _extract_only(config)

    assert result.exit_code == 0, result.output
    video_dir = dataset_dir / "frames" / "allsky-20260101"
    frames = pd.read_parquet(video_dir / "manifest.parquet")
    assert sorted(p.name for p in video_dir.glob("*.jpg")) == sorted(
        Path(str(value)).name for value in frames["frame_path"]
    )


def test_a_failed_re_extraction_leaves_the_previous_frames_in_place(
    tmp_path: Path, synthetic_dat: Path, monkeypatch: pytest.MonkeyPatch
):
    videos = tmp_path / "videos"
    videos.mkdir()
    write_overlay_video(
        videos / "allsky-20260101.mp4", [f"2026010112{minute:02d}30" for minute in range(3)]
    )
    dataset_dir = tmp_path / "dataset"
    config = _config_with_timestamps(
        tmp_path / "c.yaml",
        videos=videos,
        dat=synthetic_dat,
        dataset_dir=dataset_dir,
        source="overlay",
    )
    assert _extract_only(config).exit_code == 0
    video_dir = dataset_dir / "frames" / "allsky-20260101"
    before = sorted(p.name for p in video_dir.glob("*.jpg"))

    def _die(video: str, video_dir: Path, cfg: object) -> pd.DataFrame:
        raise OSError(f"disk full halfway through re-extracting {video} into {video_dir} ({cfg})")

    monkeypatch.setattr("allsky.cli.prepare._extract_and_qc", _die)
    _config_with_timestamps(
        config, videos=videos, dat=synthetic_dat, dataset_dir=dataset_dir, source="modelled"
    )
    result = _extract_only(config)

    assert result.exit_code != 0
    assert sorted(p.name for p in video_dir.glob("*.jpg")) == before


def test_build_manifest_alone_refuses_frames_extracted_under_another_config(
    tmp_path: Path, dark_video: Path, synthetic_dat: Path
):
    config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
    assert _extract_only(config).exit_code == 0
    config.write_text(config.read_text() + "crop:\n  enabled: true\n  top: 8\n", encoding="utf-8")

    result = runner.invoke(
        app, ["prepare-local", "--config", str(config), "--steps", "build-manifest"]
    )

    assert result.exit_code == 1, result.output
    assert "allsky-20260101" in result.output
    assert not (dataset_dir / "manifest.parquet").exists()


def test_build_manifest_alone_still_runs_on_frames_from_the_same_config(
    tmp_path: Path, dark_video: Path, synthetic_dat: Path
):
    config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
    assert _extract_only(config).exit_code == 0

    result = runner.invoke(
        app, ["prepare-local", "--config", str(config), "--steps", "build-manifest"]
    )

    assert result.exit_code == 0, result.output
    assert len(pd.read_parquet(dataset_dir / "manifest.parquet")) > 0


def test_build_manifest_alone_runs_on_frames_that_predate_the_provenance_sidecar(
    tmp_path: Path, dark_video: Path, synthetic_dat: Path
):
    config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
    assert _extract_only(config).exit_code == 0
    for sidecar in dataset_dir.rglob("frames.meta.json"):
        sidecar.unlink()

    result = runner.invoke(
        app, ["prepare-local", "--config", str(config), "--steps", "build-manifest"]
    )

    assert result.exit_code == 0, result.output
    assert "record no video/mask/crop/resize config" in result.output
    assert len(pd.read_parquet(dataset_dir / "manifest.parquet")) > 0


def test_a_non_utf8_frame_provenance_is_read_as_absent(tmp_path: Path):
    video_dir = tmp_path / "allsky-20260101"
    video_dir.mkdir()
    (video_dir / "frames.meta.json").write_bytes(b'{"frames_sha256": "\xff\xfe"}')

    assert _read_frames_key(video_dir) is None


#: One case per field the frame key claims to cover — the four pixel sections and
#: the four video time fields. A field that quietly left the formula makes the
#: resume reuse frames extracted under a different config, which is the one thing
#: the key exists to prevent.
@pytest.mark.parametrize(
    "patch",
    [
        {"video": {"filename_date_format": "allsky_%Y%m%d"}},
        {"video": {"timestamps": "modelled"}},
        {"video": {"start_time": "07:00"}},
        {"video": {"minutes_per_frame": 2.0}},
        {"mask": {"threshold": 200}},
        {"crop": {"enabled": True, "top": 40}},
        {"pad": {"enabled": True, "top": 40}},
        {"resize": 224},
    ],
)
def test_a_config_edit_that_changes_an_extracted_frame_moves_its_provenance_hash(patch: dict):
    edited = PrepareConfig.model_validate(patch)

    assert _frames_inputs_sha256(edited) != _frames_inputs_sha256(PrepareConfig())


def test_an_edit_that_reaches_no_frame_leaves_the_provenance_hash_alone():
    """The gate has to distinguish a re-extraction from an unrelated edit, or
    every config touch would discard hours of extracted frames.
    """
    unrelated = PrepareConfig.model_validate({"embeddings": {"batch_size": 3}})

    assert _frames_inputs_sha256(unrelated) == _frames_inputs_sha256(PrepareConfig())


def _write_mask(path: Path, *, blacked_rows: int, size: int = 64) -> Path:
    keep = np.full((size, size), 255, dtype=np.uint8)
    keep[:blacked_rows] = 0
    iio.imwrite(path, keep)
    return path


class TestMaskBytesAreInTheFrameProvenanceHash:
    def test_rewriting_the_mask_png_in_place_moves_the_hash(self, tmp_path: Path):
        mask = _write_mask(tmp_path / "horizon.png", blacked_rows=0)
        cfg = PrepareConfig.model_validate({"mask": {"path": str(mask)}})
        before = _frames_inputs_sha256(cfg)

        _write_mask(mask, blacked_rows=8)

        assert _frames_inputs_sha256(cfg) != before

    def test_an_embeddings_edit_leaves_the_hash_alone(self, tmp_path: Path):
        mask = _write_mask(tmp_path / "horizon.png", blacked_rows=0)
        base = _frames_inputs_sha256(PrepareConfig.model_validate({"mask": {"path": str(mask)}}))

        tuned = PrepareConfig.model_validate(
            {"mask": {"path": str(mask)}, "embeddings": {"batch_size": 8}}
        )

        assert _frames_inputs_sha256(tuned) == base

    def test_a_maskless_config_hashes_as_before_the_mask_bytes_were_folded_in(self):
        assert _frames_inputs_sha256(PrepareConfig()) == config_subset_sha256(
            PrepareConfig(),
            sections=("mask", "crop", "pad", "resize"),
            nested_fields={"video": VIDEO_TIME_FIELDS},
            subject="the frame provenance hash",
        )


def test_rewriting_the_mask_png_in_place_re_extracts_instead_of_resuming(
    tmp_path: Path, dark_video: Path, synthetic_dat: Path
):
    config, dataset_dir = _config_for(tmp_path, dark_video, synthetic_dat)
    mask = _write_mask(tmp_path / "horizon.png", blacked_rows=0)
    config.write_text(config.read_text() + f"mask:\n  path: '{mask}'\n", encoding="utf-8")
    assert _prepare(config).exit_code == 0

    _write_mask(mask, blacked_rows=32)
    result = _prepare(config)

    assert result.exit_code == 0, result.output
    assert "re-extracting" in result.output
    shipped = [
        iio.imread(jpeg) for jpeg in (dataset_dir / "frames" / "allsky-20260101").glob("*.jpg")
    ]
    assert shipped
    assert all(not frame[:32].any() for frame in shipped)


def test_a_changed_pad_section_moves_the_frame_provenance_hash():
    """``pad`` rewrites the pixels as much as ``crop`` does, yet it was missing from
    ``FRAME_PIXEL_SECTIONS``: the isotropic re-extraction hashed identically to the
    unpadded one, so frames and embeddings resumed across the change."""
    padded = PrepareConfig.model_validate({"pad": {"enabled": True, "top": 254, "bottom": 377}})

    assert _frames_inputs_sha256(padded) != _frames_inputs_sha256(PrepareConfig())
