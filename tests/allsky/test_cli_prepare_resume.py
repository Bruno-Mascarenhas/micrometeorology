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
from allsky.cli.prepare import _apply_frame_qc
from allsky.data.contracts import QCFlag
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

        # Exactly what extract_frames leaves behind when the QC pass is killed.
        vman = dataset_dir / "frames" / "allsky-20260101" / "manifest.parquet"
        frames = pd.read_parquet(vman).drop(columns=["qc_frame_flags"])
        frames.to_parquet(vman, index=False)
        (dataset_dir / "manifest.parquet").unlink()

        result = _prepare(config)
        assert result.exit_code == 0, result.output
        assert "re-extracting" in result.output

        manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
        assert (manifest["qc_flags"] & int(QCFlag.FRAME_DARK)).all()

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
        # The documented `allsky extract-frames` -> `prepare-local --steps
        # build-manifest` flow: no qc_frame_flags anywhere, must stay exit 0.
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
        # pd.concat of a QC'd manifest with a QC-less one yields float64 + NaN,
        # which used to die in .astype("int64") with a bare IntCastingNaNError.
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
