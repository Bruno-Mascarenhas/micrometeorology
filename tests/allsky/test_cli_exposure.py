"""Tests for the ``exposure-features`` CLI.

The decoding boundary is replaced by an in-memory frame source: no video is
decoded here. The source dataset is a two-video manifest with real JPEG
placeholders under ``frames/`` so the ``frames`` link can be checked to resolve.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import allsky.exposure
from allsky.cli import app
from allsky.cli.exposure import (
    FRAMES_DIRNAME,
    SHARD_DIRNAME,
    SOURCE_BUILD_KEYS,
    ExposureFeaturesError,
    run_exposure_features,
)
from allsky.config import DATASET_MANIFEST_FILENAME, DATASET_SPLIT_FILENAME, manifest_meta_path
from allsky.data.loading import load_manifest
from allsky.data.manifest import write_manifest_parquet
from allsky.exposure import EXPOSURE_FEATURE_COLUMNS, SHUFFLED_FEATURE_COLUMNS, ExposureRecord

runner = CliRunner()

_VIDEOS = ("allsky-20260101.mp4", "allsky-20260102.mp4")
_FRAMES_PER_VIDEO = 4
_UNREADABLE = {("allsky-20260102.mp4", 2)}
_SOURCE_META = {
    "dataset_version": "2",
    "feature_set": "bare",
    "feature_columns": ["solar_elevation", "doy_sin"],
    "config_sha256": "c" * 64,
    "inputs_sha256": "i" * 64,
}


def _write_source_dataset(root: Path) -> Path:
    rows = []
    for video in _VIDEOS:
        day = video.removesuffix(".mp4")
        (root / FRAMES_DIRNAME / day).mkdir(parents=True)
        for index in range(_FRAMES_PER_VIDEO):
            sample_id = f"{day}-{12 + index:02d}00"
            image = f"{FRAMES_DIRNAME}/{day}/{sample_id}.jpg"
            iio.imwrite(root / image, np.zeros((8, 8, 3), dtype=np.uint8))
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": image,
                    "frame_index": index,
                    "video": video,
                    "split": "train" if video == _VIDEOS[0] else "val",
                    "target_dhi": 100.0 + index,
                }
            )
    manifest = pd.DataFrame(rows)
    for column in ("sample_id", "image_path", "video", "split"):
        manifest[column] = manifest[column].astype("string")
    write_manifest_parquet(manifest, _SOURCE_META, root / DATASET_MANIFEST_FILENAME)
    (root / DATASET_SPLIT_FILENAME).write_text(json.dumps({"split_id": "abc"}), encoding="utf-8")
    return root


def _write_videos(videos_dir: Path, names: Iterable[str] = _VIDEOS) -> Path:
    videos_dir.mkdir()
    for name in names:
        (videos_dir / name).write_bytes(b"")
    return videos_dir


def _fake_records(path: Path, frame_indices: Iterable[int]) -> list[ExposureRecord]:
    records = []
    for index in sorted(frame_indices):
        if (path.name, index) in _UNREADABLE:
            records.append(ExposureRecord(index, None, None, None, 0.5))
        else:
            exposure_s = 1e-4 * (index + 1)
            records.append(
                ExposureRecord(index, exposure_s, float(np.log2(exposure_s)), 8.0 + index, 0.0)
            )
    return records


def _refusing_records(path: Path, _frame_indices: Iterable[int]) -> list[ExposureRecord]:
    raise AssertionError(f"decoding must not run for {path.name} on resume")


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    return {
        "data_root": _write_source_dataset(tmp_path / "dataset"),
        "videos": _write_videos(tmp_path / "videos"),
        "out": tmp_path / "dataset-exp",
    }


def _run(workspace: dict[str, Path], *, resume: bool = False, seed: int = 0) -> dict:
    return run_exposure_features(
        workspace["data_root"],
        workspace["videos"],
        workspace["out"],
        seed=seed,
        resume=resume,
        frame_records=_fake_records,
    )


def test_the_written_manifest_carries_the_feature_columns_without_the_unreadable_row(
    workspace: dict[str, Path],
):
    _run(workspace)

    manifest, meta = load_manifest(workspace["out"] / DATASET_MANIFEST_FILENAME)

    assert len(manifest) == len(_VIDEOS) * _FRAMES_PER_VIDEO - len(_UNREADABLE)
    assert set(EXPOSURE_FEATURE_COLUMNS) <= set(manifest.columns)
    assert set(SHUFFLED_FEATURE_COLUMNS) <= set(manifest.columns)
    assert "allsky-20260102-1400" not in set(manifest["sample_id"])
    assert meta["manifest_sha256"] is not None


def test_the_meta_records_where_the_rows_came_from_and_what_was_removed(
    workspace: dict[str, Path],
):
    source_meta = json.loads(
        manifest_meta_path(workspace["data_root"] / DATASET_MANIFEST_FILENAME).read_text()
    )

    written = _run(workspace, seed=5)

    block = written["exposure_features"]
    assert block["source_data_root"] == str(workspace["data_root"])
    assert block["source_manifest_sha256"] == source_meta["manifest_sha256"]
    assert block["videos_dir"] == str(workspace["videos"])
    assert (block["rows_before"], block["rows_after"]) == (8, 7)
    assert block["removed_sample_ids"] == ["allsky-20260102-1400"]
    assert block["unreadable_by_video"] == {_VIDEOS[0]: 0, _VIDEOS[1]: 1}
    assert block["shuffle_seed"] == 5
    assert written["manifest_sha256"] != source_meta["manifest_sha256"]
    assert written["feature_set"] == "bare"


def test_the_meta_declares_the_feature_columns_this_manifest_serves(workspace: dict[str, Path]):
    written = _run(workspace)

    assert written["feature_columns"] == [
        *_SOURCE_META["feature_columns"],
        *EXPOSURE_FEATURE_COLUMNS,
        *SHUFFLED_FEATURE_COLUMNS,
    ]


def test_the_prepare_build_hashes_leave_the_top_level_for_the_source_block(
    workspace: dict[str, Path],
):
    written = _run(workspace)

    block = written["exposure_features"]
    assert not {*SOURCE_BUILD_KEYS} & {*written}
    assert (block["source_config_sha256"], block["source_inputs_sha256"]) == (
        _SOURCE_META["config_sha256"],
        _SOURCE_META["inputs_sha256"],
    )
    assert block["source_feature_columns"] == _SOURCE_META["feature_columns"]


def test_the_split_artifact_is_copied_beside_the_manifest(workspace: dict[str, Path]):
    _run(workspace)

    copied = workspace["out"] / DATASET_SPLIT_FILENAME
    assert copied.read_text(encoding="utf-8") == (
        workspace["data_root"] / DATASET_SPLIT_FILENAME
    ).read_text(encoding="utf-8")


def test_frames_is_a_relative_symlink_through_which_every_image_path_resolves(
    workspace: dict[str, Path],
):
    _run(workspace)

    link = workspace["out"] / FRAMES_DIRNAME
    manifest = pd.read_parquet(workspace["out"] / DATASET_MANIFEST_FILENAME)

    assert link.is_symlink()
    assert not link.readlink().is_absolute()
    assert link.resolve() == (workspace["data_root"] / FRAMES_DIRNAME).resolve()
    assert all((workspace["out"] / path).is_file() for path in manifest["image_path"])


def test_one_shard_per_video_is_written_under_the_hidden_directory(workspace: dict[str, Path]):
    _run(workspace)

    shards = sorted(path.name for path in (workspace["out"] / SHARD_DIRNAME).iterdir())
    assert shards == [f"{video}.parquet" for video in _VIDEOS]


def test_an_existing_output_directory_is_refused_without_resume(workspace: dict[str, Path]):
    _run(workspace)

    with pytest.raises(ExposureFeaturesError, match="--resume"):
        _run(workspace)


def test_resume_reuses_the_shards_and_decodes_nothing(workspace: dict[str, Path]):
    first = _run(workspace)

    again = run_exposure_features(
        workspace["data_root"],
        workspace["videos"],
        workspace["out"],
        seed=0,
        resume=True,
        frame_records=_refusing_records,
    )

    assert again["manifest_sha256"] == first["manifest_sha256"]


def test_resume_recomputes_only_the_video_whose_shard_is_missing(workspace: dict[str, Path]):
    first = _run(workspace)
    (workspace["out"] / SHARD_DIRNAME / f"{_VIDEOS[1]}.parquet").unlink()
    decoded: list[str] = []

    def partial(path: Path, frame_indices: Iterable[int]) -> list[ExposureRecord]:
        decoded.append(path.name)
        return _fake_records(path, frame_indices)

    again = run_exposure_features(
        workspace["data_root"],
        workspace["videos"],
        workspace["out"],
        seed=0,
        resume=True,
        frame_records=partial,
    )

    assert decoded == [_VIDEOS[1]]
    assert again["manifest_sha256"] == first["manifest_sha256"]


def test_a_video_the_manifest_names_but_the_directory_lacks_is_an_error_before_any_decode(
    tmp_path: Path,
):
    data_root = _write_source_dataset(tmp_path / "dataset")
    videos = _write_videos(tmp_path / "videos", names=_VIDEOS[:1])

    with pytest.raises(ExposureFeaturesError, match=_VIDEOS[1]):
        run_exposure_features(
            data_root, videos, tmp_path / "out", seed=0, resume=False, frame_records=_fake_records
        )
    assert not (tmp_path / "out").exists()


def test_a_decode_failure_surfaces_as_the_command_error(workspace: dict[str, Path]):
    def broken(path: Path, _frame_indices: Iterable[int]) -> list[ExposureRecord]:
        raise ValueError(f"{path.name} frame 0: expected a native frame")

    with pytest.raises(ExposureFeaturesError, match="expected a native frame"):
        run_exposure_features(
            workspace["data_root"],
            workspace["videos"],
            workspace["out"],
            seed=0,
            resume=False,
            frame_records=broken,
        )


def test_a_source_without_a_split_artifact_is_refused(workspace: dict[str, Path]):
    (workspace["data_root"] / DATASET_SPLIT_FILENAME).unlink()

    with pytest.raises(ExposureFeaturesError, match="split artifact"):
        _run(workspace)


def test_the_command_writes_the_dataset_and_reports_the_counts(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(allsky.exposure, "exposure_records_for_video", _fake_records)

    result = runner.invoke(
        app,
        [
            "exposure-features",
            "--data-root",
            str(workspace["data_root"]),
            "--videos",
            str(workspace["videos"]),
            "--out",
            str(workspace["out"]),
            "--seed",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "7 of 8 rows (1 unreadable removed)" in result.output
    assert (workspace["out"] / DATASET_MANIFEST_FILENAME).is_file()


def test_the_command_exits_one_naming_the_missing_video(tmp_path: Path):
    data_root = _write_source_dataset(tmp_path / "dataset")
    videos = _write_videos(tmp_path / "videos", names=_VIDEOS[:1])

    result = runner.invoke(
        app,
        [
            "exposure-features",
            "--data-root",
            str(data_root),
            "--videos",
            str(videos),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 1
    assert _VIDEOS[1] in result.output


def test_the_command_exits_one_on_an_existing_output_without_resume(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(allsky.exposure, "exposure_records_for_video", _fake_records)
    _run(workspace)

    result = runner.invoke(
        app,
        [
            "exposure-features",
            "--data-root",
            str(workspace["data_root"]),
            "--videos",
            str(workspace["videos"]),
            "--out",
            str(workspace["out"]),
        ],
    )

    assert result.exit_code == 1
    assert "--resume" in result.output
