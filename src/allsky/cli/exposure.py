"""``exposure-features`` CLI: a dataset whose manifest carries the camera's exposure.

Reads a prepared dataset's manifest, decodes each source video once at native
resolution and, for the frames the manifest names, reads the overlay exposure
time and the sky disc's relative radiance (:mod:`allsky.exposure`). The result
is a NEW dataset directory: the source manifest with the exposure columns and
their within-split shuffled controls appended, rows with an unreadable overlay
removed, the split artifact copied, and ``frames`` a relative symlink to the
source frames so no JPEG is duplicated. The sidecar meta names the source
manifest, its hash, the rows removed and the shuffle seed.

Per-video shards live under ``<out>/.exposure/`` and make the run resumable:
``--resume`` reuses every shard already on disk and recomputes only the missing
ones. Without it an existing output directory is refused.

imageio-ffmpeg and the reader are imported inside the command, so
``allsky --help`` stays light.
"""

import json
import logging
import shutil
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer

from allsky.cli.runtime import configure_cli_logging
from allsky.config import DATASET_MANIFEST_FILENAME, DATASET_SPLIT_FILENAME

if TYPE_CHECKING:
    import pandas as pd

    from allsky.exposure import ExposureRecord

logger = logging.getLogger(__name__)

__all__ = [
    "FRAMES_DIRNAME",
    "SHARD_DIRNAME",
    "SOURCE_BUILD_KEYS",
    "ExposureFeaturesError",
    "FrameRecordsFn",
    "exposure_features_cmd",
    "register",
    "run_exposure_features",
]

SHARD_DIRNAME = ".exposure"
FRAMES_DIRNAME = "frames"
META_PROVENANCE_KEY = "exposure_features"
#: Hashes ``allsky prepare`` writes for the build it resumes from; they
#: describe the source dataset, not this one, so they move under
#: :data:`META_PROVENANCE_KEY` as ``source_*``.
SOURCE_BUILD_KEYS = ("config_sha256", "inputs_sha256")

#: ``(video path, frame indices) -> records``; the decoding boundary a test
#: replaces with an in-memory source.
FrameRecordsFn = Callable[[Path, Iterable[int]], "list[ExposureRecord]"]


class ExposureFeaturesError(RuntimeError):
    """The exposure dataset could not be built from what is on disk."""


def _check_videos_present(videos_dir: Path, videos: list[str]) -> None:
    missing = [name for name in videos if not (videos_dir / name).is_file()]
    if missing:
        raise ExposureFeaturesError(
            f"{len(missing)} video(s) the manifest names are not in {videos_dir}: "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
        )


def _shard_path(out_dir: Path, video: str) -> Path:
    return out_dir / SHARD_DIRNAME / f"{video}.parquet"


def _video_shard(
    out_dir: Path,
    videos_dir: Path,
    video: str,
    frame_indices: Iterable[int],
    frame_records: FrameRecordsFn,
    *,
    resume: bool,
) -> pd.DataFrame:
    import pandas as pd

    from allsky.exposure import records_frame
    from labmim_core.atomic import atomic_write

    shard = _shard_path(out_dir, video)
    if resume and shard.is_file():
        logger.info("%s: shard present, reusing %s", video, shard)
        return pd.read_parquet(shard)
    records = frame_records(videos_dir / video, frame_indices)
    frame = records_frame(video, records)
    atomic_write(shard, lambda tmp: frame.to_parquet(tmp, index=False))
    return frame


def _link_frames(out_dir: Path, data_root: Path, manifest: pd.DataFrame) -> Path:
    source = data_root / FRAMES_DIRNAME
    if not source.is_dir():
        raise ExposureFeaturesError(f"source dataset has no {FRAMES_DIRNAME}/ directory: {source}")
    link = out_dir / FRAMES_DIRNAME
    target = source.resolve().relative_to(out_dir.resolve(), walk_up=True)
    if link.is_symlink():
        if link.resolve() != source.resolve():
            raise ExposureFeaturesError(f"{link} already points at {link.resolve()}, not {source}")
    elif link.exists():
        raise ExposureFeaturesError(f"{link} exists and is not a symlink; refusing to replace it")
    else:
        link.symlink_to(target, target_is_directory=True)
    first_image = out_dir / str(manifest["image_path"].iloc[0])
    if not first_image.is_file():
        raise ExposureFeaturesError(
            f"{first_image} does not resolve through the {FRAMES_DIRNAME} link -> {target}"
        )
    return target


def run_exposure_features(
    data_root: Path,
    videos_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    resume: bool,
    frame_records: FrameRecordsFn | None = None,
) -> dict[str, Any]:
    """Build the exposure-feature dataset at *out_dir* from the one at *data_root*.

    Parameters
    ----------
    data_root:
        Prepared dataset directory holding ``manifest.parquet``, its meta
        sidecar, ``splits.json`` and ``frames/``.
    videos_dir:
        Directory holding every video the manifest's ``video`` column names.
    out_dir:
        Destination dataset directory. Must not exist unless *resume*.
    seed:
        Seed of the within-split permutations behind the shuffled controls.
    resume:
        Reuse the per-video shards already under ``<out_dir>/.exposure/``.
    frame_records:
        Decoding boundary, ``(video path, frame indices) -> records``; defaults
        to :func:`allsky.exposure.exposure_records_for_video`.

    Returns
    -------
    dict
        The sidecar meta written beside ``<out_dir>/manifest.parquet``: the
        source meta refreshed with this build's ``code_version`` and
        ``created_at``, the new ``manifest_sha256`` and ``row_count``,
        ``feature_columns`` extended with the columns this manifest adds, the
        prepare-build hashes (:data:`SOURCE_BUILD_KEYS`) moved into an
        ``exposure_features`` block as ``source_*`` beside the source root,
        manifest hash and feature columns, the video directory, the row counts
        before and after, the removed ``sample_id`` values, the unreadable
        count per video, the shuffle seed and the columns added.

    Raises
    ------
    ExposureFeaturesError
        If a manifest, split artifact or frames directory is missing at
        *data_root*, a named video is missing from *videos_dir*, *out_dir*
        exists without *resume*, or a video cannot be read at the frames the
        manifest names.
    """
    import pandas as pd

    import allsky.exposure as exposure_module
    from allsky.data.loading import load_manifest
    from allsky.provenance import code_version

    records_for = (
        exposure_module.exposure_records_for_video if frame_records is None else frame_records
    )
    manifest_path = data_root / DATASET_MANIFEST_FILENAME
    split_path = data_root / DATASET_SPLIT_FILENAME
    if not manifest_path.is_file():
        raise ExposureFeaturesError(f"no manifest at {manifest_path}")
    if not split_path.is_file():
        raise ExposureFeaturesError(f"no split artifact at {split_path}")
    if out_dir.exists() and not resume:
        raise ExposureFeaturesError(f"{out_dir} exists; pass --resume to reuse its shards")
    manifest, source_meta = load_manifest(manifest_path)
    videos = sorted({str(name) for name in manifest["video"]})
    _check_videos_present(videos_dir, videos)
    out_dir.mkdir(parents=True, exist_ok=True)

    shards: list[pd.DataFrame] = []
    unreadable_by_video: dict[str, int] = {}
    for position, video in enumerate(videos, start=1):
        indices = manifest.loc[manifest["video"] == video, "frame_index"].to_numpy().tolist()
        try:
            shard = _video_shard(out_dir, videos_dir, video, indices, records_for, resume=resume)
        except ValueError as exc:
            raise ExposureFeaturesError(str(exc)) from exc
        unreadable = int(shard["exposure_s"].isna().sum())
        unreadable_by_video[video] = unreadable
        shards.append(shard)
        logger.info(
            "%s: %d frame(s), %d unreadable (%d/%d videos)",
            video,
            len(shard),
            unreadable,
            position,
            len(videos),
        )

    join = exposure_module.attach_exposure_features(
        manifest, pd.concat(shards, ignore_index=True), seed=seed
    )
    columns_added = [
        *exposure_module.EXPOSURE_FEATURE_COLUMNS,
        *exposure_module.SHUFFLED_FEATURE_COLUMNS,
    ]
    source_feature_columns = list(source_meta.get("feature_columns", []))
    meta = {
        **{key: value for key, value in source_meta.items() if key not in SOURCE_BUILD_KEYS},
        "feature_columns": [*source_feature_columns, *columns_added],
        "code_version": code_version(),
        "created_at": datetime.now(UTC).isoformat(),
        META_PROVENANCE_KEY: {
            "source_data_root": str(data_root),
            "source_manifest_sha256": source_meta.get("manifest_sha256"),
            **{f"source_{key}": source_meta.get(key) for key in SOURCE_BUILD_KEYS},
            "source_feature_columns": source_feature_columns,
            "videos_dir": str(videos_dir),
            "rows_before": len(manifest),
            "rows_after": len(join.manifest),
            "removed_sample_ids": list(join.removed_sample_ids),
            "unreadable_by_video": unreadable_by_video,
            "shuffle_seed": join.seed,
            "columns_added": columns_added,
            "glyph_match_shifts": list(exposure_module.GLYPH_MATCH_SHIFTS),
        },
    }
    from allsky.data.manifest import write_manifest_parquet

    written = write_manifest_parquet(join.manifest, meta, out_dir / DATASET_MANIFEST_FILENAME)
    shutil.copyfile(split_path, out_dir / DATASET_SPLIT_FILENAME)
    target = _link_frames(out_dir, data_root, join.manifest)
    logger.info(
        "exposure-features: %d of %d rows kept (%d unreadable removed); %s -> %s",
        len(join.manifest),
        len(manifest),
        len(join.removed_sample_ids),
        out_dir / FRAMES_DIRNAME,
        target,
    )
    return written


def exposure_features_cmd(
    data_root: Annotated[
        Path,
        typer.Option(
            "--data-root",
            help="Prepared dataset (manifest.parquet, splits.json, frames/).",
            exists=True,
            file_okay=False,
        ),
    ],
    videos: Annotated[
        Path,
        typer.Option(
            "--videos",
            help="Directory holding every allsky-YYYYMMDD.mp4 the manifest names.",
            exists=True,
            file_okay=False,
        ),
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="New dataset directory to write.")],
    seed: Annotated[
        int, typer.Option("--seed", help="Seed of the within-split shuffled controls.")
    ] = 0,
    resume: Annotated[
        bool, typer.Option("--resume", help="Reuse the per-video shards already in --out.")
    ] = False,
) -> None:
    """Attach the overlay exposure and relative radiance to a prepared dataset.

    Writes ``<out>/manifest.parquet`` (+ meta), copies ``splits.json`` and links
    ``frames`` back to ``--data-root``. Rows whose overlay is unreadable are
    removed and listed in the meta, never imputed.

    Raises
    ------
    typer.Exit
        Code 1 when the inputs are incomplete or a video cannot be read.
    """
    configure_cli_logging()
    try:
        meta = run_exposure_features(data_root, videos, out, seed=seed, resume=resume)
    except ExposureFeaturesError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    summary = meta[META_PROVENANCE_KEY]
    typer.echo(
        f"Wrote {out / DATASET_MANIFEST_FILENAME}: {summary['rows_after']} of "
        f"{summary['rows_before']} rows ({len(summary['removed_sample_ids'])} unreadable "
        f"removed), manifest_sha256={str(meta['manifest_sha256'])[:12]}"
    )
    typer.echo(json.dumps({"unreadable_by_video": summary["unreadable_by_video"]}))


def register(app: typer.Typer) -> None:
    """Attach the ``exposure-features`` command onto *app*."""
    app.command("exposure-features")(exposure_features_cmd)
