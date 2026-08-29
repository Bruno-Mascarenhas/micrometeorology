"""Dataset-preparation CLI commands.

Three commands are attached to the shared app by :func:`register`:

- ``validate-dataset`` — run :func:`allsky.data.validation.validate_manifest`
  over a manifest (and its split artifact when present) and exit non-zero on
  errors (or on warnings under ``--strict``);
- ``prepare-local`` — the local end-to-end preparation pipeline (extract frames
  -> build manifest -> day splits) with ``--steps`` selection, ``--dry-run``,
  ``--force`` and resume semantics;
- ``export-colab-bundle`` — pack a prepared dataset into a Colab-ready
  ``tar.gz`` via :func:`allsky.bundle.export_colab_bundle`.

Heavy dependencies (pandas, imageio, torch-free sibling modules) are imported
lazily inside each command so ``allsky --help`` stays light and torch-free.
"""

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Annotated, Any

import typer

from allsky.atomic import atomic_write, atomic_write_json
from allsky.cli.runtime import configure_cli_logging
from allsky.config import (
    DATASET_MANIFEST_FILENAME,
    DATASET_SPLIT_FILENAME,
    FRAME_PIXEL_SECTIONS,
    VIDEO_TIME_FIELDS,
    PrepareConfig,
    load_prepare_config,
    manifest_meta_path,
)
from allsky.frame_pixels import decode_rgb

logger = logging.getLogger(__name__)

#: pandas.DataFrame at runtime. pandas is imported lazily inside each command (see
#: the module docstring) so ``allsky --help`` stays light, so it cannot be named
#: directly in these annotations.
type PandasDataFrame = Any

#: Preparation steps ``prepare-local`` can run, in execution order.
VALID_STEPS = ("extract-frames", "build-manifest", "splits")

_FRAMES_META_NAME = "frames.meta.json"

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="PrepareConfig YAML (defaults to built-in defaults when omitted).",
        exists=True,
        dir_okay=False,
    ),
]


def _load_prepare(config: Path | None) -> PrepareConfig:
    """Load a :class:`PrepareConfig` from *config*, or the defaults when None."""
    return PrepareConfig() if config is None else load_prepare_config(config)


def _config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the whole resolved config, recorded as manifest provenance."""
    from allsky.provenance import config_sha256

    return config_sha256(cfg)


#: :class:`PrepareConfig` sections that reach the manifest.  ``embeddings`` and
#: ``splits`` are deliberately excluded from :func:`_manifest_inputs_sha256`:
#: neither influences a single manifest row, so an unrelated edit (lowering
#: ``embeddings.batch_size`` after an OOM) must not force a full rebuild.
_MANIFEST_CONFIG_SECTIONS = (
    "video",
    "site",
    "features",
    "mask",
    "crop",
    "resize",
    "night_filter",
    "sensor",
    "targets",
    "alignment",
    "output",
)


def _frames_inputs_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the config that decides what an extracted frame is.

    Recorded beside each per-video frame manifest, so the resume check can tell
    frames produced under the current config from frames that merely exist.
    :data:`~allsky.config.VIDEO_TIME_FIELDS` fixes the clock every frame is
    named and stamped by, and ``mask``/``crop``/``resize`` are written into the
    JPEG, so a change to any of them makes the frames on disk a different
    artifact — one the manifest would otherwise be rebuilt from under
    provenance describing the new config.

    The mask **section** is only ``{path, threshold}``, so a horizon PNG redrawn
    in place at the same path leaves it byte-identical while every extracted
    JPEG keeps the old obstruction map; the file's own content hash is folded in
    so that edit re-extracts too.
    """
    from allsky.provenance import config_subset_sha256

    return config_subset_sha256(
        cfg,
        sections=FRAME_PIXEL_SECTIONS,
        nested_fields={"video": VIDEO_TIME_FIELDS},
        content_files=() if cfg.mask.path is None else (cfg.mask.path,),
        subject="the frame provenance hash",
    )


def _read_frames_key(video_dir: Path) -> str | None:
    """The frame-config hash recorded beside a per-video manifest, or None."""
    path = video_dir / _FRAMES_META_NAME
    if not path.is_file():
        return None
    try:
        recorded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("frame provenance %s is unreadable (%s); treating it as absent", path, exc)
        return None
    stored = recorded.get("frames_sha256") if isinstance(recorded, dict) else None
    return stored if isinstance(stored, str) else None


def _write_frames_key(video_dir: Path, frames_sha: str) -> None:
    """Record the config the frames just written were extracted under."""
    atomic_write_json(video_dir / _FRAMES_META_NAME, {"frames_sha256": frames_sha})


def _require_frames_key(video_dir: Path, stem: str, *, frames_key: str, force: bool) -> None:
    """Abort when frames the manifest is about to be built from contradict *frames_key*.

    ``--steps build-manifest`` is a supported entry point that runs no extraction,
    so the only check standing between frames of one config and a manifest stamped
    with another is this one: the manifest's own provenance records the config of
    the run that built it, which would then assert a preprocessing the JPEGs on
    disk never went through.  Nothing can be re-extracted from here — the step
    that does was not requested — so a recorded hash that DIFFERS stops the run.

    A video directory with no hash recorded at all is a different statement: it
    predates the sidecar, which only :func:`_write_frames_key` writes and which no
    step backfills for frames already extracted.  Refusing it would make
    ``--steps build-manifest`` impossible on every dataset prepared before this
    check existed, with only ``--force`` — which skips the comparison entirely —
    as a way through.  It warns and proceeds instead: the frames may well match,
    and the manifest's own provenance still records the config that built it.
    """
    if force:
        return
    recorded = _read_frames_key(video_dir)
    if recorded == frames_key:
        return
    if recorded is None:
        typer.echo(
            f"WARNING: the frames for {stem} in {video_dir} record no video/mask/crop/resize "
            "config, so build-manifest cannot confirm they went through the one it is about to "
            "stamp the manifest with. Re-run with the extract-frames step included to settle it."
        )
        return
    typer.echo(
        f"ERROR: the frames for {stem} in {video_dir} were extracted under a different "
        "video/mask/crop/resize config, so build-manifest would stamp the manifest with a "
        "config that did not produce them.\n"
        "Re-run with the extract-frames step included (or --force to build from them as they "
        "are)."
    )
    raise typer.Exit(1)


def _manifest_inputs_sha256(cfg: PrepareConfig, per_video: list[PandasDataFrame]) -> str:
    """Content hash of everything the manifest is actually built from.

    ``_config_sha256`` covers none of it, so the resume check keys on this hash
    instead: a newly extracted video day has to invalidate the manifest, and an
    edit to an irrelevant section must not.  Three inputs are folded in:

    - the extracted frame set — the ``frame_path`` values of the per-video
      manifests that ``pd.concat`` feeds to the builder;
    - each sensor file's **content** hash.  Not ``(size, mtime_ns)``: ``cp``/
      ``scp`` of an unchanged archive bumps mtime and would force a pointless
      rebuild, while ``rsync -a`` preserves mtime across a real in-place edit.
      The build step already parses these files in full, so hashing them is
      cheap relative to the work it gates;
    - only the config sections that reach the manifest
      (:data:`_MANIFEST_CONFIG_SECTIONS`).

    Note that the live, appended sensor archive changes content on essentially
    every logger write, so a rebuild on every run is the expected steady state —
    the manifest's targets are re-derived from the newest sensor data, which is
    the point.
    """
    unknown = [name for name in _MANIFEST_CONFIG_SECTIONS if name not in PrepareConfig.model_fields]
    if unknown:
        raise RuntimeError(
            f"PrepareConfig has no field(s) {unknown}; pydantic ignores bogus include keys, so "
            "the manifest inputs hash would silently stop covering them"
        )
    from allsky.provenance import canonical_config_json, file_content_sha256

    digest = hashlib.sha256()
    for frame in per_video:
        for frame_path in sorted(str(value) for value in frame["frame_path"]):
            digest.update(frame_path.encode("utf-8"))
            digest.update(b"\0")
    for sensor_path in cfg.sensor.paths:
        digest.update(file_content_sha256(Path(sensor_path)).encode("utf-8"))
    sections = cfg.model_dump(mode="json", include=set(_MANIFEST_CONFIG_SECTIONS))
    digest.update(canonical_config_json(sections).encode("utf-8"))
    return digest.hexdigest()


def validate_dataset(
    config: ConfigOption = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            help="Manifest parquet (default <dataset_dir>/manifest.parquet).", exists=True
        ),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Promote warnings to failures (exit 1).")
    ] = False,
    skip_image_check: Annotated[
        bool,
        typer.Option(
            "--skip-image-check",
            help="Do not verify that every image_path exists on disk (use on a frame-less "
            "Colab bundle, where only the embeddings travelled).",
        ),
    ] = False,
) -> None:
    """Validate a prepared manifest; exit 1 on errors (or on warnings if --strict).

    The split artifact next to the manifest is validated together with it when
    present, and the ``.meta.json`` sidecar supplies the provenance the checks
    compare against (its absence is a warning, not a failure).

    Raises
    ------
    typer.Exit
        Code 1 when the manifest is absent or the report holds errors (or, under
        ``--strict``, warnings).
    """
    configure_cli_logging()
    cfg = _load_prepare(config)

    import pandas as pd

    from allsky.data.splits import load_split_artifact
    from allsky.data.validation import validate_manifest

    manifest_path = (
        manifest
        if manifest is not None
        else Path(cfg.output.dataset_dir) / DATASET_MANIFEST_FILENAME
    )
    if not manifest_path.exists():
        typer.echo(f"ERROR: manifest not found: {manifest_path}")
        raise typer.Exit(1)

    meta_path = manifest_meta_path(manifest_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not meta_path.exists():
        typer.echo(f"WARNING: meta sidecar not found: {meta_path}")

    manifest_df = pd.read_parquet(manifest_path)
    data_root = manifest_path.parent

    split_artifact = None
    split_path = manifest_path.with_name(DATASET_SPLIT_FILENAME)
    if split_path.exists():
        split_artifact = load_split_artifact(split_path).to_dict()
        typer.echo(f"Split artifact: {split_path}")

    report = validate_manifest(
        manifest_df,
        meta,
        data_root=data_root,
        split_artifact=split_artifact,
        strict=strict,
        check_files=not skip_image_check,
    )

    for warning in report.warnings:
        typer.echo(f"WARNING: {warning}")
    for error in report.errors:
        typer.echo(f"ERROR: {error}")
    typer.echo(
        f"validate-dataset: {len(manifest_df)} rows, "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )

    failed = bool(report.errors) or (strict and bool(report.warnings))
    if failed:
        raise typer.Exit(1)
    typer.echo("OK")


def prepare_local(
    config: ConfigOption = None,
    steps: Annotated[
        str,
        typer.Option(help=f"Comma-separated subset of {list(VALID_STEPS)} (default: all)."),
    ] = ",".join(VALID_STEPS),
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Log the full plan and write nothing.")
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-extract, rebuild and regenerate regardless of state."),
    ] = False,
) -> None:
    """Prepare a local dataset: extract frames, build the manifest and day splits.

    Each step resumes on the artifacts already on disk unless ``--force`` is
    given: a video whose frame manifest is complete and whose frames were
    extracted under the current config is not re-extracted (see
    :func:`_frames_inputs_sha256`), and the manifest is rebuilt only when the
    inputs it is derived from changed (see :func:`_manifest_inputs_sha256`).

    Raises
    ------
    typer.Exit
        Code 1 on an unknown ``--steps`` name, on a sensor record that cannot be
        paired with any video day, when ``build-manifest`` runs with no extracted
        frames, or when the split artifact already exists for a different day set.
    """
    configure_cli_logging()
    cfg = _load_prepare(config)

    import glob

    step_set = _parse_steps(steps)
    dataset_dir = Path(cfg.output.dataset_dir)
    frames_root = dataset_dir / "frames"
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    meta_path = manifest_meta_path(manifest_path)
    split_path = dataset_dir / DATASET_SPLIT_FILENAME
    videos = sorted(glob.glob(cfg.video.pattern))
    config_sha = _config_sha256(cfg)

    if dry_run:
        _log_plan(
            cfg=cfg,
            steps=step_set,
            videos=videos,
            frames_root=frames_root,
            manifest_path=manifest_path,
            split_path=split_path,
            config_sha=config_sha,
        )
        return

    if not videos:
        typer.echo(f"WARNING: no videos matched pattern {cfg.video.pattern!r}")

    if videos and "build-manifest" in step_set:
        _check_sensor_coverage(cfg, videos)

    per_video = _run_extract_step(
        cfg=cfg,
        videos=videos,
        frames_root=frames_root,
        run_extract="extract-frames" in step_set,
        build_manifest="build-manifest" in step_set,
        force=force,
    )

    if "build-manifest" in step_set:
        _run_build_manifest_step(
            cfg=cfg,
            per_video=per_video,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            meta_path=meta_path,
            split_path=split_path,
            config_sha=config_sha,
            force=force,
        )

    if "splits" in step_set:
        _run_splits_step(
            cfg=cfg,
            manifest_path=manifest_path,
            split_path=split_path,
            force=force,
        )


def export_colab_bundle_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help="Destination bundle .tar.gz.")],
    config: ConfigOption = None,
    include_embeddings: Annotated[
        bool,
        typer.Option(
            "--include-embeddings/--no-include-embeddings",
            help="Include precomputed embedding shards when present.",
        ),
    ] = True,
    include_frames: Annotated[
        bool,
        typer.Option(
            "--include-frames/--no-include-frames",
            help="Include the JPEG frames the manifest references (needed by image-mode "
            "experiments; off by default because embedding-mode bundles do not use them).",
        ),
    ] = False,
) -> None:
    """Pack a prepared dataset into a Colab-ready tar.gz bundle."""
    configure_cli_logging()
    cfg = _load_prepare(config)

    from allsky.bundle import export_colab_bundle

    config_paths = [config] if config is not None else []
    summary = export_colab_bundle(
        out,
        prepare_cfg=cfg,
        config_paths=config_paths,
        include_embeddings=include_embeddings,
        include_frames=include_frames,
    )
    typer.echo(json.dumps(summary, indent=2, default=str))


def _parse_steps(steps: str) -> set[str]:
    """Parse and validate the ``--steps`` CSV; unknown steps abort."""
    requested = [s.strip() for s in steps.split(",") if s.strip()]
    unknown = [s for s in requested if s not in VALID_STEPS]
    if unknown:
        typer.echo(f"ERROR: unknown step(s) {unknown}; valid steps are {list(VALID_STEPS)}")
        raise typer.Exit(1)
    return set(requested)


def _log_plan(
    *,
    cfg: PrepareConfig,
    steps: set[str],
    videos: list[str],
    frames_root: Path,
    manifest_path: Path,
    split_path: Path,
    config_sha: str,
) -> None:
    """Emit the full prepare-local plan without writing anything (``--dry-run``)."""
    typer.echo("prepare-local DRY RUN (no files will be written)")
    typer.echo(f"  steps:          {sorted(steps)}")
    typer.echo(f"  video pattern:  {cfg.video.pattern}")
    typer.echo(f"  videos found:   {len(videos)}")
    for video in videos:
        typer.echo(f"    {video} -> {frames_root / Path(video).stem}")
    typer.echo(f"  sensor paths:   {cfg.sensor.paths}")
    typer.echo(f"  feature set:    {cfg.features.feature_set}")
    typer.echo(f"  manifest out:   {manifest_path}")
    typer.echo(f"  splits out:     {split_path}")
    typer.echo(f"  config_sha256:  {config_sha}")


def _run_extract_step(
    *,
    cfg: PrepareConfig,
    videos: list[str],
    frames_root: Path,
    run_extract: bool,
    build_manifest: bool,
    force: bool,
) -> list[PandasDataFrame]:
    """Extract (or resume) per-video frames; return the per-video frame manifests.

    When *run_extract* is False the existing per-video manifests are loaded so a
    later ``build-manifest`` step can proceed on a previously extracted dataset —
    but only while their recorded frame provenance matches the current config,
    since no re-extraction can happen from there (see :func:`_require_frames_key`).
    That demand is made only when *build_manifest* is set: a ``--steps splits``
    run reads the persisted manifest and ``cfg.splits`` alone, discarding what
    this function returns, so frame provenance it never consults must not stop it.

    Resume keys on a **complete** per-video manifest, never on the mere existence
    of the file: :func:`allsky.video.extract_frames` publishes
    ``manifest.parquet`` before the visual-QC pass runs, so an interruption
    inside that pass leaves a manifest without the ``qc_frame_flags`` column
    (every FRAME_DARK / FRAME_SATURATED bit silently lost) and an interrupted
    write leaves one that cannot be read at all.  Both are re-extracted.

    It keys on :func:`_frames_inputs_sha256` as well, recorded in each video
    directory when its frames were written.  The frames themselves carry no trace
    of the config that produced them, so without that record a changed
    ``video.timestamps`` / ``mask`` / ``crop`` / ``resize`` resumed onto the old
    JPEGs while the manifest — whose own key does cover those sections — was
    rebuilt from them and stamped with the new config's hash, asserting a build
    that never ran.  Frames with no record are re-extracted for the same reason:
    the config behind them is unknown.

    A video whose own timestamps are unusable is skipped, not fatal, and every
    skip is named again on the next run.  The overlay reader refuses a day it
    cannot timestamp — the 2026-06-04 archive video steps its clock 7 s backwards
    at frame 851 — and that refusal is right, but it describes ONE day: letting
    it end the loop throws away the other 95 days of a two-hour extraction and
    stops the daily job in ``docs/allsky-archive.md`` permanently.
    A skipped day contributes no manifest rows, which is what a night-only day
    already does.  The exit code stays zero for the same reason: the fault is a
    permanent property of that day's bytes, so a run that failed on it will fail
    on it forever, and an exit code that can never go green signals nothing.
    """
    from allsky.overlay import OverlayTimestampError

    frames_key = _frames_inputs_sha256(cfg)
    per_video: list[PandasDataFrame] = []
    unusable: list[str] = []
    for video in videos:
        stem = Path(video).stem
        video_dir = frames_root / stem
        video_manifest = video_dir / DATASET_MANIFEST_FILENAME
        existing = _read_frame_manifest(video_manifest)
        qc_complete = existing is not None and "qc_frame_flags" in existing.columns

        if not run_extract:
            if existing is None:
                if video_manifest.exists():
                    typer.echo(
                        f"WARNING: skipping {stem}: frame manifest {video_manifest} is unreadable"
                    )
                continue
            if build_manifest:
                _require_frames_key(video_dir, stem, frames_key=frames_key, force=force)
            if not qc_complete:
                typer.echo(
                    f"WARNING: frame manifest {video_manifest} has no qc_frame_flags; FRAME_DARK/"
                    "FRAME_SATURATED will be unset (re-run the extract-frames step to "
                    "populate them)"
                )
            per_video.append(existing)
            continue

        if qc_complete and not force:
            recorded = _read_frames_key(video_dir)
            if recorded == frames_key:
                typer.echo(f"resume: skipping extraction for {stem} (frames already present)")
                per_video.append(existing)
                continue
            typer.echo(
                f"resume: the frames for {stem} were extracted under "
                + ("no recorded" if recorded is None else "a different")
                + " video/mask/crop/resize config, re-extracting"
            )
        elif video_manifest.exists() and not force:
            typer.echo(
                f"resume: frame manifest for {stem} is unreadable or predates visual QC, "
                "re-extracting"
            )
        try:
            frame_manifest = _extract_replacing_frames(video, video_dir, cfg)
        except OverlayTimestampError as exc:
            unusable.append(stem)
            typer.echo(f"WARNING: skipping {stem}: {exc}")
            continue
        _write_frame_manifest(video_manifest, frame_manifest)
        _write_frames_key(video_dir, frames_key)
        per_video.append(frame_manifest)
        typer.echo(f"extract-frames: {len(frame_manifest)} frames from {stem} -> {video_dir}")
    if unusable:
        typer.echo(
            f"WARNING: {len(unusable)} video(s) could not be timestamped and contribute no "
            f"rows: {', '.join(unusable)}"
        )
    return per_video


def _read_frame_manifest(video_manifest: Path) -> PandasDataFrame | None:
    """Read a per-video frame manifest, or None when it is missing or unreadable.

    A truncated parquet (a kill or a full disk during the write) must not wedge
    every later run inside ``pd.read_parquet``: it is logged and reported as
    absent so the caller can re-extract that one video.
    """
    import pandas as pd

    if not video_manifest.exists():
        return None
    try:
        return pd.read_parquet(video_manifest)
    except Exception as exc:  # noqa: BLE001 - any parquet/IO failure means "re-extract this video"
        logger.warning(
            "frame manifest %s is unreadable (%s); treating it as absent", video_manifest, exc
        )
        return None


def _write_frame_manifest(video_manifest: Path, frame_manifest: PandasDataFrame) -> None:
    """Atomically persist a per-video frame manifest (temp file + ``os.replace``)."""
    atomic_write(video_manifest, lambda tmp: frame_manifest.to_parquet(tmp, index=False))


def _video_day(path: str, cfg: PrepareConfig) -> Any:
    from allsky.video import video_date

    return video_date(path, cfg.video)


def _check_sensor_coverage(cfg: PrepareConfig, videos: list[str]) -> None:
    """Fail before extraction when the logger cannot pair with the videos on hand.

    Extraction and visual QC run for minutes per video and every frame is then
    discarded by the pairing step, so a coverage gap has to surface here rather
    than as an empty manifest an hour later.
    """
    import datetime as dt

    import pandas as pd

    sensor_df = _load_sensor_df(cfg)
    if sensor_df.empty:
        typer.echo(f"ERROR: no sensor records read from {cfg.sensor.paths}")
        raise typer.Exit(code=1)

    index = pd.DatetimeIndex(sensor_df.index)
    sensor_start, sensor_end = index.min(), index.max()
    days = sorted({_video_day(video, cfg) for video in videos})
    covered = [
        day
        for day in days
        if sensor_start.date() <= day + dt.timedelta(days=1) and day <= sensor_end.date()
    ]

    typer.echo(
        f"sensor coverage: {sensor_start:%Y-%m-%d %H:%M} .. {sensor_end:%Y-%m-%d %H:%M} "
        f"({len(index)} records)"
    )
    typer.echo(f"video days:      {days[0]} .. {days[-1]} ({len(days)} videos)")

    if not covered:
        typer.echo(
            "ERROR: the sensor record and the videos do not overlap, so no frame could ever "
            "be paired with a measurement.\n"
            f"  sensor ends   {sensor_end:%Y-%m-%d %H:%M}\n"
            f"  videos start  {days[0]}\n"
            "Export the logger up to the video dates (or narrow video.pattern to the days the "
            "logger covers) before preparing."
        )
        raise typer.Exit(code=1)
    if len(covered) < len(days):
        typer.echo(
            f"WARNING: only {len(covered)} of {len(days)} video days fall inside the sensor "
            f"record ({covered[0]} .. {covered[-1]}); the rest will contribute no rows"
        )


def _extract_and_qc(video: str, video_dir: Path, cfg: PrepareConfig) -> PandasDataFrame:
    """Extract native frames then read them back for visual QC + preprocessing.

    The ``qc_frame_flags`` column added here is what the manifest builder later
    ORs into ``qc_flags``.

    QC describes the frame **as extracted**: with a mask/crop/resize configured
    the file is overwritten afterwards, so the flags do not describe the
    preprocessed bytes that ship.
    """
    import imageio.v3 as iio
    import numpy as np
    import pandas as pd

    from allsky.overlay import extract_frames_for
    from allsky.preprocessing import _needs_preprocessing, process_frame, resolve_mask, visual_qc
    from allsky.video import JPEG_QUALITY

    frame_manifest = extract_frames_for(video, video_dir, cfg.video)
    needs = _needs_preprocessing(cfg)
    mask = resolve_mask(cfg) if needs else None

    qc_flags: list[int] = []
    for frame_path in frame_manifest["frame_path"]:
        image = decode_rgb(frame_path)
        bits = 0
        for flag in visual_qc(image):
            bits |= int(flag)
        qc_flags.append(bits)
        if needs:
            iio.imwrite(frame_path, process_frame(image, cfg, mask=mask), quality=JPEG_QUALITY)

    result = frame_manifest.copy()
    existing = (
        result["qc_frame_flags"].to_numpy(dtype="int64")
        if "qc_frame_flags" in result.columns
        else np.zeros(len(result), dtype="int64")
    )
    result["qc_frame_flags"] = pd.array(
        existing | np.asarray(qc_flags, dtype="int64"), dtype="int64"
    )
    return result


def _extract_replacing_frames(video: str, video_dir: Path, cfg: PrepareConfig) -> PandasDataFrame:
    """Extract *video* into a staging directory, then swap it in for *video_dir*.

    Frames are named after their own timestamp, so a re-extraction under a
    changed clock writes a different file set: on the 2026-06-25 archive video an
    overlay run wrote 1366 JPEGs and the modelled re-run added 1434 beside them,
    leaving 68 orphans that the frame manifest does not list and nothing removes.
    Consumers that read the directory rather than the manifest — the archive
    uploader does — then see two clocks at once.

    The replacement is produced whole before anything is removed and the previous
    directory is moved aside rather than deleted, so a failure at any point
    leaves the earlier extraction intact.

    Returns
    -------
    pandas.DataFrame
        The frame manifest of :func:`_extract_and_qc`, with ``frame_path``
        rewritten from the staging directory onto the final *video_dir*.
    """
    staging = video_dir.with_name(f"{video_dir.name}.incoming")
    superseded = video_dir.with_name(f"{video_dir.name}.superseded")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        frame_manifest = _extract_and_qc(video, staging, cfg)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    shutil.rmtree(superseded, ignore_errors=True)
    if video_dir.exists():
        video_dir.rename(superseded)
    staging.rename(video_dir)
    shutil.rmtree(superseded, ignore_errors=True)
    replaced = frame_manifest.copy()
    replaced["frame_path"] = [
        str(video_dir / Path(str(frame_path)).name) for frame_path in frame_manifest["frame_path"]
    ]
    return replaced


def _run_build_manifest_step(
    *,
    cfg: PrepareConfig,
    per_video: list[PandasDataFrame],
    dataset_dir: Path,
    manifest_path: Path,
    meta_path: Path,
    split_path: Path,
    config_sha: str,
    force: bool,
) -> None:
    """Build + persist the v2 manifest, skipping when its inputs are unchanged."""
    import pandas as pd

    inputs_sha = _manifest_inputs_sha256(cfg, per_video) if per_video else None
    if manifest_path.exists() and meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if _manifest_is_up_to_date(existing, inputs_sha=inputs_sha, config_sha=config_sha):
            typer.echo("resume: manifest up to date (inputs unchanged), skipping build-manifest")
            return
        typer.echo(
            "build-manifest: inputs changed since last build (frame set, sensor data or a "
            "manifest-relevant config section), rebuilding"
        )

    if not per_video:
        typer.echo(
            "ERROR: build-manifest needs extracted frames; run the extract-frames step first"
        )
        raise typer.Exit(1)

    from allsky.data.manifest import build_manifest_from_prepare_config, write_manifest_parquet

    frames_manifest: PandasDataFrame = pd.concat(per_video, ignore_index=True)
    sensor_df = _load_sensor_df(cfg)
    manifest, meta = build_manifest_from_prepare_config(
        frames_manifest, sensor_df, cfg, data_root=dataset_dir, config_sha256=config_sha
    )
    meta["inputs_sha256"] = inputs_sha
    manifest = _apply_frame_qc(manifest, frames_manifest)
    manifest = _carry_split_labels(manifest, split_path)
    written = write_manifest_parquet(manifest, meta, manifest_path)
    typer.echo(
        f"build-manifest: {written['row_count']} rows -> {manifest_path} "
        f"(sha256 {str(written['manifest_sha256'])[:12]})"
    )


def _manifest_is_up_to_date(
    existing_meta: dict[str, Any], *, inputs_sha: str | None, config_sha: str
) -> bool:
    """Whether the persisted manifest was built from the current inputs.

    A sidecar carrying no ``inputs_sha256`` (or a run with no frame manifests to
    hash) falls back to the config-only comparison, so an already-prepared
    archive is not silently rebuilt — and re-split — on its next cron run.
    """
    recorded = existing_meta.get("inputs_sha256")
    if recorded is None or inputs_sha is None:
        return existing_meta.get("config_sha256") == config_sha
    return bool(recorded == inputs_sha)


def _carry_split_labels(manifest: PandasDataFrame, split_path: Path) -> PandasDataFrame:
    """Fill a freshly built manifest's ``split`` column from the existing artifact.

    ``build_manifest`` writes ``split`` as all-null, so a rebuild drops every
    label; if the splits step then aborts (a grown day set hashes to a different
    ``split_id``) the manifest is left with no split at all and no way back
    except ``--force``.  Only days the artifact already assigns are filled — no
    assignment is invented, changed or re-drawn.
    """
    if not split_path.exists():
        return manifest

    from allsky.data.splits import load_split_artifact

    try:
        split = load_split_artifact(split_path)
    except (ValueError, OSError) as exc:
        typer.echo(f"WARNING: cannot reuse split labels from {split_path} ({exc})")
        return manifest
    out = manifest.copy()
    out["split"] = manifest["day_id"].astype("string").map(split.assignment).astype("string")
    typer.echo(
        f"build-manifest: carried {int(out['split'].notna().sum())} existing split "
        f"label(s) forward from {split_path.name}"
    )
    return out


def _run_splits_step(
    *,
    cfg: PrepareConfig,
    manifest_path: Path,
    split_path: Path,
    force: bool,
) -> None:
    """Create + persist the day-level split artifact (guarded against overwrite)."""
    import pandas as pd

    from allsky.data.manifest import attach_split_column
    from allsky.data.splits import SplitExistsError, create_day_splits, save_split_artifact

    if not manifest_path.exists():
        typer.echo(f"ERROR: splits step needs a manifest at {manifest_path}")
        raise typer.Exit(1)

    manifest_df = pd.read_parquet(manifest_path)
    day_ids = manifest_df["day_id"].astype(str).tolist()
    try:
        split = create_day_splits(
            day_ids,
            cfg.splits.val_fraction,
            cfg.splits.test_fraction,
            cfg.splits.seed,
            strategy=cfg.splits.strategy,
            gap_days=cfg.splits.gap_days,
        )
    except ValueError as exc:
        typer.echo(f"ERROR: cannot create splits: {exc}")
        raise typer.Exit(1) from exc

    try:
        save_split_artifact(split, split_path, force=force)
    except SplitExistsError as exc:
        typer.echo(f"ERROR: {exc}")
        typer.echo(
            "ERROR: the manifest's day set or the split parameters changed since that artifact "
            "was written; re-run the splits step with --force to re-derive it — which re-draws "
            "every day, so metrics from models trained on the old split stop being comparable"
        )
        raise typer.Exit(1) from exc
    typer.echo(f"splits: {split.split_id[:12]} -> {split_path}")

    attach_split_column(manifest_path, split)
    typer.echo(f"splits: attached 'split' column to {manifest_path} (manifest_sha256 changed)")


def _load_sensor_df(cfg: PrepareConfig) -> PandasDataFrame:
    """Read all configured TOA5 files into one deduplicated time-indexed frame.

    Raw logger columns are kept as-is (the manifest builder selects and validates
    the policy columns it needs); ``cfg.sensor.column_map`` optionally renames
    logger columns to the policy source names before building.
    """
    import pandas as pd

    from micrometeorology.sensors.archive import mask_sentinels
    from micrometeorology.sensors.ingestion import read_campbell_dat

    frames = [read_campbell_dat(path) for path in cfg.sensor.paths]
    sensor_df = pd.concat(frames).sort_index()
    sensor_df = sensor_df.loc[~sensor_df.index.duplicated(keep="first")]
    # read_campbell_dat's own -900 default catches nothing in the LabMiM
    # archive: the real sentinels are 1000 degC, 999 %RH, -273.1 degC and a
    # windowed 0, all of them finite. The manifest builder filters on
    # np.isfinite alone, so without this the rails reach air_temp_c /
    # dew_point_c / rel_humidity and the feature normaliser fits its mean and
    # std over them. This is the archive's own sentinel table rather than a
    # second one, so the two cannot drift.
    sensor_df, _removed = mask_sentinels(sensor_df)
    if cfg.sensor.column_map:
        sensor_df = sensor_df.rename(columns=cfg.sensor.column_map)
    return sensor_df


def _apply_frame_qc(manifest: PandasDataFrame, frames_manifest: PandasDataFrame) -> PandasDataFrame:
    """OR the per-frame visual QC bits into the manifest ``qc_flags`` by sample_id.

    Frames extracted by the low-level ``allsky extract-frames`` command carry no
    ``qc_frame_flags`` at all, and concatenating such a video with a QC'd one
    yields a float column with gaps.  Neither is fatal — both are reported and
    the affected frames simply contribute no extra QC bits.
    """
    import pandas as pd

    if "qc_frame_flags" not in frames_manifest.columns:
        typer.echo(
            "WARNING: frame manifest has no qc_frame_flags; FRAME_DARK/FRAME_SATURATED will "
            "be unset (re-run the extract-frames step to populate them)"
        )
        return manifest
    frame_flags = frames_manifest["qc_frame_flags"]
    unflagged = frame_flags.isna()
    if unflagged.any():
        videos = sorted(frames_manifest.loc[unflagged, "video"].astype(str).unique())
        typer.echo(
            f"WARNING: no visual QC for video(s) {videos}; their FRAME_DARK/FRAME_SATURATED "
            "bits will be unset (re-run the extract-frames step for them)"
        )
    timestamps = pd.to_datetime(frames_manifest["timestamp"])
    sample_ids = [f"allsky-{ts:%Y%m%d-%H%M}" for ts in timestamps]
    qc_by_sample = dict(zip(sample_ids, frame_flags.fillna(0).astype("int64"), strict=False))
    extra = manifest["sample_id"].map(qc_by_sample).fillna(0).astype("int64")
    out = manifest.copy()
    out["qc_flags"] = (out["qc_flags"].astype("int64") | extra).astype("int64")
    return out


def register(app: typer.Typer) -> None:
    """Attach the prepare-family commands (``validate-dataset``, ``prepare-local``,
    ``export-colab-bundle``) onto *app*.
    """
    app.command("validate-dataset")(validate_dataset)
    app.command("prepare-local")(prepare_local)
    app.command("export-colab-bundle")(export_colab_bundle_cmd)
