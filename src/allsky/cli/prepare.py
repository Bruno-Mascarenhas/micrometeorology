"""Dataset-preparation CLI commands (Wave C2b).

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

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from allsky.config import PrepareConfig, load_prepare_config

logger = logging.getLogger(__name__)

#: pandas.DataFrame at runtime. pandas is imported lazily inside each command (see
#: the module docstring) so ``allsky --help`` stays light, so it cannot be named
#: directly in these annotations.
type PandasDataFrame = Any

#: Preparation steps ``prepare-local`` can run, in execution order.
VALID_STEPS = ("extract-frames", "build-manifest", "splits")

_MANIFEST_NAME = "manifest.parquet"
_SPLIT_NAME = "splits.json"

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


def _configure_logging() -> None:
    """Enable structured INFO logging once (idempotent-ish, matches legacy CLI)."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )


def _load_prepare(config: Path | None) -> PrepareConfig:
    """Load a :class:`PrepareConfig` from *config*, or the defaults when None."""
    return PrepareConfig() if config is None else load_prepare_config(config)


def _config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the resolved config (used to detect a stale manifest)."""
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# validate-dataset
# ---------------------------------------------------------------------------


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
) -> None:
    """Validate a prepared manifest; exit 1 on errors (or on warnings if --strict)."""
    _configure_logging()
    cfg = _load_prepare(config)

    import pandas as pd

    from allsky.data.splits import load_split_artifact
    from allsky.data.validation import validate_manifest

    manifest_path = (
        manifest if manifest is not None else Path(cfg.output.dataset_dir) / _MANIFEST_NAME
    )
    if not manifest_path.exists():
        typer.echo(f"ERROR: manifest not found: {manifest_path}")
        raise typer.Exit(1)

    meta_path = manifest_path.with_name(f"{manifest_path.name}.meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if not meta_path.exists():
        typer.echo(f"WARNING: meta sidecar not found: {meta_path}")

    manifest_df = pd.read_parquet(manifest_path)
    data_root = manifest_path.parent

    split_artifact = None
    split_path = manifest_path.with_name(_SPLIT_NAME)
    if split_path.exists():
        split_artifact = load_split_artifact(split_path).to_dict()
        typer.echo(f"Split artifact: {split_path}")

    report = validate_manifest(
        manifest_df,
        meta,
        data_root=data_root,
        split_artifact=split_artifact,
        strict=strict,
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


# ---------------------------------------------------------------------------
# prepare-local
# ---------------------------------------------------------------------------


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
    """Prepare a local dataset: extract frames, build the manifest and day splits."""
    _configure_logging()
    cfg = _load_prepare(config)

    import glob

    step_set = _parse_steps(steps)
    dataset_dir = Path(cfg.output.dataset_dir)
    frames_root = dataset_dir / "frames"
    manifest_path = dataset_dir / _MANIFEST_NAME
    meta_path = manifest_path.with_name(f"{manifest_path.name}.meta.json")
    split_path = dataset_dir / _SPLIT_NAME
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

    per_video = _run_extract_step(
        cfg=cfg,
        videos=videos,
        frames_root=frames_root,
        run_extract="extract-frames" in step_set,
        force=force,
    )

    if "build-manifest" in step_set:
        _run_build_manifest_step(
            cfg=cfg,
            per_video=per_video,
            dataset_dir=dataset_dir,
            manifest_path=manifest_path,
            meta_path=meta_path,
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


# ---------------------------------------------------------------------------
# export-colab-bundle
# ---------------------------------------------------------------------------


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
) -> None:
    """Pack a prepared dataset into a Colab-ready tar.gz bundle."""
    _configure_logging()
    cfg = _load_prepare(config)

    from allsky.bundle import export_colab_bundle

    config_paths = [config] if config is not None else []
    summary = export_colab_bundle(
        out,
        prepare_cfg=cfg,
        config_paths=config_paths,
        include_embeddings=include_embeddings,
    )
    typer.echo(json.dumps(summary, indent=2, default=str))


# ---------------------------------------------------------------------------
# step helpers
# ---------------------------------------------------------------------------


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
    force: bool,
) -> list[PandasDataFrame]:
    """Extract (or resume) per-video frames; return the per-video frame manifests.

    When *run_extract* is False the existing per-video manifests are loaded so a
    later ``build-manifest`` step can proceed on a previously extracted dataset.
    """
    import pandas as pd

    per_video: list[PandasDataFrame] = []
    for video in videos:
        stem = Path(video).stem
        vdir = frames_root / stem
        vman = vdir / _MANIFEST_NAME
        if not run_extract:
            if vman.exists():
                per_video.append(pd.read_parquet(vman))
            continue
        if vman.exists() and not force:
            typer.echo(f"resume: skipping extraction for {stem} (frames already present)")
            per_video.append(pd.read_parquet(vman))
            continue
        frame_manifest = _extract_and_qc(video, vdir, cfg)
        frame_manifest.to_parquet(vman, index=False)
        per_video.append(frame_manifest)
        typer.echo(f"extract-frames: {len(frame_manifest)} frames from {stem} -> {vdir}")
    return per_video


def _extract_and_qc(video: str, vdir: Path, cfg: PrepareConfig) -> PandasDataFrame:
    """Extract native frames then read them back for visual QC + preprocessing.

    ``extract_frames`` writes native-resolution JPEGs; each is then decoded once
    to compute :func:`allsky.preprocessing.visual_qc` flags (stored in a
    ``qc_frame_flags`` column the manifest builder later ORs into ``qc_flags``)
    and, when the config configures a mask/crop/resize, rewritten through
    :func:`allsky.preprocessing.process_frame`.  The read-back costs one JPEG
    decode per frame — cheap relative to extraction, and it keeps the QC and
    preprocessing on the exact bytes that ship in the dataset.
    """
    import imageio.v3 as iio
    import numpy as np
    import pandas as pd

    from allsky.preprocessing import _needs_preprocessing, process_frame, visual_qc
    from allsky.video import JPEG_QUALITY, extract_frames

    frame_manifest = extract_frames(video, vdir, cfg.video, step=1, resize=None)
    needs = _needs_preprocessing(cfg)

    qc_flags: list[int] = []
    for frame_path in frame_manifest["frame_path"]:
        image = np.asarray(iio.imread(frame_path), dtype=np.uint8)
        bits = 0
        for flag in visual_qc(image):
            bits |= int(flag)
        qc_flags.append(bits)
        if needs:
            iio.imwrite(frame_path, process_frame(image, cfg), quality=JPEG_QUALITY)

    result = frame_manifest.copy()
    result["qc_frame_flags"] = pd.array(qc_flags, dtype="int64")
    return result


def _run_build_manifest_step(
    *,
    cfg: PrepareConfig,
    per_video: list[PandasDataFrame],
    dataset_dir: Path,
    manifest_path: Path,
    meta_path: Path,
    config_sha: str,
    force: bool,
) -> None:
    """Build + persist the v2 manifest, skipping when up to date (resume)."""
    import pandas as pd

    if manifest_path.exists() and meta_path.exists() and not force:
        existing = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") == config_sha:
            typer.echo("resume: manifest up to date (config unchanged), skipping build-manifest")
            return
        typer.echo("build-manifest: config changed since last build, rebuilding")

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
    manifest = _apply_frame_qc(manifest, frames_manifest)
    written = write_manifest_parquet(manifest, meta, manifest_path)
    typer.echo(
        f"build-manifest: {written['row_count']} rows -> {manifest_path} "
        f"(sha256 {str(written['manifest_sha256'])[:12]})"
    )


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
            day_ids, cfg.splits.val_fraction, cfg.splits.test_fraction, cfg.splits.seed
        )
    except ValueError as exc:
        typer.echo(f"ERROR: cannot create splits: {exc}")
        raise typer.Exit(1) from exc

    try:
        save_split_artifact(split, split_path, force=force)
    except SplitExistsError as exc:
        typer.echo(f"ERROR: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(f"splits: {split.split_id[:12]} -> {split_path}")

    # Fill the manifest's 'split' column from the fresh assignment (rewrites the
    # parquet atomically and re-hashes manifest_sha256 — attach changes the hash).
    attach_split_column(manifest_path, split)
    typer.echo(f"splits: attached 'split' column to {manifest_path} (manifest_sha256 changed)")


def _load_sensor_df(cfg: PrepareConfig) -> PandasDataFrame:
    """Read all configured TOA5 files into one deduplicated time-indexed frame.

    Raw logger columns are kept as-is (the manifest builder selects and validates
    the policy columns it needs); ``cfg.sensor.column_map`` optionally renames
    logger columns to the policy source names before building.
    """
    import pandas as pd

    from micrometeorology.sensors.ingestion import read_campbell_dat

    frames = [read_campbell_dat(path) for path in cfg.sensor.paths]
    sensor_df = pd.concat(frames).sort_index()
    sensor_df = sensor_df.loc[~sensor_df.index.duplicated(keep="first")]
    if cfg.sensor.column_map:
        sensor_df = sensor_df.rename(columns=cfg.sensor.column_map)
    return sensor_df


def _apply_frame_qc(manifest: PandasDataFrame, frames_manifest: PandasDataFrame) -> PandasDataFrame:
    """OR the per-frame visual QC bits into the manifest ``qc_flags`` by sample_id."""
    import pandas as pd

    if "qc_frame_flags" not in frames_manifest.columns:
        return manifest
    timestamps = pd.to_datetime(frames_manifest["timestamp"])
    sample_ids = [f"allsky-{ts:%Y%m%d-%H%M}" for ts in timestamps]
    qc_by_sample = dict(
        zip(sample_ids, frames_manifest["qc_frame_flags"].astype("int64"), strict=False)
    )
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
