"""``precompute-embeddings`` CLI: DINOv2 (or fake) embeddings for a manifest.

Reads a :class:`~allsky.config.PrepareConfig` YAML (whose ``embeddings`` section
pins backbone / pooling / batch / device / shard-size / dtype), loads the v2
manifest, builds the visual backbone and runs the resumable, atomically-written
extraction loop in :func:`allsky.embeddings.extract.extract_embeddings`.

The backbone name ``"fake"`` selects the deterministic, network-free
:class:`~allsky.embeddings.backbone.FakeBackbone` (a documented test/dev hook);
``"dinov2_vits14"`` selects the real DINOv2 backbone.  Any other name fails with
a message listing the available backbones.

Heavy dependencies (torch, safetensors, the backbone model) are imported lazily
inside the command, so importing :mod:`allsky.cli` never pulls them.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from allsky.config import PrepareConfig, VideoConfig

logger = logging.getLogger("allsky.embeddings")


def _configure_logging() -> None:
    """Attach a stderr handler at INFO once, so progress is visible in the CLI."""
    root = logging.getLogger("allsky")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


#: Sections a stored vector depends on whole: the encoder itself and the
#: preprocessing ``prepare-local`` bakes into the JPEG the encoder reads.
_EMBEDDING_CONFIG_SECTIONS = ("embeddings", "mask", "crop", "resize")

#: ``video`` reaches the hash by field, not whole: the clock decides which
#: capture ends up under a given ``sample_id``, while ``pattern`` only widens the
#: day set — and a day added to the glob must still resume, since growing the
#: dataset one day at a time is how this store is filled.
_EMBEDDING_VIDEO_FIELDS = ("timestamps", "start_time", "minutes_per_frame")


def _config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the config the stored vectors depend on (order-independent).

    This is the only content gate on resume, so it covers the encoder section
    together with everything that decides the pixels it encodes: ``mask``,
    ``crop`` and ``resize`` are written into the frame on disk, and
    ``video.timestamps`` (with the pair the modelled mapping reads) decides which
    capture a ``sample_id`` names.  Without them a re-prepared dataset resumes
    onto vectors of the old pixels, every id already being in the index.

    Raises
    ------
    RuntimeError
        When a name here is not a config field: pydantic drops unknown include
        keys silently, which would shrink the hash without any sign of it.
    """
    unknown = [
        name for name in _EMBEDDING_CONFIG_SECTIONS if name not in PrepareConfig.model_fields
    ]
    unknown += [
        f"video.{name}" for name in _EMBEDDING_VIDEO_FIELDS if name not in VideoConfig.model_fields
    ]
    if unknown:
        raise RuntimeError(
            f"PrepareConfig has no field(s) {unknown}; the resume hash would stop covering them"
        )
    include: dict[str, Any] = dict.fromkeys(_EMBEDDING_CONFIG_SECTIONS, True)
    include["video"] = set(_EMBEDDING_VIDEO_FIELDS)
    sections = cfg.model_dump(mode="json", include=include)
    canonical = json.dumps(sections, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def precompute_embeddings(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="PrepareConfig YAML (its 'embeddings' section pins the backbone).",
            exists=True,
            dir_okay=False,
        ),
    ],
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help="Manifest parquet override (default: <dataset_dir>/manifest.parquet).",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", "-o", help="Embeddings output dir (default: <dataset_dir>/embeddings)."
        ),
    ] = None,
    device: Annotated[
        str | None,
        typer.Option("--device", help="Device override (auto|cpu|cuda|mps)."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip sample_ids already embedded."),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report the plan and write nothing."),
    ] = False,
) -> None:
    """Precompute visual embeddings for a dataset manifest (DINOv2 or fake).

    The manifest's ``image_path`` values are relative POSIX paths against the
    manifest's own directory, so that directory is the data root the extraction
    loop resolves frames against.

    Raises
    ------
    typer.Exit
        Code 1 when the manifest is absent, the configured backbone name is
        unknown, or the extraction loop fails.
    """
    import pandas as pd

    from allsky.config import load_prepare_config
    from allsky.embeddings import build_backbone, extract_embeddings
    from allsky.embeddings.backbone import AVAILABLE_BACKBONES

    _configure_logging()

    cfg = load_prepare_config(config)
    dataset_dir = Path(cfg.output.dataset_dir)
    manifest_path = manifest if manifest is not None else dataset_dir / "manifest.parquet"
    out_dir = out if out is not None else dataset_dir / "embeddings"
    device_pref = device if device is not None else cfg.embeddings.device
    data_root = manifest_path.parent

    if not manifest_path.exists():
        typer.echo(f"error: manifest not found: {manifest_path}", err=True)
        raise typer.Exit(code=1)

    backbone_name = cfg.embeddings.backbone
    try:
        backbone = build_backbone(
            backbone_name,
            pooling=cfg.embeddings.pooling,
            device=device_pref,
            dtype=cfg.embeddings.dtype,
        )
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        typer.echo(f"available backbones: {', '.join(AVAILABLE_BACKBONES)}", err=True)
        raise typer.Exit(code=1) from exc

    logger.info(
        "precompute-embeddings: backbone=%s pooling=%s device=%s manifest=%s out=%s",
        backbone_name,
        cfg.embeddings.pooling,
        device_pref,
        manifest_path,
        out_dir,
    )

    try:
        manifest_df = pd.read_parquet(manifest_path)
        summary = extract_embeddings(
            manifest_df,
            backbone,
            out_dir,
            data_root=data_root,
            batch_size=cfg.embeddings.batch_size,
            device=device_pref,
            shard_size=cfg.embeddings.shard_size,
            resume=resume,
            dry_run=dry_run,
            config_sha256=_config_sha256(cfg),
        )
    except Exception as exc:  # surface any failure as a non-zero exit
        typer.echo(f"error: embedding extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach ``precompute-embeddings`` onto *app*."""
    app.command("precompute-embeddings")(precompute_embeddings)
