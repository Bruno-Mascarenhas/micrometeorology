"""``precompute-embeddings`` CLI: DINOv2 (or fake) embeddings for a manifest.

Reads a :class:`~allsky.config.PrepareConfig` YAML (whose ``embeddings`` section
pins backbone / pooling / batch / device / shard-size / dtype), loads the v2
manifest, builds the visual backbone and runs the resumable, atomically-written
extraction loop in :func:`allsky.embeddings.extract.extract_embeddings`.

The backbone name ``"fake"`` selects the deterministic, network-free
:class:`~allsky.embeddings.backbone.FakeBackbone` (a documented test/dev hook);
the ``dinov2_vit{s,b,l,g}14`` names select the real DINOv2 backbones.  Any other
name fails with a message listing the available backbones.

Heavy dependencies (torch, safetensors, the backbone model) are imported lazily
inside the command, so importing :mod:`allsky.cli` never pulls them.
"""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from allsky.cli.runtime import configure_cli_logging
from allsky.cli.train import DeviceChoice  # typer resolves this annotation at runtime
from allsky.config import (
    DATASET_MANIFEST_FILENAME,
    FRAME_PIXEL_SECTIONS,
    VIDEO_TIME_FIELDS,
    PrepareConfig,
)

logger = logging.getLogger(__name__)


#: Sections a stored vector depends on whole: the encoder itself and the
#: preprocessing ``prepare-local`` bakes into the JPEG the encoder reads.
_EMBEDDING_CONFIG_SECTIONS = ("embeddings", *FRAME_PIXEL_SECTIONS)


def _mask_content_files(cfg: PrepareConfig) -> tuple[str, ...]:
    """The files whose bytes shape the encoded pixels without being config values."""
    return () if cfg.mask.path is None else (cfg.mask.path,)


def _pixel_config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the config deciding which pixels a stored vector encodes.

    Recorded in ``embeddings.meta.json`` beside the full resume digest, because
    the two answer different questions.  The full digest moves whenever its own
    formula widens, which says nothing about whether a single pixel changed; this
    one moves only when the frames themselves would.

    It is PROVENANCE, not a migration key: ``_check_resume_compatible`` refuses
    any divergence of ``config_sha256`` outright, and nothing reads this hash to
    carry a store across a formula change. The way out of a refused resume is
    ``--no-resume``, which re-encodes; what this hash buys is a reader of the
    store being able to tell "the formula widened" from "the pixels changed".
    """
    from allsky.provenance import config_subset_sha256

    return config_subset_sha256(
        cfg,
        sections=FRAME_PIXEL_SECTIONS,
        nested_fields={"video": VIDEO_TIME_FIELDS},
        content_files=_mask_content_files(cfg),
        subject="the embedding pixel provenance hash",
    )


def _config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the config the stored vectors depend on (order-independent).

    This is the only content gate on resume, so it covers the encoder section
    together with everything that decides the pixels it encodes: ``mask``,
    ``crop`` and ``resize`` are written into the frame on disk, and
    :data:`~allsky.config.VIDEO_TIME_FIELDS` decides which capture a
    ``sample_id`` names.  Without them a re-prepared dataset resumes onto
    vectors of the old pixels, every id already being in the index.

    The mask **section** is only ``{path, threshold}``, so the mask file's own
    content hash is folded in as well: a horizon PNG redrawn in place at the
    same path changes every masked pixel the backbone sees while leaving the
    section byte-identical, which is precisely a resume onto vectors of the old
    pixels.  The frame gate in :func:`allsky.cli.prepare._frames_inputs_sha256`
    folds the same bytes in, so both stages invalidate together.
    """
    from allsky.provenance import config_subset_sha256

    return config_subset_sha256(
        cfg,
        sections=_EMBEDDING_CONFIG_SECTIONS,
        nested_fields={"video": VIDEO_TIME_FIELDS},
        content_files=_mask_content_files(cfg),
        subject="the embedding resume hash",
    )


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
        DeviceChoice | None,
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

    configure_cli_logging()

    cfg = load_prepare_config(config)
    dataset_dir = Path(cfg.output.dataset_dir)
    manifest_path = manifest if manifest is not None else dataset_dir / DATASET_MANIFEST_FILENAME
    out_dir = out if out is not None else dataset_dir / "embeddings"
    device_pref = str(device) if device is not None else cfg.embeddings.device
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
            pixel_config_sha256=_pixel_config_sha256(cfg),
        )
    except Exception as exc:  # surface any failure as a non-zero exit
        typer.echo(f"error: embedding extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach ``precompute-embeddings`` onto *app*."""
    app.command("precompute-embeddings")(precompute_embeddings)
