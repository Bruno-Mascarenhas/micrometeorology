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

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from allsky.config import PrepareConfig

logger = logging.getLogger("allsky.embeddings")


def _configure_logging() -> None:
    """Attach a stderr handler at INFO once, so progress is visible in the CLI."""
    root = logging.getLogger("allsky")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def _config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the embeddings config section (stable, order-independent)."""
    canonical = json.dumps(cfg.embeddings.model_dump(), sort_keys=True, default=str)
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
    """Precompute visual embeddings for a dataset manifest (DINOv2 or fake)."""
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
    # Manifest image paths are relative POSIX against the manifest's directory.
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
