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
from typing import Annotated

import typer

from allsky.config import VIDEO_TIME_FIELDS, PrepareConfig

logger = logging.getLogger("allsky.embeddings")


def _configure_logging() -> None:
    """Attach a stderr handler at INFO once, so progress is visible in the CLI."""
    root = logging.getLogger("allsky")
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO)


#: Sections deciding the pixels a stored vector was computed from, as opposed to
#: the encoder that computed it: what ``prepare-local`` bakes into the JPEG.
_PIXEL_CONFIG_SECTIONS = ("mask", "crop", "resize")

#: Sections a stored vector depends on whole: the encoder itself and the
#: preprocessing ``prepare-local`` bakes into the JPEG the encoder reads.
_EMBEDDING_CONFIG_SECTIONS = ("embeddings", *_PIXEL_CONFIG_SECTIONS)


def _mask_content_files(cfg: PrepareConfig) -> tuple[str, ...]:
    """The files whose bytes shape the encoded pixels without being config values."""
    return () if cfg.mask.path is None else (cfg.mask.path,)


def _pixel_config_sha256(cfg: PrepareConfig) -> str:
    """Content hash of the config deciding which pixels a stored vector encodes.

    Recorded in ``embeddings.meta.json`` beside the full resume digest, because
    the two answer different questions.  The full digest moves whenever its own
    formula widens, which says nothing about whether a single pixel changed; this
    one moves only when the frames themselves would.  That is what lets
    :func:`allsky.embeddings.extract._check_resume_compatible` migrate a store
    across a formula change without having to take the encoder's word for the
    preprocessing behind it.
    """
    from allsky.provenance import config_subset_sha256

    return config_subset_sha256(
        cfg,
        sections=_PIXEL_CONFIG_SECTIONS,
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


def _legacy_config_sha256(cfg: PrepareConfig) -> str:
    """The resume digest under the formula every store on disk was stamped with.

    :func:`_config_sha256` widened both the covered sections and the JSON
    encoding, so every previously extracted store records a digest no current
    config can reproduce and would be refused on resume — a full re-encode of a
    dataset in which neither the encoder nor a single JPEG changed.  This
    reproduces the old formula verbatim (python-mode dump, ``default=str``,
    default separators) so
    :func:`allsky.embeddings.extract._check_resume_compatible` can recognise
    such a store — and migrate it in place when, and only when, the store also
    records the pixel provenance that formula never covered.
    """
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
            legacy_config_sha256=_legacy_config_sha256(cfg),
            pixel_config_sha256=_pixel_config_sha256(cfg),
        )
    except Exception as exc:  # surface any failure as a non-zero exit
        typer.echo(f"error: embedding extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(json.dumps(summary, indent=2, default=str))


def register(app: typer.Typer) -> None:
    """Attach ``precompute-embeddings`` onto *app*."""
    app.command("precompute-embeddings")(precompute_embeddings)
