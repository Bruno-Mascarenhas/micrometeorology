"""Export a self-contained Colab bundle (tar.gz) of a prepared dataset.

:func:`export_colab_bundle` packs everything a Colab (or any offline) session
needs to train on a locally prepared dataset into a single ``tar.gz``:

- the manifest parquet and its ``.meta.json`` sidecar;
- the persisted split artifact (when present);
- the precomputed embedding shards + index + meta (optional);
- every config YAML used, plus a resolved dump of the
  :class:`~allsky.config.PrepareConfig`;
- a generated ``BUNDLE_README.md`` describing the contents and how to consume
  them on Colab.

Every member lives under a single top-level bundle directory and is stored with
a **relative POSIX** name; absolute or ``..``-escaping names are refused on both
write and read.  Members are added in a deterministic (sorted) order and the
archive is written atomically (temp file + :func:`os.replace`).

:func:`validate_bundle` re-opens an archive, lists its members and verifies the
manifest content hash against the value recorded in the sidecar meta — the same
``manifest_sha256`` :func:`allsky.data.manifest.write_manifest_parquet` writes.

Pure stdlib + pandas/pyarrow + PyYAML: importing this module never pulls torch.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

if TYPE_CHECKING:
    from collections.abc import Iterable

    from allsky.config import PrepareConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BUNDLE_README_NAME",
    "export_colab_bundle",
    "validate_bundle",
]

#: Generated readme filename inside the bundle.
BUNDLE_README_NAME = "BUNDLE_README.md"
#: Default manifest / split filenames looked up under a dataset directory.
_MANIFEST_NAME = "manifest.parquet"
_SPLIT_NAME = "splits.json"
_EMBEDDINGS_DIRNAME = "embeddings"


def _content_sha256(manifest: pd.DataFrame) -> str:
    """Container-independent content hash of a manifest (order-sensitive).

    Mirrors :func:`allsky.data.manifest._content_sha256` exactly (column names
    joined with commas, then the index-free CSV) so a bundle's manifest can be
    checked against the ``manifest_sha256`` its sidecar recorded at write time.
    """
    digest = hashlib.sha256()
    digest.update(",".join(manifest.columns).encode("utf-8"))
    digest.update(manifest.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()


def _safe_arcname(name: str) -> str:
    """Return *name* if it is a relative POSIX path, else raise.

    Rejects absolute paths and any ``..`` component so a bundle can never encode
    a path that escapes its top-level directory on extraction.
    """
    posix = PurePosixPath(name)
    if posix.is_absolute() or name.startswith("/") or ".." in posix.parts:
        raise ValueError(f"unsafe bundle member name {name!r} (absolute or contains '..')")
    return name


def export_colab_bundle(
    out_path: str | Path,
    *,
    prepare_cfg: PrepareConfig | None = None,
    manifest_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    split_path: str | Path | None = None,
    embeddings_dir: str | Path | None = None,
    config_paths: Iterable[str | Path] = (),
    include_embeddings: bool = True,
    bundle_name: str = "allsky_bundle",
) -> dict[str, Any]:
    """Pack a prepared dataset into a Colab-ready ``tar.gz`` at *out_path*.

    Provide either *prepare_cfg* (its ``output.dataset_dir`` supplies the
    default manifest / split / embeddings locations) or the explicit paths;
    explicit paths always win.  When *prepare_cfg* is given its fully resolved
    form is also written into the bundle as ``config/prepare.resolved.yaml``.

    Parameters
    ----------
    out_path:
        Destination ``.tar.gz`` (parent directories are created).
    prepare_cfg:
        Config whose ``output.dataset_dir`` roots the default paths.
    manifest_path, meta_path, split_path, embeddings_dir:
        Explicit overrides.  *meta_path* defaults to ``<manifest>.meta.json``;
        *split_path* / *embeddings_dir* default to the standard names beside the
        manifest and are skipped when absent.
    config_paths:
        Config YAML files copied verbatim into ``config/`` in the bundle.
    include_embeddings:
        When ``False`` (or the embeddings directory is missing) embedding shards
        are omitted.
    bundle_name:
        Top-level directory every member is nested under.

    Returns
    -------
    dict
        ``{"path", "size_bytes", "members"}`` — the written archive path, its
        size in bytes and the sorted list of member arcnames.

    Raises
    ------
    ValueError
        If no manifest can be resolved, the manifest/meta is missing on disk, or
        a constructed member name would be unsafe.
    """
    manifest_file, meta_file, split_file, emb_dir = _resolve_sources(
        prepare_cfg=prepare_cfg,
        manifest_path=manifest_path,
        meta_path=meta_path,
        split_path=split_path,
        embeddings_dir=embeddings_dir,
    )
    if not manifest_file.exists():
        raise ValueError(f"manifest not found: {manifest_file}")
    if not meta_file.exists():
        raise ValueError(f"manifest meta sidecar not found: {meta_file}")

    root = PurePosixPath(bundle_name)
    # arcname -> source Path; generated text members are collected separately.
    file_members: dict[str, Path] = {
        _safe_arcname((root / _MANIFEST_NAME).as_posix()): manifest_file,
        _safe_arcname((root / f"{_MANIFEST_NAME}.meta.json").as_posix()): meta_file,
    }
    if split_file is not None and split_file.exists():
        file_members[_safe_arcname((root / _SPLIT_NAME).as_posix())] = split_file

    embedded_files = 0
    if include_embeddings and emb_dir is not None and emb_dir.is_dir():
        for path in sorted(p for p in emb_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(emb_dir).as_posix()
            arc = _safe_arcname((root / _EMBEDDINGS_DIRNAME / rel).as_posix())
            file_members[arc] = path
            embedded_files += 1

    for cfg_path in config_paths:
        src = Path(cfg_path)
        arc = _safe_arcname((root / "config" / src.name).as_posix())
        file_members[arc] = src

    # Generated text members (resolved config dump + readme).
    text_members: dict[str, str] = {}
    if prepare_cfg is not None:
        resolved = yaml.safe_dump(prepare_cfg.model_dump(mode="json"), sort_keys=True)
        text_members[_safe_arcname((root / "config" / "prepare.resolved.yaml").as_posix())] = (
            resolved
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    all_names = sorted(
        [*file_members, *text_members, _safe_arcname((root / BUNDLE_README_NAME).as_posix())]
    )
    readme = _render_readme(bundle_name=bundle_name, members=all_names, meta=meta)
    text_members[_safe_arcname((root / BUNDLE_README_NAME).as_posix())] = readme

    members = sorted([*file_members, *text_members])
    _write_tar_atomic(out_path, file_members=file_members, text_members=text_members, order=members)

    out = Path(out_path)
    size = out.stat().st_size
    logger.info(
        "export_colab_bundle: wrote %s (%d members, %d embedding files, %d bytes)",
        out,
        len(members),
        embedded_files,
        size,
    )
    return {"path": str(out), "size_bytes": size, "members": members}


def validate_bundle(path: str | Path) -> dict[str, Any]:
    """List a bundle's members and verify its manifest content hash.

    Reads members through :meth:`tarfile.TarFile.extractfile` only (never
    extracting to disk) and rejects any absolute or ``..`` member name.  The
    manifest parquet is read back and re-hashed with :func:`_content_sha256`;
    the result is compared against ``manifest_sha256`` in the sidecar meta.

    Returns
    -------
    dict
        ``{"members", "manifest_member", "manifest_sha256", "expected_sha256",
        "manifest_sha256_ok"}``.

    Raises
    ------
    ValueError
        If a member name is unsafe, or the manifest / meta member is missing.
    """
    with tarfile.open(path, "r:gz") as tar:
        names = [member.name for member in tar.getmembers()]
        for name in names:
            _safe_arcname(name)

        manifest_member = _find_member(names, _MANIFEST_NAME)
        meta_member = _find_member(names, f"{_MANIFEST_NAME}.meta.json")
        if manifest_member is None:
            raise ValueError(f"bundle {path} has no {_MANIFEST_NAME} member")
        if meta_member is None:
            raise ValueError(f"bundle {path} has no manifest meta sidecar member")

        manifest_bytes = _read_member(tar, manifest_member)
        meta = json.loads(_read_member(tar, meta_member).decode("utf-8"))

    manifest = pd.read_parquet(io.BytesIO(manifest_bytes))
    recomputed = _content_sha256(manifest)
    expected = meta.get("manifest_sha256")
    ok = expected is not None and recomputed == expected
    if not ok:
        logger.warning(
            "validate_bundle: manifest sha256 mismatch (recomputed %s, expected %s)",
            recomputed[:12],
            str(expected)[:12],
        )
    return {
        "members": sorted(names),
        "manifest_member": manifest_member,
        "manifest_sha256": recomputed,
        "expected_sha256": expected,
        "manifest_sha256_ok": ok,
    }


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _resolve_sources(
    *,
    prepare_cfg: PrepareConfig | None,
    manifest_path: str | Path | None,
    meta_path: str | Path | None,
    split_path: str | Path | None,
    embeddings_dir: str | Path | None,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Resolve manifest/meta/split/embeddings sources from explicit args or *cfg*."""
    dataset_dir = Path(prepare_cfg.output.dataset_dir) if prepare_cfg is not None else None

    if manifest_path is not None:
        manifest_file = Path(manifest_path)
    elif dataset_dir is not None:
        manifest_file = dataset_dir / _MANIFEST_NAME
    else:
        raise ValueError("no manifest_path given and no prepare_cfg to derive one from")

    meta_file = (
        Path(meta_path)
        if meta_path is not None
        else manifest_file.with_name(f"{manifest_file.name}.meta.json")
    )

    if split_path is not None:
        split_file: Path | None = Path(split_path)
    elif dataset_dir is not None:
        split_file = dataset_dir / _SPLIT_NAME
    else:
        split_file = manifest_file.with_name(_SPLIT_NAME)

    if embeddings_dir is not None:
        emb_dir: Path | None = Path(embeddings_dir)
    elif dataset_dir is not None:
        emb_dir = dataset_dir / _EMBEDDINGS_DIRNAME
    else:
        emb_dir = manifest_file.with_name(_EMBEDDINGS_DIRNAME)

    return manifest_file, meta_file, split_file, emb_dir


def _write_tar_atomic(
    out_path: str | Path,
    *,
    file_members: dict[str, Path],
    text_members: dict[str, str],
    order: list[str],
) -> None:
    """Write the gzip tar to a temp file then :func:`os.replace` it into place."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    try:
        with tarfile.open(tmp, "w:gz") as tar:
            for arc in order:
                if arc in file_members:
                    tar.add(file_members[arc], arcname=arc)
                else:
                    data = text_members[arc].encode("utf-8")
                    info = tarfile.TarInfo(name=arc)
                    info.size = len(data)
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def _find_member(names: list[str], suffix: str) -> str | None:
    """First member whose basename equals *suffix* (deterministic by sort)."""
    matches = sorted(n for n in names if PurePosixPath(n).name == suffix)
    return matches[0] if matches else None


def _read_member(tar: tarfile.TarFile, name: str) -> bytes:
    """Read a member's bytes via ``extractfile`` (never touches disk)."""
    handle = tar.extractfile(name)
    if handle is None:
        raise ValueError(f"bundle member {name!r} is not a regular file")
    with handle:
        return handle.read()


def _render_readme(*, bundle_name: str, members: list[str], meta: dict[str, Any]) -> str:
    """Render the ``BUNDLE_README.md`` describing the archive and how to use it."""
    member_lines = "\n".join(f"- `{name}`" for name in members)
    dataset_version = meta.get("dataset_version", "?")
    row_count = meta.get("row_count", "?")
    feature_set = meta.get("feature_set", "?")
    manifest_sha = str(meta.get("manifest_sha256", "?"))
    return f"""# All-sky dataset bundle

Self-contained export of a locally prepared all-sky dataset (manifest v{dataset_version}).

## Contents

Every path below is relative to the `{bundle_name}/` directory in this archive.

{member_lines}

## Dataset summary

- dataset_version: `{dataset_version}`
- rows: `{row_count}`
- feature_set: `{feature_set}`
- manifest_sha256: `{manifest_sha}`

## Consuming this bundle on Colab

Colab runtimes may ship a Python older than this package requires, so provision
a matching interpreter with `uv` rather than relying on the system Python:

```bash
pip install uv
uv python install 3.14
# unpack the bundle (members are all relative — safe to extract anywhere)
tar -xzf {bundle_name}.tar.gz
uv run --python 3.14 python - <<'PY'
import pandas as pd
manifest = pd.read_parquet("{bundle_name}/{_MANIFEST_NAME}")
print(manifest.shape)
print(manifest.head())
PY
```

Image paths in the manifest are **relative POSIX** paths resolved against the
`{bundle_name}/` directory (the data root); resolve them with
`allsky.data.contracts.resolve(image_path, data_root)`.  If embedding shards are
included they live under `{bundle_name}/{_EMBEDDINGS_DIRNAME}/` with their index
and `embeddings.meta.json`.

Verify integrity after unpacking with
`allsky.bundle.validate_bundle("{bundle_name}.tar.gz")`, which re-hashes the
manifest and checks it against `manifest_sha256` above.
"""
