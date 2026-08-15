"""Reproducibility provenance: git commit, code version and content hashing.

Single home for the provenance stamps shared across the allsky pipeline, so
there is exactly one implementation of each:

- :func:`git_commit` / :func:`code_version` — the package-version + git-commit
  reproducibility stamp baked into manifest meta sidecars
  (:mod:`allsky.data.manifest`) and training checkpoints
  (:mod:`allsky.training.checkpointing`).
- :func:`content_sha256` — the container-independent manifest content hash
  (``manifest_sha256``) written by
  :func:`allsky.data.manifest.write_manifest_parquet` and re-verified by
  :func:`allsky.bundle.validate_bundle`.
- :func:`config_subset_sha256` — the resume gate every stage hashes the config
  subset its artifact depends on with (:mod:`allsky.cli.prepare`,
  :mod:`allsky.cli.embeddings`).

Pure stdlib + pandas: importing this module never pulls torch.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

from micrometeorology.common.git import run_git, source_root

__all__ = [
    "code_version",
    "config_subset_sha256",
    "content_sha256",
    "file_content_sha256",
    "git_commit",
]

_DISTRIBUTION = "labmim-micrometeorology"


def git_commit() -> str | None:
    """Current git commit hash, or None when unavailable (best-effort)."""
    # `or None` because an empty stdout is no more useful than a failed call
    # here, even though run_git keeps the two distinguishable for callers that
    # do care (see solrad_correction.utils.metadata's dirty-tree probe).
    return run_git(["rev-parse", "HEAD"], cwd=source_root()) or None


def code_version() -> dict[str, str | None]:
    """Package version plus a best-effort git commit (reproducibility stamp).

    Returns
    -------
    dict
        ``{"package_version", "git_commit"}``. Either value is ``None`` when
        it cannot be determined — the package is not installed, or the source
        tree is not a git checkout — so a stamp is always writable.
    """
    try:
        version: str | None = importlib_metadata.version(_DISTRIBUTION)
    except importlib_metadata.PackageNotFoundError:
        version = None
    return {"package_version": version, "git_commit": git_commit()}


def content_sha256(manifest: pd.DataFrame) -> str:
    """Container-independent content hash of a manifest (order-sensitive).

    The digest folds the comma-joined column names followed by the index-free
    CSV bytes, so it tracks the manifest's *content* (values and column order)
    independently of the parquet container it is stored in.  This is the
    ``manifest_sha256`` recorded in a manifest's meta sidecar and re-verified
    when a Colab bundle is validated.
    """
    digest = hashlib.sha256()
    digest.update(",".join(manifest.columns).encode("utf-8"))
    # ``lineterminator`` pinned: pandas defaults it to ``os.linesep``, which would
    # make the digest platform-dependent — a manifest hashed on Windows would never
    # match itself re-hashed on Linux, and ``validate_bundle`` would report a
    # byte-intact bundle as corrupt on exactly the cross-machine handoff it exists
    # for. "\n" is what Linux already produced, so no recorded digest moves.
    digest.update(manifest.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def file_content_sha256(path: Path) -> str:
    """Content hash of *path*, or a stable marker naming it when it is absent."""
    if not path.is_file():
        return f"absent:{path}"
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config_subset_sha256(
    config: BaseModel,
    *,
    sections: Sequence[str],
    nested_fields: Mapping[str, Sequence[str]] | None = None,
    content_files: Sequence[str | Path] = (),
    subject: str,
) -> str:
    """Order-independent content hash of a chosen subset of a config model.

    A pipeline stage resumes onto its own artifacts only while the config that
    shaped them is unchanged, and each stage depends on a different slice of the
    tree: *sections* names the sub-models taken whole, *nested_fields* the
    parents taken field by field (a section whose siblings must not invalidate
    the artifact, such as a glob that only widens the day set).

    A config value that only *points* at a file leaves the digest blind to that
    file's bytes — a mask PNG rewritten in place at the same path keeps a
    ``{path, threshold}`` section byte-identical while changing every pixel the
    stage produces — so *content_files* folds those bytes in.  An empty
    *content_files* leaves the digest exactly where it was without them, so a
    stage that references no file keeps resuming onto its existing artifacts.

    Parameters
    ----------
    config:
        The populated config model to hash a subset of.
    sections:
        Top-level field names of *config*, each dumped whole.
    nested_fields:
        ``{parent: (field, ...)}`` for parents dumped field by field.
    content_files:
        Paths, in a caller-fixed order, whose bytes shape the artifact as much
        as the config does.  A missing file hashes to a stable marker, so its
        later appearance moves the digest.
    subject:
        What the digest gates, named in the error message below.

    Returns
    -------
    str
        Hex sha256 over the canonical JSON of the subset followed by the content
        hash of each entry of *content_files*, so two configs agreeing on both
        hash alike whatever the key order in their YAML.

    Raises
    ------
    RuntimeError
        When a name is not a field of its model: pydantic drops unknown include
        keys silently, which would shrink the hash without any sign of it.
    """
    nested = dict(nested_fields or {})
    unknown = [name for name in sections if name not in type(config).model_fields]
    for parent, names in nested.items():
        child = getattr(config, parent, None)
        if not isinstance(child, BaseModel):
            unknown.append(parent)
            continue
        unknown += [f"{parent}.{name}" for name in names if name not in type(child).model_fields]
    if unknown:
        raise RuntimeError(
            f"{type(config).__name__} has no field(s) {unknown}; {subject} would stop covering them"
        )
    include: dict[str, Any] = dict.fromkeys(sections, True)
    for parent, names in nested.items():
        include[parent] = set(names)
    canonical = json.dumps(
        config.model_dump(mode="json", include=include), sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8"))
    for file in content_files:
        digest.update(file_content_sha256(Path(file)).encode("utf-8"))
    return digest.hexdigest()
