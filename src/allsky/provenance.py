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

Pure stdlib + pandas: importing this module never pulls torch.
"""

import hashlib
from importlib import metadata as importlib_metadata

import pandas as pd

from micrometeorology.common.git import run_git, source_root

__all__ = ["code_version", "content_sha256", "git_commit"]

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
