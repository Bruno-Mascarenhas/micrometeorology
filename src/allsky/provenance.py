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

from micrometeorology.common.git import run_git

__all__ = ["code_version", "content_sha256", "git_commit"]

#: Installed distribution queried for the package-version stamp.
_DISTRIBUTION = "labmim-micrometeorology"


def git_commit() -> str | None:
    """Current git commit hash, or None when unavailable (best-effort)."""
    # `or None` because an empty stdout is no more useful than a failed call
    # here, even though run_git keeps the two distinguishable for callers that
    # do care (see solrad_correction.utils.metadata's dirty-tree probe).
    return run_git(["rev-parse", "HEAD"]) or None


def code_version() -> dict[str, str | None]:
    """Package version plus a best-effort git commit (reproducibility stamp)."""
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
    digest.update(manifest.to_csv(index=False).encode("utf-8"))
    return digest.hexdigest()
