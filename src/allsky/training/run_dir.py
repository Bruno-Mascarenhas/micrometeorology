"""The life cycle of one training run's directory, in one place.

A run directory is a small state machine the engine drives and nothing else
owns: it is created, possibly resumed into, rotated when a previous run or a
previous monitor left artifacts behind, appended to each epoch, and truncated
back when a resume rewinds the history. Those transitions were spread across
the engine, where the order between them is easy to break and hard to see.

Two invariants hold across all of them:

- **A surviving ``best.ckpt`` is never overwritten in place.** It is the one
  artifact in the directory nothing can recompute, so every path that would
  replace it rotates it aside first, onto a destination
  :func:`free_rotation_destination` guarantees is unused.
- **``metrics.csv`` and ``metrics.json`` describe the weights beside them.**
  A resume that rewinds the epoch counter truncates both, and a fresh run into
  a used directory rotates them together with ``last.ckpt``, so a reader never
  finds metrics for weights that no longer exist.

Imports only the checkpoint filenames and the experiment config, so the engine
depends on this and not the other way round.
"""

import csv
import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from allsky.config import ExperimentConfig
from allsky.training.checkpointing import BEST_CHECKPOINT, LAST_CHECKPOINT
from labmim_core.atomic import atomic_write, atomic_write_json

logger = logging.getLogger(__name__)

__all__ = [
    "MONITOR_CHANGE_SUFFIX",
    "STALE_RUN_SUFFIX",
    "append_csv",
    "csv_fields",
    "free_rotation_destination",
    "reset_stale_run_artifacts",
    "resolve_resume_path",
    "rewrite_csv",
    "rotate_best",
    "truncate_metrics",
]

#: Destination suffix when a monitor change invalidates the stored best. Distinct
#: from STALE_RUN_SUFFIX on purpose: sharing one name would let a fresh run replace
#: the only surviving copy of the previous monitor's weights, the one artifact in a
#: run directory nothing can recompute.
MONITOR_CHANGE_SUFFIX = ".stale-monitor"

#: Destination suffix when a fresh run supersedes a previous run's best.
STALE_RUN_SUFFIX = ".stale"


def resolve_resume_path(resume: str | Path | None, run_dir: Path) -> Path | None:
    """Resolve the checkpoint to resume from (``"auto"`` finds ``last.ckpt``)."""
    if resume is None:
        return None
    if isinstance(resume, str) and resume == "auto":
        candidate = run_dir / LAST_CHECKPOINT
        if candidate.exists():
            return candidate
        logger.info("resume='auto' but %s does not exist; starting fresh", candidate)
        return None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return path


def rotate_best(run_dir: Path, *, suffix: str, reason: str) -> None:
    """Move an existing ``best.ckpt`` aside instead of letting it be overwritten.

    Whatever invalidates the stored best — a changed monitor, a fresh run into
    the same directory — the next improving epoch rewrites ``best.ckpt``, and
    those weights are the one artifact in a run directory nothing can
    recompute. Every destination goes through
    :func:`free_rotation_destination`, so a second rotation onto the same
    suffix gets its own numbered name rather than deleting what the first
    preserved.

    Parameters
    ----------
    run_dir:
        Run directory holding ``best.ckpt``. A missing file is a silent return:
        both callers can reach here with nothing to rotate.
    suffix:
        Destination suffix, :data:`MONITOR_CHANGE_SUFFIX` or
        :data:`STALE_RUN_SUFFIX` — never shared, for the reason recorded there.
    reason:
        Parenthetical for the warning, saying what invalidated the checkpoint.
    """
    path = run_dir / BEST_CHECKPOINT
    if not path.exists():
        return
    backup = free_rotation_destination(path.with_name(f"{BEST_CHECKPOINT}{suffix}"))
    os.replace(path, backup)
    logger.warning("rotated %s aside to %s (%s)", path, backup.name, reason)


def free_rotation_destination(preferred: Path) -> Path:
    """*preferred* if free, else the first ``<preferred>.<n>`` (n from 2) that is.

    A rotation destination is never overwritten: the weights it holds were selected
    under a monitor no later run recomputes.
    """
    destination = preferred
    ordinal = 2
    while destination.exists():
        destination = preferred.with_name(f"{preferred.name}.{ordinal}")
        ordinal += 1
    return destination


def csv_fields(cfg: ExperimentConfig) -> list[str]:
    """Stable, config-derived CSV column order (identical across resumes).

    ``lr_backbone`` is always emitted rather than made to depend on the optimizer's
    group count: the header must not change mid-run, and a run without a separate
    backbone rate simply leaves the cell blank.
    """
    fields = ["epoch", "lr", "lr_backbone"]
    for split in ("train", "val"):
        fields.append(f"{split}_loss")
        if cfg.targets.dhi.enabled:
            fields += [f"{split}_loss_dhi", f"{split}_dhi_mae"]
        if cfg.targets.kindex.enabled:
            fields += [f"{split}_loss_kindex", f"{split}_kindex_mae"]
        if cfg.targets.sky.enabled:
            fields += [f"{split}_loss_sky", f"{split}_sky_acc"]
        if cfg.targets.cloud_fraction.enabled:
            fields.append(f"{split}_loss_cloud_fraction")
    return fields


def append_csv(path: Path, fields: list[str], row: Mapping[str, Any]) -> None:
    """Append *row* to the metrics CSV (writing the header only when new)."""
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def rewrite_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomically rewrite the metrics CSV as *fields* header + *rows*."""

    def _write(tmp: Path) -> None:
        with open(tmp, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    atomic_write(path, _write)


def truncate_metrics(run_dir: Path, fields: list[str], resumed_epoch: int) -> list[dict[str, Any]]:
    """Drop metrics rows past *resumed_epoch* and rewrite CSV + JSON from the rest.

    ``metrics.csv``/``metrics.json`` are written before ``last.ckpt`` each epoch,
    so a crash in that gap can leave rows for an epoch the resumed checkpoint never
    completed.  Only rows with ``epoch <= resumed_epoch`` (completed epochs) are
    kept; both files are atomically rewritten from them and the truncated history
    is returned for the loop to keep appending to.  ``metrics.json`` is the source
    of truth (it is always present once a checkpoint exists); if it is somehow
    absent the files are left untouched rather than risking data loss.
    """
    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_json.exists():
        if metrics_csv.exists():
            logger.warning(
                "resume: metrics.json is missing but metrics.csv is present; leaving the "
                "metrics files untouched (cannot safely truncate without the JSON history)"
            )
        return []
    loaded = json.loads(metrics_json.read_text(encoding="utf-8"))
    history = [row for row in loaded if int(row.get("epoch", 0)) <= resumed_epoch]
    dropped = len(loaded) - len(history)
    if dropped:
        logger.info("resume: dropped %d stale metrics row(s) past epoch %d", dropped, resumed_epoch)
    rewrite_csv(metrics_csv, fields, history)
    atomic_write_json(metrics_json, history)
    return history


def reset_stale_run_artifacts(run_dir: Path) -> None:
    """Rotate a previous run's metrics and checkpoints aside on a fresh run.

    A fresh (non-resume) run into a reused run directory must not append to the
    previous run's metrics.  Each stale file is renamed to
    ``<name>.stale`` (replacing an older backup) rather than deleted, so the prior
    run's numbers are still recoverable; the fresh run then re-creates the files
    from scratch.

    ``last.ckpt`` is rotated with them: it is overwritten at the end of epoch 1,
    which would leave the preserved metrics describing weights that no longer
    exist anywhere.  ``best.ckpt`` is *not* rotated here —
    :func:`rotate_best` does it at the first epoch that improves, so a fresh run
    that dies before producing a replacement leaves the previous best where it is
    instead of emptying the directory.

    Only these three names are rotated, and only onto their own ``.stale``
    destination: a ``best.ckpt.stale-monitor`` rotated under an earlier monitor is
    left untouched.
    """
    for name in ("metrics.csv", "metrics.json", LAST_CHECKPOINT):
        path = run_dir / name
        if path.exists():
            backup = path.with_name(f"{name}{STALE_RUN_SUFFIX}")
            os.replace(path, backup)
            logger.warning(
                "fresh run: rotated stale %s aside to %s (a previous run wrote it)",
                path,
                backup.name,
            )
