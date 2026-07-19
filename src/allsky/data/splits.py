"""Deterministic day-level train/val/test splits with a persisted artifact.

Splitting is by **calendar day** (``day_id``), never by row, so no day straddles
two splits — the only way to keep image-sensor samples from the same day out of
both training and evaluation.  :func:`create_day_splits` is deterministic in
``(day_ids, fractions, seed)``; the resulting :class:`DaySplit` carries a
content-addressed ``split_id`` (sha256 over the canonical assignment + params).

The artifact is written once and never silently regenerated: saving a different
assignment over an existing file raises :class:`SplitExistsError` unless
``force=True``.  Every construct/load path runs a leakage self-check.

Pure stdlib/numpy; importing this module never pulls torch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from allsky.data.contracts import DATASET_VERSION

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "SPLIT_NAMES",
    "DaySplit",
    "SplitExistsError",
    "check_split_leakage",
    "create_day_splits",
    "load_split_artifact",
    "save_split_artifact",
]

#: Canonical split names.
SPLIT_NAMES = ("train", "val", "test")


class SplitExistsError(FileExistsError):
    """Raised when saving a *different* split over an existing artifact without force."""


@dataclass(frozen=True)
class DaySplit:
    """A deterministic day-level partition into train/val/test.

    Attributes
    ----------
    assignment:
        ``day_id -> split`` map (each day in exactly one split).
    seed, val_fraction, test_fraction:
        The parameters that produced the partition (recorded for reproducibility
        and folded into :attr:`split_id`).
    created_at:
        ISO-8601 UTC creation timestamp (excluded from :attr:`split_id`).
    """

    assignment: dict[str, str]
    seed: int
    val_fraction: float
    test_fraction: float
    created_at: str

    @property
    def split_id(self) -> str:
        """Content hash over the assignment + params (creation time excluded)."""
        return _split_id(self.assignment, self.seed, self.val_fraction, self.test_fraction)

    def days_for(self, split: str) -> list[str]:
        """Sorted ``day_id`` list assigned to *split*."""
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return sorted(day for day, name in self.assignment.items() if name == split)

    def check_leakage(self) -> None:
        """Assert the three split day-sets are pairwise disjoint (self-check)."""
        check_split_leakage(self.days_for("train"), self.days_for("val"), self.days_for("test"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable artifact form."""
        return {
            "dataset_version": DATASET_VERSION,
            "split_id": self.split_id,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "created_at": self.created_at,
            "assignment": dict(sorted(self.assignment.items())),
            "counts": {name: len(self.days_for(name)) for name in SPLIT_NAMES},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DaySplit:
        """Inverse of :meth:`to_dict`; verifies the stored ``split_id``."""
        assignment = {str(k): str(v) for k, v in payload["assignment"].items()}
        split = cls(
            assignment=assignment,
            seed=int(payload["seed"]),
            val_fraction=float(payload["val_fraction"]),
            test_fraction=float(payload["test_fraction"]),
            created_at=str(payload.get("created_at", "")),
        )
        stored = payload.get("split_id")
        if stored is not None and stored != split.split_id:
            raise ValueError(
                f"split artifact is corrupt: stored split_id {stored!r} does not match "
                f"the recomputed {split.split_id!r}"
            )
        split.check_leakage()
        return split


def create_day_splits(
    day_ids: Sequence[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> DaySplit:
    """Deterministically partition unique *day_ids* into train/val/test.

    Days are de-duplicated, sorted, then shuffled with a seeded RNG.  Sizes are
    ``floor(n * fraction)`` for test then val, but each non-zero fraction is
    guaranteed at least one day when the day count allows; the remainder is
    train.  Deterministic in ``(sorted unique day_ids, fractions, seed)``.

    Raises
    ------
    ValueError
        If a fraction is out of ``[0, 1)``, their sum is >= 1, there are no
        days, or there are too few days to honour the requested non-zero splits.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            f"val_fraction + test_fraction must be < 1, got {val_fraction + test_fraction}"
        )

    unique_days = sorted({str(d) for d in day_ids})
    n = len(unique_days)
    if n == 0:
        raise ValueError("no day_ids provided")

    n_test = _split_size(n, test_fraction)
    n_val = _split_size(n, val_fraction)
    if n_test + n_val >= n:
        raise ValueError(
            f"too few days ({n}) for val_fraction={val_fraction}, "
            f"test_fraction={test_fraction}: no days left for train"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    shuffled = [unique_days[i] for i in order]

    test_days = shuffled[:n_test]
    val_days = shuffled[n_test : n_test + n_val]
    train_days = shuffled[n_test + n_val :]

    assignment: dict[str, str] = {}
    for day in train_days:
        assignment[day] = "train"
    for day in val_days:
        assignment[day] = "val"
    for day in test_days:
        assignment[day] = "test"

    split = DaySplit(
        assignment=assignment,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        created_at=datetime.now(UTC).isoformat(),
    )
    split.check_leakage()
    return split


def save_split_artifact(split: DaySplit, path: str | Path, *, force: bool = False) -> Path:
    """Atomically write *split* to *path* as JSON; guard against silent regeneration.

    If *path* already holds a split with a **different** ``split_id`` and
    *force* is False, :class:`SplitExistsError` is raised.  An identical
    existing split is left untouched (idempotent).

    Returns the artifact path.
    """
    out = Path(path)
    if out.exists() and not force:
        existing = load_split_artifact(out)
        if existing.split_id == split.split_id:
            return out
        raise SplitExistsError(
            f"a different split already exists at {out} "
            f"(existing split_id {existing.split_id[:12]}, new {split.split_id[:12]}); "
            "pass force=True to overwrite"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(split.to_dict(), handle, indent=2, ensure_ascii=False)
    os.replace(tmp, out)
    return out


def load_split_artifact(path: str | Path) -> DaySplit:
    """Load and verify a split artifact (checks ``split_id`` and leakage)."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return DaySplit.from_dict(payload)


def check_split_leakage(
    train_days: Sequence[str], val_days: Sequence[str], test_days: Sequence[str]
) -> None:
    """Assert the three day collections are pairwise disjoint.

    Raises
    ------
    ValueError
        Naming the days shared between any two splits.
    """
    train, val, test = set(train_days), set(val_days), set(test_days)
    overlaps = {
        "train&val": sorted(train & val),
        "train&test": sorted(train & test),
        "val&test": sorted(val & test),
    }
    leaked = {pair: days for pair, days in overlaps.items() if days}
    if leaked:
        raise ValueError(f"day-level split leakage detected: {leaked}")


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _split_size(n: int, fraction: float) -> int:
    """Day count for a fraction: ``floor(n*fraction)``, >= 1 when fraction > 0."""
    if fraction <= 0.0:
        return 0
    return max(1, int(np.floor(n * fraction)))


def _split_id(
    assignment: dict[str, str], seed: int, val_fraction: float, test_fraction: float
) -> str:
    """sha256 over the canonical (sorted) assignment and split parameters."""
    canonical = json.dumps(
        {
            "assignment": dict(sorted(assignment.items())),
            "seed": seed,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
