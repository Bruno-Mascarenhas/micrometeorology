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

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from allsky.data.contracts import DATASET_VERSION
from allsky.provenance import canonical_config_json
from labmim_core.atomic import atomic_write

__all__ = [
    "SPLIT_NAMES",
    "DaySplit",
    "SplitExistsError",
    "check_split_leakage",
    "create_day_splits",
    "load_split_artifact",
    "save_split_artifact",
]

SPLIT_NAMES = ("train", "val", "test")

#: How the day-level partition is drawn.  ``chronological`` puts the earliest
#: days in train and the latest in test, separated by :data:`DEFAULT_GAP_DAYS`
#: dropped days; ``random`` interleaves them, which is only defensible for
#: instantaneous estimation and never for forecasting.
SplitStrategy = Literal["chronological", "random"]

#: Days discarded between consecutive splits so no training day sits within a
#: day of an evaluation day. Sky state is strongly autocorrelated across a
#: midnight boundary, so adjacent days are close to duplicates.
DEFAULT_GAP_DAYS = 1


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
    strategy:
        How the partition was drawn (see :data:`SplitStrategy`).
    gap_days:
        Days dropped at each chronological boundary; ``0`` for a random
        partition, which has no boundary to guard.
    """

    assignment: dict[str, str]
    seed: int
    val_fraction: float
    test_fraction: float
    created_at: str
    strategy: SplitStrategy = "chronological"
    gap_days: int = DEFAULT_GAP_DAYS

    @property
    def split_id(self) -> str:
        """Content hash over the assignment + params (creation time excluded)."""
        return _split_id(
            self.assignment,
            self.seed,
            self.val_fraction,
            self.test_fraction,
            self.strategy,
            self.gap_days,
        )

    def days_for(self, split: str) -> list[str]:
        """Sorted ``day_id`` list assigned to *split*.

        Raises
        ------
        ValueError
            If *split* is not one of :data:`SPLIT_NAMES`.
        """
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
        return sorted(day for day, name in self.assignment.items() if name == split)

    def check_leakage(self) -> None:
        """Assert the three split day-sets are pairwise disjoint (self-check).

        Raises
        ------
        ValueError
            Naming every day that landed in more than one split.
        """
        check_split_leakage(self.days_for("train"), self.days_for("val"), self.days_for("test"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable artifact form."""
        return {
            "dataset_version": DATASET_VERSION,
            "split_id": self.split_id,
            "seed": self.seed,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "strategy": self.strategy,
            "gap_days": self.gap_days,
            "created_at": self.created_at,
            "assignment": dict(sorted(self.assignment.items())),
            "counts": {name: len(self.days_for(name)) for name in SPLIT_NAMES},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DaySplit:
        """Inverse of :meth:`to_dict`; verifies the stored ``split_id``.

        Raises
        ------
        KeyError
            If *payload* is missing a required key.
        ValueError
            If the stored ``split_id`` disagrees with the one recomputed from
            the assignment (a corrupt artifact), if the strategy name is one
            this build does not implement, or if the assignment leaks a day
            across splits.
        """
        assignment = {str(k): str(v) for k, v in payload["assignment"].items()}
        split = cls(
            assignment=assignment,
            seed=int(payload["seed"]),
            val_fraction=float(payload["val_fraction"]),
            test_fraction=float(payload["test_fraction"]),
            created_at=str(payload.get("created_at", "")),
            strategy=_split_strategy(payload["strategy"]),
            gap_days=int(payload["gap_days"]),
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
    *,
    strategy: SplitStrategy = "chronological",
    gap_days: int = DEFAULT_GAP_DAYS,
) -> DaySplit:
    """Deterministically partition unique *day_ids* into train/val/test.

    Days are de-duplicated and sorted. Sizes are ``floor(n * fraction)`` for test
    then val, but each non-zero fraction is guaranteed at least one day when the
    day count allows; the remainder is train.

    Parameters
    ----------
    day_ids:
        Local calendar days (``YYYY-MM-DD``); duplicates are collapsed.
    val_fraction, test_fraction:
        Share of days for validation and test; each in ``[0, 1)`` and summing
        below 1.
    seed:
        Seed for ``strategy="random"``. Recorded either way so the artifact
        describes what produced it.
    strategy:
        ``"chronological"`` (default) assigns the earliest days to train and the
        latest to test, so no evaluation day precedes a training day.
        ``"random"`` interleaves them: legitimate for estimating a quantity from
        a simultaneous image, indefensible for forecasting, and never the
        default because that error is invisible in the metrics.
    gap_days:
        Days dropped from every split at each chronological boundary. Ignored by
        ``strategy="random"``, where no boundary exists to guard.

    Returns
    -------
    DaySplit
        The partition, its parameters and a content-addressed ``split_id``.

    Raises
    ------
    ValueError
        If a fraction is out of ``[0, 1)``, their sum is >= 1, *gap_days* is
        negative, there are no days, or there are too few days to honour the
        requested non-zero splits (the gap consumes days too).
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in [0, 1), got {test_fraction}")
    if gap_days < 0:
        raise ValueError(f"gap_days must be >= 0, got {gap_days}")
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

    if strategy == "random":
        rng = np.random.default_rng(seed)
        shuffled = [unique_days[i] for i in rng.permutation(n)]
        test_days = shuffled[:n_test]
        val_days = shuffled[n_test : n_test + n_val]
        train_days = shuffled[n_test + n_val :]
    else:
        train_days, val_days, test_days = _chronological_blocks(
            unique_days, n_val=n_val, n_test=n_test, gap_days=gap_days
        )

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
        strategy=strategy,
        gap_days=gap_days if strategy == "chronological" else 0,
    )
    split.check_leakage()
    return split


def save_split_artifact(split: DaySplit, path: str | Path, *, force: bool = False) -> Path:
    """Atomically write *split* to *path* as JSON; guard against silent regeneration.

    An identical existing split is left untouched (idempotent), so re-running a
    pipeline never rewrites the artifact a checkpoint was trained against.

    Parameters
    ----------
    split:
        The partition to persist, serialized with :meth:`DaySplit.to_dict`.
    path:
        Destination JSON path.
    force:
        Overwrite an existing artifact whose ``split_id`` differs.

    Returns
    -------
    pathlib.Path
        The artifact path.

    Raises
    ------
    SplitExistsError
        If *path* already holds a split with a **different** ``split_id`` and
        *force* is False.
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

    def write_payload(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(split.to_dict(), handle, indent=2, ensure_ascii=False)

    return atomic_write(out, write_payload)


def load_split_artifact(path: str | Path) -> DaySplit:
    """Load and verify a split artifact (checks ``split_id`` and leakage).

    Raises
    ------
    ValueError
        For any of the corruption modes listed on :meth:`DaySplit.from_dict`.
    """
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


def _split_size(n: int, fraction: float) -> int:
    """Day count for a fraction: ``floor(n*fraction)``, >= 1 when fraction > 0."""
    if fraction <= 0.0:
        return 0
    return max(1, int(np.floor(n * fraction)))


def _split_strategy(value: Any) -> SplitStrategy:
    """Read a strategy name off an artifact, refusing one this build cannot honour."""
    name = str(value)
    if name not in ("chronological", "random"):
        raise ValueError(
            f"split artifact declares strategy {name!r}, which this build does not implement"
        )
    return "chronological" if name == "chronological" else "random"


def _chronological_blocks(
    days: list[str], *, n_val: int, n_test: int, gap_days: int
) -> tuple[list[str], list[str], list[str]]:
    """Cut sorted *days* into train | gap | val | gap | test, oldest first.

    The gap days belong to no split: dropping them is what stops the last
    training day and the first validation day from being consecutive, which for
    a sky record makes them near-duplicates of each other.
    """
    needed = n_val + n_test + (2 * gap_days if n_val and n_test else gap_days)
    if needed >= len(days):
        raise ValueError(
            f"{len(days)} day(s) cannot carry val={n_val}, test={n_test} and "
            f"gap_days={gap_days}: no days left for train"
        )
    test_days = days[len(days) - n_test :] if n_test else []
    remaining = days[: len(days) - n_test - (gap_days if n_test else 0)]
    val_days = remaining[len(remaining) - n_val :] if n_val else []
    train_days = remaining[: len(remaining) - n_val - (gap_days if n_val else 0)]
    return train_days, val_days, test_days


def _split_id(
    assignment: dict[str, str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
    strategy: str,
    gap_days: int,
) -> str:
    """sha256 over the canonical (sorted) assignment and split parameters."""
    canonical = canonical_config_json(
        {
            "assignment": dict(sorted(assignment.items())),
            "seed": seed,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
            "strategy": strategy,
            "gap_days": gap_days,
        }
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
