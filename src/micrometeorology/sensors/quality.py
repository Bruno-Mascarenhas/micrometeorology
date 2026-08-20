"""Statistical quality control: the faults a range gate structurally cannot see.

The range gate in :func:`~micrometeorology.sensors.ingestion.apply_physical_limits`
rejects the impossible.  It cannot reject the improbable, and that is where this
archive's real faults live: 997 hPa is a perfectly possible pressure, so a
barometer that drops 17 hPa for one five-minute sample and returns passes every
bound and lands in the published hourly mean.  Measured on this archive, 626 such
excursions survived the gate and shifted 533 published hours by up to 4.26 hPa.

Two tests are implemented, and the distinction between them is what each can see:

- an **excursion** is a value that leaves the series and comes back within a
  sample or two.  The return is the discriminator, not the size of the jump: a
  plain step test at 1.5 degC flags 305 temperature samples of which 50% coincide
  with rain, against a 6.1% archive baseline, because a convective cold pool IS a
  large step; requiring the series to recover to its previous level within two
  samples drops that coincidence to 16%, because a cold pool does not recover.
- a **persistent run** is a value that stops moving.  A railed sensor reads inside
  every bound forever, and only the absence of variation gives it away.

Both mask to NaN and neither interpolates: a rejected sample becomes missing, and
the discontinuity stays explicit for whatever reads the frame next.

Thresholds are configuration, not constants here, and each carries in
``configs/micromet/default.yaml`` the measurement or citation it came from.  They
are LOCAL: the WMO minimum-variability forms flag 3.6% of one temperature channel
and 10.1% of one humidity channel at this site, because a damped maritime tropical
climate genuinely holds still longer than the continental stations those criteria
were written for.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from micrometeorology.common.config import SensorPersistenceLimit, SensorStepLimit

#: Longest hop between two samples still differenced against each other, as a
#: multiple of the sampling interval: one masked or missing sample between them is
#: tolerated, two are not. Deliberately NOT tied to ``max_len``, which would let
#: the tolerated hop grow with the excursion length, against thresholds calibrated
#: on five-minute variability.
MAX_ADJACENT_INTERVALS = 2

#: Sampling interval a first difference is only meaningful across.  Comparing
#: index spacing against this as a ``Timedelta`` is deliberate: the archive index
#: carries MICROSECOND resolution, so an integer nanosecond conversion silently
#: matches nothing and every test quietly becomes a no-op that reports zero
#: removals — a passing run that checked nothing.
SAMPLING_INTERVAL = pd.Timedelta(minutes=5)


def _on_grid(index: pd.DatetimeIndex) -> NDArray:
    """Whether each sample follows its predecessor by exactly one sampling interval.

    Parameters
    ----------
    index:
        Monotonic sample stamps, ``(N,)``.

    Returns
    -------
    numpy.ndarray
        Boolean, ``(N,)``. The first sample is always ``False``: it has no
        predecessor, so no difference across it is defined.
    """
    spacing = pd.Series(index).diff()
    return np.asarray((spacing == SAMPLING_INTERVAL).to_numpy(), dtype=bool)


def _reachable(stamps: NDArray, reach: pd.Timedelta) -> NDArray:
    """Whether each sample is close enough in TIME to its predecessor to difference across.

    Used on the finite subsequence of a channel, where "predecessor" is the last
    sample that carried a value.  A dropout that an earlier gate already masked
    leaves a hole, and requiring the raw five-minute grid across that hole is what
    made this test blind to the very episodes it exists for: the gate NaNs one
    side of a spike, and the spike then has no neighbour to be measured against.

    Parameters
    ----------
    stamps:
        Sample stamps of the finite subsequence, ``(M,)``, ``datetime64``.
    reach:
        Longest separation still treated as adjacent.

    Returns
    -------
    numpy.ndarray
        Boolean, ``(M,)``, first entry ``False``.
    """
    adjacent = np.zeros(stamps.size, dtype=bool)
    if stamps.size > 1:
        # A backwards jump is not adjacency. Without the lower bound a
        # non-monotonic index differences across arbitrary time, which the
        # exact-grid predicate this replaced was immune to by construction.
        deltas = np.diff(stamps)
        adjacent[1:] = (deltas > np.timedelta64(0, "us")) & (deltas <= np.timedelta64(reach))
    return adjacent


def mask_step_excursions(
    frame: pd.DataFrame, limits: list[SensorStepLimit]
) -> tuple[pd.DataFrame, int]:
    """Mask samples that leave the series and return within *max_len* samples.

    Parameters
    ----------
    frame:
        Five-minute archive frame indexed by naive station-local stamps.
    limits:
        One entry per column, with ``column``, ``threshold`` in the column's own
        unit, ``return_tol`` for how close the series must come back to where it
        left, and ``max_len`` for how many samples the excursion may last. A
        column absent from *frame* is skipped, as the range gate skips it: the
        logger's headers change between eras.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The same object, mutated in place, and the number of samples masked.
        Only the excursion's INTERIOR is masked — the sample that returns is
        good data and stays.
    """
    stamps = pd.DatetimeIndex(frame.index).to_numpy()
    removed = 0

    for limit in limits:
        column = limit.column
        if column not in frame.columns:
            continue
        threshold = limit.threshold
        return_tol = limit.return_tol
        max_len = limit.max_len
        if max_len < 1:
            raise ValueError(
                f"{column}: max_len must be at least 1, got {max_len}. A shorter one "
                "detects nothing and reports zero removals, which reads as a clean run."
            )

        # Work on the samples that carry a value, not on the raw grid: an earlier
        # gate has usually already masked one side of the very dropout being
        # looked for, and differencing across that hole is what blinded this test.
        values = frame[column].to_numpy(dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if finite.size < 3:
            continue
        present = values[finite]
        adjacent = _reachable(stamps[finite], MAX_ADJACENT_INTERVALS * SAMPLING_INTERVAL)
        steps = np.diff(present, prepend=np.nan)
        departures = np.flatnonzero(adjacent & (np.abs(steps) > threshold))

        excursion = np.zeros(values.size, dtype=bool)
        for departure in departures:
            start = int(departure)
            # On a double dip the sample that RECOVERS one excursion is also a
            # departure from it, and its baseline would be the rejected value. That
            # inverts the test: the ambient reading becomes the excursion and gets
            # masked. A baseline this function already rejected is no baseline.
            if excursion[finite[start - 1]]:
                continue
            baseline = present[start - 1]
            for end in range(start + 1, min(start + max_len + 1, present.size)):
                if not adjacent[end]:
                    break
                returns = (
                    np.sign(steps[end]) == -np.sign(steps[start])
                    and abs(steps[end]) > threshold
                    and abs(present[end] - baseline) <= return_tol
                )
                if returns:
                    excursion[finite[start:end]] = True
                    break

        if excursion.any():
            frame.loc[excursion, column] = np.nan
            removed += int(excursion.sum())

    return frame, removed


def _run_starts(values: NDArray, on_grid: NDArray) -> NDArray:
    """Index each sample by the run of identical values it belongs to.

    A gap BREAKS a run: two identical values either side of missing hours are not
    evidence that anything held still in between.  A NaN breaks it too.
    """
    changed = np.ones(values.size, dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    return np.cumsum(changed | ~on_grid | np.isnan(values))


def mask_persistent_runs(
    frame: pd.DataFrame, limits: list[SensorPersistenceLimit]
) -> tuple[pd.DataFrame, int]:
    """Mask runs of bitwise-identical values longer than a column tolerates.

    Parameters
    ----------
    frame:
        Five-minute archive frame indexed by naive station-local stamps.
    limits:
        One entry per column, with ``column``, ``min_run`` as the longest run
        left alone. There is deliberately no calm exemption: measured on this
        archive the two populations do not overlap, so ``min_run`` separates them
        on its own. Genuine calm runs reach 2 samples at p99 and 6 at their
        longest, while the rails run 1,318 to 3,060 samples — 110 to 255 hours of
        a coastal site reporting no wind at all. An exemption keyed to the calm
        LEVEL would have protected exactly those rails, since a sensor that dies
        reporting zero reads calm forever.

    Returns
    -------
    tuple of (pandas.DataFrame, int)
        The same object, mutated in place, and the number of samples masked.
    """
    on_grid = _on_grid(pd.DatetimeIndex(frame.index))
    removed = 0

    for limit in limits:
        column = limit.column
        if column not in frame.columns:
            continue
        min_run = limit.min_run

        values = frame[column].to_numpy(dtype=float)
        runs = _run_starts(values, on_grid)
        lengths = np.bincount(runs)
        stuck = (lengths[runs] > min_run) & np.isfinite(values)

        if stuck.any():
            frame.loc[stuck, column] = np.nan
            removed += int(stuck.sum())

    return frame, removed
