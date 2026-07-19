"""Temporal alignment strategies for image <-> sensor pairing.

Two families share one :class:`AlignmentStrategy` protocol (each carries a
stable string ``id`` stored in the manifest sidecar meta):

- **Build-time pairing** — :class:`CenterFrame` (the default) maps each video
  frame to the single nearest sensor record within a maximum distance.  It is
  what :func:`allsky.data.manifest.build_manifest` uses to attach met/target
  values to a frame.
- **Dataset-level windowing** — :class:`MeanEmbedding` and
  :class:`AttentionPooling` return, for a sample timestamp, the ordered list of
  frame positions falling inside an alignment window.  The dataset then pools
  the corresponding embeddings (mean / attention); those poolers land in a
  later wave, so here they only resolve the per-sample frame list.

Pure numpy/pandas; importing this module never pulls torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AlignmentResult",
    "AlignmentStrategy",
    "AttentionPooling",
    "CenterFrame",
    "MeanEmbedding",
    "available_strategies",
    "get_strategy",
    "register_strategy",
]

#: Nanoseconds per minute (int64 timestamp arithmetic is done in ns).
_NS_PER_MINUTE = 60_000_000_000


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Int64 nanoseconds-since-epoch for a (naive) DatetimeIndex, unit-pinned."""
    values: np.ndarray = index.as_unit("ns").to_numpy().astype("int64")
    return values


@runtime_checkable
class AlignmentStrategy(Protocol):
    """Temporal alignment strategy with a stable string identity.

    ``id`` is persisted in the manifest meta so a rebuilt/loaded dataset knows
    which pairing produced it.  Every strategy can resolve the ordered frame
    positions inside its window for a sample timestamp.
    """

    id: str

    def select_frames(self, sample_time: pd.Timestamp, frame_index: pd.DatetimeIndex) -> list[int]:
        """Positions in *frame_index* inside this strategy's window, time-ordered."""
        ...


@dataclass(frozen=True)
class AlignmentResult:
    """Result of pairing frames to sensor records (see :meth:`CenterFrame.pair`).

    Attributes
    ----------
    sensor_pos:
        For each frame, the positional index into the (monotonic) sensor index
        of the paired record, or ``-1`` when no record fell within tolerance.
    distance_minutes:
        Absolute time distance to the paired record in minutes; ``NaN`` for
        unmatched frames.
    """

    sensor_pos: np.ndarray
    distance_minutes: np.ndarray

    @property
    def matched(self) -> np.ndarray:
        """Boolean mask of frames that found a sensor record within tolerance."""
        result: np.ndarray = self.sensor_pos >= 0
        return result


def _window_positions(
    sample_time: pd.Timestamp, frame_index: pd.DatetimeIndex, window_minutes: float
) -> list[int]:
    """Time-ordered positions of *frame_index* within a centred window."""
    half = pd.Timedelta(minutes=window_minutes / 2.0)
    low = pd.Timestamp(sample_time) - half
    high = pd.Timestamp(sample_time) + half
    values = frame_index.as_unit("ns").to_numpy()
    mask = (values >= low.to_datetime64()) & (values <= high.to_datetime64())
    positions = np.nonzero(mask)[0]
    order = np.argsort(values[positions], kind="stable")
    ordered: list[int] = [int(p) for p in positions[order]]
    return ordered


class CenterFrame:
    """Pair each frame to the nearest sensor record within ``max_distance_minutes``.

    The default build-time strategy.  ``window_minutes`` is retained for
    interface symmetry with the windowed poolers (and drives
    :meth:`select_frames`), while :meth:`pair` — the method the manifest builder
    calls — matches on ``max_distance_minutes`` only.
    """

    id = "center_frame"

    def __init__(self, window_minutes: float = 10.0, max_distance_minutes: float = 5.0) -> None:
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        if max_distance_minutes <= 0:
            raise ValueError(f"max_distance_minutes must be positive, got {max_distance_minutes}")
        self.window_minutes = float(window_minutes)
        self.max_distance_minutes = float(max_distance_minutes)

    def pair(
        self, frame_times: pd.DatetimeIndex, sensor_times: pd.DatetimeIndex
    ) -> AlignmentResult:
        """Match every frame to the nearest sensor record within tolerance.

        Parameters
        ----------
        frame_times:
            Naive-local frame timestamps (any order).
        sensor_times:
            Naive-local sensor timestamps, **monotonic increasing** (dedup the
            sensor frame before calling; a non-monotonic index raises).

        Returns
        -------
        AlignmentResult
            Positional sensor index (``-1`` unmatched) and distance in minutes
            (``NaN`` unmatched) per frame, aligned 1:1 with *frame_times*.
        """
        n_frames = len(frame_times)
        n_sensors = len(sensor_times)
        if n_sensors and not sensor_times.is_monotonic_increasing:
            raise ValueError("sensor_times must be monotonic increasing")

        sensor_pos = np.full(n_frames, -1, dtype=np.int64)
        distance_minutes = np.full(n_frames, np.nan, dtype=np.float64)
        if n_frames == 0 or n_sensors == 0:
            return AlignmentResult(sensor_pos=sensor_pos, distance_minutes=distance_minutes)

        # pandas' int8 view is unit-dependent (defaults to us): pin both to ns.
        sensor_ns = _ns(sensor_times)
        frame_ns = _ns(frame_times)
        insert = np.searchsorted(sensor_ns, frame_ns)
        left = np.clip(insert - 1, 0, n_sensors - 1)
        right = np.clip(insert, 0, n_sensors - 1)
        dist_left = np.abs(frame_ns - sensor_ns[left])
        dist_right = np.abs(frame_ns - sensor_ns[right])
        take_right = dist_right < dist_left
        nearest = np.where(take_right, right, left)
        nearest_dist_ns = np.where(take_right, dist_right, dist_left)

        tolerance_ns = self.max_distance_minutes * _NS_PER_MINUTE
        within = nearest_dist_ns <= tolerance_ns
        sensor_pos[within] = nearest[within]
        distance_minutes[within] = nearest_dist_ns[within] / _NS_PER_MINUTE
        return AlignmentResult(sensor_pos=sensor_pos, distance_minutes=distance_minutes)

    def select_frames(self, sample_time: pd.Timestamp, frame_index: pd.DatetimeIndex) -> list[int]:
        """Position of the single nearest frame within ``max_distance_minutes``.

        Returns a one- or zero-element list, so it composes with the windowed
        strategies' list return type.
        """
        if len(frame_index) == 0:
            return []
        frame_ns = _ns(frame_index)
        sample_ns = np.int64(pd.Timestamp(sample_time).as_unit("ns").value)
        distances = np.abs(frame_ns - sample_ns)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self.max_distance_minutes * _NS_PER_MINUTE:
            return [nearest]
        return []


class _WindowStrategy:
    """Shared implementation for the dataset-level windowed poolers."""

    id = "window"

    def __init__(self, window_minutes: float = 10.0) -> None:
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        self.window_minutes = float(window_minutes)

    def select_frames(self, sample_time: pd.Timestamp, frame_index: pd.DatetimeIndex) -> list[int]:
        """All frame positions inside the centred window, ordered by time."""
        return _window_positions(sample_time, frame_index, self.window_minutes)


class MeanEmbedding(_WindowStrategy):
    """Window strategy whose per-sample frames are mean-pooled at dataset level."""

    id = "mean_embedding"


class AttentionPooling(_WindowStrategy):
    """Window strategy whose per-sample frames are attention-pooled at dataset level."""

    id = "attention_pooling"


#: Name -> strategy class registry.  Extend via :func:`register_strategy`.
_STRATEGIES: dict[str, type] = {
    CenterFrame.id: CenterFrame,
    MeanEmbedding.id: MeanEmbedding,
    AttentionPooling.id: AttentionPooling,
}


def register_strategy(name: str, cls: type) -> None:
    """Register a new alignment strategy class under *name* (overwrites)."""
    _STRATEGIES[name] = cls


def available_strategies() -> tuple[str, ...]:
    """Registered alignment strategy names, in registration order."""
    return tuple(_STRATEGIES)


def get_strategy(name: str, **kwargs: float) -> AlignmentStrategy:
    """Instantiate the registered alignment strategy *name* with *kwargs*.

    Raises
    ------
    KeyError
        If *name* is not registered (message lists the known strategies).
    """
    try:
        cls = _STRATEGIES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown alignment strategy {name!r}; known: {sorted(_STRATEGIES)}"
        ) from exc
    strategy: AlignmentStrategy = cls(**kwargs)
    return strategy


def _ordered_sample_frames(
    strategy: AlignmentStrategy,
    sample_times: Sequence[pd.Timestamp],
    frame_index: pd.DatetimeIndex,
) -> list[list[int]]:
    """Per-sample ordered frame positions (helper for windowed dataset use)."""
    return [strategy.select_frames(pd.Timestamp(t), frame_index) for t in sample_times]
