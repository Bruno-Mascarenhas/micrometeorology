"""Build-time temporal alignment for image <-> sensor pairing.

:class:`CenterFrame` maps each video frame to the single nearest sensor record
within a maximum distance, carrying a stable string ``id`` that is stored in the
manifest sidecar meta so a loaded dataset knows which pairing produced it. It is
what :func:`allsky.data.manifest.build_manifest` uses to attach met/target
values to a frame.

Dataset-level windowing (``mean_embedding`` / ``attention_pooling``) is NOT
here: it is implemented once, vectorized per day, by
:class:`allsky.data.datasets.MultimodalEmbeddingDataset`, and the name set it
accepts is owned by :data:`allsky.config.AlignmentStrategyName`. This module
briefly carried a second, unreachable windowing implementation with different
semantics (no same-day restriction), which is what made the two disagree.

Pure numpy/pandas; importing this module never pulls torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

__all__ = [
    "AlignmentResult",
    "AlignmentStrategy",
    "CenterFrame",
]

#: Nanoseconds per minute (int64 timestamp arithmetic is done in ns).
_NS_PER_MINUTE = 60_000_000_000


def _ns(index: pd.DatetimeIndex) -> np.ndarray:
    """Int64 nanoseconds-since-epoch for a (naive) DatetimeIndex, unit-pinned."""
    values: np.ndarray = index.as_unit("ns").to_numpy().astype("int64")
    return values


@runtime_checkable
class AlignmentStrategy(Protocol):
    """Build-time frame/sensor pairing with a stable string identity.

    ``id`` is persisted in the manifest meta so a rebuilt/loaded dataset knows
    which pairing produced it. This is the contract
    :func:`allsky.data.manifest.build_manifest` annotates and isinstance-checks.
    """

    id: str

    def pair(
        self, frame_times: pd.DatetimeIndex, sensor_times: pd.DatetimeIndex
    ) -> AlignmentResult:
        """Pair every frame with a sensor record (``-1`` where none is close enough)."""
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


class CenterFrame:
    """Pair each frame to the nearest sensor record within ``max_distance_minutes``.

    The one build-time strategy.  ``window_minutes`` is carried for the manifest
    meta and for the dataset that reads it back as its window width, while
    :meth:`pair` — the method the manifest builder calls — matches on
    ``max_distance_minutes`` only.
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
