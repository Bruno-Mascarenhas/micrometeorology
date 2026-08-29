"""Sequential dataset for PyTorch (LSTM / Transformer)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, default_collate

from solrad_correction.datasets.sidecars import (
    FEATURE_NAMES_FILENAME,
    read_feature_names,
    read_optional_index,
    write_feature_names,
    write_optional_index,
)
from solrad_correction.utils.memory import assert_array_size

SequenceArray = np.ndarray | np.memmap | torch.Tensor


class PreCollatedBatch(NamedTuple):
    """A ``(features, targets)`` batch a dataset already assembled itself.

    Marks the output of :meth:`WindowedSequenceDataset.__getitems__` so
    :func:`collate_sequence_batch` hands it straight to the model instead of
    trying to collate it a second time.

    Attributes
    ----------
    features:
        ``float32``, shape ``(batch, sequence_length, n_features)``.
    targets:
        ``float32``, shape ``(batch,)``, one target per window.
    """

    features: torch.Tensor
    targets: torch.Tensor


def collate_sequence_batch(
    samples: PreCollatedBatch | list[Any],
) -> tuple[torch.Tensor, torch.Tensor] | Any:
    """Collate for any loader that may be fed a batched dataset.

    ``WindowedSequenceDataset`` assembles whole batches in ``__getitems__``,
    which torch's fetcher prefers over per-index ``__getitem__`` at EVERY
    DataLoader site — it is not opt-in. This must therefore be the ``collate_fn``
    of every loader over that dataset; anything else (a plain
    ``SequenceDataset``, a ``TensorDataset``) falls through to the default.

    Parameters
    ----------
    samples:
        Either a :class:`PreCollatedBatch` the dataset assembled itself, or the
        list of per-index samples torch's fetcher gathers otherwise.

    Returns
    -------
    tuple of (features, targets)
        For a pre-collated batch, its two tensors unwrapped: ``float32`` of
        shape ``(batch, sequence_length, n_features)`` and ``(batch,)``.
        Otherwise whatever ``torch.utils.data.default_collate`` builds from the
        sample list.
    """
    if isinstance(samples, PreCollatedBatch):
        return samples.features, samples.targets
    return default_collate(samples)


class SequenceDataset(Dataset):
    """PyTorch Dataset for time-series sequences.

    Each sample is ``(x_window, y_target)`` where:
    - ``x_window``: ``float32`` tensor of shape ``(sequence_length, n_features)``
    - ``y_target``: ``float32`` scalar tensor (regression target at end of window)

    The windows are already materialized — typically by
    :func:`solrad_correction.features.sequence.create_sequences` — and held in
    memory as one dense tensor. :class:`WindowedSequenceDataset` is the lazy
    alternative for series too large to expand.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Parameters
        ----------
        features:
            Dense windows, shape ``(n_samples, sequence_length, n_features)``,
            cast to ``float32``.
        targets:
            One target per window, shape ``(n_samples,)``, cast to ``float32``.

        Raises
        ------
        MemoryError
            If either array exceeds the ``SOLRAD_MAX_ARRAY_GB`` guardrail.
        """
        assert_array_size(features.shape, np.float32, context="dense sequence feature tensor")
        assert_array_size(targets.shape, np.float32, context="dense sequence target vector")
        self.X = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class WindowedSequenceDataset(Dataset):
    """Lazy PyTorch Dataset for sliding-window time-series samples.

    Unlike ``SequenceDataset``, this stores the base 2-D feature matrix and
    target vector, then slices each window on demand. This preserves the
    ``create_sequences()`` alignment — window rows ``[i, i + sequence_length)``
    predict ``y[i + sequence_length - 1]``, the target concurrent with the last
    row inside the window — without materializing the full 3-D
    ``(n_windows, sequence_length, n_features)`` array.

    When a ``DatetimeIndex`` is provided, windows that span a temporal gap
    larger than the base sampling frequency (inferred as the median timestamp
    delta, overridable via ``max_gap``) are dropped so no window mixes
    discontinuous history.
    """

    def __init__(
        self,
        features: SequenceArray,
        targets: SequenceArray,
        sequence_length: int,
        *,
        target_offset: int | None = None,
        index: pd.DatetimeIndex | None = None,
        max_gap: pd.Timedelta | str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        features:
            2-D base feature matrix of shape ``(n_samples, n_features)``, cast
            to ``float32``. A ``np.memmap`` is kept on disk rather than read in.
        targets:
            1-D target vector of shape ``(n_samples,)``, cast to ``float32``.
        sequence_length:
            Number of time steps per input window.
        target_offset:
            Target row offset relative to the window start. Defaults to
            ``sequence_length - 1`` (the last row inside the window), matching
            ``create_sequences()``.
        index:
            Optional ``DatetimeIndex`` aligned with the base rows. When given,
            windows spanning a temporal gap larger than ``max_gap`` are dropped.
        max_gap:
            Maximum allowed timestamp delta between consecutive rows inside a
            window. Defaults to the inferred base frequency (median delta).

        Raises
        ------
        ValueError
            If ``sequence_length`` is not positive or is not shorter than the
            series, if ``target_offset`` is negative or past the last target, if
            ``features``/``targets``/``index`` disagree in length, if
            ``features`` is not 2-D, or if gap filtering leaves no usable
            window.
        MemoryError
            If a materialized base array would exceed the
            ``SOLRAD_MAX_ARRAY_GB`` guardrail.
        """
        if sequence_length <= 0:
            raise ValueError(f"sequence_length ({sequence_length}) must be positive")

        self.sequence_length = sequence_length
        self.target_offset = sequence_length - 1 if target_offset is None else target_offset

        if self.target_offset < 0:
            raise ValueError(f"target_offset ({self.target_offset}) must be non-negative")

        self.X = self._prepare_features(features)
        self.y = self._prepare_targets(targets)
        self.index = None if index is None else pd.DatetimeIndex(index)

        if len(self.X) != len(self.y):
            raise ValueError(
                f"features ({len(self.X)}) and target ({len(self.y)}) must have same length"
            )
        if self.index is not None and len(self.index) != len(self.X):
            raise ValueError(
                f"index ({len(self.index)}) and features ({len(self.X)}) must have same length"
            )
        if sequence_length >= len(self.X):
            raise ValueError(f"sequence_length ({sequence_length}) >= data length ({len(self.X)})")
        if self.target_offset >= len(self.y):
            raise ValueError(
                f"target_offset ({self.target_offset}) >= target length ({len(self.y)})"
            )

        base_length = min(len(self.X) - sequence_length + 1, len(self.y) - self.target_offset)
        self._starts = self._valid_window_starts(base_length, max_gap)
        self._length = base_length if self._starts is None else len(self._starts)
        if self._length <= 0:
            raise ValueError("No sequence windows can be generated with the provided offsets")

    def _valid_window_starts(
        self,
        base_length: int,
        max_gap: pd.Timedelta | str | None,
    ) -> np.ndarray | None:
        """Return window start positions whose rows contain no temporal gap."""
        if self.index is None:
            return None
        timestamps = self.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        deltas = np.diff(timestamps)
        limit = float(np.median(deltas)) if max_gap is None else float(pd.Timedelta(max_gap).value)
        gap_after = np.zeros(len(self.index), dtype=np.int64)
        gap_after[1:] = deltas > limit
        cumulative = np.cumsum(gap_after)
        starts = np.arange(max(base_length, 0))
        span = max(self.sequence_length - 1, self.target_offset)
        valid: np.ndarray = starts[(cumulative[starts + span] - cumulative[starts]) == 0]
        return valid

    @staticmethod
    def _prepare_features(features: SequenceArray) -> SequenceArray:
        if isinstance(features, torch.Tensor):
            if features.ndim != 2:
                raise ValueError(f"features must be 2-D, got shape {tuple(features.shape)}")
            return features.to(dtype=torch.float32)

        arr = np.asarray(features)
        if arr.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {arr.shape}")
        assert_array_size(arr.shape, np.float32, context="windowed sequence feature matrix")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    @staticmethod
    def _prepare_targets(targets: SequenceArray) -> SequenceArray:
        if isinstance(targets, torch.Tensor):
            y = targets.flatten()
            return y.to(dtype=torch.float32)

        arr = np.asarray(targets).reshape(-1)
        assert_array_size(arr.shape, np.float32, context="windowed sequence target vector")
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return arr

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice one window out of the base matrix.

        Parameters
        ----------
        idx:
            Window position; negative values count back from the last window.
            Windows are numbered over the gap-filtered set, so ``idx`` is not a
            base-row position whenever an index was supplied.

        Returns
        -------
        tuple of (x_window, y_target)
            ``float32`` tensors of shape ``(sequence_length, n_features)`` and
            ``()``.

        Raises
        ------
        IndexError
            If ``idx`` is outside the window range.
        """
        if idx < 0:
            idx += self._length
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)

        start = idx if self._starts is None else int(self._starts[idx])
        window = self.X[start : start + self.sequence_length]
        target = self.y[start + self.target_offset]

        if isinstance(window, torch.Tensor):
            x_tensor = window
        else:
            x_tensor = torch.as_tensor(window, dtype=torch.float32)

        if isinstance(target, torch.Tensor):
            y_tensor = target.reshape(())
        else:
            y_tensor = torch.as_tensor(target, dtype=torch.float32)

        return x_tensor, y_tensor

    def __getitems__(self, indices: list[int]) -> PreCollatedBatch:
        """Gather a whole batch of windows in one indexing operation.

        ``torch.utils.data._MapDatasetFetcher`` calls this instead of
        ``__getitem__`` per index whenever it exists, so a batch costs one fancy
        index into the base matrix rather than ``batch_size`` Python-level
        slices, tensor conversions and a stack. The result is already batched,
        which is why every loader over this dataset must use
        :func:`collate_sequence_batch` — the default collate would try to stack
        the ``(B, L, F)`` block with the ``(B,)`` targets.

        Parameters
        ----------
        indices:
            Window positions, negative values counting back from the last
            window, as torch's sampler emits them.

        Returns
        -------
        PreCollatedBatch
            ``float32`` tensors of shape ``(B, sequence_length, n_features)``
            and ``(B,)``, with ``B = len(indices)``.

        Raises
        ------
        IndexError
            If any position is outside the window range.
        """
        length = self._length
        positions = np.fromiter(
            (idx + length if idx < 0 else idx for idx in indices),
            dtype=np.int64,
            count=len(indices),
        )
        if positions.size and (positions.min() < 0 or positions.max() >= length):
            raise IndexError(f"index out of range for {length} windows")
        starts = positions if self._starts is None else self._starts[positions]
        window_rows = starts[:, None] + np.arange(self.sequence_length, dtype=np.int64)
        target_rows = starts + self.target_offset

        if isinstance(self.X, torch.Tensor):
            x_batch = self.X[torch.from_numpy(window_rows)].to(dtype=torch.float32)
        else:
            x_batch = torch.as_tensor(self.X[window_rows], dtype=torch.float32)
        if isinstance(self.y, torch.Tensor):
            y_batch = self.y[torch.from_numpy(target_rows)].to(dtype=torch.float32)
        else:
            y_batch = torch.as_tensor(self.y[target_rows], dtype=torch.float32)
        return PreCollatedBatch(x_batch, y_batch)

    @property
    def n_features(self) -> int:
        """Number of features in each time step."""
        return int(self.X.shape[1])

    def _target_positions(self) -> np.ndarray:
        """Base-row positions of each window's target."""
        starts = np.arange(self._length) if self._starts is None else self._starts
        return starts + self.target_offset

    def target_values(self) -> np.ndarray:
        """Return targets aligned with the lazy sequence windows.

        Returns
        -------
        np.ndarray
            ``float32``, shape ``(len(self),)``, in the targets' own units. The
            i-th entry is the target :meth:`__getitem__` pairs with window i, so
            this is the ground-truth vector predictions are scored against —
            gap-dropped windows are absent from it too.
        """
        positions = self._target_positions()
        if isinstance(self.y, torch.Tensor):
            values = self.y[torch.as_tensor(positions, dtype=torch.long)]
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
        selected: np.ndarray = np.asarray(self.y, dtype=np.float32)[positions]
        return selected

    def prediction_index(self) -> pd.DatetimeIndex | None:
        """Return the timestamps of each window's target row, if an index is set.

        Returns
        -------
        pd.DatetimeIndex or None
            Shape ``(len(self),)``, aligned entry-for-entry with
            :meth:`target_values`, so predictions can be written back against
            the time they belong to. ``None`` when the dataset was built without
            an index.
        """
        if self.index is None:
            return None
        return self.index[self._target_positions()]

    def save(
        self,
        path: str | Path,
        *,
        feature_names: list[str] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> None:
        """Save the lazy dataset backing arrays without materializing windows.

        Parameters
        ----------
        path:
            Directory to write; created if missing.
        feature_names:
            Column names of the base matrix, length ``n_features``. Stored so a
            reloaded dataset can be matched against a model's feature order.
        index:
            Timestamps of the base rows, shape ``(n_samples,)``. Defaults to the
            dataset's own index when omitted.
        """
        from solrad_correction.datasets.serialization import save_windowed_sequence_dataset

        save_windowed_sequence_dataset(self, path, feature_names=feature_names, index=index)

    @classmethod
    def load(cls, path: str | Path) -> WindowedSequenceDataset:
        """Load a lazy dataset saved by ``WindowedSequenceDataset.save``."""
        from solrad_correction.datasets.serialization import load_windowed_sequence_dataset

        return load_windowed_sequence_dataset(path)


@dataclass(frozen=True, slots=True)
class WindowedSequenceDatasetMeta:
    """Metadata for lazy sequence datasets saved without dense windows.

    The on-disk form of a :class:`WindowedSequenceDataset`: the base arrays plus
    the windowing parameters needed to reproduce exactly the same windows, so
    the dense ``(n_windows, sequence_length, n_features)`` block is never written
    out.

    Attributes
    ----------
    features:
        Base feature matrix, ``float32`` on disk, shape
        ``(n_samples, n_features)``, columns ordered as ``feature_names``.
    targets:
        Base target vector, ``float32`` on disk, shape ``(n_samples,)``.
    feature_names:
        Column names of ``features``, length ``n_features``.
    sequence_length:
        Rows per window.
    target_offset:
        Target row offset from the window start; ``None`` means the last row
        inside the window.
    index:
        Timestamps of the base rows, shape ``(n_samples,)``, or ``None``. Drives
        the gap filtering when the dataset is rebuilt.
    """

    features: np.ndarray
    targets: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    sequence_length: int = 24
    target_offset: int | None = None
    index: pd.DatetimeIndex | None = None

    @classmethod
    def from_dataset(
        cls,
        dataset: WindowedSequenceDataset,
        *,
        feature_names: list[str] | None = None,
        index: pd.DatetimeIndex | None = None,
    ) -> WindowedSequenceDatasetMeta:
        """Create metadata from a lazy dataset without expanding windows.

        The base arrays are taken as they are held (a torch tensor is moved to
        CPU and unwrapped), so no window is ever materialized. ``feature_names``
        and ``index`` default to the dataset's own index when not supplied.
        """
        features = (
            dataset.X.detach().cpu().numpy()
            if isinstance(dataset.X, torch.Tensor)
            else np.asarray(dataset.X)
        )
        targets = (
            dataset.y.detach().cpu().numpy()
            if isinstance(dataset.y, torch.Tensor)
            else np.asarray(dataset.y)
        )
        return cls(
            features=features,
            targets=targets,
            feature_names=feature_names or [],
            sequence_length=dataset.sequence_length,
            target_offset=dataset.target_offset,
            index=index if index is not None else dataset.index,
        )

    def to_torch_dataset(self) -> WindowedSequenceDataset:
        """Create a lazy PyTorch dataset from the saved backing arrays.

        Rebuilds the same windows the saved dataset exposed, gap filtering
        included: it is re-derived from ``index`` at the default ``max_gap``
        rather than stored, so an artifact saved with a non-default ``max_gap``
        comes back with the default one.
        """
        return WindowedSequenceDataset(
            self.features,
            self.targets,
            self.sequence_length,
            target_offset=self.target_offset,
            index=None if self.index is None else pd.DatetimeIndex(self.index),
        )

    def save(self, path: str | Path) -> None:
        """Save base arrays and metadata without dense sequence materialization.

        Writes ``path`` as a directory holding ``windowed_sequences.npz`` (the
        ``float32`` base arrays, the windowing parameters and a
        ``format_version`` stamp), ``feature_names.csv`` and, when an index is
        set, ``index.csv``. Read back by :meth:`load`.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(
            p / "windowed_sequences.npz",
            X_base=np.asarray(self.features, dtype=np.float32),
            y_base=np.asarray(self.targets, dtype=np.float32).reshape(-1),
            sequence_length=np.array([self.sequence_length]),
            target_offset=np.array(
                [self.sequence_length - 1 if self.target_offset is None else self.target_offset]
            ),
            format_version=np.array([1]),
        )
        write_feature_names(p, self.feature_names)
        write_optional_index(p, self.index)

    @classmethod
    def load(cls, path: str | Path) -> WindowedSequenceDatasetMeta:
        """Load a directory written by :meth:`save`.

        Both sidecars are optional, so an artifact saved without feature names
        or without timestamps loads back with those fields empty rather than
        failing.
        """
        p = Path(path)
        data = np.load(p / "windowed_sequences.npz")
        features = data["X_base"]
        targets = data["y_base"]
        seq_len = int(data["sequence_length"][0])
        target_offset = int(data["target_offset"][0])

        feature_names: list[str] = (
            read_feature_names(p) if (p / FEATURE_NAMES_FILENAME).exists() else []
        )
        index = read_optional_index(p)

        return cls(
            features=features,
            targets=targets,
            feature_names=feature_names,
            sequence_length=seq_len,
            target_offset=target_offset,
            index=index,
        )
