"""Sequential dataset for PyTorch (LSTM / Transformer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from solrad_correction.utils.memory import assert_array_size

SequenceArray = np.ndarray | np.memmap | torch.Tensor


class SequenceDataset(Dataset):
    """PyTorch Dataset for time-series sequences.

    Each sample is ``(x_window, y_target)`` where:
    - ``x_window``: tensor of shape ``(sequence_length, n_features)``
    - ``y_target``: scalar tensor (regression target at end of window)
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Parameters
        ----------
        features:
            3-D array of shape ``(n_samples, sequence_length, n_features)``.
        targets:
            1-D array of shape ``(n_samples,)``.
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
            2-D base feature matrix of shape ``(n_samples, n_features)``.
        targets:
            1-D target vector of shape ``(n_samples,)``.
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

    @property
    def n_features(self) -> int:
        """Number of features in each time step."""
        return int(self.X.shape[1])

    def _target_positions(self) -> np.ndarray:
        """Base-row positions of each window's target."""
        starts = np.arange(self._length) if self._starts is None else self._starts
        return starts + self.target_offset

    def target_values(self) -> np.ndarray:
        """Return targets aligned with the lazy sequence windows."""
        positions = self._target_positions()
        if isinstance(self.y, torch.Tensor):
            values = self.y[torch.as_tensor(positions, dtype=torch.long)]
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
        selected: np.ndarray = np.asarray(self.y, dtype=np.float32)[positions]
        return selected

    def prediction_index(self) -> pd.DatetimeIndex | None:
        """Return the timestamps of each window's target row, if an index is set."""
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
        """Save the lazy dataset backing arrays without materializing windows."""
        from solrad_correction.datasets.serialization import save_windowed_sequence_dataset

        save_windowed_sequence_dataset(self, path, feature_names=feature_names, index=index)

    @classmethod
    def load(cls, path: str | Path) -> WindowedSequenceDataset:
        """Load a lazy dataset saved by ``WindowedSequenceDataset.save``."""
        from solrad_correction.datasets.serialization import load_windowed_sequence_dataset

        return load_windowed_sequence_dataset(path)


@dataclass
class WindowedSequenceDatasetMeta:
    """Metadata for lazy sequence datasets saved without dense windows."""

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
        """Create metadata from a lazy dataset without expanding windows."""
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
        """Create a lazy PyTorch dataset from the saved backing arrays."""
        return WindowedSequenceDataset(
            self.features,
            self.targets,
            self.sequence_length,
            target_offset=self.target_offset,
            index=None if self.index is None else pd.DatetimeIndex(self.index),
        )

    def save(self, path: str | Path) -> None:
        """Save base arrays and metadata without dense sequence materialization."""
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
        meta = pd.DataFrame({"feature_names": self.feature_names})
        meta.to_csv(p / "feature_names.csv", index=False)
        if self.index is not None:
            pd.Series(self.index).to_csv(p / "index.csv", index=False)

    @classmethod
    def load(cls, path: str | Path) -> WindowedSequenceDatasetMeta:
        """Load a saved lazy sequence dataset."""
        p = Path(path)
        data = np.load(p / "windowed_sequences.npz")
        features = data["X_base"]
        targets = data["y_base"]
        seq_len = int(data["sequence_length"][0])
        target_offset = int(data["target_offset"][0])

        feature_names: list[str] = []
        meta_path = p / "feature_names.csv"
        if meta_path.exists():
            meta = pd.read_csv(meta_path)
            feature_names = meta["feature_names"].tolist()

        index = None
        idx_path = p / "index.csv"
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path)
            index = pd.to_datetime(idx_df.iloc[:, 0])

        return cls(
            features=features,
            targets=targets,
            feature_names=feature_names,
            sequence_length=seq_len,
            target_offset=target_offset,
            index=index,  # type: ignore
        )
