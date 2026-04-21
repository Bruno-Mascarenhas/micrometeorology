"""Sequential dataset for PyTorch (LSTM / Transformer)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        self.X = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class SequenceDatasetMeta:
    """Metadata wrapper for a SequenceDataset — enables save/load."""

    X_raw: np.ndarray
    y_raw: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    sequence_length: int = 24
    index: pd.DatetimeIndex | None = None

    def to_torch_dataset(self) -> SequenceDataset:
        """Create a PyTorch SequenceDataset from the raw arrays."""
        return SequenceDataset(self.X_raw, self.y_raw)

    def save(self, path: str | Path) -> None:
        """Save sequence dataset for reproducibility."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.savez(
            p / "sequences.npz",
            X=self.X_raw,
            y=self.y_raw,
            sequence_length=np.array([self.sequence_length]),
        )
        meta = pd.DataFrame({"feature_names": self.feature_names})
        meta.to_csv(p / "feature_names.csv", index=False)
        if self.index is not None:
            pd.Series(self.index).to_csv(p / "index.csv", index=False)

    @classmethod
    def load(cls, path: str | Path) -> SequenceDatasetMeta:
        """Load a previously saved sequence dataset."""
        p = Path(path)
        data = np.load(p / "sequences.npz")
        features, targets = data["X"], data["y"]
        seq_len = int(data["sequence_length"][0])

        meta = pd.read_csv(p / "feature_names.csv")
        feature_names = meta["feature_names"].tolist()

        index = None
        idx_path = p / "index.csv"
        if idx_path.exists():
            idx_df = pd.read_csv(idx_path)
            index = pd.to_datetime(idx_df.iloc[:, 0])

        return cls(
            X_raw=features,
            y_raw=targets,
            feature_names=feature_names,
            sequence_length=seq_len,
            index=index,  # type: ignore
        )
