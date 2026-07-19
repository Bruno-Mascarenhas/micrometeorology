"""Map-style multimodal datasets emitting the new-stack batch contract.

Two datasets share one batch contract (see :meth:`MultimodalImageDataset.__getitem__`):

- :class:`MultimodalImageDataset` loads sky JPEGs (paths relative to
  ``data_root``) end-to-end with an imageio read -> PIL bilinear resize -> CHW
  float32 in ``[0, 1]`` recipe.
- :class:`MultimodalEmbeddingDataset` reads a precomputed visual embedding per
  sample through an :class:`EmbeddingReader` (the real safetensors reader lands
  in wave C2; here it is a minimal ``sample_id -> np.ndarray`` protocol).

Both standardize the engineered feature vector with a **train-only**
:class:`allsky.features.FeatureNormalizer` (validation/test must be handed the
training-split normalizer — computing one locally is refused as a leakage
guard).  Targets are emitted in **raw physical units**;
``sky_class == -1`` and NaN regression targets mark missing labels for the loss
to mask.

``torch`` is imported lazily inside ``__getitem__`` so importing
``allsky.data.datasets`` never pulls torch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from allsky.data.contracts import resolve
from allsky.features.normalization import FeatureNormalizer

__all__ = [
    "EmbeddingReader",
    "MultimodalEmbeddingDataset",
    "MultimodalImageDataset",
]

#: One dataset item: a ``str -> torch.Tensor`` mapping. ``torch`` is imported
#: lazily inside ``__getitem__``, so importing this module never pulls it.
type SampleTensors = dict[str, Any]

#: Regression target columns (raw physical units; NaN = missing).
_REGRESSION_TARGETS = ("target_dhi", "target_kindex", "cloud_fraction")

#: Dataset-level temporal windowing modes for the embedding dataset.
WindowMode = Literal["center_frame", "mean_embedding", "attention_pooling"]
_WINDOW_MODES: tuple[WindowMode, ...] = ("center_frame", "mean_embedding", "attention_pooling")
#: Nanoseconds per minute (int64 timestamp arithmetic is done in ns).
_NS_PER_MINUTE = 60_000_000_000


@runtime_checkable
class EmbeddingReader(Protocol):
    """Minimal reader interface: ``sample_id -> (D,) float embedding``.

    The concrete safetensors-backed reader arrives in wave C2; any callable (or
    object with ``__call__``) returning a 1-D array for a ``sample_id`` — and
    optionally exposing an integer ``dim`` — satisfies this protocol.
    """

    def __call__(self, sample_id: str) -> np.ndarray:
        """Return the embedding vector for *sample_id*."""
        ...


class _BaseMultimodalDataset:
    """Shared feature/target handling for the multimodal datasets."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        train: bool = True,
        stats: FeatureNormalizer | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.train = train
        if not self.feature_columns:
            raise ValueError("feature_columns must be non-empty")
        missing = [c for c in self.feature_columns if c not in self.manifest.columns]
        if missing:
            raise ValueError(f"manifest is missing feature columns: {missing}")

        if stats is None:
            if not train:
                raise ValueError(
                    "train=False requires a FeatureNormalizer fit on the training "
                    "split (pass stats=train_dataset.stats) — fitting on a "
                    "validation/test split would leak information"
                )
            stats = FeatureNormalizer.fit(self.manifest, self.feature_columns)
        elif list(stats.columns) != self.feature_columns:
            raise ValueError(
                f"stats columns {list(stats.columns)} do not match "
                f"feature columns {self.feature_columns}"
            )
        self.stats = stats

        self._features = stats.transform(self.manifest).astype(np.float32)
        self._dhi = self._raw_target("target_dhi")
        self._kindex = self._raw_target("target_kindex")
        self._cloud_fraction = self._raw_target("cloud_fraction")
        self._sky_class = self.manifest["sky_class"].to_numpy(dtype=np.int64)
        self._sample_ids = [str(s) for s in self.manifest["sample_id"]]

    def _raw_target(self, column: str) -> np.ndarray:
        """Raw physical target column as float32 (NaN preserved as missing)."""
        if column in self.manifest.columns:
            return self.manifest[column].to_numpy(dtype=np.float32)
        return np.full(len(self.manifest), np.nan, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def _target_item(self, idx: int) -> SampleTensors:
        """Build the shared target tensors for row *idx* (torch imported lazily)."""
        import torch

        return {
            "features": torch.from_numpy(np.ascontiguousarray(self._features[idx])),
            "dhi": torch.tensor(self._dhi[idx], dtype=torch.float32),
            "kindex": torch.tensor(self._kindex[idx], dtype=torch.float32),
            "sky_class": torch.tensor(self._sky_class[idx], dtype=torch.long),
            "cloud_fraction": torch.tensor(self._cloud_fraction[idx], dtype=torch.float32),
        }


class MultimodalImageDataset(_BaseMultimodalDataset):
    """Sky-image + sensor dataset serving the new-stack batch contract.

    Each item is a dict of torch tensors:

    - ``features`` — float32 ``(F,)`` standardized sensor vector;
    - ``image`` — float32 ``(3, H, W)`` in ``[0, 1]``, resized to *image_size*;
    - ``dhi`` — float32 raw diffuse target (W/m2), NaN when missing;
    - ``kindex`` — float32 raw k-index target, NaN when missing;
    - ``sky_class`` — int64 label, ``-1`` when missing;
    - ``cloud_fraction`` — float32 in ``[0, 1]``, NaN when missing.

    Parameters
    ----------
    manifest:
        v2 manifest DataFrame.
    feature_columns:
        Engineered feature names to serve (must be manifest columns).
    data_root:
        Root the manifest ``image_path`` values resolve against.
    image_size:
        Square output size for each frame.
    train, stats:
        Train-only standardization: on the training split ``stats`` is fit from
        *manifest*; validation/test must be handed ``train_dataset.stats``.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        data_root: str | Path,
        image_size: int = 224,
        train: bool = True,
        stats: FeatureNormalizer | None = None,
    ) -> None:
        super().__init__(manifest, feature_columns, train=train, stats=stats)
        self.data_root = data_root
        self.image_size = image_size
        self._paths = [str(p) for p in self.manifest["image_path"]]

    def _load_image(self, relative_path: str) -> np.ndarray:
        """Load a JPEG as float32 CHW in [0, 1], resized to ``image_size``.

        imageio read, grayscale->RGB safety net, PIL bilinear resize; resolves
        the manifest's relative POSIX path against ``data_root``.
        """
        import imageio.v3 as iio

        full = resolve(relative_path, self.data_root)
        image = iio.imread(full)
        if image.ndim == 2:  # pragma: no cover - grayscale safety net
            image = np.stack([image] * 3, axis=-1)
        size = self.image_size
        if image.shape[0] != size or image.shape[1] != size:
            from PIL import Image

            image = np.asarray(
                Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
            )
        scaled = image.astype(np.float32) / 255.0
        return np.ascontiguousarray(scaled.transpose(2, 0, 1))

    def __getitem__(self, idx: int) -> SampleTensors:
        import torch

        item = self._target_item(idx)
        item["image"] = torch.from_numpy(self._load_image(self._paths[idx]))
        return item


class MultimodalEmbeddingDataset(_BaseMultimodalDataset):
    """Precomputed-embedding + sensor dataset serving the batch contract.

    Like :class:`MultimodalImageDataset` but emits a visual **embedding** per
    sample (read per ``sample_id`` through *embedding_reader*) instead of a raw
    ``image``.  The embedding dimension is discovered from the first read (or the
    reader's ``dim`` attribute) — no magic constant.

    Temporal windowing (``window``) controls how each row's neighbouring frames
    contribute, using the manifest's ``day_id`` / ``timestamp_utc`` to resolve a
    per-row window (same ``day_id``, ``|t - t_row| <= window_minutes / 2``,
    time-ordered; the row's own ``sample_id`` is always included):

    - ``"center_frame"`` (default) — item carries ``embedding`` ``(D,)`` (the
      sample's own embedding only; no windowing);
    - ``"mean_embedding"`` — item carries ``embedding`` ``(D,)`` = the mean of the
      window's *available* embeddings (missing co-frame reads are skipped;
      all-missing falls back to the row's own embedding);
    - ``"attention_pooling"`` — item carries ``embedding_seq`` ``(T, D)`` fp32
      zero-padded to a fixed ``T = ceil(window_minutes) + 1`` (simple collation)
      plus a bool ``frame_mask`` ``(T,)`` (True = a real frame). Encoder-side
      pooling (mask-aware mean or learned attention) lives in
      :class:`allsky.modeling.visual_encoder.PrecomputedEmbedding`.

    Parameters
    ----------
    embedding_reader:
        Callable ``sample_id -> (D,) np.ndarray`` (see :class:`EmbeddingReader`).
    window:
        Temporal windowing mode (see above).
    window_minutes:
        Full window width in minutes for the windowed modes.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        embedding_reader: EmbeddingReader,
        train: bool = True,
        stats: FeatureNormalizer | None = None,
        window: WindowMode = "center_frame",
        window_minutes: float = 10.0,
    ) -> None:
        super().__init__(manifest, feature_columns, train=train, stats=stats)
        if window not in _WINDOW_MODES:
            raise ValueError(f"window must be one of {_WINDOW_MODES}, got {window!r}")
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        self.embedding_reader = embedding_reader
        declared = getattr(embedding_reader, "dim", None)
        self._embedding_dim = int(declared) if declared is not None else None
        self.window = window
        self.window_minutes = float(window_minutes)
        #: Fixed padded window length for ``attention_pooling`` (simple collation).
        self.seq_len = math.ceil(self.window_minutes) + 1
        #: Per-row positional window members (empty list for ``center_frame``).
        self._windows: list[list[int]] = self._resolve_windows() if window != "center_frame" else []

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (discovered lazily from the first sample)."""
        if self._embedding_dim is None:
            self._embedding_dim = int(
                np.asarray(self.embedding_reader(self._sample_ids[0])).shape[-1]
            )
        return self._embedding_dim

    def _resolve_windows(self) -> list[list[int]]:
        """Per-row positional window members (same day_id, within window, ordered).

        For each row the members are the positions whose ``day_id`` matches and
        whose ``timestamp_utc`` is within ``window_minutes / 2`` of the row's own
        time, in time order.  The row's own position is always included (distance
        zero), so a window is never empty.
        """
        index = pd.DatetimeIndex(self.manifest["timestamp_utc"]).tz_convert("UTC").tz_localize(None)
        times_ns = index.as_unit("ns").to_numpy().astype("int64")
        days = self.manifest["day_id"].astype(str).to_numpy()
        half_ns = round(self.window_minutes / 2.0 * _NS_PER_MINUTE)
        windows: list[list[int]] = [[] for _ in range(len(self.manifest))]
        for day in np.unique(days):
            idx = np.nonzero(days == day)[0]
            order = np.argsort(times_ns[idx], kind="stable")
            idx_sorted = idx[order]
            t_sorted = times_ns[idx_sorted]
            for k, position in enumerate(idx_sorted):
                in_window = (t_sorted >= t_sorted[k] - half_ns) & (
                    t_sorted <= t_sorted[k] + half_ns
                )
                windows[int(position)] = [int(p) for p in idx_sorted[in_window]]
        return windows

    def _read(self, sample_id: str) -> np.ndarray:
        """Read + validate the ``(D,)`` float32 embedding for *sample_id*."""
        embedding = np.asarray(self.embedding_reader(sample_id), dtype=np.float32)
        if embedding.ndim != 1:
            raise ValueError(
                f"embedding for {sample_id!r} must be 1-D, got shape {embedding.shape}"
            )
        if self._embedding_dim is None:
            self._embedding_dim = int(embedding.shape[0])
        elif embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"embedding dim {embedding.shape[0]} for {sample_id!r} does not "
                f"match the expected {self._embedding_dim}"
            )
        return embedding

    def _read_optional(self, sample_id: str) -> np.ndarray | None:
        """Read a co-frame embedding, returning ``None`` when it is absent."""
        try:
            return self._read(sample_id)
        except KeyError:
            return None

    def _window_embeddings(self, idx: int) -> list[np.ndarray]:
        """Available embeddings for row *idx*'s window, in time order."""
        return [
            vector
            for member in self._windows[idx]
            if (vector := self._read_optional(self._sample_ids[member])) is not None
        ]

    def __getitem__(self, idx: int) -> SampleTensors:
        import torch

        item = self._target_item(idx)
        if self.window == "center_frame":
            embedding = self._read(self._sample_ids[idx])
            item["embedding"] = torch.from_numpy(np.ascontiguousarray(embedding))
            return item

        vectors = self._window_embeddings(idx)
        if not vectors:  # all co-frames missing -> fall back to the row's own frame
            vectors = [self._read(self._sample_ids[idx])]

        if self.window == "mean_embedding":
            pooled = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
            item["embedding"] = torch.from_numpy(np.ascontiguousarray(pooled))
            return item

        # attention_pooling: zero-padded (T, D) sequence + bool frame_mask.
        take = vectors[: self.seq_len]
        dim = take[0].shape[0]
        seq = np.zeros((self.seq_len, dim), dtype=np.float32)
        mask = np.zeros(self.seq_len, dtype=bool)
        for i, vector in enumerate(take):
            seq[i] = vector
            mask[i] = True
        item["embedding_seq"] = torch.from_numpy(seq)
        item["frame_mask"] = torch.from_numpy(mask)
        return item
