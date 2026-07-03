"""Frame-sensor pairing index and the PyTorch-facing dataset.

:func:`build_index` matches each extracted video frame to the nearest
radiation-sensor record within a configurable time tolerance, producing the
table that :class:`AllSkyDataset` serves to a ``DataLoader``.

The sensor frame is expected to carry the target columns added by
``allsky.sensors.derive_targets`` (``kt``, ``diffuse``, ``cloud_class``,
``target_source``).  When ``target_source == "erbs_pseudo"`` the diffuse
values are Erbs-decomposition pseudo-targets derived from GHI, not
measurements — they bootstrap the pipeline until a shaded pyranometer exists.

``torch`` is imported lazily inside :class:`AllSkyDataset` methods so this
module (and :func:`build_index`) can be used in environments without torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import numpy as np
import pandas as pd

from allsky.config import AllSkyConfig, ModelConfig, SensorConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

#: Pairing-index columns that are never model features.
NON_FEATURE_COLUMNS = frozenset(
    {
        "frame_path",
        "timestamp",
        "video",
        "index",
        "sensor_timestamp",
        "kt",
        "diffuse",
        "cloud_class",
        "target_source",
        "day",
    }
)

#: Target columns required in every pairing-index row.
TARGET_COLUMNS = ("kt", "diffuse", "cloud_class")

#: Guard against division by ~zero when standardizing constant features.
_MIN_FEATURE_STD = 1e-6


def build_index(
    manifest: pd.DataFrame,
    sensor_df: pd.DataFrame,
    cfg: SensorConfig | AllSkyConfig,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Pair extracted frames with the nearest sensor record within tolerance.

    Performs an as-of merge (``direction="nearest"``) between the frame
    manifest and the sensor frame, then drops rows that are unusable for
    training:

    - frames with no sensor record within ``tolerance_minutes`` — this also
      removes night frames, because ``derive_targets`` drops low-sun sensor
      rows before they reach this function;
    - rows with missing targets (``kt``, ``diffuse``, ``cloud_class``) or
      missing feature values.

    Parameters
    ----------
    manifest:
        Frame manifest from :func:`allsky.video.extract_frames` (columns
        ``frame_path``, ``timestamp``, ``video``, ``index``).
    sensor_df:
        Sensor frame with a ``DatetimeIndex`` (naive local), the configured
        feature columns, and the target columns from ``derive_targets``.
    cfg:
        Sensor config (or root config) providing ``feature_columns`` and
        ``tolerance_minutes``.
    out_path:
        Optional parquet destination; parent directories are created.

    Returns
    -------
    pd.DataFrame
        One row per matched frame: manifest columns, ``sensor_timestamp``,
        sensor features and targets.  ``cloud_class`` is ``int64``.
    """
    scfg = cfg.sensor if isinstance(cfg, AllSkyConfig) else cfg

    missing_features = [c for c in scfg.feature_columns if c not in sensor_df.columns]
    if missing_features:
        raise ValueError(f"sensor frame is missing feature columns: {missing_features}")
    missing_targets = [c for c in TARGET_COLUMNS if c not in sensor_df.columns]
    if missing_targets:
        raise ValueError(
            f"sensor frame is missing target columns {missing_targets}; "
            "run allsky.sensors.derive_targets first"
        )

    frames = manifest.sort_values("timestamp").reset_index(drop=True)
    frames["timestamp"] = pd.to_datetime(frames["timestamp"]).astype("datetime64[ns]")
    sensors = sensor_df.reset_index(names="sensor_timestamp")
    sensors["sensor_timestamp"] = pd.to_datetime(sensors["sensor_timestamp"]).astype(
        "datetime64[ns]"
    )
    sensors = sensors.sort_values("sensor_timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        frames,
        sensors,
        left_on="timestamp",
        right_on="sensor_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=scfg.tolerance_minutes),
    )

    required = ["sensor_timestamp", *TARGET_COLUMNS, *scfg.feature_columns]
    index_df = merged.dropna(subset=required).reset_index(drop=True)
    index_df["cloud_class"] = index_df["cloud_class"].astype("int64")

    n_dropped = len(merged) - len(index_df)
    logger.info(
        "build_index: matched %d/%d frames (%d dropped: unmatched/night/missing targets)",
        len(index_df),
        len(merged),
        n_dropped,
    )

    if out_path is not None:
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        index_df.to_parquet(out_file, index=False)
        logger.info("build_index: wrote %s", out_file)
    return index_df


def infer_feature_columns(index_df: pd.DataFrame) -> list[str]:
    """Numeric pairing-index columns that are not reserved metadata/targets."""
    return [
        c
        for c in index_df.columns
        if c not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(index_df[c])
    ]


@dataclass(frozen=True)
class FeatureStats:
    """Per-feature standardization statistics.

    Must be computed from the **training split only** and reused for
    validation/test datasets (and stored in checkpoint metadata) so no
    information leaks across splits.
    """

    columns: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_frame(cls, index_df: pd.DataFrame, columns: list[str]) -> FeatureStats:
        """Compute mean/std over *columns*; ~zero stds are clamped to 1."""
        values = index_df.loc[:, columns].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        std = np.where(std < _MIN_FEATURE_STD, 1.0, std).astype(np.float32)
        return cls(columns=tuple(columns), mean=mean.astype(np.float32), std=std)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form for checkpoint/index metadata."""
        return {
            "columns": list(self.columns),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureStats:
        """Inverse of :meth:`to_dict`."""
        return cls(
            columns=tuple(payload["columns"]),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
        )


class AllSkyDataset:
    """Map-style dataset pairing sky JPEGs with sensor features and targets.

    Each item is a dict of torch tensors:

    - ``"image"``: ``float32`` CHW in ``[0, 1]``, resized to
      ``model.image_size``;
    - ``"features"``: ``float32`` standardized sensor feature vector;
    - ``"cloud_class"``: ``int64`` weak label (0 clear / 1 partial /
      2 overcast);
    - ``"diffuse"``: ``float32`` diffuse irradiance target in W/m2 (an Erbs
      pseudo-target when ``target_source == "erbs_pseudo"``).

    Deliberately **not** a ``torch.utils.data.Dataset`` subclass: map-style
    ``DataLoader`` consumption only requires ``__len__``/``__getitem__``, and
    keeping torch out of module scope lets ``allsky.dataset`` import without
    torch installed.  torch is imported lazily inside ``__getitem__``.

    Leakage guard: validation/test datasets (``train=False``) must be given
    the :class:`FeatureStats` computed on the training split (available as
    ``.stats`` on the train dataset); computing stats locally is refused.

    Parameters
    ----------
    index_df:
        Pairing index from :func:`build_index`.
    model_cfg:
        Model config (or root config) providing ``image_size``.
    train:
        Whether this is the training split.
    feature_columns:
        Feature column names; inferred via :func:`infer_feature_columns`
        when ``None``.
    stats:
        Standardization statistics; computed from *index_df* only when
        ``train=True`` and ``stats is None``.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        model_cfg: ModelConfig | AllSkyConfig,
        train: bool = True,
        *,
        feature_columns: list[str] | None = None,
        stats: FeatureStats | None = None,
    ) -> None:
        mcfg = model_cfg.model if isinstance(model_cfg, AllSkyConfig) else model_cfg
        self.index = index_df.reset_index(drop=True)
        self.image_size = mcfg.image_size
        self.train = train
        self.feature_columns = (
            list(feature_columns)
            if feature_columns is not None
            else infer_feature_columns(self.index)
        )
        if not self.feature_columns:
            raise ValueError("no feature columns found in the pairing index")

        if stats is None:
            if not train:
                raise ValueError(
                    "train=False requires FeatureStats computed from the training "
                    "split (pass stats=train_dataset.stats) — computing them from "
                    "a validation/test split would leak information"
                )
            stats = FeatureStats.from_frame(self.index, self.feature_columns)
        elif list(stats.columns) != self.feature_columns:
            raise ValueError(
                f"stats columns {list(stats.columns)} do not match "
                f"feature columns {self.feature_columns}"
            )
        self.stats = stats

        raw = self.index.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        self._features = (raw - stats.mean) / stats.std
        self._cloud_class = self.index["cloud_class"].to_numpy(dtype=np.int64)
        self._diffuse = self.index["diffuse"].to_numpy(dtype=np.float32)
        self._paths = [str(p) for p in self.index["frame_path"]]

    def __len__(self) -> int:
        return len(self.index)

    def _load_image(self, path: str) -> np.ndarray:
        """Load a JPEG as float32 CHW in [0, 1], resized to ``image_size``."""
        image = iio.imread(path)
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        import torch

        return {
            "image": torch.from_numpy(self._load_image(self._paths[idx])),
            "features": torch.from_numpy(self._features[idx]),
            "cloud_class": torch.tensor(self._cloud_class[idx], dtype=torch.long),
            "diffuse": torch.tensor(self._diffuse[idx], dtype=torch.float32),
        }
