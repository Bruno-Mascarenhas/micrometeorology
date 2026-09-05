"""Map-style multimodal datasets emitting the new-stack batch contract.

Two datasets share one batch contract (see :class:`MultimodalImageDataset`):

- :class:`MultimodalImageDataset` loads sky JPEGs (paths relative to
  ``data_root``) end-to-end with a PIL decode -> RGB -> bilinear resize -> CHW
  float32 recipe, standardized by the DINOv2 channel statistics so the image
  path feeds the backbone what the offline embedding path feeds it.
- :class:`MultimodalEmbeddingDataset` reads a precomputed visual embedding per
  sample through an :class:`EmbeddingReader`, the minimal
  ``sample_id -> np.ndarray`` protocol that
  :class:`allsky.embeddings.storage.SafetensorsEmbeddingReader` implements.

Both standardize the engineered feature vector with a **train-only**
:class:`allsky.features.FeatureNormalizer` (validation/test must be handed the
training-split normalizer — computing one locally is refused as a leakage
guard).  Targets are emitted in **raw physical units**;
``sky_class == -1`` and NaN regression targets mark missing labels for the loss
to mask.

``torch`` is imported lazily on the first item (never at module import), so
importing ``allsky.data.datasets`` never pulls torch.
"""

import itertools
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, get_args, runtime_checkable

import numpy as np
import pandas as pd

from allsky.clearsky import clearsky_diffuse
from allsky.config import DEFAULT_IMAGE_SIZE, AlignmentStrategyName, DHIParameterization
from allsky.data.contracts import NS_PER_MINUTE, resolve
from allsky.features.normalization import FeatureNormalizer
from allsky.geometry import solar_geometry_maps
from allsky.lens import LensCalibration, isotropic_calibration
from labmim_core.site import STATION_UTC_OFFSET_HOURS

if TYPE_CHECKING:
    # allsky.preprocessing reaches back into allsky.data.contracts, so importing
    # it at runtime would close a cycle through this package's __init__.
    from allsky.augmentation import AugmentationPipeline
    from allsky.preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingReader",
    "MultimodalEmbeddingDataset",
    "MultimodalImageDataset",
]

#: One dataset item: a ``str -> torch.Tensor`` mapping. ``torch`` is imported
#: lazily on the first item, so importing this module never pulls it.
type SampleTensors = dict[str, Any]


#: Dataset-level temporal windowing modes for the embedding dataset. The name
#: set is owned by :data:`allsky.config.AlignmentStrategyName` (a leaf module),
#: so the config that selects a mode and the dataset that implements it can
#: never disagree about which modes exist.
type WindowMode = AlignmentStrategyName
_WINDOW_MODES: tuple[WindowMode, ...] = get_args(AlignmentStrategyName)


@runtime_checkable
class EmbeddingReader(Protocol):
    """Minimal reader interface: ``sample_id -> (D,) float embedding``.

    :class:`allsky.embeddings.storage.SafetensorsEmbeddingReader` is the shipped
    implementation; any callable (or object with ``__call__``) returning a 1-D
    array for a ``sample_id`` — and optionally exposing an integer ``dim`` —
    satisfies this protocol.
    """

    def __call__(self, sample_id: str) -> np.ndarray:
        """Return the ``(D,)`` float embedding stored for *sample_id*."""
        ...


def resolve_time_windows(
    manifest: pd.DataFrame, window_minutes: float, *, max_frames: int | None = None
) -> list[list[int]]:
    """Per-row positional window members: same ``day_id``, within the window.

    For each row the members are the positions whose ``day_id`` matches and whose
    ``timestamp_utc`` is within ``window_minutes / 2`` of the row's own time, in
    time order.  The row's own position is always included (distance zero), so a
    window is never empty.

    Parameters
    ----------
    manifest:
        Frame carrying ``day_id`` and ``timestamp_utc``.
    window_minutes:
        Full width of the window, in minutes.
    max_frames:
        Cap on members per row, evenly subsampled and always keeping the first
        and last. ``None`` keeps every member.

    Returns
    -------
    list of list of int
        One list of manifest positions per row, in time order.
    """
    index = pd.DatetimeIndex(manifest["timestamp_utc"]).tz_convert("UTC").tz_localize(None)
    times_ns = index.as_unit("ns").to_numpy().astype("int64")
    day_codes = pd.factorize(manifest["day_id"].astype(str), sort=False)[0]
    half_ns = round(window_minutes / 2.0 * NS_PER_MINUTE)
    n_rows = len(manifest)
    windows: list[list[int]] = [[] for _ in range(n_rows)]
    order = np.lexsort((np.arange(n_rows), times_ns, day_codes))
    sorted_days = day_codes[order]
    sorted_times = times_ns[order]
    day_starts = np.concatenate(
        ([0], np.flatnonzero(sorted_days[1:] != sorted_days[:-1]) + 1, [n_rows])
    )
    for start, stop in itertools.pairwise(day_starts):
        idx_sorted = order[start:stop]
        t_sorted = sorted_times[start:stop]
        low = np.searchsorted(t_sorted, t_sorted - half_ns, side="left")
        high = np.searchsorted(t_sorted, t_sorted + half_ns, side="right")
        for row, window_start, window_stop in zip(idx_sorted, low, high, strict=True):
            members = idx_sorted[window_start:window_stop].tolist()
            windows[int(row)] = _subsample_window(members, max_frames)
    return windows


def _local_block_ends_ns(
    manifest: pd.DataFrame, block_minutes: float, utc_offset_hours: float
) -> np.ndarray:
    """End of the datalogger block each row falls in, as naive-local ns since epoch."""
    index = pd.DatetimeIndex(manifest["timestamp_utc"]).tz_convert("UTC").tz_localize(None)
    local = index + pd.Timedelta(hours=utc_offset_hours)
    return local.ceil(f"{block_minutes:g}min").as_unit("ns").to_numpy().astype("int64")


def resolve_sensor_block_windows(
    manifest: pd.DataFrame,
    block_minutes: float,
    utc_offset_hours: float,
    *,
    max_frames: int | None = None,
) -> list[list[int]]:
    """Per-row positional window members: the frames that share the row's datalogger block.

    The logger end-stamps a ``block_minutes`` average, so two same-day frames
    belong to one row of it exactly when their local stamps round up to the same
    block end. Members are returned in time order and always include the row.

    Parameters
    ----------
    manifest:
        Frame carrying ``day_id`` and ``timestamp_utc``.
    block_minutes:
        The logger's averaging interval, in minutes.
    utc_offset_hours:
        The site's fixed clock offset; the block boundaries are local.
    max_frames:
        Cap on members per row, evenly subsampled keeping the ends.

    Returns
    -------
    list of list of int
        One list of manifest positions per row, in time order.
    """
    ends = _local_block_ends_ns(manifest, block_minutes, utc_offset_hours)
    times_ns = (
        pd.DatetimeIndex(manifest["timestamp_utc"])
        .tz_convert("UTC")
        .tz_localize(None)
        .as_unit("ns")
        .to_numpy()
        .astype("int64")
    )
    day_codes = pd.factorize(manifest["day_id"].astype(str), sort=False)[0]
    n_rows = len(manifest)
    order = np.lexsort((np.arange(n_rows), times_ns, ends, day_codes))
    keys = np.stack([day_codes[order], ends[order]], axis=1)
    starts = np.concatenate(
        ([0], np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1, [n_rows])
    )
    windows: list[list[int]] = [[] for _ in range(n_rows)]
    for start, stop in itertools.pairwise(starts):
        members = order[start:stop].tolist()
        picked = _subsample_window(members, max_frames)
        for row in members:
            windows[int(row)] = picked
    return windows


def representative_rows_per_block(
    manifest: pd.DataFrame, block_minutes: float, utc_offset_hours: float
) -> np.ndarray:
    """Mask keeping one row per datalogger block: the frame nearest the block centroid.

    The centroid — ``block_minutes / 2`` before the block end — is where the
    average the row carries is centred, and the frame closest to it is the one
    whose instantaneous sky best represents that average.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` bool, aligned to *manifest*'s rows.
    """
    ends = _local_block_ends_ns(manifest, block_minutes, utc_offset_hours)
    local_ns = (
        (
            pd.DatetimeIndex(manifest["timestamp_utc"]).tz_convert("UTC").tz_localize(None)
            + pd.Timedelta(hours=utc_offset_hours)
        )
        .as_unit("ns")
        .to_numpy()
        .astype("int64")
    )
    centroid = ends - round(block_minutes / 2.0 * NS_PER_MINUTE)
    distance = np.abs(local_ns - centroid)
    table = pd.DataFrame(
        {"day": manifest["day_id"].astype(str).to_numpy(), "end": ends, "distance": distance}
    )
    nearest = table.groupby(["day", "end"], sort=False)["distance"].idxmin().to_numpy()
    keep = np.zeros(len(manifest), dtype=bool)
    keep[nearest] = True
    return keep


def _windows_for(
    window: WindowMode,
    manifest: pd.DataFrame,
    window_minutes: float,
    utc_offset_hours: float,
    *,
    max_frames: int | None,
) -> list[list[int]]:
    """The per-row members a windowed *window* mode implies; empty under ``center_frame``."""
    if window == "center_frame":
        return []
    if window == "sensor_block":
        return resolve_sensor_block_windows(
            manifest, window_minutes, utc_offset_hours, max_frames=max_frames
        )
    return resolve_time_windows(manifest, window_minutes, max_frames=max_frames)


def _subsample_window(members: list[int], max_frames: int | None) -> list[int]:
    if max_frames is None or len(members) <= max_frames:
        return members
    picks = np.linspace(0, len(members) - 1, max_frames).round().astype(int)
    return [members[i] for i in picks.tolist()]


class _BaseMultimodalDataset:
    """Shared feature/target handling for the multimodal datasets."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        train: bool = True,
        stats: FeatureNormalizer | None = None,
        dhi_parameterization: DHIParameterization = "raw",
        utc_offset_hours: float = STATION_UTC_OFFSET_HOURS,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.train = train
        self.epoch = 0
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

        self._features = stats.transform(self.manifest)
        self._utc_offset_hours = float(utc_offset_hours)
        self._dhi_scale = self._dhi_scale_column(dhi_parameterization, utc_offset_hours)
        self._dhi = self._raw_target("target_dhi") / self._dhi_scale
        self._kindex = self._raw_target("target_kindex")
        self._cloud_fraction = self._raw_target("cloud_fraction")
        self._sky_class = self.manifest["sky_class"].to_numpy(dtype=np.int64, copy=True)
        self._sample_ids = [str(s) for s in self.manifest["sample_id"]]
        self._columns: SampleTensors | None = None

    @property
    def served_targets(self) -> dict[str, np.ndarray]:
        """The regression target arrays this dataset actually serves."""
        return {"dhi": self._dhi, "kindex": self._kindex}

    def set_epoch(self, epoch: int) -> None:
        """Tell the dataset which pass over the data it is on.

        Only the image dataset reads it — its augmentation seeds on
        ``(seed, epoch, idx)`` — but the training loop calls this on whatever
        dataset it was handed, so the attribute lives here rather than the loop
        asking which kind it holds. Probing for it instead would let a rename
        silently freeze augmentation on the first epoch.
        """
        self.epoch = epoch

    def _dhi_scale_column(
        self, parameterization: DHIParameterization, utc_offset_hours: float
    ) -> np.ndarray:
        if parameterization == "raw":
            return np.ones(len(self.manifest), dtype=np.float32)
        if parameterization != "clearsky_index":
            raise ValueError(
                f"unknown dhi_parameterization {parameterization!r}; expected 'raw' or "
                "'clearsky_index'"
            )
        missing = [c for c in ("solar_zenith", "timestamp_utc") if c not in self.manifest.columns]
        if missing:
            raise ValueError(f"the clear-sky-index DHI target needs the manifest columns {missing}")
        times = pd.to_datetime(self.manifest["timestamp_utc"], utc=True)
        scale = clearsky_diffuse(self.manifest["solar_zenith"], times, utc_offset_hours)
        if not np.all(np.isfinite(scale)) or float(np.min(scale)) <= 0.0:
            raise ValueError(
                "the clear-sky DHI reference is non-positive or non-finite on some rows, so "
                "DHI / DHI_clearsky is undefined there; the night filter is what normally "
                "keeps it away from zero"
            )
        return scale.astype(np.float32)

    def _raw_target(self, column: str) -> np.ndarray:
        """Raw physical target column as float32 (NaN preserved as missing).

        The copy is deliberate: pandas hands back a read-only block view when no
        cast is needed, and ``torch.from_numpy`` on a read-only array warns and
        yields a tensor whose writes are undefined behaviour.
        """
        if column in self.manifest.columns:
            return self.manifest[column].to_numpy(dtype=np.float32, copy=True)
        return np.full(len(self.manifest), np.nan, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def _column_tensors(self) -> SampleTensors:
        """Whole-column tensors over the resident numpy columns, built once.

        Built on first use rather than in ``__init__`` so importing this module
        stays torch-free and so each DataLoader worker builds its own instead of
        having them pickled in (Python 3.14 defaults to the forkserver start
        method).  ``features`` is copied C-contiguous once here, which turns
        every per-sample row read into a contiguous view.
        """
        if self._columns is None:
            import torch

            self._columns = {
                "features": torch.from_numpy(np.ascontiguousarray(self._features)),
                "dhi": torch.from_numpy(self._dhi),
                "dhi_scale": torch.from_numpy(self._dhi_scale),
                "kindex": torch.from_numpy(self._kindex),
                "sky_class": torch.from_numpy(self._sky_class),
                "cloud_fraction": torch.from_numpy(self._cloud_fraction),
            }
        return self._columns

    def _target_item(self, idx: int) -> SampleTensors:
        """Shared target tensors for row *idx*, as views into the column tensors.

        Indexing the prebuilt columns replaces four per-sample ``torch.tensor``
        allocations.  The emitted tensors are **views** into dataset-owned
        buffers: ``default_collate`` stacks (and therefore copies) them, so
        batches are unaffected, but a caller that mutates an item in place would
        be mutating the dataset.
        """
        return {name: column[idx] for name, column in self._column_tensors().items()}


class MultimodalImageDataset(_BaseMultimodalDataset):
    """Sky-image + sensor dataset serving the new-stack batch contract.

    Each item is a dict of torch tensors:

    - ``features`` — float32 ``(F,)`` standardized sensor vector;
    - ``image`` — float32 ``(C, H, W)`` under ``center_frame``, or ``image_seq``
      ``(T, C, H, W)`` plus ``frame_mask`` ``(T,)`` under a windowed strategy;
      resized to *image_size*, with three
      standardized RGB planes, plus the
      :data:`~allsky.geometry.GEOMETRY_CHANNEL_NAMES` maps when
      *geometry_channels* is set;
    - ``dhi`` — float32 diffuse target, NaN when missing: W/m2 under the default
      ``raw`` parameterization, and ``DHI / DHI_clearsky`` (dimensionless) under
      ``clearsky_index``;
    - ``dhi_scale`` — float32 W/m2 divisor that produced ``dhi``; exactly ``1.0``
      under ``raw``. Multiplying ``dhi`` by it always gives W/m2, which is how
      the engine and the evaluator report one number whatever the head fitted;
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
    geometry_channels:
        Names of the per-pixel solar geometry maps of
        :func:`allsky.geometry.solar_geometry_maps` to append to every frame,
        empty for none.  They are built at
        *image_size* from :func:`allsky.lens.isotropic_calibration` and the row's
        own solar angles.  Only valid for frames from the isotropic
        re-extraction: on an anisotropically resized frame the maps would
        describe optics the image does not have.

    Raises
    ------
    ValueError
        If *feature_columns* is empty or names a column *manifest* lacks, if
        *stats* covers different columns, if ``train=False`` is passed without
        the training split's *stats* (the leakage guard), or if
        *geometry_channels* is asked for without finite solar angles or
        alongside a translating augmentation.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_columns: Sequence[str],
        *,
        data_root: str | Path,
        image_size: int = DEFAULT_IMAGE_SIZE,
        train: bool = True,
        stats: FeatureNormalizer | None = None,
        augment: AugmentationPipeline | None = None,
        preprocess: PreprocessingPipeline | None = None,
        seed: int = 0,
        geometry_channels: Sequence[str] = (),
        frame_geometry: Mapping[str, Any] | None = None,
        dhi_parameterization: DHIParameterization = "raw",
        utc_offset_hours: float = STATION_UTC_OFFSET_HOURS,
        window: WindowMode = "center_frame",
        window_minutes: float = 10.0,
        window_max_frames: int = 5,
    ) -> None:
        super().__init__(
            manifest,
            feature_columns,
            train=train,
            stats=stats,
            dhi_parameterization=dhi_parameterization,
            utc_offset_hours=utc_offset_hours,
        )
        self.data_root = data_root
        self.image_size = image_size
        # Resolved once here rather than per __getitem__: the join rebuilds a
        # PurePosixPath and two Paths, and a dataloader worker pays it on every
        # sample of every epoch.
        self._paths = [resolve(str(p), self.data_root) for p in self.manifest["image_path"]]
        # Augmentation is a training-split transform by definition: applying it
        # to val/test would measure the model on pixels the sensor never saw.
        self.augment = augment if train else None
        # Preprocessing is NOT gated on `train`: it must be identical wherever
        # the model runs, or inference sees pixels training never produced.
        self.preprocess = preprocess
        self._seed = int(seed)
        if window not in _WINDOW_MODES:
            raise ValueError(f"window must be one of {_WINDOW_MODES}, got {window!r}")
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        if window_max_frames < 1:
            raise ValueError(f"window_max_frames must be at least 1, got {window_max_frames}")
        self.window = window
        self.window_minutes = float(window_minutes)
        self.seq_len = int(window_max_frames)
        self._windows: list[list[int]] = _windows_for(
            window, manifest, self.window_minutes, utc_offset_hours, max_frames=self.seq_len
        )
        self._geometry_channels = tuple(geometry_channels)
        self.frame_geometry = frame_geometry
        self._geometry = (
            self._geometry_source(manifest, augment if train else None)
            if self._geometry_channels
            else None
        )

    def _geometry_source(
        self, manifest: pd.DataFrame, augment: AugmentationPipeline | None
    ) -> tuple[LensCalibration, np.ndarray, np.ndarray]:
        missing = [c for c in ("solar_zenith", "solar_azimuth") if c not in manifest.columns]
        if missing:
            raise ValueError(
                f"geometry channels need the manifest columns {missing}, which it lacks"
            )
        if augment is not None and augment.p_translate > 0:
            raise ValueError(
                "geometry channels are incompatible with p_translate > 0: the frame would shift "
                "while the geometry maps, built from the lens, would not follow it"
            )
        zenith_deg = manifest["solar_zenith"].to_numpy(dtype=np.float64)
        azimuth_deg = manifest["solar_azimuth"].to_numpy(dtype=np.float64)
        if not (np.isfinite(zenith_deg).all() and np.isfinite(azimuth_deg).all()):
            raise ValueError(
                "geometry channels need finite solar_zenith / solar_azimuth on every row"
            )
        self._refuse_geometry_over_unknown_frames()
        return isotropic_calibration(self.image_size), zenith_deg, azimuth_deg

    def _refuse_geometry_over_unknown_frames(self) -> None:
        """Refuse the geometry planes for frames not written isotropically.

        :func:`~allsky.lens.isotropic_calibration` describes ONE geometry: the
        disc centred and inscribed by the prepare crop and pad, then resized
        square. Applied to a frame that went through the plain 1920x1080 resize
        it puts the horizon where the frame does not have one, and the planes
        describe a lens the pixels were never taken through — silently, because
        the shapes agree.

        The manifest's ``frame_geometry`` is what says which of the two a
        dataset holds. A manifest built before it was recorded says nothing, so
        that case warns rather than refusing: every dataset of that vintage would
        otherwise stop loading.

        Raises
        ------
        ValueError
            When the recorded geometry enables neither the crop nor the pad, so
            the frames are not the inscribed disc the calibration describes.
        """
        if self.frame_geometry is None:
            logger.warning(
                "geometry channels are built on the isotropic lens calibration, and this "
                "manifest records no frame_geometry to confirm its frames were written that "
                "way; re-run prepare-local to record it"
            )
            return
        crop = self.frame_geometry.get("crop") or {}
        pad = self.frame_geometry.get("pad") or {}
        if not crop.get("enabled") and not pad.get("enabled"):
            raise ValueError(
                "geometry channels need frames written through the isotropic crop/pad, but "
                "this manifest's frame_geometry enables neither; the planes would describe a "
                "lens these pixels were never taken through"
            )

    def _load_image(self, image_path: Path, idx: int = 0) -> np.ndarray:
        """Load a JPEG as a standardized float32 CHW array, resized to ``image_size``.

        The chain is decode -> ``[0, 1]`` -> preprocess -> resize -> augment
        -> standardize.

        Preprocessing runs at NATIVE resolution, before the resize: filling the
        timestamp band after downscaling leaves glyph pixels smeared into the
        sky rows below it (measured on this camera: up to 0.031 of residual in
        the six rows under the band). Augmentation runs on the ``[0, 1]`` frame
        because every transform in :mod:`allsky.augmentation` is defined there,
        and standardisation stays last so the backbone always receives its
        pretraining distribution. ``idx`` seeds the augmentation together with the seed and the epoch, and it is
        the SERVED row's index even when the frame read is a co-frame of that
        row's window.

        PIL decode -> RGB (``convert`` channel-replicates grayscale) -> bilinear
        resize. ``image_path`` is already resolved against ``data_root``. On the
        no-preprocessing path, decoding straight with PIL is pixel-identical to
        reading through imageio and wrapping the array back into an image to
        resize it, without the two full-frame numpy<->PIL copies; the
        preprocessing path necessarily pays them, because the stage is defined
        on float CHW and PIL cannot bilinear-resize one.
        """
        # Imported here because allsky.preprocessing reaches back into
        # allsky.data.contracts, so a module-level import would close a cycle
        # through this package's __init__.
        from allsky.preprocessing import imagenet_standardize, model_input_frame

        chw = model_input_frame(
            image_path,
            size=self.image_size,
            preprocess=self.preprocess,
        )
        # `chw` was allocated there, so standardising in place costs no copy.
        standardized = imagenet_standardize(self._augmented(chw, idx), copy=False)
        if self._geometry is None:
            return standardized
        calibration, zenith_deg, azimuth_deg = self._geometry
        maps = solar_geometry_maps(
            calibration,
            (self.image_size, self.image_size),
            sun_zenith_rad=float(np.radians(zenith_deg[idx])),
            sun_azimuth_rad=float(np.radians(azimuth_deg[idx])),
            channels=self._geometry_channels,
        )
        return np.concatenate([standardized, maps], axis=0)

    def _augmented(self, chw: np.ndarray, idx: int) -> np.ndarray:
        """Augment one frame with a per-sample seeded generator.

        The generator is derived from ``(seed, epoch, idx)`` rather than drawn
        from a shared stream, so a dataloader worker cannot change which
        transform a sample gets — the engine's own docstring warns that worker
        RNG would otherwise leak into the batch — while the epoch term keeps the
        draw varying across passes.
        """
        if self.augment is None or not self.augment.enabled:
            return chw
        rng = np.random.default_rng((self._seed, self.epoch, idx))
        augmented: np.ndarray = self.augment(chw, rng, self._imaged_pixels(chw.shape[1:]))
        return augmented

    def _imaged_pixels(self, shape: tuple[int, int]) -> np.ndarray | None:
        """``(H, W)`` bool mask of the pixels the camera actually imaged.

        Derived from the ROI this run's own preprocessing applies, which is the
        one absent region the dataset can see. The isotropic pad is NOT covered:
        it is written by ``PrepareConfig``, which no ``ExperimentConfig`` carries,
        so a padded dataset still has its fill treated as sky by the two
        transforms below. ``None`` when the run masks nothing.
        """
        from allsky.preprocessing import roi_keep_mask

        radius = self.preprocess.roi_radius_fraction if self.preprocess is not None else None
        if radius is None:
            return None
        return roi_keep_mask(shape[0], shape[1], radius)

    def __getitem__(self, idx: int) -> SampleTensors:
        """Row *idx*: the shared targets plus its frame, or its window of frames.

        Under ``center_frame`` the item carries ``image`` ``(C, H, W)``. Under any
        windowed strategy it carries ``image_seq`` ``(T, C, H, W)`` zero-padded to
        ``seq_len`` and ``frame_mask`` ``(T,)`` bool marking the real frames.
        """
        import torch

        item = self._target_item(idx)
        if self.window == "center_frame":
            item["image"] = torch.from_numpy(self._load_image(self._paths[idx], idx))
            return item
        members = self._windows[idx]
        frames = np.zeros((self.seq_len, *self._frame_shape()), dtype=np.float32)
        mask = np.zeros(self.seq_len, dtype=bool)
        for slot, position in enumerate(members):
            # Seeded on the SERVED row, never on the co-frame's own position: a
            # per-frame draw gives each frame of one window an independent
            # exposure and noise realisation, which is scintillation the sky did
            # not produce — and the window exists precisely to average the sky.
            frames[slot] = self._load_image(self._paths[position], idx)
            mask[slot] = True
        item["image_seq"] = torch.from_numpy(frames)
        item["frame_mask"] = torch.from_numpy(mask)
        return item

    def _frame_shape(self) -> tuple[int, int, int]:
        return (3 + len(self._geometry_channels), self.image_size, self.image_size)


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

    Raises
    ------
    ValueError
        If *window* is not one of the modes above, if *window_minutes* is not
        positive, or for any of the feature/normalizer failures listed on
        :class:`MultimodalImageDataset`.
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
        dhi_parameterization: DHIParameterization = "raw",
        utc_offset_hours: float = STATION_UTC_OFFSET_HOURS,
    ) -> None:
        super().__init__(
            manifest,
            feature_columns,
            train=train,
            stats=stats,
            dhi_parameterization=dhi_parameterization,
            utc_offset_hours=utc_offset_hours,
        )
        if window not in _WINDOW_MODES:
            raise ValueError(f"window must be one of {_WINDOW_MODES}, got {window!r}")
        if window_minutes <= 0:
            raise ValueError(f"window_minutes must be positive, got {window_minutes}")
        self.embedding_reader = embedding_reader
        declared = getattr(embedding_reader, "dim", None)
        self._embedding_dim = int(declared) if declared is not None else None
        self.window = window
        self.window_minutes = float(window_minutes)
        #: Fixed padded window length ``T`` for ``attention_pooling``.
        self.seq_len = math.ceil(self.window_minutes) + 1
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
        return _windows_for(
            self.window, self.manifest, self.window_minutes, self._utc_offset_hours, max_frames=None
        )

    def _read(self, sample_id: str) -> np.ndarray:
        """Read + validate the ``(D,)`` float32 embedding for *sample_id*.

        The array may be a read-only view into the preloaded store, so every
        caller copies before ``torch.from_numpy``: that wraps the buffer without
        copying, and a tensor backed by read-only memory is undefined behaviour
        the moment anything writes through it.
        """
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
        """Row *idx*: the shared target tensors plus its visual payload.

        The payload is ``embedding`` ``(D,)`` float32 under ``center_frame`` and
        ``mean_embedding``; under ``attention_pooling`` it is ``embedding_seq``
        ``(T, D)`` float32 zero-padded to :attr:`seq_len` plus ``frame_mask``
        ``(T,)`` bool, True where the row of the sequence is a real frame.
        """
        import torch

        item = self._target_item(idx)
        if self.window == "center_frame":
            embedding = self._read(self._sample_ids[idx])
            item["embedding"] = torch.from_numpy(np.array(embedding, copy=True))
            return item

        vectors = self._window_embeddings(idx)
        if not vectors:
            vectors = [self._read(self._sample_ids[idx])]

        if self.window in ("mean_embedding", "sensor_block"):
            pooled = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
            item["embedding"] = torch.from_numpy(np.array(pooled, copy=True))
            return item

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
