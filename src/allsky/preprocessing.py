"""All-sky frame preprocessing: static mask, crop, resize and visual QC.

The preparation pipeline turns a raw decoded video frame (``uint8`` RGB, shape
``(H, W, 3)``) into the analysis-ready image the manifest points at, and flags
frames that are unusable for radiometric reasons:

- :func:`apply_static_mask` blacks out everything outside the sky region — a
  PNG mask when one is supplied, otherwise a **heuristic** circular fisheye
  estimate (:func:`estimate_circular_mask`);
- :func:`center_crop` extracts a centred box (``top`` / ``left`` shift the box
  off-centre when the sky disc is not centred);
- :func:`resize_image` bilinearly resizes (the same PIL recipe as
  :func:`allsky.video.extract_frames`);
- :func:`visual_qc` returns the :class:`~allsky.data.contracts.QCFlag` bits
  ``FRAME_DARK`` (mean luminance below a threshold), ``FRAME_SATURATED``
  (too large a fraction of fully-clipped white pixels) and ``FRAME_UNREADABLE``
  (a frame with no pixels at all, on which nothing can be measured);
- :func:`process_frame` composes mask -> crop -> pad -> resize from a
  :class:`~allsky.config.PrepareConfig`, optionally reusing a mask decoded once
  by :func:`resolve_mask` instead of re-reading the PNG for every frame.

Everything is pure numpy + PIL: importing this module never pulls torch.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from allsky.config import (
    TIMESTAMP_BAND_FRACTION,
    CropConfig,
    ExperimentConfig,
    OverlayPolicy,
    PadConfig,
    PrepareConfig,
)
from allsky.data.contracts import QCFlag
from allsky.frame_pixels import as_rgb_uint8 as _as_rgb_uint8
from allsky.frame_pixels import decode_rgb, decode_rgb_resized, resize_bilinear
from allsky.lens import LensCalibration

__all__ = [
    "DARK_LUMINANCE_THRESHOLD",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "SATURATED_FRACTION_THRESHOLD",
    "SATURATED_LEVEL",
    "TIMESTAMP_BAND_FRACTION",
    "OverlayPolicy",
    "PreprocessingPipeline",
    "apply_static_mask",
    "center_crop",
    "estimate_circular_mask",
    "imagenet_standardize",
    "load_mask",
    "model_input_frame",
    "pad_frame",
    "process_frame",
    "remove_timestamp_band",
    "resize_image",
    "resolve_mask",
    "visual_qc",
]

#: Channel mean/std DINOv2 was pretrained with. Every path that feeds the
#: backbone must standardize with these: the model was trained on inputs with
#: roughly zero mean and unit variance per channel, and a raw ``[0, 1]`` frame
#: sits about 1.3 sigma low on red with a quarter of the expected spread.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

_IMAGENET_MEAN_CHW = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_STD_CHW = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
_IMAGENET_MEAN_CHW.flags.writeable = False
_IMAGENET_STD_CHW.flags.writeable = False

#: The ``fill`` constant on the native uint8 scale. The float route writes
#: :data:`IMAGENET_MEAN` and the dataset re-quantises it; these are the levels
#: that round trip lands on, so the uint8 route writes them directly.
_FILL_LEVELS = np.round(np.asarray(IMAGENET_MEAN, dtype=np.float64) * 255.0).astype(np.uint8)
_FILL_LEVELS.flags.writeable = False


def imagenet_standardize(chw: np.ndarray, *, copy: bool = True) -> np.ndarray:
    """Standardize a CHW float array in ``[0, 1]`` by the DINOv2 channel stats.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 array scaled to ``[0, 1]``.
    copy:
        ``False`` standardizes in place and returns *chw* itself, which saves
        two full-frame allocations per call — 0.5 ms on a 224 px frame, about
        19 s of worker CPU per epoch, measured. Pass it only when the buffer is
        yours; it requires float32, since in place there is no cast to land on.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32, dimensionless, standardized per channel.

    Raises
    ------
    TypeError
        If ``copy=False`` is passed for an array that is not float32.
    """
    if not copy:
        if chw.dtype != np.float32:
            raise TypeError(f"in-place standardisation needs float32, got {chw.dtype}")
        chw -= _IMAGENET_MEAN_CHW
        chw /= _IMAGENET_STD_CHW
        return chw
    standardized = (chw - _IMAGENET_MEAN_CHW) / _IMAGENET_STD_CHW
    return standardized.astype(np.float32, copy=False)


def remove_timestamp_band(
    chw: np.ndarray,
    *,
    policy: OverlayPolicy = "fill",
    band_fraction: float = TIMESTAMP_BAND_FRACTION,
    fill_value: tuple[float, float, float] = IMAGENET_MEAN,
) -> np.ndarray:
    """Deal with the camera's burned-in timestamp at the top of the frame.

    The overlay is static furniture: it is not sky, it carries no irradiance
    information, and at 224 px it covers three of the sixteen patch rows (36 of
    224 rows, the third of them only in part). An
    occlusion probe on a trained checkpoint measured the band at 1.4-1.6x the
    weight of an equal-area sky band — real, but far from the dominant signal,
    so do not expect removing it to transform the model.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``.
    policy:
        ``keep`` leaves the frame untouched (historical default).

        ``fill`` paints the band with :data:`IMAGENET_MEAN`. **This is the
        recommended setting.** It removes the glyphs, fabricates nothing, and
        leaves the rest of the geometry intact. It does create a hard horizontal
        edge, but a ViT reads that as a set of constant tokens, which is far more
        benign than it would be for a CNN. The constant standardises to ``0``,
        though exactly ``0`` only on the direct path: the dataset re-quantises
        through uint8 and resizes first, which leaves the band near 8e-3
        (measured) and blurs its lower edge.

        ``inpaint`` mirrors the rows below the band back over it. The seam is
        smoother, but that smoothness is the problem: a diffuse-irradiance
        regressor consumes cloud texture, and mirroring fabricates plausible sky
        texture across 14.6 % of the frame — a systematic, deterministic
        hallucination the model will read as real cloud. Offered for ablation.

        ``crop`` removes the band and pads at the bottom. It discards physics:
        those rows are genuine low-elevation dome, and horizon brightness is a
        real contributor to DHI. Ablation only.
    band_fraction:
        Height of the band as a fraction of the frame. Measured on this camera:
        the overlay reaches y=75 of 512 (0.1465); the default adds a margin.
    fill_value:
        Per-channel constant for ``fill``; defaults to :data:`IMAGENET_MEAN`.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32; the same array when *policy* is ``keep``.
    """
    if policy == "keep":
        return chw
    band = _band_rows(chw.shape[1], band_fraction)
    if policy == "fill":
        out = chw.copy()
        out[:, :band, :] = np.asarray(fill_value, dtype=np.float32).reshape(3, 1, 1)
        return out
    if policy == "inpaint":
        source = chw[:, band : 2 * band, :]
        if source.shape[1] < band:
            source = np.repeat(chw[:, band : band + 1, :], band, axis=1)
        out = chw.copy()
        out[:, :band, :] = source[:, ::-1, :]
        return out
    if policy == "crop":
        cropped = chw[:, band:, :]
        pad = np.repeat(cropped[:, -1:, :], band, axis=1)
        return np.ascontiguousarray(np.concatenate([cropped, pad], axis=1))
    raise ValueError(f"unknown overlay policy {policy!r}")


def _band_rows(height: int, band_fraction: float) -> int:
    """How many rows the overlay band covers, shared by both frame layouts.

    Raises
    ------
    ValueError
        If the band would cover no row at all, or the whole frame.
    """
    band = round(band_fraction * height)
    if band < 1:
        raise ValueError(
            f"band fraction {band_fraction} covers no row of a frame of height {height}"
        )
    if band >= height:
        raise ValueError(f"band {band} covers the whole frame of height {height}")
    return band


@lru_cache(maxsize=8)
def roi_keep_mask(height: int, width: int, radius_fraction: float) -> np.ndarray:
    """Read-only ``(H, W)`` BOOL keep-mask for the ROI disc.

    The same disc :func:`_roi_keep` multiplies by, as the mask the augmentation
    needs to tell an absent pixel from a dark one.
    """
    keep = _roi_keep(height, width, radius_fraction) > 0.0
    keep.flags.writeable = False
    return keep


def _roi_keep(height: int, width: int, radius_fraction: float) -> np.ndarray:
    """Read-only ``(H, W)`` float32 keep-mask, built once per geometry.

    The disc depends only on the frame shape and the radius, all fixed for a
    run, but this is called once per sample in the dataloader.
    """
    keep = estimate_circular_mask((height, width), radius_fraction=radius_fraction)
    as_float = keep.astype(np.float32)
    as_float.flags.writeable = False
    return as_float


@lru_cache(maxsize=8)
def _roi_drop_uint8(height: int, width: int, radius_fraction: float) -> np.ndarray:
    """Read-only ``(H, W)`` bool array marking the pixels the ROI discards."""
    drop = ~estimate_circular_mask((height, width), radius_fraction=radius_fraction)
    drop.flags.writeable = False
    return drop


@dataclass(frozen=True, slots=True)
class PreprocessingPipeline:
    """Deterministic transforms applied to EVERY split, train and inference alike.

    This is the difference from :class:`allsky.augmentation.AugmentationPipeline`:
    augmentation is random and training-only, preprocessing is fixed and must be
    byte-identical wherever the model runs. The settings therefore travel in
    ``checkpoint["config"]``, and every path that turns a frame into model input
    rebuilds this pipeline from there.

    Every field defaults to the historical behaviour, so a config that does not
    mention preprocessing reproduces the numbers it reproduced before.

    Attributes
    ----------
    overlay:
        What to do with the burned-in timestamp band; see
        :func:`remove_timestamp_band`.
    band_fraction:
        Height of that band as a fraction of the frame.
    roi_radius_fraction:
        When set, keep only a centred disc of this fraction of ``min(H, W) / 2``
        and zero the rest — the sky dome, without the frame furniture around it.
        ``None`` disables it. The disc is centred on the image because the
        isotropic re-extraction (``configs/allsky/data/local_prepare_iso.yaml``)
        already makes the dome concentric with the frame; a measured centre lives
        in :class:`allsky.lens.LensCalibration`, built for that geometry by
        :func:`allsky.lens.isotropic_calibration`.
    """

    overlay: OverlayPolicy = "keep"
    band_fraction: float = TIMESTAMP_BAND_FRACTION
    roi_radius_fraction: float | None = None

    @classmethod
    def from_config(cls, cfg: ExperimentConfig) -> PreprocessingPipeline:
        """The pipeline *cfg* declares, as every path that scores a frame rebuilds it.

        Training, offline evaluation and live snapshot scoring must each hold
        the same transforms or the model sees pixels it was not fitted on, and
        nothing reports the difference. One reader, so a field added to
        :class:`~allsky.config.PreprocessingConfig` reaches all three at once.

        This covers ``PreprocessingConfig`` — the overlay band and the ROI — and
        nothing else. The mask/crop/pad/resize that shaped the frames on disk
        live on :class:`~allsky.config.PrepareConfig`, which no
        ``ExperimentConfig`` carries; they travel as the checkpoint's own
        ``frame_geometry`` and reach :func:`model_input_frame` through its
        *geometry* argument.
        """
        return cls(**cfg.preprocessing.model_dump())

    @property
    def enabled(self) -> bool:
        """True when the pipeline changes any pixel."""
        return self.overlay != "keep" or self.roi_radius_fraction is not None

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        """Apply the pipeline to one ``(3, H, W)`` float32 frame in ``[0, 1]``.

        No production path calls this: every frame that reaches a model goes
        through :meth:`apply_uint8_hwc`, which runs before the ``[0, 1]`` scaling
        and is the cheaper of the two. This route exists as the equality oracle
        the tests hold that one against — same pixels, different arithmetic — so
        deleting it would remove the only independent check on it.
        """
        out = remove_timestamp_band(chw, policy=self.overlay, band_fraction=self.band_fraction)
        if self.roi_radius_fraction is not None:
            out = out * _roi_keep(out.shape[1], out.shape[2], self.roi_radius_fraction)
        return np.ascontiguousarray(out, dtype=np.float32)

    def apply_uint8_hwc(self, hwc: np.ndarray) -> np.ndarray:
        """The same transform, on the frame as decoded — ``(H, W, 3)`` ``uint8``.

        Callers that hold a decoded frame and want a decoded frame back should
        use this instead of :meth:`__call__`: the float route costs a
        ``[0, 1]`` conversion, a transpose, six full-frame float buffers and a
        re-quantisation, measured at 9.13 ms per 512x512 frame against 2.61 ms
        here, and the dataloader pays it once per sample.

        It produces the **same pixels**, not merely similar ones, because every
        stage is a constant write, a data move, or a multiply by exactly 0 or 1:
        the ``fill`` constant ``IMAGENET_MEAN * 255`` lands on the exact levels
        ``(124, 116, 104)`` the float route re-quantises to, ``inpaint`` and
        ``crop`` only move pixels, and the ROI writes exact zeros.
        ``tests/allsky/test_preprocessing_pipeline.py`` pins the equality over
        every policy so the two routes cannot drift apart.

        Parameters
        ----------
        hwc:
            ``(H, W, 3)`` ``uint8`` RGB frame on the native 0-255 scale.

        Returns
        -------
        numpy.ndarray
            ``(H, W, 3)`` ``uint8``; the input array itself when the pipeline is
            disabled.
        """
        if not self.enabled:
            return hwc
        out = hwc
        if self.overlay != "keep":
            band = _band_rows(hwc.shape[0], self.band_fraction)
            if self.overlay == "fill":
                out = hwc.copy()
                out[:band] = _FILL_LEVELS
            elif self.overlay == "inpaint":
                source = hwc[band : 2 * band]
                if source.shape[0] < band:
                    source = np.repeat(hwc[band : band + 1], band, axis=0)
                out = hwc.copy()
                out[:band] = source[::-1]
            elif self.overlay == "crop":
                cropped = hwc[band:]
                pad = np.repeat(cropped[-1:], band, axis=0)
                out = np.concatenate([cropped, pad], axis=0)
            else:
                raise ValueError(f"unknown overlay policy {self.overlay!r}")
        if self.roi_radius_fraction is not None:
            if out is hwc:
                out = hwc.copy()
            out[_roi_drop_uint8(out.shape[0], out.shape[1], self.roi_radius_fraction)] = 0
        return np.ascontiguousarray(out)


def model_input_frame(
    source: str | Path | bytes,
    *,
    size: int,
    preprocess: PreprocessingPipeline | None = None,
    geometry: PrepareConfig | None = None,
) -> np.ndarray:
    """Decode one frame into the array a visual backbone is fed, minus the standardisation.

    The chain the dataset's own frames went through, in one place: decode ->
    prepare geometry (mask/crop/pad/resize, *geometry*) -> preprocess at the
    resulting resolution -> resize -> ``[0, 1]`` CHW. It deliberately stops
    short of :func:`imagenet_standardize` because the training dataset augments
    between the two and the serving path does not — that one step is the only
    difference the two sides are allowed to have, and keeping the rest here is
    what stops them drifting.

    *geometry* is the ``PrepareConfig`` stage that ran once, at extraction time,
    to write the JPEGs the dataset reads; the dataset itself never sees it.
    Serving therefore has to re-apply it, and that is exactly what
    :class:`~allsky.config.ExperimentConfig` could not describe: a checkpoint
    trained on the isotropic dataset was fed a 1.78x horizontally squeezed
    full frame, with nothing detecting the mismatch.

    The preprocessing runs before the final resize on purpose: the overlay band
    is a fixed fraction of the frame, and a bilinear resize would smear its edge
    into the sky below it.

    Parameters
    ----------
    source:
        Path to the frame, or its encoded bytes.
    size:
        Side of the square the model expects, in pixels.
    preprocess:
        Deterministic pipeline from the run's config; ``None`` or a disabled one
        leaves the pixels alone.
    geometry:
        The prepare geometry the training frames were written through; ``None``
        leaves the decoded pixels as they are.

    Returns
    -------
    numpy.ndarray
        ``(3, size, size)`` float32 in ``[0, 1]``, C-contiguous and freshly
        allocated — so the caller may standardize it in place.
    """
    if geometry is not None:
        arr = process_frame(decode_rgb(source), geometry)
        if preprocess is not None and preprocess.enabled:
            arr = preprocess.apply_uint8_hwc(arr)
        if arr.shape[0] != size or arr.shape[1] != size:
            arr = resize_bilinear(arr, size)
    elif preprocess is not None and preprocess.enabled:
        arr = preprocess.apply_uint8_hwc(decode_rgb(source))
        if arr.shape[0] != size or arr.shape[1] != size:
            arr = resize_bilinear(arr, size)
    else:
        arr = decode_rgb_resized(source, size)
    chw = np.ascontiguousarray(np.asarray(arr, dtype=np.uint8).transpose(2, 0, 1))
    scaled = chw.astype(np.float32)
    scaled /= 255.0
    return scaled


#: BT.601 luminance weights (R, G, B) used by :func:`visual_qc`.
_LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)

#: Mean luminance (0-255) below which a frame is flagged ``FRAME_DARK`` — the
#: default night/twilight threshold, overridable per call.
DARK_LUMINANCE_THRESHOLD = 10.0

#: A pixel is "saturated" when every channel is at or above this level.
SATURATED_LEVEL = 255

#: Fraction of saturated pixels above which a frame is flagged ``FRAME_SATURATED``.
SATURATED_FRACTION_THRESHOLD = 0.2

#: Grayscale threshold used to binarize a PNG mask when the config leaves it auto.
_DEFAULT_MASK_THRESHOLD = 127.0


def load_mask(path: str | Path, *, threshold: float | None = None) -> np.ndarray:
    """Load a PNG mask as a boolean keep-array (``True`` = keep the pixel).

    The image is read as grayscale and binarized at *threshold* (default
    :data:`_DEFAULT_MASK_THRESHOLD`): pixels strictly above the threshold are
    kept.  Any PIL-readable format works; the ``.png`` convention is a naming
    hint only.

    Returns
    -------
    numpy.ndarray
        Keep-array of shape ``(H, W)``, ``bool``, in image (row, column)
        coordinates matching the frames it will be applied to.
    """
    from PIL import Image

    with Image.open(path) as handle:
        gray = np.asarray(handle.convert("L"))
    thr = _DEFAULT_MASK_THRESHOLD if threshold is None else float(threshold)
    return gray > thr


def estimate_circular_mask(
    shape: tuple[int, ...],
    *,
    radius_fraction: float = 1.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Heuristic circular fisheye mask for an image of *shape*.

    A disc of radius ``radius_fraction * min(H, W) / 2`` centred on the image
    (or on *center* ``(row, col)``) is kept; everything outside is dropped.

    Thin wrapper over :meth:`allsky.lens.LensCalibration.centred_in`, which owns
    the geometry — the same description of the optics that
    :meth:`~allsky.lens.LensCalibration.pixel_of` uses to place the sun.

    Parameters
    ----------
    shape:
        Image shape; only the leading ``(H, W)`` entries are read, so a
        ``(H, W, 3)`` frame shape can be passed straight through.
    radius_fraction:
        Disc radius as a fraction of half the shorter image dimension.
    center:
        Disc centre in image ``(row, col)`` pixel coordinates; the image
        centre when None.

    Returns
    -------
    numpy.ndarray
        Keep-array of shape ``(H, W)``, ``bool``.

    Limitation
    ----------
    This is a **heuristic**; see
    :meth:`~allsky.lens.LensCalibration.centred_in` for what it assumes and what
    it cannot represent.
    """
    calibration = LensCalibration.centred_in(shape, radius_fraction=radius_fraction, centre=center)
    return calibration.keep_mask(shape)


def apply_static_mask(
    image: np.ndarray,
    mask: str | Path | np.ndarray | None = None,
    *,
    threshold: float | None = None,
) -> np.ndarray:
    """Black out everything outside *mask*; return a masked copy of *image*.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame.
    mask:
        A PNG path (loaded via :func:`load_mask`), a boolean/keep array matching
        the image's ``(H, W)``, or ``None`` to fall back to the heuristic
        :func:`estimate_circular_mask`.
    threshold:
        Grayscale binarization threshold forwarded to :func:`load_mask` when
        *mask* is a path (ignored otherwise).

    Returns
    -------
    numpy.ndarray
        Masked copy of the frame, shape ``(H, W, 3)``, ``uint8``, with every
        dropped pixel set to 0 in all three channels.  The input is not
        modified.

    Raises
    ------
    ValueError
        If a supplied mask's 2-D shape does not match the image.
    """
    arr = _as_rgb_uint8(image)
    if mask is None:
        keep = estimate_circular_mask(arr.shape)
    elif isinstance(mask, (str, Path)):
        keep = load_mask(mask, threshold=threshold)
    else:
        keep = np.asarray(mask, dtype=bool)
    if keep.shape != arr.shape[:2]:
        raise ValueError(
            f"mask shape {keep.shape} does not match image spatial shape {arr.shape[:2]}"
        )
    out = arr.copy()
    out[~keep] = 0
    return out


def center_crop(image: np.ndarray, crop: CropConfig) -> np.ndarray:
    """Extract the centred crop described by *crop*; a no-op when disabled.

    The crop box is ``(crop.height, crop.width)`` (each falling back to the full
    extent when ``None``), placed at the image centre.  ``crop.top`` /
    ``crop.left`` shift that centred box by the given pixel offsets — useful when
    the sky disc is not centred — and the result is clipped to stay inside the
    frame.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame.
    crop:
        Crop box description; ``crop.enabled`` False returns the frame as-is.

    Returns
    -------
    numpy.ndarray
        ``(crop_h, crop_w, 3)`` ``uint8`` view of the input — a slice, not a
        copy, so it shares the caller's buffer.
    """
    arr = _as_rgb_uint8(image)
    if not crop.enabled:
        return arr
    height, width = arr.shape[:2]
    box_h = min(int(crop.height) if crop.height is not None else height, height)
    box_w = min(int(crop.width) if crop.width is not None else width, width)
    top = (height - box_h) // 2 + int(crop.top)
    left = (width - box_w) // 2 + int(crop.left)
    top = int(np.clip(top, 0, height - box_h))
    left = int(np.clip(left, 0, width - box_w))
    return arr[top : top + box_h, left : left + box_w]


def resize_image(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Bilinearly resize *image* to *size* (``int`` = square, else ``(W, H)``).

    Mirrors the PIL recipe used by :func:`allsky.video.extract_frames` so a
    frame resized here is byte-identical to one resized at extraction time.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame.
    size:
        Target size in pixels: an ``int`` for a square, otherwise a
        ``(width, height)`` pair — PIL's axis order, the transpose of the
        array's ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Resized frame, shape ``(height, width, 3)``, ``uint8``.
    """
    return resize_bilinear(_as_rgb_uint8(image), size)


def visual_qc(
    image: np.ndarray,
    *,
    dark_threshold: float = DARK_LUMINANCE_THRESHOLD,
    saturated_fraction_threshold: float = SATURATED_FRACTION_THRESHOLD,
    saturated_level: int = SATURATED_LEVEL,
) -> set[QCFlag]:
    """Flag radiometrically unusable frames.

    Returns the subset of ``{FRAME_DARK, FRAME_SATURATED, FRAME_UNREADABLE}``
    that applies:

    - ``FRAME_DARK`` when the mean BT.601 luminance is below *dark_threshold*
      (night/twilight frames captured below the usable-sun horizon);
    - ``FRAME_SATURATED`` when the fraction of fully-clipped white pixels (every
      channel ``>= saturated_level``) exceeds *saturated_fraction_threshold*
      (over-exposure / direct-sun bloom washing out the sky texture);
    - ``FRAME_UNREADABLE``, alone, when the frame has no pixels: neither
      radiometric quantity exists, so neither threshold can be evaluated.

    The thresholds default to the module-level constants
    (:data:`DARK_LUMINANCE_THRESHOLD`, :data:`SATURATED_FRACTION_THRESHOLD`,
    :data:`SATURATED_LEVEL`) and may be overridden per call.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame, on the native 0-255 scale (this
        runs before any float normalization).
    dark_threshold:
        Mean-luminance floor on the same 0-255 scale.
    saturated_fraction_threshold:
        Ceiling on the fraction of clipped pixels, in [0, 1].
    saturated_level:
        Per-channel level at or above which a pixel counts as clipped, 0-255.

    Returns
    -------
    set of QCFlag
        Possibly empty; a clean frame flags nothing. The flags describe
        radiometric usability only — geometry (mask, crop) is not checked here.
        A frame with no pixels at all (a truncated decode) is unmeasurable and
        carries ``FRAME_UNREADABLE``, so the bitmask persisted for it is never
        zero and no downstream reader can mistake it for a clean frame.
    """
    arr = _as_rgb_uint8(image)
    flags: set[QCFlag] = set()

    channels = arr.reshape(-1, 3)
    if channels.shape[0] == 0:
        flags.add(QCFlag.FRAME_UNREADABLE)
        return flags

    # Summing each channel in exact integer arithmetic and weighting the three
    # sums is equal to weighting every pixel and averaging, without the
    # (H, W, 3) float64 copy the frame-wide dot product materializes.
    channel_sums = np.array(
        [channels[:, c].sum(dtype=np.uint64) for c in range(3)], dtype=np.float64
    )
    if float(channel_sums @ _LUMINANCE_WEIGHTS) / channels.shape[0] < dark_threshold:
        flags.add(QCFlag.FRAME_DARK)

    # Counting the per-channel comparisons costs a fraction of materializing the
    # (H, W, 3) boolean and reducing it along the channel axis, and is exactly
    # equal: both are the same integer count divided by the same pixel total.
    saturated = np.count_nonzero(
        (channels[:, 0] >= saturated_level)
        & (channels[:, 1] >= saturated_level)
        & (channels[:, 2] >= saturated_level)
    )
    if saturated / channels.shape[0] > saturated_fraction_threshold:
        flags.add(QCFlag.FRAME_SATURATED)

    return flags


def _needs_preprocessing(cfg: PrepareConfig) -> bool:
    """True when :func:`process_frame` would change the pixels of a frame."""
    return (
        cfg.mask.path is not None or cfg.crop.enabled or cfg.pad.enabled or cfg.resize is not None
    )


def resolve_mask(cfg: PrepareConfig) -> np.ndarray | None:
    """Decode *cfg*'s static mask once, for reuse across every frame of a run.

    Returns
    -------
    numpy.ndarray or None
        None when no ``cfg.mask.path`` is configured, else the keep-array
        :func:`load_mask` produces, shape ``(H, W)``, ``bool``. The buffer is
        marked read-only because every :func:`process_frame` call in the loop
        shares it.
    """
    if cfg.mask.path is None:
        return None
    keep = load_mask(cfg.mask.path, threshold=cfg.mask.threshold)
    keep.setflags(write=False)
    return keep


def pad_frame(image: np.ndarray, pad: PadConfig) -> np.ndarray:
    """Pad an RGB frame on each side with a constant level; a no-op when disabled.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame.
    pad:
        Per-side pixel counts and the fill level.

    Returns
    -------
    numpy.ndarray
        ``(H + top + bottom, W + left + right, 3)`` ``uint8``; the input array
        itself when ``pad.enabled`` is False.
    """
    arr = _as_rgb_uint8(image)
    if not pad.enabled:
        return arr
    padded: np.ndarray = np.pad(
        arr,
        ((pad.top, pad.bottom), (pad.left, pad.right), (0, 0)),
        mode="constant",
        constant_values=pad.fill,
    )
    return padded


def process_frame(
    image: np.ndarray, cfg: PrepareConfig, *, mask: np.ndarray | None = None
) -> np.ndarray:
    """Compose mask -> crop -> pad -> resize from a :class:`~allsky.config.PrepareConfig`.

    Each stage is skipped when its config leaves it unset: the static mask is
    applied only when ``cfg.mask.path`` is supplied (a PNG mask), the crop only
    when ``cfg.crop.enabled``, the pad only when ``cfg.pad.enabled``, and the
    resize only when ``cfg.resize`` is set.
    A decentred/auto circular mask is intentionally **not** applied by default
    (it would silently zero pixels); call :func:`apply_static_mask` with
    ``mask=None`` explicitly to opt into the heuristic estimate.

    *mask* is an already-decoded keep-array from :func:`resolve_mask`; passing it
    skips the per-frame PNG decode and yields the same pixels.  ``mask=None``
    falls back to decoding ``cfg.mask.path`` — it never means "estimate a
    circular mask", which is why the two branches stay explicit here.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame as decoded from the video.
    cfg:
        Prepare config supplying ``mask``, ``crop`` and ``resize``.
    mask:
        Keep-array of shape ``(H, W)``, ``bool``, decoded once by
        :func:`resolve_mask` and shared across the run.

    Returns
    -------
    numpy.ndarray
        Analysis-ready frame, ``(H', W', 3)`` ``uint8``, still on the 0-255
        scale — normalization belongs to the model's own transform. With every
        stage unset the input array is returned unchanged.
    """
    out = image
    if mask is not None:
        out = apply_static_mask(out, mask)
    elif cfg.mask.path is not None:
        out = apply_static_mask(out, cfg.mask.path, threshold=cfg.mask.threshold)
    out = center_crop(out, cfg.crop)
    out = pad_frame(out, cfg.pad)
    if cfg.resize is not None:
        out = resize_image(out, cfg.resize)
    return out
