"""Where the served network's output depends on the pixels, probed by occlusion.

Two probes, both causal and both built on a :class:`~allsky.snapshot.ServedModel`
loaded once:

- :func:`occlusion_map` slides a square window over the standardized input and
  records, per position, the change of one head when the window is replaced
  by the network's mean level (zero after ImageNet standardization); the
  result is a signed grid at window/stride resolution, in the input frame.
- :func:`neutralise_region` scores the frame once more with one region set to
  that same mean level — the overlay text band the camera burns into every
  frame, or any other box — for a single counterfactual number.

The geometry helpers map between the three pixel frames a published map lives
in: the network input (``S x S``, padded square), the camera frame the input
was cut from, and the re-encoded ``allsky.jpg`` the page shows.

Timestamps are naive station-local, as everywhere in serving.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from allsky.config import PrepareConfig, SiteConfig
from allsky.lens import PLANETARIO_NATIVE
from allsky.snapshot import ScalarFeatures, ServedModel

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_OCCLUSION_BATCH",
    "OCCLUSION_FILL",
    "OCCLUSION_STRIDE_PX",
    "OCCLUSION_WINDOW_PX",
    "OVERLAY_TEXT_BAND_RAW",
    "Box",
    "FrameGeometry",
    "OcclusionMap",
    "RegionMasks",
    "frame_geometry_of",
    "neutralise_region",
    "occlusion_map",
    "region_masks",
    "render_attribution_rgba",
]

#: Side of the occluding square and the step between positions, in input
#: pixels: 96/32 on a 512 px input gives a 14 x 14 grid whose cells overlap
#: three to one, smooth enough to read and cheap enough to run on a CPU
#: between two watch polls.
OCCLUSION_WINDOW_PX = 96
OCCLUSION_STRIDE_PX = 32
#: The value an occluded pixel takes, in standardized units: zero is the
#: ImageNet channel mean, the level the network treats as "no information".
OCCLUSION_FILL = 0.0
DEFAULT_OCCLUSION_BATCH = 32

#: The text the camera burns into the top-left corner (clock, site name, sensor
#: temperature, exposure time), measured on the Planetario 1920 x 1080 frames:
#: four lines of ~22 px, the longest ("Planetario e Observatorio da UFBA")
#: ~330 px wide.
OVERLAY_TEXT_BAND_RAW = (0, 0, 90, 340)

ATTRIBUTION_ALPHA_MAX = 0.85
#: Sequential ramp (dark violet -> orange -> pale yellow), the magma-like
#: family the site's density charts already use for "more".
_RAMP_STOPS = (
    np.array(
        [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 253, 191]], dtype=np.float64
    )
    / 255.0
)


@dataclass(frozen=True, slots=True)
class Box:
    """A pixel rectangle: ``top``/``left`` inclusive origin, ``height``/``width`` extents."""

    top: float
    left: float
    height: float
    width: float

    def as_dict(self, decimals: int = 2) -> dict[str, float]:
        """The four fields, rounded, in the key order the page reads."""
        return {
            "left": round(self.left, decimals) + 0.0,
            "top": round(self.top, decimals) + 0.0,
            "width": round(self.width, decimals) + 0.0,
            "height": round(self.height, decimals) + 0.0,
        }

    def scaled(self, factor_y: float, factor_x: float | None = None) -> Box:
        """The same rectangle in a frame scaled by *factor_y* vertically and *factor_x* horizontally."""
        fx = factor_y if factor_x is None else factor_x
        return Box(self.top * factor_y, self.left * fx, self.height * factor_y, self.width * fx)


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    """How the network input was cut from a camera frame of ``raw_height x raw_width``.

    Attributes
    ----------
    crop:
        Rectangle of the camera frame that survived the crop, in camera pixels.
    padded_height, padded_width:
        Size of the frame after padding, before the resize; equal for the
        isotropic geometry, unequal for a checkpoint that recorded none (the
        whole camera frame squeezed onto the square input).
    pad_top, pad_left:
        Where the crop sits inside the padded frame.
    input_size:
        Side of the network input, after the resize.
    """

    raw_height: int
    raw_width: int
    crop: Box
    padded_height: int
    padded_width: int
    pad_top: int
    pad_left: int
    input_size: int

    @property
    def scale_y(self) -> float:
        """Input pixels per padded-frame pixel, vertically."""
        return self.input_size / self.padded_height

    @property
    def scale_x(self) -> float:
        """Input pixels per padded-frame pixel, horizontally."""
        return self.input_size / self.padded_width

    @property
    def content_box(self) -> Box:
        """The rectangle of the input frame holding camera pixels (the rest is padding)."""
        return Box(self.pad_top, self.pad_left, self.crop.height, self.crop.width).scaled(
            self.scale_y, self.scale_x
        )

    def input_to_raw(self, box: Box) -> Box:
        """Map a rectangle in input pixels to camera-frame pixels."""
        unscaled = box.scaled(1.0 / self.scale_y, 1.0 / self.scale_x)
        return Box(
            unscaled.top - self.pad_top + self.crop.top,
            unscaled.left - self.pad_left + self.crop.left,
            unscaled.height,
            unscaled.width,
        )

    def raw_to_input(self, box: Box) -> Box:
        """Map a rectangle in camera-frame pixels to input pixels, clipped to the input."""
        shifted = Box(
            box.top - self.crop.top + self.pad_top,
            box.left - self.crop.left + self.pad_left,
            box.height,
            box.width,
        ).scaled(self.scale_y, self.scale_x)
        top = float(np.clip(shifted.top, 0, self.input_size))
        left = float(np.clip(shifted.left, 0, self.input_size))
        bottom = float(np.clip(shifted.top + shifted.height, 0, self.input_size))
        right = float(np.clip(shifted.left + shifted.width, 0, self.input_size))
        return Box(top, left, max(bottom - top, 0.0), max(right - left, 0.0))


def frame_geometry_of(
    geometry: PrepareConfig | None, *, raw_height: int, raw_width: int, input_size: int
) -> FrameGeometry:
    """Resolve the checkpoint's prepare geometry against a camera frame of the given size.

    Mirrors :func:`allsky.preprocessing.center_crop` (a centred box shifted by
    ``crop.top``/``crop.left`` and clipped) and :func:`allsky.preprocessing.pad_frame`.
    A checkpoint recording no geometry maps the whole frame, squeezed, onto the
    input — which is what its model actually saw.
    """
    if geometry is None or not geometry.crop.enabled:
        crop = Box(0.0, 0.0, float(raw_height), float(raw_width))
    else:
        box_h = min(int(geometry.crop.height or raw_height), raw_height)
        box_w = min(int(geometry.crop.width or raw_width), raw_width)
        top = int(
            np.clip((raw_height - box_h) // 2 + int(geometry.crop.top), 0, raw_height - box_h)
        )
        left = int(
            np.clip((raw_width - box_w) // 2 + int(geometry.crop.left), 0, raw_width - box_w)
        )
        crop = Box(float(top), float(left), float(box_h), float(box_w))
    pad_top = pad_left = 0
    padded_h, padded_w = int(crop.height), int(crop.width)
    if geometry is not None and geometry.pad.enabled:
        pad_top, pad_left = int(geometry.pad.top), int(geometry.pad.left)
        padded_h += pad_top + int(geometry.pad.bottom)
        padded_w += pad_left + int(geometry.pad.right)
    if padded_h != padded_w:
        logger.info(
            "the padded frame is %dx%d, not square: the resize to %d px is anisotropic, as the "
            "network was trained",
            padded_h,
            padded_w,
            input_size,
        )
    return FrameGeometry(
        raw_height=raw_height,
        raw_width=raw_width,
        crop=crop,
        padded_height=padded_h,
        padded_width=padded_w,
        pad_top=pad_top,
        pad_left=pad_left,
        input_size=input_size,
    )


@dataclass(frozen=True, slots=True)
class RegionMasks:
    """Boolean ``(S, S)`` masks of the four regions a sensitivity mass is split into.

    The regions are disjoint and cover the input: ``pad`` is everything
    outside the camera pixels, ``overlay_band`` the burned-in text,
    ``disc`` the sky inside the lens horizon minus that band, ``other`` the
    remaining camera pixels (the frame furniture around the disc).
    """

    disc: np.ndarray
    overlay_band: np.ndarray
    pad: np.ndarray
    other: np.ndarray

    def mass_of(self, magnitude: np.ndarray) -> dict[str, float]:
        """Fraction of *magnitude*'s mass under each region; the four sum to one.

        Parameters
        ----------
        magnitude:
            ``(S, S)`` float32, non-negative, in the input frame (a
            :meth:`OcclusionMap.rasterised` map).
        """
        total = float(magnitude.sum())
        if total <= 0.0:
            return {"disc": 0.0, "overlay_band": 0.0, "pad": 0.0, "other": 0.0}
        return {
            "disc": float(magnitude[self.disc].sum() / total),
            "overlay_band": float(magnitude[self.overlay_band].sum() / total),
            "pad": float(magnitude[self.pad].sum() / total),
            "other": float(magnitude[self.other].sum() / total),
        }


def _box_mask(box: Box, size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    top, left = int(np.floor(box.top)), int(np.floor(box.left))
    bottom, right = int(np.ceil(box.top + box.height)), int(np.ceil(box.left + box.width))
    mask[max(top, 0) : min(bottom, size), max(left, 0) : min(right, size)] = True
    return mask


def _horizon_mask(frame: FrameGeometry) -> np.ndarray:
    """Pixels inside the lens horizon, the native calibration mapped through *frame*.

    The camera's optical centre and horizon radius are measured on the native
    1920 x 1080 frame; through a non-isotropic geometry the circle becomes an
    ellipse, which is what a checkpoint without geometry (the whole frame
    squeezed onto the square input) actually sees.
    """
    size = frame.input_size
    centre_row = (PLANETARIO_NATIVE.centre_row - frame.crop.top + frame.pad_top) * frame.scale_y
    centre_col = (PLANETARIO_NATIVE.centre_col - frame.crop.left + frame.pad_left) * frame.scale_x
    radius_rows = PLANETARIO_NATIVE.radius_px * frame.scale_y
    radius_cols = PLANETARIO_NATIVE.radius_px * frame.scale_x
    rows, cols = np.mgrid[0:size, 0:size]
    inside: np.ndarray = ((rows - centre_row) / radius_rows) ** 2 + (
        (cols - centre_col) / radius_cols
    ) ** 2 <= 1.0
    return inside


def region_masks(frame: FrameGeometry) -> RegionMasks:
    """The sky disc, the overlay text band, the padding and the rest, in input pixels, disjoint."""
    size = frame.input_size
    horizon = _horizon_mask(frame)
    band_top, band_left, band_h, band_w = OVERLAY_TEXT_BAND_RAW
    band = _box_mask(frame.raw_to_input(Box(band_top, band_left, band_h, band_w)), size)
    content = _box_mask(frame.content_box, size)
    overlay_band = band & content
    disc = horizon & content & ~overlay_band
    return RegionMasks(
        disc=disc, overlay_band=overlay_band, pad=~content, other=content & ~disc & ~overlay_band
    )


@dataclass(frozen=True, slots=True)
class OcclusionMap:
    """One occlusion sweep over one frame.

    Attributes
    ----------
    grid:
        ``(rows, cols)`` float32, the signed change of the target's *physical*
        value when the window at that position is occluded (occluded minus
        base). Row-major over window positions, top-left first.
    base_value:
        The target's physical value on the untouched frame.
    positions:
        ``(rows * cols, 2)`` int, the ``(top, left)`` input pixel of each window.
    """

    target: str
    window_px: int
    stride_px: int
    grid: np.ndarray
    base_value: float
    positions: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        """``|grid|`` max-normalised to ``[0, 1]``; all zero when nothing moved."""
        magnitude = np.abs(self.grid)
        peak = float(magnitude.max())
        return magnitude / peak if peak > 0.0 else magnitude

    @property
    def peak(self) -> tuple[int, int]:
        """``(row, col)`` of the largest absolute change."""
        row, col = np.unravel_index(int(np.argmax(np.abs(self.grid))), self.grid.shape)
        return int(row), int(col)

    def rasterised(self, size: int) -> np.ndarray:
        """``(size, size)`` float32 magnitude, each pixel the mean of the windows covering it."""
        heat = np.zeros((size, size), dtype=np.float32)
        count = np.zeros((size, size), dtype=np.float32)
        weights = self.magnitude.reshape(-1)
        for weight, (top, left) in zip(weights, self.positions, strict=True):
            heat[top : top + self.window_px, left : left + self.window_px] += weight
            count[top : top + self.window_px, left : left + self.window_px] += 1.0
        return np.where(count > 0.0, heat / np.maximum(count, 1.0), 0.0).astype(np.float32)


def _target_values(
    served: ServedModel,
    outputs: dict[str, Any],
    target: str,
    timestamp: pd.Timestamp,
    site: SiteConfig,
) -> np.ndarray:
    """Physical values of *target* for every row of a batched forward."""
    return served.physical_values(outputs, target, timestamp=timestamp, site=site)


def occlusion_map(
    served: ServedModel,
    planes: np.ndarray,
    features: ScalarFeatures,
    *,
    timestamp: pd.Timestamp,
    site: SiteConfig,
    target: str,
    window_px: int = OCCLUSION_WINDOW_PX,
    stride_px: int = OCCLUSION_STRIDE_PX,
    batch_size: int = DEFAULT_OCCLUSION_BATCH,
) -> OcclusionMap:
    """Slide a mean-level window over the RGB planes and record the target's change.

    Parameters
    ----------
    served:
        The loaded checkpoint; must consume the image.
    planes:
        ``(3 + G, S, S)`` float32 standardized input from
        :meth:`ServedModel.image_planes`. Only the three RGB planes are
        occluded; geometry planes, when present, are left intact.
    features:
        The scalar vector of the same frame.
    target:
        Head to read: ``"kindex"`` or ``"dhi"`` (physical units).

    Returns
    -------
    OcclusionMap
        Signed change per window position, occluded minus base.

    Raises
    ------
    ValueError
        If the model reads no image, the target head is absent, or the window
        does not fit the input.
    """
    import torch

    if not served.consumes_image:
        raise ValueError(f"{served.cfg.name} reads no image; nothing to occlude")
    size = served.image_size
    if window_px > size or window_px <= 0 or stride_px <= 0:
        raise ValueError(f"window {window_px}/stride {stride_px} do not fit a {size} px input")
    base_outputs = served.forward(served.batch(features, planes=planes))
    if target not in base_outputs:
        raise ValueError(f"{served.cfg.name} has no {target!r} head")
    base_value = float(_target_values(served, base_outputs, target, timestamp, site)[0])

    starts = list(range(0, size - window_px + 1, stride_px))
    positions = np.array([(top, left) for top in starts for left in starts], dtype=np.int64)
    deltas = np.empty(len(positions), dtype=np.float64)
    planes_tensor = torch.from_numpy(np.ascontiguousarray(planes))
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        images = planes_tensor.unsqueeze(0).repeat(len(chunk), 1, 1, 1)
        for row, (top, left) in enumerate(chunk):
            images[row, :3, top : top + window_px, left : left + window_px] = OCCLUSION_FILL
        batch = {
            "features": torch.from_numpy(features.standardized)
            .repeat(len(chunk), 1)
            .to(served.device),
            "image": images.to(served.device),
        }
        outputs = served.forward(batch)
        deltas[start : start + len(chunk)] = (
            _target_values(served, outputs, target, timestamp, site) - base_value
        )
    grid = deltas.reshape(len(starts), len(starts)).astype(np.float32)
    return OcclusionMap(
        target=target,
        window_px=window_px,
        stride_px=stride_px,
        grid=grid,
        base_value=base_value,
        positions=positions,
    )


def neutralise_region(
    served: ServedModel,
    planes: np.ndarray,
    features: ScalarFeatures,
    *,
    box: Box,
    timestamp: pd.Timestamp,
    site: SiteConfig,
) -> dict[str, Any]:
    """Score the frame with the RGB planes inside *box* set to the mean level.

    Parameters
    ----------
    served:
        The loaded checkpoint; must consume the image.
    planes:
        ``(3 + G, S, S)`` float32 standardized input from
        :meth:`ServedModel.image_planes`; only the three RGB planes are altered.
    features:
        The scalar vector of the same frame.
    box:
        Rectangle in **input** pixels (not camera pixels); see
        :meth:`FrameGeometry.raw_to_input`.
    timestamp, site:
        The frame's naive local capture time and the station, for the
        physical-unit conversion.

    Returns
    -------
    dict
        The physical prediction record of the altered frame, as
        :meth:`ServedModel.physical` returns it.

    Raises
    ------
    ValueError
        If the model reads no image.
    """
    if not served.consumes_image:
        raise ValueError(f"{served.cfg.name} reads no image; nothing to neutralise")
    altered = np.array(planes, copy=True)
    mask = _box_mask(box, served.image_size)
    altered[:3, mask] = OCCLUSION_FILL
    outputs = served.forward(served.batch(features, planes=altered))
    return served.physical(outputs, timestamp=timestamp, site=site)


def _ramp(values: np.ndarray) -> np.ndarray:
    stops = np.linspace(0.0, 1.0, len(_RAMP_STOPS))
    return np.stack(
        [np.interp(values, stops, _RAMP_STOPS[:, channel]) for channel in range(3)], axis=-1
    )


def render_attribution_rgba(
    occlusion: OcclusionMap,
    frame: FrameGeometry,
    *,
    out_width: int,
    out_height: int,
) -> np.ndarray:
    """Warp the sensitivity magnitude into the published image's pixel grid.

    Rasterised at the input resolution, resized to the padded frame, the
    padding cut away, placed where the crop sits in the camera frame and
    resized to ``out_width x out_height``. Alpha carries the magnitude
    (``ATTRIBUTION_ALPHA_MAX`` at the peak), transparent outside the field of
    view.

    Returns
    -------
    numpy.ndarray
        ``(out_height, out_width, 4)`` uint8 RGBA.
    """
    from PIL import Image

    heat_input = occlusion.rasterised(frame.input_size)
    padded = Image.fromarray((heat_input * 255.0).astype(np.uint8)).resize(
        (frame.padded_width, frame.padded_height), Image.Resampling.BILINEAR
    )
    heat_padded = np.asarray(padded, dtype=np.float32) / 255.0
    crop_h, crop_w = int(frame.crop.height), int(frame.crop.width)
    heat_crop = heat_padded[
        frame.pad_top : frame.pad_top + crop_h, frame.pad_left : frame.pad_left + crop_w
    ]
    heat_raw = np.zeros((frame.raw_height, frame.raw_width), dtype=np.float32)
    top, left = int(frame.crop.top), int(frame.crop.left)
    heat_raw[top : top + crop_h, left : left + crop_w] = heat_crop
    heat_out = (
        np.asarray(
            Image.fromarray((heat_raw * 255.0).astype(np.uint8)).resize(
                (out_width, out_height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    rgb = (_ramp(heat_out) * 255.0).astype(np.uint8)
    alpha = (np.clip(heat_out, 0.0, 1.0) * ATTRIBUTION_ALPHA_MAX * 255.0).astype(np.uint8)
    return np.dstack([rgb, alpha])
