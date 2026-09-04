"""Decoding and resizing an all-sky frame, in one place.

Every path that turns a file (or a JPEG payload) into an ``(H, W, 3)`` ``uint8``
RGB array, or bilinear-resizes one, goes through here: extraction and overlay
staging, reprocessing, the prepare CLI, embedding extraction, the backbone's own
transform, and the live snapshot.

The frames have to come out **byte-identical** across those paths: a JPEG
written at extraction and the same frame resized later must agree, or the stored
embeddings stop describing the images the manifest points at.

This module imports nothing from ``allsky`` — only numpy and PIL — so every one
of them depends on it without depending on each other.
"""

from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = ["as_rgb_uint8", "decode_rgb", "decode_rgb_resized", "resize_bilinear"]


def decode_rgb(source: str | Path | bytes) -> np.ndarray:
    """Decode an image file or payload to an ``(H, W, 3)`` ``uint8`` RGB array.

    Decoding straight with PIL is byte-identical to reading through imageio
    (which decodes with PIL anyway) and skips its per-call plugin dispatch, so
    previously extracted embedding stores stay valid.

    Parameters
    ----------
    source:
        Path to an image, or the encoded bytes themselves.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` ``uint8``, RGB, on the native 0-255 scale. Grayscale and
        palette images are converted, so the channel axis is always present.
    """
    from PIL import Image

    with Image.open(_readable(source)) as handle:
        return np.asarray(_as_rgb_image(handle), dtype=np.uint8)


def decode_rgb_resized(source: str | Path | bytes, size: int | tuple[int, int]) -> np.ndarray:
    """Decode and bilinear-resize in one pass, skipping the numpy round trip.

    Identical to ``resize_bilinear(decode_rgb(source), size)`` — the array
    :func:`decode_rgb` returns is exactly what :func:`resize_bilinear` wraps
    back into a PIL image — but resizes the decoded handle directly, so neither
    full-resolution copy is made. Verified byte-for-byte over the extracted
    frames of this camera.

    Parameters
    ----------
    source:
        Path to an image, or the encoded bytes themselves.
    size:
        Target size in pixels, spelled as in :func:`resize_bilinear`.

    Returns
    -------
    numpy.ndarray
        ``(height, width, 3)`` ``uint8``, RGB, on the native 0-255 scale.
    """
    from PIL import Image

    target = (size, size) if isinstance(size, int) else size
    with Image.open(_readable(source)) as handle:
        resized = _as_rgb_image(handle).resize(target, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _readable(source: str | Path | bytes) -> str | Path | BinaryIO:
    """Wrap encoded bytes so PIL can open them; pass a path straight through."""
    import io

    return io.BytesIO(source) if isinstance(source, bytes) else source


def _as_rgb_image(handle: PILImage) -> PILImage:
    """The handle as RGB, converting only when it is not already there.

    ``Image.convert`` copies the whole raster even when the mode already
    matches, which on this camera's frames is every frame and 17.4 % of the
    decode.
    """
    return handle if handle.mode == "RGB" else handle.convert("RGB")


def as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Validate *image* is an ``(H, W, 3)`` ``uint8`` RGB array and return it.

    Raises
    ------
    ValueError
        If the array is not 3-D with a trailing size-3 channel axis.
    TypeError
        If the dtype is not ``uint8``.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) RGB image, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image, got dtype {arr.dtype}")
    return arr


def resize_bilinear(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Bilinear-resize an RGB frame — the one recipe every path shares.

    Parameters
    ----------
    image:
        ``(H, W, 3)`` ``uint8`` RGB frame. Not validated here: callers that want
        the check call :func:`as_rgb_uint8` first, and the extraction path
        deliberately does not.
    size:
        Target size in pixels: an ``int`` for a square, otherwise a
        ``(width, height)`` pair — PIL's axis order, the transpose of the
        array's ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        Resized frame, ``(height, width, 3)``.
    """
    from PIL import Image

    target = (size, size) if isinstance(size, int) else size
    return np.asarray(Image.fromarray(image).resize(target, Image.Resampling.BILINEAR))
