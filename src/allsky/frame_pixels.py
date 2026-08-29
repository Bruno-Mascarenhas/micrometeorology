"""Decoding and resizing an all-sky frame, in one place.

Four modules turn a file (or a JPEG payload) into an ``(H, W, 3)`` ``uint8`` RGB
array and bilinear-resize one: :mod:`allsky.video` at extraction time,
:mod:`allsky.preprocessing` when a frame is reprocessed,
:mod:`allsky.embeddings.backbone` before encoding, and :mod:`allsky.snapshot`
when a live frame is scored.

The frames have to come out **byte-identical** across those paths: a JPEG
written at extraction and the same frame resized later must agree, or the stored
embeddings stop describing the images the manifest points at.

This module imports nothing from ``allsky`` — only numpy and PIL — so all four
depend on it without depending on each other.
"""

from pathlib import Path

import numpy as np

__all__ = ["as_rgb_uint8", "decode_rgb", "resize_bilinear"]


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
    import io

    from PIL import Image

    handle_source = io.BytesIO(source) if isinstance(source, bytes) else source
    with Image.open(handle_source) as handle:
        return np.asarray(handle.convert("RGB"), dtype=np.uint8)


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
