"""The decode/resize pair whose byte-for-byte identity the dataset relies on.

``decode_rgb_resized`` exists to skip a full-frame numpy round trip, and it is
only ever the right thing to call while it produces exactly what
``resize_bilinear(decode_rgb(...))`` produces. Nothing pinned that, so the fast
path could have drifted from the slow one and every image the model reads would
have moved with it.
"""

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from allsky.frame_pixels import (
    as_rgb_uint8,
    decode_rgb,
    decode_rgb_resized,
    resize_bilinear,
)
from allsky.video import JPEG_QUALITY


@pytest.fixture
def rgb_jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "frame.jpg"
    pixels = np.random.default_rng(11).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    iio.imwrite(path, pixels, quality=JPEG_QUALITY)
    return path


@pytest.mark.parametrize("size", [16, (24, 12)])
def test_the_fast_decode_equals_decode_then_resize(rgb_jpeg: Path, size):
    np.testing.assert_array_equal(
        decode_rgb_resized(rgb_jpeg, size), resize_bilinear(decode_rgb(rgb_jpeg), size)
    )


@pytest.mark.parametrize("mode", ["L", "P"])
def test_a_grayscale_or_palette_jpeg_decodes_to_three_channels(tmp_path: Path, mode: str):
    """The camera writes RGB, but a re-encode elsewhere can produce either, and
    both must reach the model as the three channels its first layer expects."""
    path = tmp_path / f"{mode}.png"
    Image.fromarray(
        np.random.default_rng(5).integers(0, 256, (32, 32), dtype=np.uint8), mode="L"
    ).convert(mode).save(path)

    decoded = decode_rgb(path)

    assert decoded.shape == (32, 32, 3)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, resize_bilinear(decode_rgb(path), (32, 32)))


def test_bytes_and_path_decode_identically(rgb_jpeg: Path):
    np.testing.assert_array_equal(decode_rgb(rgb_jpeg.read_bytes()), decode_rgb(rgb_jpeg))


class TestAsRgbUint8Refusals:
    def test_a_frame_that_is_not_three_dimensional_is_refused(self):
        with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
            as_rgb_uint8(np.zeros((8, 8), dtype=np.uint8))

    def test_a_frame_with_the_wrong_channel_count_is_refused(self):
        with pytest.raises(ValueError, match=r"\(8, 8, 2\)"):
            as_rgb_uint8(np.zeros((8, 8, 2), dtype=np.uint8))
