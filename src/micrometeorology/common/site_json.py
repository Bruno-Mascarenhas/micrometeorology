"""The byte contract of every JSON this project publishes to the site.

Compact separators, UTF-8 without escapes and ``allow_nan=False`` are one
constant here, and every writer takes the encoding from it, so the published
files cannot disagree about the contract.

``allow_nan=False`` is the load-bearing one: ``NaN`` is not valid JSON and a
browser parser rejects the *whole file*, so a non-finite number has to become
``null`` before it is written. :func:`finite` and :func:`rounded` are that step.

The writer that stages atomically lives here rather than inside the 1200-line
climatology module, so exporters with nothing climatological about them do not
pay its import-time bibliography validation.

Depends only on the standard library and :mod:`allsky.atomic`. Modules with
their own I/O (the streaming GeoJSON writer, the measured dumps-then-write in
the WRF batch) import :data:`JSON_ENCODING` alone and keep their own writing.
"""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from allsky.atomic import atomic_write

logger = logging.getLogger(__name__)

__all__ = ["JSON_ENCODING", "finite", "rounded", "rounded_list", "write_json"]

#: The published encoding. Compact because these files are fetched by a browser;
#: ``ensure_ascii=False`` because the pages are in Portuguese and escaping every
#: accent both inflates the payload and makes the artifact unreadable by eye.
JSON_ENCODING: dict[str, Any] = {
    "separators": (",", ":"),
    "ensure_ascii": False,
    "allow_nan": False,
}


def finite(value: float | None) -> float | None:
    """Map a non-finite number to ``None`` so the strict writer accepts it.

    An empty subset legitimately produces NaN parameters; they travel as
    ``null`` rather than failing the whole document in the browser.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def rounded(value: float | None, decimals: int) -> float | None:
    """:func:`finite`, then rounded to *decimals* places."""
    number = finite(value)
    return None if number is None else round(number, decimals)


def rounded_list(values: Sequence[float], decimals: int) -> list[float | None]:
    """:func:`rounded` over a sequence, preserving order and length."""
    return [rounded(float(value), decimals) for value in values]


def write_json(output_path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write one artifact atomically, in the encoding the site pipeline uses.

    Serialised to a private sibling and ``os.replace``-d into place by
    :func:`allsky.atomic.atomic_write`, so a reader fetching the directory
    mid-run sees the old file or the new one, never a truncated parse error.

    Raises
    ------
    ValueError
        If the payload still contains a non-finite float. Route every number
        through :func:`finite` before calling this.
    """
    encoded = json.dumps(payload, **JSON_ENCODING)
    out = atomic_write(output_path, lambda tmp: tmp.write_text(encoded, encoding="utf-8"))
    logger.info("wrote %s (%d bytes)", out, len(encoded.encode("utf-8")))
    return out
