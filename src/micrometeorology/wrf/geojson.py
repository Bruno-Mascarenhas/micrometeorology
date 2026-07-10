"""GeoJSON and JSON generation for WRF grids.

Produces the grid GeoJSON and per-timestep value JSON files for external visualization.

Optimisations applied:
  - GeoJSON coordinates rounded to 10 decimal places (~0.01 mm precision).
  - JSON output uses compact separators (no indent / whitespace).
  - Custom float encoder avoids Python's excessive float precision
    (e.g. ``20.450000762939453`` → ``20.45``).
  - Whole floats in values arrays are serialized as integers (``0.0`` → ``0``);
    the parsed numeric values are unchanged.
  - A compact ``.grid.json`` companion is written next to each ``.geojson``
    (see :func:`write_grid_compact_json_stream`): same cell rectangles at a
    fraction of the size, preferred by the site front-end with the legacy
    GeoJSON as fallback.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from micrometeorology.wrf.safety import assert_reasonable_array_size

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
JSON_VALUE_CHUNK_SIZE = 65_536
GEOJSON_FEATURE_CHUNK_SIZE = 4_096


def write_values_json_stream(
    output_path: str | Path,
    var: NDArray,
    scale_min: float,
    scale_max: float,
    date_str: str,
    wind_data: dict | None = None,
    *,
    chunk_size: int = JSON_VALUE_CHUNK_SIZE,
) -> Path:
    """Write value JSON without materializing the full flattened values list."""
    arr = var.filled(np.nan) if isinstance(var, np.ma.MaskedArray) else np.asarray(var)
    assert_reasonable_array_size(
        arr.shape,
        arr.dtype,
        context=f"streamed values JSON generation for {output_path}",
        multiplier=2.0,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "scale_values": [round(float(x), 2) for x in np.linspace(scale_min, scale_max, 6)],
        "date_time": date_str,
    }
    if wind_data is not None:
        metadata["wind"] = wind_data

    with open(out, "w", encoding="utf-8") as f:
        f.write('{"metadata":')
        json.dump(metadata, f, separators=(",", ":"), ensure_ascii=False)
        f.write(',"values":[')
        _write_flat_values_chunks(f, arr, chunk_size=chunk_size)
        f.write("]}")

    logger.info("Saved streamed JSON: %s", out)
    return out


def create_wind_vectors_json(
    u: NDArray,
    v: NDArray,
    date_time: datetime | None,
    downsampling: int = 4,
) -> dict[str, Any]:
    """Build a standalone wind-vectors JSON payload (no grid values).

    Used to provide wind arrow overlays for ANY variable on the
    interactive maps, without embedding wind data in every variable's
    JSON file.

    Parameters
    ----------
    u, v:
        2-D arrays of wind components (m/s) for a single time step.
    date_time:
        Forecast datetime (local).
    downsampling:
        Stride for spatial downsampling of the arrow grid.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise ValueError(f"wind vector shapes differ: {u.shape!r} vs {v.shape!r}")
    assert_reasonable_array_size(
        u.shape,
        u.dtype,
        context="wind vector JSON generation",
        multiplier=4.0,
    )

    magnitude = np.hypot(u, v)
    # Meteorological convention: angle is direction wind comes FROM
    angle = np.degrees(np.arctan2(u, v))
    angle = np.where(angle < 0, angle + 360.0, angle)

    ny, nx = u.shape

    # Vectorized downsampling — replaces nested Python loop
    i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    angles_sampled = np.round(angle[i_flat, j_flat], 1)
    mags_sampled = np.round(magnitude[i_flat, j_flat], 2)
    linear_indices = i_flat * nx + j_flat

    valid = ~np.isnan(angles_sampled)

    # Date formatting
    if date_time is None:
        date_str = "N/A"
    else:
        try:
            dt = date_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            date_str = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            date_str = str(date_time)

    return {
        "metadata": {"date_time": date_str},
        "downsampled_angles": angles_sampled[valid].tolist(),
        "downsampled_magnitudes": mags_sampled[valid].tolist(),
        "downsampled_linear_indices": linear_indices[valid].tolist(),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def write_grid_geojson_stream(
    output_path: str | Path,
    lon: NDArray,
    lat: NDArray,
    resolution_x: float,
    resolution_y: float,
    _colormap: str = "",
) -> Path:
    """Write grid GeoJSON feature-by-feature without building a full feature list.

    Corner coordinates are computed vectorized (same element arithmetic and
    operand order as the historical per-cell writer, in the input dtype) and the
    feature text is assembled in chunks with f-strings. The output bytes are
    identical to serialising each :func:`_grid_cell_feature` dict with
    ``json.dump(..., separators=(",", ":"), ensure_ascii=False)``.
    """
    n_rows, n_cols = _validate_grid(lon, lat, context=f"streamed GeoJSON grid for {output_path}")
    metadata = {"resolucao_m": [float(resolution_x), float(resolution_y)]}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lon_left, lon_right, lat_top, lat_bottom = _grid_cell_corners(lon, lat)
    n_cells = n_rows * n_cols

    with open(out, "w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","metadata":')
        json.dump(metadata, f, separators=(",", ":"), ensure_ascii=False)
        f.write(',"features":[')
        for start in range(0, n_cells, GEOJSON_FEATURE_CHUNK_SIZE):
            stop = min(start + GEOJSON_FEATURE_CHUNK_SIZE, n_cells)
            chunk = ",".join(
                f'{{"type":"Feature","geometry":{{"type":"Polygon","coordinates":'
                f"[[[{lon_left[k]!r},{lat_bottom[k]!r}],"
                f"[{lon_right[k]!r},{lat_bottom[k]!r}],"
                f"[{lon_right[k]!r},{lat_top[k]!r}],"
                f"[{lon_left[k]!r},{lat_top[k]!r}],"
                f"[{lon_left[k]!r},{lat_bottom[k]!r}]]]}}"
                f',"properties":{{"linear_index":{k}}}}}'
                for k in range(start, stop)
            )
            if start:
                f.write(",")
            f.write(chunk)
        f.write("]}")

    return out


GRID_COMPACT_DECIMALS = 7


def write_grid_compact_json_stream(
    output_path: str | Path,
    lon: NDArray,
    lat: NDArray,
    resolution_x: float,
    resolution_y: float,
) -> Path:
    """Write the compact grid companion (``D0X.grid.json``).

    Cell ``k`` (row-major; ``k`` equals the legacy GeoJSON ``linear_index``)
    is the axis-aligned rectangle with longitude edges
    ``(lon_left[k], lon_right[k])`` and latitude edges
    ``(lat_top[k], lat_bottom[k])`` — the SAME corner arithmetic as the
    legacy GeoJSON features, rounded to 7 decimals (~1.1 cm). Seven decimals
    is the smallest count that still uniquely identifies the float32 source
    coordinates at these magnitudes, so nothing beyond float32 noise is lost
    (verified against the production D01-D04 grids; 6 decimals is NOT enough).

    Two layouts, discriminated by the ``format`` key:

    - ``grid-edges-v1`` — when the grid is separable (every row shares the
      same longitude edges and every column the same latitude edges, as in
      regular lat/lon WRF projections), only the 1-D edge vectors are stored:
      ``lon_edges`` (``n_cols + 1``) and ``lat_edges`` (``n_rows + 1``).
      Cell ``(i, j)`` spans lon ``lon_edges[j]..lon_edges[j+1]`` and lat
      ``lat_edges[i]..lat_edges[i+1]`` (edge vectors keep the writer's
      top/bottom orientation; consumers must not assume a sign direction).
      ~3 KB instead of 1.2-2.6 MB for the production domains.

    - ``grid-bounds-v1`` — fallback for non-separable (curvilinear) grids:
      per-cell ``[lon_left, lat_bottom, lon_right, lat_top]``.
    """
    n_rows, n_cols = _validate_grid(lon, lat, context=f"compact grid JSON for {output_path}")
    lon_left, lon_right, lat_top, lat_bottom = _grid_cell_corner_arrays(lon, lat)

    payload: dict[str, Any] = {}
    if _grid_is_separable(lon_left, lon_right, lat_top, lat_bottom):
        # Structural sharing (lon_right[:, j] IS lon_left[:, j+1], both
        # views of lon_mid; same for lat) means row 0 / column 0 carry the
        # full edge information.
        lon_edges = np.concatenate([lon_left[0, :], lon_right[0, -1:]])
        lat_edges = np.concatenate([lat_top[:, 0], lat_bottom[-1:, 0]])
        payload["format"] = "grid-edges-v1"
        payload["metadata"] = {"resolucao_m": [float(resolution_x), float(resolution_y)]}
        payload["shape"] = [n_rows, n_cols]
        payload["lon_edges"] = [round(v, GRID_COMPACT_DECIMALS) for v in lon_edges.tolist()]
        payload["lat_edges"] = [round(v, GRID_COMPACT_DECIMALS) for v in lat_edges.tolist()]
    else:
        west = [round(v, GRID_COMPACT_DECIMALS) for v in lon_left.ravel().tolist()]
        east = [round(v, GRID_COMPACT_DECIMALS) for v in lon_right.ravel().tolist()]
        top = [round(v, GRID_COMPACT_DECIMALS) for v in lat_top.ravel().tolist()]
        bottom = [round(v, GRID_COMPACT_DECIMALS) for v in lat_bottom.ravel().tolist()]
        payload["format"] = "grid-bounds-v1"
        payload["metadata"] = {"resolucao_m": [float(resolution_x), float(resolution_y)]}
        payload["shape"] = [n_rows, n_cols]
        payload["bounds"] = [[west[k], bottom[k], east[k], top[k]] for k in range(n_rows * n_cols)]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), ensure_ascii=False)
    logger.info("Saved compact grid JSON: %s (%s)", out, payload["format"])
    return out


def _grid_is_separable(
    lon_left: NDArray,
    lon_right: NDArray,
    lat_top: NDArray,
    lat_bottom: NDArray,
) -> bool:
    """True when longitude edges repeat identically across rows and latitude
    edges across columns (exact equality on the unrounded values)."""
    return bool(
        (lon_left == lon_left[0:1, :]).all()
        and (lon_right == lon_right[0:1, :]).all()
        and (lat_top == lat_top[:, 0:1]).all()
        and (lat_bottom == lat_bottom[:, 0:1]).all()
    )


def _grid_cell_corner_arrays(
    lon: NDArray,
    lat: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute unrounded 2-D corner arrays ``(lon_left, lon_right, lat_top, lat_bottom)``.

    The arithmetic preserves the historical per-cell writer exactly — same
    expressions and operand order, evaluated in the input dtype (float32
    grids stay float32). Shared by the legacy GeoJSON writer and the compact
    grid writer so both serialize the SAME corner values.
    """
    # WRF readers hand back MaskedArrays (mask all False); np.ma arithmetic
    # promotes ``/ 2`` to float64, unlike the per-element scalar path.
    # Normalize to the underlying ndarray so float32 grids stay float32.
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    lat_mid = (lat[:-1, :] + lat[1:, :]) / 2
    lat_top = np.concatenate(
        [lat[:1, :] + (lat[:1, :] - lat[1:2, :]) / 2, lat_mid],
        axis=0,
    )
    lat_bottom = np.concatenate(
        [lat_mid, lat[-1:, :] - (lat[-2:-1, :] - lat[-1:, :]) / 2],
        axis=0,
    )

    lon_mid = (lon[:, :-1] + lon[:, 1:]) / 2
    lon_left = np.concatenate(
        [lon[:, :1] - (lon[:, 1:2] - lon[:, :1]) / 2, lon_mid],
        axis=1,
    )
    lon_right = np.concatenate(
        [lon_mid, lon[:, -1:] + (lon[:, -1:] - lon[:, -2:-1]) / 2],
        axis=1,
    )

    return lon_left, lon_right, lat_top, lat_bottom


def _grid_cell_corners(
    lon: NDArray,
    lat: NDArray,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute per-cell corner coordinates for every grid cell, vectorized.

    Returns ``(lon_left, lon_right, lat_top, lat_bottom)`` as flat row-major
    Python lists rounded with builtin ``round(v, 10)`` per element.
    ``np.round`` must NOT be used here: it disagrees with builtin ``round``
    at decimal ties for float64 inputs (e.g. ``-14.000000000050001``).
    """
    lon_left, lon_right, lat_top, lat_bottom = _grid_cell_corner_arrays(lon, lat)
    return (
        _rounded_coordinate_list(lon_left),
        _rounded_coordinate_list(lon_right),
        _rounded_coordinate_list(lat_top),
        _rounded_coordinate_list(lat_bottom),
    )


def _rounded_coordinate_list(arr: NDArray) -> list[float]:
    """Flatten row-major and apply builtin ``round(v, 10)`` per element."""
    return [round(v, 10) for v in np.asarray(arr).ravel().tolist()]


def _validate_grid(lon: NDArray, lat: NDArray, *, context: str) -> tuple[int, int]:
    if lon.shape != lat.shape:
        raise ValueError(f"lon/lat grid shapes differ: {lon.shape!r} vs {lat.shape!r}")
    if len(lon.shape) != 2:
        raise ValueError(f"lon/lat grids must be 2-D, got {lon.shape!r}")
    n_rows, n_cols = lon.shape
    if n_rows < 2 or n_cols < 2:
        raise ValueError("GeoJSON grid generation requires at least a 2x2 grid")
    assert_reasonable_array_size(
        lon.shape,
        np.float64,
        context=context,
        multiplier=4.0,
    )
    return int(n_rows), int(n_cols)


# json.dumps writes whole floats with a redundant ".0" ("0.0" instead of "0").
# Stripping it changes no parsed value (JSON "0" and "0.0" are the same number,
# and "-0.0" -> "-0" preserves the negative-zero sign) but saves ~2.4% across
# the values corpus — over 40% on rain files full of dry-hour zeros. The
# values text is only numbers/null separated by commas, so a ".0" directly
# before a separator (or chunk end) can only be a whole float's suffix.
_WHOLE_FLOAT_RE = re.compile(r"(-?\d+)\.0(?=,|$)")


def _write_flat_values_chunks(f: TextIO, arr: NDArray, *, chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    flat = np.ravel(arr)
    first = True
    for start in range(0, flat.size, chunk_size):
        chunk = np.round(flat[start : start + chunk_size].astype(np.float64, copy=False), 2)
        values: list[float | None] = chunk.tolist()
        invalid = np.flatnonzero(~np.isfinite(chunk))
        for idx in invalid:
            values[idx] = None
        if not values:
            continue
        text = _WHOLE_FLOAT_RE.sub(r"\1", json.dumps(values, separators=(",", ":"))[1:-1])
        if not first:
            f.write(",")
        first = False
        f.write(text)
