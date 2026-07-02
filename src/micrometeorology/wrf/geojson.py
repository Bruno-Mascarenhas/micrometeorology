"""GeoJSON and JSON generation for WRF grids.

Produces the grid GeoJSON and per-timestep value JSON files for external visualization.

Optimisations applied:
  - GeoJSON coordinates rounded to 10 decimal places (~0.01 mm precision).
  - JSON output uses compact separators (no indent / whitespace).
  - Custom float encoder avoids Python's excessive float precision
    (e.g. ``20.450000762939453`` → ``20.45``).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from micrometeorology.common.paths import ensure_dir
from micrometeorology.wrf.safety import assert_reasonable_array_size

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
JSON_VALUE_CHUNK_SIZE = 65_536
GEOJSON_FEATURE_CHUNK_SIZE = 4_096


def create_grid_geojson(
    lon: NDArray,
    lat: NDArray,
    resolution_x: float,
    resolution_y: float,
    _colormap: str,
) -> dict:
    """Build a GeoJSON FeatureCollection representing the WRF grid cells.

    Each feature is a rectangular polygon with a ``linear_index`` property
    so that the JavaScript front-end can map values to cells by index.
    """
    features: list[dict] = []
    n_rows, n_cols = _validate_grid(lon, lat, context="GeoJSON grid feature generation")

    features.extend(
        _grid_cell_feature(lon, lat, i, j, n_rows, n_cols)
        for i in range(n_rows)
        for j in range(n_cols)
    )

    metadata = {
        "resolucao_m": [float(resolution_x), float(resolution_y)],
    }

    return OrderedDict(
        [
            ("type", "FeatureCollection"),
            ("metadata", metadata),
            ("features", features),
        ]
    )


def create_values_json(
    var: NDArray,
    scale_min: float,
    scale_max: float,
    date_time: datetime | None,
    wind_data: dict | None = None,
) -> dict[str, Any]:
    """Build the per-timestep JSON payload.

    Parameters
    ----------
    var:
        2-D array of values for a single time step.
    scale_min, scale_max:
        Colour-scale boundaries.
    date_time:
        Forecast datetime (local).
    wind_data:
        Optional wind-vector data (from ``compute_wind_vectors_at_height``).
    """
    arr = var.filled(np.nan) if isinstance(var, np.ma.MaskedArray) else np.asarray(var)
    assert_reasonable_array_size(
        arr.shape,
        arr.dtype,
        context="values JSON payload generation",
        multiplier=4.0,
    )

    # Vectorized: round, flatten, convert to Python list in one pass
    flat = np.round(arr.astype(np.float64), 2).ravel()
    values_rounded: list[float | None] = flat.tolist()  # type: ignore
    # Replace NaN with None — only touch NaN positions (O(nan_count) vs O(N))
    nan_indices = np.flatnonzero(np.isnan(flat))
    for idx in nan_indices:
        values_rounded[idx] = None

    # Date formatting
    if date_time is None:
        date_str = "N/A"
    else:
        try:
            dt = date_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            date_str = dt.strftime("%d/%m/%Y %H:%M:%S")
        except Exception:
            date_str = str(date_time)

    scale_values = [float(round(x, 2)) for x in np.linspace(scale_min, scale_max, 6)]

    metadata: dict[str, Any] = {
        "scale_values": scale_values,
        "date_time": date_str,
    }
    if wind_data is not None:
        metadata["wind"] = wind_data

    return {"metadata": metadata, "values": values_rounded}


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


def save_geojson(
    output_dir: str | Path,
    filename_prefix: str,
    lon: NDArray,
    lat: NDArray,
    dx: float,
    dy: float,
    colormap: str = "",
) -> Path:
    """Create and save a grid GeoJSON file.

    The ``colormap`` parameter is accepted for backward-compatibility but
    is **no longer stored** in the output — external clients should read it from
    their own configuration.
    """
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{filename_prefix}.geojson"
    write_grid_geojson_stream(out_path, lon, lat, dx, dy, colormap)
    logger.info("Saved GeoJSON: %s", out_path)
    return out_path


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
    operand order as :func:`_grid_cell_feature`, in the input dtype) and the
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


def _grid_cell_corners(
    lon: NDArray,
    lat: NDArray,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Compute per-cell corner coordinates for every grid cell, vectorized.

    Returns ``(lon_left, lon_right, lat_top, lat_bottom)`` as flat row-major
    Python lists. The arithmetic mirrors :func:`_grid_cell_feature` exactly —
    same expressions and operand order, evaluated in the input dtype (float32
    grids stay float32) — followed by Python's builtin ``round(v, 10)`` per
    element. ``np.round`` must NOT be used here: it disagrees with builtin
    ``round`` at decimal ties for float64 inputs (e.g. ``-14.000000000050001``).
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

    return (
        _rounded_coordinate_list(lon_left),
        _rounded_coordinate_list(lon_right),
        _rounded_coordinate_list(lat_top),
        _rounded_coordinate_list(lat_bottom),
    )


def _rounded_coordinate_list(arr: NDArray) -> list[float]:
    """Flatten row-major and apply builtin ``round(v, 10)`` per element."""
    return [round(v, 10) for v in np.asarray(arr).ravel().tolist()]


def _write_grid_geojson_stream_reference(
    output_path: str | Path,
    lon: NDArray,
    lat: NDArray,
    resolution_x: float,
    resolution_y: float,
) -> Path:
    """Reference (pre-vectorization) stream writer.

    Kept module-private as the byte-identity oracle for tests of
    :func:`write_grid_geojson_stream` — do not use in production code paths.
    """
    n_rows, n_cols = _validate_grid(lon, lat, context=f"streamed GeoJSON grid for {output_path}")
    metadata = {"resolucao_m": [float(resolution_x), float(resolution_y)]}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","metadata":')
        json.dump(metadata, f, separators=(",", ":"), ensure_ascii=False)
        f.write(',"features":[')
        first = True
        for i in range(n_rows):
            for j in range(n_cols):
                if not first:
                    f.write(",")
                first = False
                json.dump(
                    _grid_cell_feature(lon, lat, i, j, n_rows, n_cols),
                    f,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
        f.write("]}")

    return out


def save_values_json(
    output_dir: str | Path,
    name: str,
    json_obj: dict,
) -> Path:
    """Save a per-timestep values JSON file (compact format)."""
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, separators=(",", ":"), ensure_ascii=False)
    logger.info("Saved JSON: %s", out_path)
    return out_path


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


def _grid_cell_feature(
    lon: NDArray,
    lat: NDArray,
    i: int,
    j: int,
    n_rows: int,
    n_cols: int,
) -> dict[str, Any]:
    if i == 0:
        lat_top = float(lat[i, j] + (lat[i, j] - lat[i + 1, j]) / 2)
        lat_bottom = float((lat[i, j] + lat[i + 1, j]) / 2)
    elif i == n_rows - 1:
        lat_top = float((lat[i - 1, j] + lat[i, j]) / 2)
        lat_bottom = float(lat[i, j] - (lat[i - 1, j] - lat[i, j]) / 2)
    else:
        lat_top = float((lat[i - 1, j] + lat[i, j]) / 2)
        lat_bottom = float((lat[i, j] + lat[i + 1, j]) / 2)

    if j == 0:
        lon_left = float(lon[i, j] - (lon[i, j + 1] - lon[i, j]) / 2)
        lon_right = float((lon[i, j] + lon[i, j + 1]) / 2)
    elif j == n_cols - 1:
        lon_left = float((lon[i, j - 1] + lon[i, j]) / 2)
        lon_right = float(lon[i, j] + (lon[i, j] - lon[i, j - 1]) / 2)
    else:
        lon_left = float((lon[i, j - 1] + lon[i, j]) / 2)
        lon_right = float((lon[i, j] + lon[i, j + 1]) / 2)

    polygon_coords = [
        [
            [round(lon_left, 10), round(lat_bottom, 10)],
            [round(lon_right, 10), round(lat_bottom, 10)],
            [round(lon_right, 10), round(lat_top, 10)],
            [round(lon_left, 10), round(lat_top, 10)],
            [round(lon_left, 10), round(lat_bottom, 10)],
        ]
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": polygon_coords},
        "properties": {"linear_index": int(i * n_cols + j)},
    }


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
        text = json.dumps(values, separators=(",", ":"), ensure_ascii=False)
        if not first:
            f.write(",")
        first = False
        f.write(text[1:-1])
