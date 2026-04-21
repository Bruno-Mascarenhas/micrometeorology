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
from pathlib import Path  # noqa: TC003 — used at runtime in save_geojson/save_values_json
from typing import TYPE_CHECKING, Any

import numpy as np

from micrometeorology.common.paths import ensure_dir

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


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
    n_rows, n_cols = lon.shape

    for i in range(n_rows):
        for j in range(n_cols):
            # Cell corners via averaging with neighbours
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

            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": polygon_coords},
                    "properties": {"linear_index": int(i * n_cols + j)},
                }
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
    geojson_obj = create_grid_geojson(lon, lat, dx, dy, colormap)
    out_path = out_dir / f"{filename_prefix}.geojson"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f, separators=(",", ":"), ensure_ascii=False)
    logger.info("Saved GeoJSON: %s", out_path)
    return out_path


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
