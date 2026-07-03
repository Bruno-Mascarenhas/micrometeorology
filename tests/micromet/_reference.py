"""Frozen byte oracles for the fast WRF JSON/GeoJSON implementations.

VERBATIM copies (only imports adjusted) of the pre-deletion implementations
removed from ``src/micrometeorology/wrf`` on branch
``refactor/remove-legacy-paths`` (source: ``git show master:src/...``).

The values/GeoJSON output bytes are a hard contract with the site front-end.
These reference functions pin that contract: production writers are compared
against them bitwise, so the arithmetic here must never be "modernised",
reordered, or otherwise touched.

Provenance:

- ``_validate_grid``, ``_grid_cell_feature``, ``create_grid_geojson``,
  ``create_values_json`` and ``reference_write_grid_geojson_stream``
  (formerly ``_write_grid_geojson_stream_reference``) come from
  ``master:src/micrometeorology/wrf/geojson.py``.
- ``interpolate_speed_to_height`` and ``compute_wind_vectors_at_height``
  come from ``master:src/micrometeorology/wrf/interpolation.py`` (numpy
  branches only; the xarray branches were dropped).
- ``compute_adjusted_heights`` comes from
  ``master:src/micrometeorology/wrf/variables.py`` (numpy branch only; the
  ``safe_binary_op`` numpy path applies the operator directly, which is what
  is inlined here).
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from micrometeorology.wrf.interpolation import vertical_interpolate
from micrometeorology.wrf.safety import assert_reasonable_array_size

if TYPE_CHECKING:
    from datetime import datetime

    from numpy.typing import NDArray

    from micrometeorology.wrf.reader import WRFDataset


# ---------------------------------------------------------------------------
# geojson.py oracles
# ---------------------------------------------------------------------------


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
    values_rounded: list[float | None] = list(flat.tolist())
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


def reference_write_grid_geojson_stream(
    output_path: str | Path,
    lon: NDArray,
    lat: NDArray,
    resolution_x: float,
    resolution_y: float,
) -> Path:
    """Reference (pre-vectorization) stream writer.

    Byte-identity oracle for ``geojson.write_grid_geojson_stream`` — the
    per-cell ``_grid_cell_feature`` + ``json.dump`` loop.
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


# ---------------------------------------------------------------------------
# interpolation.py oracles (numpy branches)
# ---------------------------------------------------------------------------


def interpolate_speed_to_height(
    speed_4d: NDArray,
    heights: NDArray,
    target_height: float,
) -> NDArray:
    """Interpolate wind speed to a target height for all time steps.

    Numpy path of the pre-deletion ``interpolate_speed_to_height``.
    """
    return vertical_interpolate(speed_4d, heights, target_height, axis=1)


def compute_wind_vectors_at_height(
    u_central: NDArray,
    v_central: NDArray,
    height_adjusted: NDArray,
    target_height: float,
    downsampling: int = 4,
) -> dict:
    """Compute wind vectors interpolated to *target_height* with down-sampling.

    Returns a dict with keys:
    - ``downsampled_angles``: wind direction angles (degrees, meteorological convention)
    - ``downsampled_magnitudes``: wind speed (m/s)
    - ``downsampled_linear_indices``: row-major linear indices for the sampled points
    """
    ny, nx = u_central.shape[2], u_central.shape[3]

    # Vectorized interpolation for all time steps at once
    u_all = vertical_interpolate(u_central, height_adjusted, target_height, axis=1)
    v_all = vertical_interpolate(v_central, height_adjusted, target_height, axis=1)

    # Average over time ignoring NaNs
    with np.errstate(invalid="ignore"):
        u_target = np.nanmean(u_all, axis=0)
        v_target = np.nanmean(v_all, axis=0)

    magnitude = np.hypot(u_target, v_target)
    angle = np.arctan2(u_target, v_target) * 180.0 / np.pi
    angle = np.where(angle < 0, angle + 360.0, angle)

    # Fast downsampling with advanced slicing
    i_idx, j_idx = np.mgrid[0:ny:downsampling, 0:nx:downsampling]
    i_flat = i_idx.ravel()
    j_flat = j_idx.ravel()

    angles_flat = angle[i_flat, j_flat]
    mags_flat = magnitude[i_flat, j_flat]

    valid = ~np.isnan(angles_flat)

    linear_indices = (i_flat * nx + j_flat)[valid]

    return {
        "downsampled_angles": angles_flat[valid].tolist(),
        "downsampled_magnitudes": mags_flat[valid].tolist(),
        "downsampled_linear_indices": linear_indices.tolist(),
    }


# ---------------------------------------------------------------------------
# variables.py oracle (numpy branch)
# ---------------------------------------------------------------------------


def compute_adjusted_heights(
    ds: WRFDataset,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute adjusted heights above terrain for vertical interpolation.

    Returns ``(U_central, V_central, height_adjusted, speed_4d)`` where:
    - ``U_central``, ``V_central``: wind components at grid cell centers
    - ``height_adjusted``: height above terrain at layer midpoints
    - ``speed_4d``: resulting wind speed at all levels
    """
    u_raw = ds.get_variable("U")
    v_raw = ds.get_variable("V")

    # Interpolate staggered grids to mass-grid cell centers positionally.
    u_shape = list(u_raw.shape)
    u_shape[3] -= 1
    assert_reasonable_array_size(u_shape, u_raw.dtype, context="U wind destagger")
    u_central = (u_raw[:, :, :, :-1] + u_raw[:, :, :, 1:]) / 2.0

    v_shape = list(v_raw.shape)
    v_shape[2] -= 1
    assert_reasonable_array_size(v_shape, v_raw.dtype, context="V wind destagger")
    v_central = (v_raw[:, :, :-1, :] + v_raw[:, :, 1:, :]) / 2.0

    # Geopotential height
    ph = ds.get_variable("PH")
    phb = ds.get_variable("PHB")
    hgt = ds.get_variable("HGT")

    geopot_total = ph + phb
    height = geopot_total / 9.81

    # Midpoint heights
    height_shape = list(height.shape)
    height_shape[1] -= 1
    assert_reasonable_array_size(
        height_shape,
        height.dtype,
        context="geopotential height destagger",
    )
    height_central = (height[:, :-1, :, :] + height[:, 1:, :, :]) / 2.0

    # Adjust for terrain.
    assert_reasonable_array_size(
        height_central.shape,
        height_central.dtype,
        context="height above terrain",
    )
    height_adjusted = height_central - hgt[:, np.newaxis, :, :]

    # Speed at all levels
    speed_4d = np.hypot(u_central, v_central)

    return u_central, v_central, height_adjusted, speed_4d
