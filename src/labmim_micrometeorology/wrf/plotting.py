"""Cartopy-based map plotting for WRF output.

Replaces all Basemap usage with Cartopy.  The visual output aims to match
the legacy maps as closely as possible.

Cartopy Data Requirements
-------------------------
Cartopy needs Natural Earth data for coastlines and borders.  On systems
without internet access, pre-download the data::

    python -c "import cartopy; cartopy.config['data_dir'] = '/path/to/data'"

See https://scitools.org.uk/cartopy/docs/latest/installing.html#data
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from labmim_micrometeorology.common.paths import ensure_dir
from labmim_micrometeorology.common.types import GridLevel

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colormap utilities
# ---------------------------------------------------------------------------


def saturated_cmap(cmap_name: str, saturation_factor: float = 2.0) -> mcolors.ListedColormap:
    """Return a colormap with adjusted colour saturation."""
    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, cmap.N))
    hsv = mcolors.rgb_to_hsv(colors[:, :3])
    hsv[:, 1] *= saturation_factor
    hsv[:, 1] = np.clip(hsv[:, 1], 0, 1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.ListedColormap(rgb)


# ---------------------------------------------------------------------------
# Map creation
# ---------------------------------------------------------------------------


def create_map_axes(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    figsize: tuple[int, int] = (8, 6),
) -> tuple[Figure, Axes]:
    """Create a figure with Cartopy axes over the given extent."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    return fig, ax


def add_map_features(
    ax: Axes,
    grid_level: GridLevel,
    shapes_dir: str | Path | None = None,
) -> None:
    """Add coastlines, state borders, lat/lon grid, and optional municipality shapes."""
    # Coastlines
    coast_width = {GridLevel.D03: 2, GridLevel.D04: 3}.get(grid_level, 1)
    ax.coastlines(resolution="10m", linewidth=coast_width)

    # State borders
    state_width = 2 if grid_level == GridLevel.D03 else 1
    ax.add_feature(
        cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces_lines", "10m"),
        linewidth=state_width,
        edgecolor="black",
        facecolor="none",
    )

    # Municipality shapes (from local shapefiles)
    if grid_level in (GridLevel.D03, GridLevel.D04, GridLevel.D05) and shapes_dir is not None:
        shp_path = Path(shapes_dir) / "BRMUE250GC_SIR.shp"
        if shp_path.exists():
            reader = shapereader.Reader(str(shp_path))
            ax.add_geometries(
                reader.geometries(),
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="gray",
                linewidth=0.5,
            )
        else:
            logger.warning("Municipality shapefile not found: %s", shp_path)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False


def plot_scalar_field(
    ax: Axes,
    lon: NDArray,
    lat: NDArray,
    data: NDArray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    alpha: float = 0.4,
    saturation: float = 2.0,
) -> None:
    """Plot a filled contour + pcolormesh overlay of a scalar field."""
    cmap = saturated_cmap(cmap_name, saturation)
    transform = ccrs.PlateCarree()

    ax.contourf(lon, lat, data, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, transform=transform)
    mesh = ax.pcolormesh(
        lon, lat, data, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, transform=transform
    )
    cb = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.04)
    cb.ax.tick_params(labelsize=10)


def plot_wind_field(
    ax: Axes,
    lon: NDArray,
    lat: NDArray,
    u: NDArray,
    v: NDArray,
    speed: NDArray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    grid_level: GridLevel,
    alpha: float = 0.4,
    saturation: float = 2.0,
) -> None:
    """Plot wind speed colour fill with quiver arrows."""
    cmap = saturated_cmap(cmap_name, saturation)
    transform = ccrs.PlateCarree()

    ax.contourf(
        lon, lat, speed, alpha=alpha, cmap="Blues", vmin=vmin, vmax=vmax, transform=transform
    )
    mesh = ax.pcolormesh(
        lon, lat, speed, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, transform=transform
    )
    cb = plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.04)
    cb.ax.tick_params(labelsize=10)

    # Sub-sample for quiver
    stride = {
        GridLevel.D01: 6,
        GridLevel.D02: 3,
        GridLevel.D03: 4,
        GridLevel.D04: 4,
        GridLevel.D05: 4,
    }.get(grid_level, 4)
    ax.quiver(
        lon[::stride, ::stride],
        lat[::stride, ::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        transform=transform,
        scale=50,
        width=0.003,
    )


def plot_contour_overlay(
    ax: Axes,
    lon: NDArray,
    lat: NDArray,
    data: NDArray,
    levels: list[float] | None = None,
) -> None:
    """Overlay labelled contour lines (e.g. pressure isobars)."""
    if levels is None:
        levels = [880, 900, 950, 1000, 1013]
    transform = ccrs.PlateCarree()
    cs = ax.contour(
        lon, lat, data, levels=levels, linewidths=0.8, colors="black", transform=transform
    )
    ax.clabel(cs, colors="black", fmt="%.0f")


def save_figure(fig: Figure, output_dir: str | Path, name: str) -> Path:
    """Save a figure to PNG."""
    out_dir = ensure_dir(output_dir)
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path)
    logger.info("Saved figure: %s", out_path)
    plt.close(fig)
    return out_path
