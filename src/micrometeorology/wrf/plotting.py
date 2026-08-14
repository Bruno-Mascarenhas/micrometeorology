"""Colormap utilities for WRF map figures.

The map drawing itself lives in :mod:`micrometeorology.wrf.batch`, which owns
the single renderer (``_render_figure``) used by every worker process.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def saturated_cmap(cmap_name: str, saturation_factor: float = 2.0) -> mcolors.ListedColormap:
    """Return a copy of a matplotlib colormap with its HSV saturation scaled.

    Parameters
    ----------
    cmap_name:
        Name of any registered matplotlib colormap.
    saturation_factor:
        Multiplier applied to the S channel; above 1 it deepens the ramp, and
        the channel is clipped to [0, 1] so the extremes stay in gamut. The
        hue and value channels, and therefore the ramp's ordering and its
        perceived lightness sequence, are untouched.

    Returns
    -------
    mcolors.ListedColormap
        A listed colormap of ``cmap.N`` colours sampled evenly from the source.
    """
    cmap = plt.colormaps[cmap_name]
    colors = cmap(np.linspace(0, 1, cmap.N))
    hsv = mcolors.rgb_to_hsv(colors[:, :3])
    hsv[:, 1] *= saturation_factor
    hsv[:, 1] = np.clip(hsv[:, 1], 0, 1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return mcolors.ListedColormap(rgb)
