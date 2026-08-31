"""Solar geometry rendered into the frame's own pixels.

Angles are radians and the coordinate frame is the station's horizontal one:
``x`` toward north, ``y`` toward east, ``z`` toward the zenith.
"""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from allsky.lens import LensCalibration

__all__ = [
    "GEOMETRY_CHANNEL_NAMES",
    "SOLAR_DISC_SIGMA_RAD",
    "resolve_geometry_channels",
    "solar_geometry_maps",
]

GEOMETRY_CHANNEL_NAMES: tuple[str, ...] = (
    "cos_sun_angle",
    "cos_pixel_zenith",
    "solar_disc",
)

#: Angular width of the soft solar disc. The sun subtends 0.53 degrees, which at
#: a 224 px model input (0.80 degrees per pixel) is below one pixel and far below
#: the 14 px patch the backbone tokenises with. The channel therefore marks the
#: sun at the scale the tokeniser can represent — one patch is ~11 degrees — and
#: 5 degrees is that choice, not a property of the sun.
SOLAR_DISC_SIGMA_RAD: float = np.radians(5.0)


@lru_cache(maxsize=8)
def _pixel_directions(calibration: LensCalibration, height: int, width: int) -> np.ndarray:
    directions = calibration.direction_of((height, width))
    directions.flags.writeable = False
    return directions


def solar_geometry_maps(
    calibration: LensCalibration,
    shape: tuple[int, int],
    *,
    sun_zenith_rad: float,
    sun_azimuth_rad: float,
    channels: tuple[str, ...] = GEOMETRY_CHANNEL_NAMES,
) -> np.ndarray:
    """Per-pixel solar geometry for one frame.

    Parameters
    ----------
    calibration:
        Lens projection of the frame the maps are built for — in the model's own
        pixel coordinates, so a resized frame wants a calibration at that size
        (:func:`allsky.lens.isotropic_calibration`), never one at another.
    shape:
        ``(H, W)`` of the frame, in pixels.
    sun_zenith_rad:
        Solar zenith angle in radians, measured from straight up.
    sun_azimuth_rad:
        Solar azimuth in radians, clockwise from north — the station convention,
        before the mount rotation, which the calibration applies.
    channels:
        Which maps to stack, named from :data:`GEOMETRY_CHANNEL_NAMES`. They are
        always returned in that constant's order, whatever order they are asked
        for, so the plane a trained weight belongs to cannot move between a run
        and its reload.

    Returns
    -------
    numpy.ndarray
        ``(len(channels), H, W)`` float32, dimensionless, in the order of
        :data:`GEOMETRY_CHANNEL_NAMES`:

        - ``cos_sun_angle`` in ``[-1, 1]`` — cosine of the angle between the
          pixel's direction and the sun's;
        - ``cos_pixel_zenith`` in ``[-1, 1]`` — cosine of the pixel's own zenith
          angle, negative outside the horizon;
        - ``solar_disc`` in ``(0, 1]`` — a Gaussian of angular distance to the
          sun with width :data:`SOLAR_DISC_SIGMA_RAD`, peaking at 1 on the solar
          direction.

    Raises
    ------
    ValueError
        If *channels* is empty or names a map this module does not build.
    """
    selected = resolve_geometry_channels(channels)
    height, width = int(shape[0]), int(shape[1])
    directions = _pixel_directions(calibration, height, width)
    sun = np.array(
        [
            np.sin(sun_zenith_rad) * np.cos(sun_azimuth_rad),
            np.sin(sun_zenith_rad) * np.sin(sun_azimuth_rad),
            np.cos(sun_zenith_rad),
        ],
        dtype=np.float32,
    )
    cos_sun_angle = np.tensordot(sun, directions, axes=(0, 0))
    built = {"cos_sun_angle": cos_sun_angle, "cos_pixel_zenith": directions[2]}
    if "solar_disc" in selected:
        angle_to_sun = np.arccos(np.clip(cos_sun_angle, -1.0, 1.0))
        built["solar_disc"] = np.exp(-0.5 * (angle_to_sun / SOLAR_DISC_SIGMA_RAD) ** 2)
    return np.stack([built[name] for name in selected]).astype(np.float32, copy=False)


def resolve_geometry_channels(requested: bool | Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a ``geometry_channels`` config value to channel names.

    Parameters
    ----------
    requested:
        ``True`` for every channel, ``False``/``None`` for none, or a sequence of
        names from :data:`GEOMETRY_CHANNEL_NAMES`.

    Returns
    -------
    tuple of str
        The selected names in :data:`GEOMETRY_CHANNEL_NAMES` order, empty when
        no channels were asked for.

    Raises
    ------
    ValueError
        If a name is unknown, if the same one is asked for twice, or if the
        sequence is empty.
    """
    if requested is None or requested is False:
        return ()
    if requested is True:
        return GEOMETRY_CHANNEL_NAMES
    names = [str(name) for name in requested]
    if not names:
        raise ValueError(
            "geometry_channels is an empty list; write false to disable the channels "
            f"or name some of {list(GEOMETRY_CHANNEL_NAMES)}"
        )
    unknown = [name for name in names if name not in GEOMETRY_CHANNEL_NAMES]
    if unknown:
        raise ValueError(
            f"unknown geometry channel(s) {unknown}; available: {list(GEOMETRY_CHANNEL_NAMES)}"
        )
    if len(set(names)) != len(names):
        raise ValueError(f"geometry_channels repeats a channel: {names}")
    return tuple(name for name in GEOMETRY_CHANNEL_NAMES if name in set(names))
