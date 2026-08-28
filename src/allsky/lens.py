"""The fisheye optics of one all-sky camera, in one place.

Two questions in this project are the same question: *where in the frame does a
given sky direction land* (which is what a sun-centred representation needs) and
*which pixels are sky at all* (which is what a region-of-interest mask needs).
Both are answered by the optical centre, the pixel radius of the horizon and the
projection law, so :class:`LensCalibration` answers both and nothing else
describes the lens.

Keeping them apart is how a calibration gets written down twice, in two
conventions, and drifts. This module imports only numpy, so both
:mod:`allsky.preprocessing` (deterministic, every split) and
:mod:`allsky.augmentation` (random, training only) can depend on it without
either depending on the other.

Pixel coordinates are image ``(row, col)``, origin at the top-left. Angles are
radians: zenith measured from straight up, azimuth clockwise from north — the
convention the station's solar geometry already uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LensCalibration"]


@dataclass(frozen=True, slots=True)
class LensCalibration:
    """Where the sky disc sits in the frame and how directions map onto it.

    Attributes
    ----------
    centre_row, centre_col:
        The optical axis in image pixel coordinates. Not necessarily the image
        centre: a fisheye is only as centred as its mount.
    radius_px:
        Pixel radius of the 90-degree zenith circle, i.e. the horizon.
    equidistant:
        ``True`` selects the ``r = f * theta`` law, standard for meteorological
        fisheyes; ``False`` selects the equisolid ``r = 2f sin(theta/2)``.
    azimuth_offset_rad:
        Rotation of the camera about its optical axis. A property of the mount,
        not of the lens, which is why it is a separate term.

    Notes
    -----
    A camera that looks up sees the sky mirrored east-west against a map drawn
    looking down: with north toward the top of the frame, east falls to the
    **left**. Fitting the Planetario sun track both ways settles it — mirrored
    gives a 6.67 px median residual and an optical centre inside the frame,
    unmirrored gives 31.69 px and a centre 9 px above the top edge. No azimuth
    offset can convert one into the other; they differ by a reflection.
    """

    centre_row: float
    centre_col: float
    radius_px: float
    equidistant: bool = True
    azimuth_offset_rad: float = 0.0

    @classmethod
    def centred_in(
        cls,
        shape: tuple[int, ...],
        *,
        radius_fraction: float = 1.0,
        centre: tuple[float, float] | None = None,
    ) -> LensCalibration:
        """The heuristic calibration for an image of *shape*, with no measurement.

        Assumes the common all-sky geometry: the projection fills the shorter
        image dimension and is roughly centred. It does not know about a
        decentred optical axis, vignetting, static horizon obstructions or a
        non-circular sensor crop — fit those against the sun's known position,
        or supply a measured PNG mask (:func:`allsky.preprocessing.load_mask`).

        Parameters
        ----------
        shape:
            Image shape; only the leading ``(H, W)`` entries are read, so a
            ``(H, W, 3)`` frame shape can be passed straight through.
        radius_fraction:
            Horizon radius as a fraction of half the shorter image dimension.
        centre:
            Optical axis in ``(row, col)``; the image centre when None.

        Returns
        -------
        LensCalibration
            Equidistant, unrotated, centred as described.
        """
        height, width = int(shape[0]), int(shape[1])
        row, col = (
            (height / 2.0, width / 2.0) if centre is None else (float(centre[0]), float(centre[1]))
        )
        return cls(
            centre_row=row,
            centre_col=col,
            radius_px=radius_fraction * min(height, width) / 2.0,
        )

    def pixel_of(self, zenith_rad: float, azimuth_rad: float) -> tuple[float, float]:
        """Pixel ``(row, col)`` a sky direction lands on.

        Parameters
        ----------
        zenith_rad:
            Angle from straight up, in radians; ``0`` is the zenith and
            ``pi / 2`` the horizon.
        azimuth_rad:
            Bearing clockwise from north, in radians.

        Returns
        -------
        tuple of float
            ``(row, col)``, unrounded, and outside the frame for a direction the
            sensor does not image.
        """
        if self.equidistant:
            r = self.radius_px * (zenith_rad / (np.pi / 2.0))
        else:
            r = self.radius_px * (np.sin(zenith_rad / 2.0) / np.sin(np.pi / 4.0))
        bearing = azimuth_rad + self.azimuth_offset_rad
        return (
            self.centre_row - r * float(np.cos(bearing)),
            self.centre_col - r * float(np.sin(bearing)),
        )

    def keep_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        """Boolean keep-array for an image of *shape*: ``True`` inside the horizon.

        Parameters
        ----------
        shape:
            Image shape; only the leading ``(H, W)`` entries are read.

        Returns
        -------
        numpy.ndarray
            ``(H, W)`` bool. The disc is the one this calibration describes, so
            a decentred axis produces a decentred mask rather than a silently
            wrong one.
        """
        height, width = int(shape[0]), int(shape[1])
        rows = np.arange(height)[:, None]
        cols = np.arange(width)[None, :]
        dist_sq = (rows - self.centre_row) ** 2 + (cols - self.centre_col) ** 2
        return dist_sq <= self.radius_px**2
