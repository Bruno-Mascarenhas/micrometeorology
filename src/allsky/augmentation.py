"""Augmentation for fixed all-sky camera frames.

Every network that trains on sky images imports its transforms from here, so the
physical argument for each one lives in a single place and cannot drift between
experiments.

WHY MOST STANDARD AUGMENTATION IS ILLEGAL HERE
----------------------------------------------
The camera is fixed and points up. The sun's position in the frame is a
deterministic function of time, and solar elevation/zenith/azimuth are *also*
fed to the model as features. A horizontal flip or a frame-centred rotation
therefore moves the sun in the image while leaving both the label and the
conditioning vector unchanged — it manufactures a physically impossible sample.

Nie, Zamzam & Brandt (Solar Energy 224, 2021) reach the same conclusion for a
model that did not even have geometry features: "Given the fact that PV output
is closely related to the position of the sun in a sky image, geometric
transformations, such as flipping and rotation, are not suitable in this task".

Do not add ``RandomHorizontalFlip``, ``RandomRotation`` or ``RandomAffine`` in
the frame's own coordinates. Two rotations survive that argument. One is about
the *sun*, which :func:`polar_unwrap` turns into a translation instead. The
other is :func:`rotate_about_zenith`, a rotation about the *zenith* applied to
the RGB planes AND the solar-geometry planes together: the sun moves in the
image and the channel that tells the model where the sun is moves with it, so
the sample stays physically consistent — the sky a mount turned by that angle
would have imaged. Without the geometry channel the same rotation is the
illegal one above, which is why the engine warns when ``p_rotate`` is set on a
run without ``model.geometry_channels``.

WHAT IS LEGAL, AND WHY
----------------------
:func:`exposure_jitter`
    A channel-common gain applied in linearised space. Since ``R*g / B*g ==
    R/B``, it leaves the red/blue ratio — the classical cloud discriminator —
    invariant up to the float32 gamma round-trip (measured: max deviation
    ~3e-3, and larger only where the gain saturates a channel, which is what an
    over-exposed sensor does too), while imitating the auto-exposure the camera
    really does
    (Roman et al., AMT 14, 2021, operate the same class of camera across seven
    exposure times precisely because one exposure cannot cover both the
    circumsolar region and the dark sky).
:func:`sensor_noise`
    Additive Gaussian noise: the sensor's own read noise, which is real and
    label-preserving.
:func:`random_erasing`
    Occlusion robustness (Zhong et al., arXiv:1708.04896). Physically it stands
    for a bird, a water drop or dirt on the dome, none of which change the
    irradiance reaching the pyranometer.
:func:`translate`
    A few pixels of camera shift. Mount flex and servicing really do move the
    frame slightly; the sun moves with the scene, so geometry stays consistent.
:func:`rotate_about_zenith`
    A rotation of the whole channel stack about the zenith, the regulariser
    Steiner et al. (2022, arXiv:2106.10270) find substitutes for data when a
    ViT is fine-tuned on little of it. Legal only with the solar-geometry
    planes in the stack, see above.
:func:`polar_unwrap`
    Sun-centred polar re-parameterisation (SPIN, Paletta et al., CVPR 2022
    OmniCV workshop, arXiv:2111.14507). Rotational invariance about the sun
    becomes translational invariance, and the circumsolar annulus — which
    governs the diffuse/direct split — is magnified. **Requires the sun's pixel
    position**, which :meth:`~allsky.lens.LensCalibration.pixel_of` computes from
    a calibration and a solar direction. The calibration exists — 415 sun
    detections over 9 days, equidistant law, 6.67 px median residual, recorded in
    ``configs/allsky/data/local_prepare_iso.yaml`` — and the isotropic
    re-extraction makes it analytic in model coordinates (concentric disc, radius
    = half the frame). What is still missing is the plumbing: no caller hands a
    :class:`~allsky.lens.LensCalibration` and a per-sample sun direction to this
    module, so the function stays unused by the training path.

All functions take and return ``(3, H, W)`` float32 CHW arrays in ``[0, 1]`` —
BEFORE the DINOv2 standardisation, which must stay last so the backbone always
receives the distribution it was pretrained on. The two geometric transforms,
:func:`translate` and :func:`rotate_about_zenith`, also accept the stack with
the geometry planes appended, ``(3 + G, H, W)``, and move every plane alike.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

__all__ = [
    "AugmentationPipeline",
    "exposure_jitter",
    "polar_unwrap",
    "random_erasing",
    "rotate_about_zenith",
    "rotate_frame",
    "sensor_noise",
    "translate",
]

#: sRGB display gamma. Exposure is a linear-space operation, so a gain applied
#: to gamma-encoded pixels would not be a gain at all.
SRGB_GAMMA = 2.2
ERASE_PLACEMENT_ATTEMPTS = 10


def exposure_jitter(
    chw: np.ndarray, rng: np.random.Generator, *, log2_range: float = 0.35
) -> np.ndarray:
    """Multiply by a channel-common gain in linearised space.

    The gain is drawn log-uniformly in ``[2**-log2_range, 2**+log2_range]``, so
    the default spans about +/- a quarter stop either way.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``, gamma-encoded (i.e. as decoded from
        the JPEG).
    rng:
        Seeded generator; the caller owns reproducibility.
    log2_range:
        Half-width of the gain, in stops.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32 in ``[0, 1]``, gamma-encoded again.

    Notes
    -----
    Applying the same gain to all three channels is what keeps the red/blue
    ratio invariant; a per-channel gain would be a white-balance change, and
    white balance is signal here, not nuisance.

    The invariance is exact in real arithmetic but not in float32: the
    ``x**2.2`` / ``x**(1/2.2)`` round trip costs about 3e-3 on the ratio, and
    clipping breaks it outright wherever the gain drives a channel past 1.0.
    """
    gain = float(2.0 ** rng.uniform(-log2_range, log2_range))
    linear: np.ndarray = np.power(chw, SRGB_GAMMA, dtype=np.float32)
    linear *= gain
    np.clip(linear, 0.0, 1.0, out=linear)
    encoded: np.ndarray = np.power(linear, 1.0 / SRGB_GAMMA, dtype=np.float32)
    return encoded


def sensor_noise(
    chw: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float = 0.01,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation *sigma*.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``, gamma-encoded.
    rng:
        Seeded generator; the caller owns reproducibility.
    sigma:
        Standard deviation of the noise, in the same ``[0, 1]`` intensity units
        as *chw*.
    valid:
        ``(H, W)`` bool keep-mask of the pixels the camera actually imaged.
        Where it is False the pixel is ABSENCE, not a dark reading — the ROI mask
        zeroes it and the isotropic pad fills it — and noising it turns "nothing
        was measured here" into a faint signal the model can learn from. ``None``
        noises every pixel, which is right for a frame with no absent region.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32, clipped back into ``[0, 1]``.
    """
    noisy = chw + rng.normal(0.0, sigma, size=chw.shape).astype(np.float32)
    np.clip(noisy, 0.0, 1.0, out=noisy)
    if valid is not None:
        noisy = np.where(valid[None, :, :], noisy, chw)
    return noisy


def random_erasing(
    chw: np.ndarray,
    rng: np.random.Generator,
    *,
    area_range: tuple[float, float] = (0.01, 0.06),
    aspect_range: tuple[float, float] = (0.4, 2.5),
    keep_solar_disc: tuple[int, int] | None = None,
    disc_radius: int = 12,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Erase one random rectangle, filled with the frame's own mean.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``.
    rng:
        Seeded generator; the caller owns reproducibility.
    area_range:
        Bounds of the rectangle's area as a fraction of the frame, drawn
        uniformly.
    aspect_range:
        Bounds of its height-to-width ratio, drawn uniformly.
    keep_solar_disc:
        ``(row, col)`` of the sun in image coordinates. When given, a patch
        overlapping the solar disc is redrawn: occluding the sun itself is not a
        nuisance, it changes the physics the model is being asked about.
    disc_radius:
        Radius in pixels of the protected disc.
    valid:
        ``(H, W)`` bool keep-mask of the pixels the camera actually imaged; the
        fill averages only those.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32 copy with one rectangle erased — or an untouched
        copy, when every placement drawn fell outside the frame or on the
        protected disc.
    """
    _, height, width = chw.shape
    out = chw.copy()
    # Averaged over the imaged pixels only: the ROI mask and the isotropic pad
    # write exact zeros over regions the camera never saw, and folding those into
    # the mean drags the fill toward black — an erasure darker than any sky.
    if valid is None:
        channel_mean = chw.mean(axis=(1, 2))
    else:
        imaged = int(valid.sum())
        channel_mean = (
            (chw * valid[None, :, :]).sum(axis=(1, 2)) / imaged if imaged else chw.mean(axis=(1, 2))
        )
    fill = np.asarray(channel_mean, dtype=np.float32).reshape(3, 1, 1)
    for _ in range(ERASE_PLACEMENT_ATTEMPTS):
        area = rng.uniform(*area_range) * height * width
        aspect = rng.uniform(*aspect_range)
        h = round(float(np.sqrt(area * aspect)))
        w = round(float(np.sqrt(area / aspect)))
        if h < 1 or w < 1 or h >= height or w >= width:
            continue
        top = int(rng.integers(0, height - h))
        left = int(rng.integers(0, width - w))
        if keep_solar_disc is not None:
            sr, sc = keep_solar_disc
            if not (
                top > sr + disc_radius
                or top + h - 1 < sr - disc_radius
                or left > sc + disc_radius
                or left + w - 1 < sc - disc_radius
            ):
                continue
        out[:, top : top + h, left : left + w] = fill
        return out
    return out


def translate(chw: np.ndarray, rng: np.random.Generator, *, max_shift: int = 4) -> np.ndarray:
    """Shift the frame by up to *max_shift* pixels, replicating the edge.

    A small rigid shift of the whole scene is what a bumped mount produces. The
    sun moves with the scene, so the image stays consistent with the geometry
    features — which is exactly what a flip would break.

    The vacated strip repeats the edge row/column. It must NOT wrap: rolling
    pastes the horizon from one side of the dome onto the other, which is the
    same class of physically impossible content that rules flips out in the
    first place.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``.
    rng:
        Seeded generator; the caller owns reproducibility.
    max_shift:
        Largest shift in pixels; the row and column shifts are drawn
        independently and uniformly from ``[-max_shift, max_shift]``.

    Returns
    -------
    numpy.ndarray
        ``(3, H, W)`` float32; the input array itself when both draws are zero.
    """
    dr = int(rng.integers(-max_shift, max_shift + 1))
    dc = int(rng.integers(-max_shift, max_shift + 1))
    if dr == 0 and dc == 0:
        return chw
    padded = np.pad(
        chw,
        ((0, 0), (abs(dr), abs(dr)), (abs(dc), abs(dc))),
        mode="edge",
    )
    top = abs(dr) - dr
    left = abs(dc) - dc
    height, width = chw.shape[1], chw.shape[2]
    return np.ascontiguousarray(padded[:, top : top + height, left : left + width])


def rotate_frame(
    chw: np.ndarray, angle_deg: float, *, fill: float | np.ndarray = 0.0
) -> np.ndarray:
    """Rotate every plane of *chw* by *angle_deg* about the centre of the pixel grid.

    Bilinear resampling; the pixels the rotation pulls in from outside the
    frame take *fill*.

    Parameters
    ----------
    chw:
        ``(..., H, W)`` float32 — a ``(C, H, W)`` frame, or a batch
        ``(B, C, H, W)`` / window ``(B, T, C, H, W)`` of them; every leading
        axis is carried and only the last two are rotated. RGB in ``[0, 1]`` or
        standardized, and the geometry planes, are all rotated alike.
    angle_deg:
        Rotation in degrees. Positive turns the frame the way
        ``numpy.rot90(plane)`` does — a pixel on the top edge moves to the left
        edge — and ``90`` reproduces ``numpy.rot90`` up to resampling error.
    fill:
        Value written where the source lies outside the frame; a scalar, or an
        array broadcastable to *chw* for a per-plane fill.

    Returns
    -------
    numpy.ndarray
        Same shape and dtype as *chw*; the input array itself when *angle_deg*
        is zero.

    Notes
    -----
    The centre of rotation is the centre of the pixel grid, ``((H - 1) / 2,
    (W - 1) / 2)``, and it is the zenith only for frames of the isotropic
    re-extraction: there the crop and pad of ``local_prepare_iso.yaml`` put the
    disc concentric with the frame, and :func:`allsky.lens.isotropic_calibration`
    places the zenith at ``112.03`` px in a 224 px frame against the grid
    centre's ``111.5`` — 0.53 px off, at most 1.06 px of displacement at 180
    degrees, against the 14 px patch the backbone tokenises. On a frame from the
    plain 1920x1080 resize this would be wrong: the fitted optical axis sits
    62 px below the sensor centre in the native frame
    (:data:`allsky.lens.PLANETARIO_NATIVE`), ``62 * S / 1080`` px in a frame
    resized to ``S``, and rotating about the grid centre there would swing the
    sun along an arc it never travels.

    The per-plane *fill* goes through a scalar-only resampler as
    ``rotate(x - fill) + fill``, which equals rotating ``x`` with *fill*
    outside because bilinear interpolation is linear.

    The test for a zero angle is an exact one on purpose, against the
    tolerance rule for floats: it is an identity shortcut, not a numerical
    comparison — a draw of exactly ``0.0`` (a ``max_deg`` of zero, or the
    first of the ``k * 360 / N`` test-time turns) hands back the input array
    untouched, and any other angle, however small, is resampled.
    """
    if angle_deg == 0.0:
        return chw
    offset = np.asarray(fill, dtype=chw.dtype)
    shifted = chw - offset if np.any(offset) else chw
    rotated: np.ndarray = ndimage.rotate(
        shifted, angle_deg, axes=(-2, -1), reshape=False, order=1, mode="constant", cval=0.0
    )
    return rotated + offset if np.any(offset) else rotated


def rotate_about_zenith(chw: np.ndarray, rng: np.random.Generator, *, max_deg: float) -> np.ndarray:
    """Rotate the channel stack about the zenith by a uniform random angle.

    The angle is drawn uniformly in ``[-max_deg, max_deg]``; the rotation is
    :func:`rotate_frame` with a fill of ``0`` — black, the level the prepare pad
    writes into the corners the camera never imaged, which is what the corners
    of an isotropic frame hold before the rotation too. That is why this runs on
    the ``[0, 1]`` frame rather than the standardized one: a zero after
    standardization is mid-grey, not the pad.

    Parameters
    ----------
    chw:
        ``(3 + G, H, W)`` float32: RGB in ``[0, 1]`` followed by the ``G``
        solar-geometry planes, all rotated together so the sun's pixel and the
        plane that marks it stay on the same bearing. A bare ``(3, H, W)``
        frame is accepted but physically illegal, see the module docstring.
    rng:
        Seeded generator; the caller owns reproducibility.
    max_deg:
        Half-width of the angle range, in degrees; ``180`` is a uniform
        rotation over the whole circle.

    Returns
    -------
    numpy.ndarray
        ``(3 + G, H, W)`` float32; the input array itself when the draw is zero.
    """
    angle = float(rng.uniform(-max_deg, max_deg))
    return rotate_frame(chw, angle)


def polar_unwrap(
    chw: np.ndarray,
    *,
    sun_row: float,
    sun_col: float,
    out_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Re-parameterise the frame into polar coordinates centred on the sun.

    Rows of the output are polar angle, columns are radius from the sun, so a
    rotation of the cloud field about the sun becomes a vertical roll — a
    translation the convolution/attention stack already handles. The circumsolar
    annulus, which governs how much of the beam is scattered into the diffuse
    component, occupies proportionally more of the output than it does of the
    input.

    Nearest-neighbour sampling is deliberate: it introduces no new pixel values,
    so it cannot invent colours the sensor never produced.

    Parameters
    ----------
    chw:
        ``(3, H, W)`` float32 in ``[0, 1]``.
    sun_row, sun_col:
        The sun's pixel position in image (row, column) coordinates, from
        :meth:`~allsky.lens.LensCalibration.pixel_of`.
    out_shape:
        ``(angles, radii)``; defaults to the input's own ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``(3, angles, radii)`` float32. Row 0 is ``theta = 0``, which points
        along +row (down the image) and grows toward +column; that is the
        image's own convention, not the clockwise-from-north azimuth
        :meth:`~allsky.lens.LensCalibration.pixel_of` takes. Columns run from
        the sun outward.

    Notes
    -----
    SPIN (arXiv:2111.14507) reports this as a *preprocessing* step rather than a
    random augmentation, and that is how it should be used: applied to train and
    eval alike. Rotating about the sun as an augmentation would displace static
    horizon content — buildings, the mount, dome dirt — to azimuths where it
    physically cannot be.
    """
    _, height, width = chw.shape
    n_theta, n_r = out_shape if out_shape is not None else (height, width)
    max_r = float(np.hypot(max(sun_row, height - sun_row), max(sun_col, width - sun_col)))

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False, dtype=np.float32)
    radius = np.linspace(0.0, max_r, n_r, dtype=np.float32)
    rr, tt = np.meshgrid(radius, theta, indexing="xy")
    rows = np.clip(np.round(sun_row + rr * np.cos(tt)).astype(np.int64), 0, height - 1)
    cols = np.clip(np.round(sun_col + rr * np.sin(tt)).astype(np.int64), 0, width - 1)
    return np.ascontiguousarray(chw[:, rows, cols])


@dataclass(frozen=True, slots=True)
class AugmentationPipeline:
    """Ordered, seeded augmentation applied to the training split only.

    Every probability defaults to ``0.0``, so constructing one without arguments
    is a no-op and an existing experiment keeps its numbers. Order is fixed:
    rotation first, then photometric (exposure, then noise), then translation
    and erasing, so an erased rectangle keeps the frame's mean fill rather than
    a value the noise then perturbs, and the rotation's resampling never
    correlates the noise the sensor draws independently per pixel. A transform
    consumes the generator only when its probability is set, so a pipeline with
    ``p_rotate = 0`` draws exactly what it drew before rotation existed and
    reproduces the same frames under the same seed.

    Attributes
    ----------
    p_rotate, rotate_max_deg:
        Probability and half-width, in degrees, of :func:`rotate_about_zenith`.
    p_exposure, exposure_log2:
        Probability and half-width, in stops, of :func:`exposure_jitter`.
    p_noise, noise_sigma:
        Probability and sigma of :func:`sensor_noise`.
    p_translate, translate_px:
        Probability and maximum shift of :func:`translate`.
    p_erase:
        Probability of :func:`random_erasing`.

    Notes
    -----
    :func:`random_erasing` is called WITHOUT ``keep_solar_disc``: protecting the
    sun needs its pixel position, and this pipeline receives neither a
    :class:`~allsky.lens.LensCalibration` nor a per-sample solar direction — only
    a frame and a generator. The site's calibration is no longer the obstacle
    (see the module docstring); the missing piece is a sun position on the call.
    Until it arrives an erased rectangle can land on the solar disc, which
    changes the physics rather than adding a nuisance, so ``p_erase`` should stay
    low.
    """

    p_exposure: float = 0.0
    exposure_log2: float = 0.35
    p_noise: float = 0.0
    noise_sigma: float = 0.01
    p_translate: float = 0.0
    translate_px: int = 4
    p_erase: float = 0.0
    p_rotate: float = 0.0
    rotate_max_deg: float = 180.0

    @property
    def enabled(self) -> bool:
        """True when at least one transform can fire."""
        return (
            max(self.p_exposure, self.p_noise, self.p_translate, self.p_erase, self.p_rotate) > 0.0
        )

    def __call__(
        self,
        chw: np.ndarray,
        rng: np.random.Generator,
        valid: np.ndarray | None = None,
        geometry: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply the pipeline to one CHW frame in ``[0, 1]``.

        *valid* is the ``(H, W)`` bool mask of the pixels the camera imaged. The
        two transforms that would otherwise treat an absent pixel as a dark
        reading take it: :func:`sensor_noise` leaves the absent ones alone and
        :func:`random_erasing` averages its fill over the imaged ones. Exposure
        and translation need no mask — the first is multiplicative, so zero stays
        zero, and the second moves absence with the frame.

        *geometry* is the ``(G, H, W)`` float32 stack of solar-geometry planes of
        the same frame, when the run feeds them to the model. Rotation and
        translation move it together with the RGB planes; the photometric
        transforms and the erasing never touch it. The return is then
        ``(3 + G, H, W)`` with the moved planes appended — RGB still in
        ``[0, 1]``, for the caller to standardize — and ``(3, H, W)`` without it.
        """
        out = chw
        planes = geometry
        if self.p_rotate and rng.random() < self.p_rotate:
            out, planes = _split(
                rotate_about_zenith(_stack(out, planes), rng, max_deg=self.rotate_max_deg), planes
            )
        if self.p_exposure and rng.random() < self.p_exposure:
            out = exposure_jitter(out, rng, log2_range=self.exposure_log2)
        if self.p_noise and rng.random() < self.p_noise:
            out = sensor_noise(out, rng, sigma=self.noise_sigma, valid=valid)
        if self.p_translate and rng.random() < self.p_translate:
            out, planes = _split(
                translate(_stack(out, planes), rng, max_shift=self.translate_px), planes
            )
        if self.p_erase and rng.random() < self.p_erase:
            out = random_erasing(out, rng, valid=valid)
        result = np.ascontiguousarray(out, dtype=np.float32)
        return result if planes is None else np.concatenate([result, planes], axis=0)


def _stack(rgb: np.ndarray, geometry: np.ndarray | None) -> np.ndarray:
    return rgb if geometry is None else np.concatenate([rgb, geometry], axis=0)


def _split(
    stacked: np.ndarray, geometry: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    if geometry is None:
        return stacked, None
    return stacked[:3], stacked[3:]
