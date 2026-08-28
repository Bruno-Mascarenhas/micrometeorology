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
the frame's own coordinates. The only rotation the physics admits is one about
the *sun*, which :func:`polar_unwrap` turns into a translation instead.

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
:func:`polar_unwrap`
    Sun-centred polar re-parameterisation (SPIN, Paletta et al., CVPR 2022
    OmniCV workshop, arXiv:2111.14507). Rotational invariance about the sun
    becomes translational invariance, and the circumsolar annulus — which
    governs the diffuse/direct split — is magnified. **Requires the lens
    projection**, so it stays inert until a calibration supplies the sun's pixel
    position; see :class:`~allsky.lens.LensCalibration`.

All functions take and return ``(3, H, W)`` float32 CHW arrays in ``[0, 1]`` —
BEFORE the DINOv2 standardisation, which must stay last so the backbone always
receives the distribution it was pretrained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allsky.lens import LensCalibration

__all__ = [
    "AugmentationPipeline",
    "LensCalibration",
    "exposure_jitter",
    "polar_unwrap",
    "random_erasing",
    "sensor_noise",
    "translate",
]

#: sRGB display gamma. Exposure is a linear-space operation, so a gain applied
#: to gamma-encoded pixels would not be a gain at all.
SRGB_GAMMA = 2.2


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


def sensor_noise(chw: np.ndarray, rng: np.random.Generator, *, sigma: float = 0.01) -> np.ndarray:
    """Add zero-mean Gaussian noise of standard deviation *sigma*."""
    noisy = chw + rng.normal(0.0, sigma, size=chw.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0, out=noisy)


def random_erasing(
    chw: np.ndarray,
    rng: np.random.Generator,
    *,
    area_range: tuple[float, float] = (0.01, 0.06),
    aspect_range: tuple[float, float] = (0.4, 2.5),
    keep_solar_disc: tuple[int, int] | None = None,
    disc_radius: int = 12,
) -> np.ndarray:
    """Erase one random rectangle, filled with the frame's own mean.

    Parameters
    ----------
    keep_solar_disc:
        ``(row, col)`` of the sun. When given, a patch overlapping the solar
        disc is redrawn: occluding the sun itself is not a nuisance, it changes
        the physics the model is being asked about.
    disc_radius:
        Radius in pixels of the protected disc.

    Returns
    -------
    numpy.ndarray
        A copy with one rectangle erased.
    """
    _, height, width = chw.shape
    out = chw.copy()
    fill = np.asarray(chw.mean(axis=(1, 2)), dtype=np.float32).reshape(3, 1, 1)
    for _ in range(10):
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
            # Rectangle-vs-square overlap. Testing whether the sun's CENTRE
            # falls inside the rectangle is not enough: a corner can clip the
            # protected square while the centre stays outside it.
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
        The sun's pixel position, from :meth:`LensCalibration.pixel_of`.
    out_shape:
        ``(angles, radii)``; defaults to the input's own ``(H, W)``.

    Returns
    -------
    numpy.ndarray
        ``(3, angles, radii)`` float32.

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
    photometric first (exposure, then noise), geometry last (translation, then
    erasing), so an erased rectangle keeps the frame's mean fill rather than a
    value the noise then perturbs.

    Attributes
    ----------
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
    sun needs its pixel position, which needs the lens projection this site does
    not have yet (see :class:`~allsky.lens.LensCalibration`). Until then an erased rectangle
    can land on the solar disc, which changes the physics rather than adding a
    nuisance — so ``p_erase`` should stay low, and the guard should be wired
    through the moment a calibration exists.
    """

    p_exposure: float = 0.0
    exposure_log2: float = 0.35
    p_noise: float = 0.0
    noise_sigma: float = 0.01
    p_translate: float = 0.0
    translate_px: int = 4
    p_erase: float = 0.0

    @property
    def enabled(self) -> bool:
        """True when at least one transform can fire."""
        return max(self.p_exposure, self.p_noise, self.p_translate, self.p_erase) > 0.0

    def __call__(self, chw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply the pipeline to one CHW frame in ``[0, 1]``."""
        out = chw
        if self.p_exposure and rng.random() < self.p_exposure:
            out = exposure_jitter(out, rng, log2_range=self.exposure_log2)
        if self.p_noise and rng.random() < self.p_noise:
            out = sensor_noise(out, rng, sigma=self.noise_sigma)
        if self.p_translate and rng.random() < self.p_translate:
            out = translate(out, rng, max_shift=self.translate_px)
        if self.p_erase and rng.random() < self.p_erase:
            out = random_erasing(out, rng)
        return np.ascontiguousarray(out, dtype=np.float32)
