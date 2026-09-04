"""Visual backbones that turn sky frames into fixed-length embeddings.

The pipeline codes against the :class:`VisualBackbone` protocol so extraction is
agnostic to *how* an embedding is produced:

- :class:`DinoV2Backbone` wraps Meta's self-supervised DINOv2 ViT-S/14 loaded
  through :func:`torch.hub.load`, **pinned to a fixed commit**
  (:data:`DINOV2_REVISION`) so the same weights and code are fetched forever —
  never the moving ``main`` branch.  The hub model is downloaded and built once
  per process (``torch.hub`` caches weights under its default cache dir); with a
  single-process, batched extraction loop no worker ever re-downloads it.
- :class:`FakeBackbone` produces deterministic hash-of-bytes embeddings with no
  network and no model download; it is the backbone every test uses.  It imports
  ``torch`` only inside :meth:`FakeBackbone.encode` (never at import), so tests
  that do not touch ``encode`` need no torch either.

Limitation
----------
DINOv2 requires a network round-trip on the very first local run (to fetch the
repo revision + weights into the ``torch.hub`` cache); it must therefore never
run in tests or CI — use :class:`FakeBackbone` there.
"""

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, get_args, runtime_checkable

import numpy as np

from allsky.config import DEFAULT_IMAGE_SIZE
from allsky.frame_pixels import resize_bilinear
from allsky.preprocessing import IMAGENET_MEAN, IMAGENET_STD

#: ``torch.hub`` GitHub repo hosting the DINOv2 entrypoints.
DINOV2_REPO = "facebookresearch/dinov2"
#: Default DINOv2 entrypoint (ViT-S/14, 384-dim tokens, patch size 14).
DINOV2_MODEL = "dinov2_vits14"
#: Pinned commit of ``facebookresearch/dinov2`` (the ``main`` HEAD resolved at
#: implementation time, 2026-07-19).  DINOv2 publishes no release tags, so a
#: full commit SHA is the only stable pin: ``torch.hub`` fetches this exact
#: revision (``repo:ref`` syntax) instead of the moving ``main`` branch, keeping
#: model code + weights reproducible across machines and time.
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"

#: DINOv2 entrypoint -> token width. All four share patch size 14, so
#: ``image_size`` stays a multiple of 14 regardless of which one is selected.
_TOKEN_DIM: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

Pooling = Literal["cls", "mean", "cls+mean"]

#: Pooling names understood by every DINOv2 entrypoint.
POOLINGS: tuple[str, ...] = get_args(Pooling)


def embedding_dim(model: str, pooling: str) -> int:
    """Output width of *model* under *pooling*.

    ``cls+mean`` concatenates the CLS token with the patch-token mean, so it is
    twice the model's token width; the other poolings are one token wide. A
    convolutional backbone has no tokens at all and answers only to ``mean``, the
    global average its own classifier head consumes.

    Raises
    ------
    ValueError
        For an unknown *model* or *pooling*.
    """
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling {pooling!r}; expected one of {sorted(POOLINGS)}")
    if model in TorchvisionBackbone.HEADS:
        if pooling != "mean":
            raise ValueError(
                f"{model} emits no tokens, so pooling {pooling!r} does not exist for it; "
                "it offers only 'mean'"
            )
        return TorchvisionBackbone.HEADS[model][0]
    token_widths = {**_TOKEN_DIM, **Dinov3Backbone.TOKEN_DIM}
    if model not in token_widths:
        raise ValueError(f"unknown backbone {model!r}; expected one of {sorted(token_widths)}")
    return token_widths[model] * (2 if pooling == "cls+mean" else 1)


__all__ = [
    "AVAILABLE_BACKBONES",
    "DINOV2_MODEL",
    "DINOV2_REPO",
    "DINOV2_REVISION",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "POOLINGS",
    "DinoV2Backbone",
    "FakeBackbone",
    "VisualBackbone",
    "build_backbone",
    "embedding_dim",
]


@runtime_checkable
class VisualBackbone(Protocol):
    """Interface every backbone satisfies: identity + transform + encode.

    Attributes
    ----------
    name:
        Backbone identity (e.g. ``"dinov2_vits14"`` or ``"fake"``).
    revision:
        Pinned code/weight revision (a commit SHA for DINOv2).
    dim:
        Output embedding dimension (columns of the ``(B, dim)`` encode result).
    pooling:
        How the backbone reduces its output to one vector (``"cls"``, ``"mean"``,
        ``"cls+mean"``; ``"fake"`` for the test backbone).  It is recorded in the
        store's provenance and compared on resume, so it belongs to the interface
        rather than being read off whichever backbone happens to carry it.

    Methods
    -------
    transform(images):
        Map a sequence of ``uint8`` HWC RGB frames to a model-ready **batch**
        (a stacked float tensor for DINOv2; the raw frames for the fake
        backbone).  The batch object is opaque and only :meth:`encode` consumes
        it.
    encode(batch):
        Turn a batch from :meth:`transform` into a ``(B, dim)`` float embedding
        matrix (a ``torch.Tensor`` or array; extraction converts to numpy).
    """

    name: str
    revision: str
    dim: int
    pooling: str

    def transform(self, images: Sequence[np.ndarray]) -> Any:
        """Map ``uint8`` HWC frames to a model-ready batch."""
        ...

    def encode(self, batch: Any) -> Any:
        """Encode a :meth:`transform` batch to a ``(B, dim)`` embedding matrix."""
        ...


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a ``uint8`` HWC frame to ``size x size`` with PIL bilinear.

    Grayscale frames are promoted to 3-channel RGB first (safety net matching
    the image-loading recipe used by :mod:`allsky.data.datasets`).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:  # pragma: no cover - grayscale safety net
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.astype(np.uint8, copy=False)
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(resize_bilinear(arr, size), dtype=np.uint8)


def imagenet_batch(images: Sequence[np.ndarray], size: int) -> Any:
    """Stack ``uint8`` HWC frames into a standardized ``(B, 3, size, size)`` tensor.

    The one recipe every torch backbone here feeds its model: resize, scale to
    ``[0, 1]``, standardize by :data:`IMAGENET_MEAN` / :data:`IMAGENET_STD`.
    Those statistics are not DINOv2's alone — the torchvision ResNet and
    EfficientNet weight enums declare exactly the same mean and std, verified
    against ``weights.transforms()`` rather than assumed — so one function serves
    every family and they cannot drift apart.

    Parameters
    ----------
    images:
        Sequence of ``(H, W, 3)`` ``uint8`` RGB frames (channels-last, 0-255).
    size:
        Square side to resize to, in pixels.

    Returns
    -------
    torch.Tensor
        ``(B, 3, size, size)`` float32, channels-first, dimensionless.
    """
    import torch

    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for image in images:
        arr = _resize_uint8(image, size)
        chw = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0
        tensors.append((chw - mean) / std)
    return torch.stack(tensors)


class _TorchModuleBackbone:
    """Shared machinery for a backbone that IS one torch module.

    DINOv3 and the torchvision families differ only in how their module is
    built; everything after that — the ImageNet batch, the lazy load onto the run
    device, the autocast policy, the pooling — is the same, and the pooling is
    the same because :mod:`allsky.modeling.backbone_families` already owns it for
    the training path. Sharing it here is what keeps extraction and fine-tuning
    from pooling a model two different ways.

    Subclasses implement :meth:`_build_module` and set :attr:`name`,
    :attr:`revision` and :attr:`dim`.
    """

    name: str
    revision: str
    dim: int

    def __init__(
        self,
        *,
        pooling: str,
        device: str,
        dtype: Literal["fp16", "fp32"],
        image_size: int,
    ) -> None:
        self.pooling = pooling
        self.dtype = dtype
        self.image_size = image_size
        self._device_pref = device
        self._model: Any = None
        self._device: Any = None
        self.transform_description = (
            f"imagenet-norm, resize {image_size}x{image_size} bilinear, pooling={pooling}"
        )

    def _build_module(self) -> Any:
        raise NotImplementedError

    @property
    def _family(self) -> Any:
        from allsky.modeling.backbone_families import family_for

        return family_for(self.name, self.pooling)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch

        from allsky.training import resolve_device

        self._device = torch.device(resolve_device(self._device_pref))
        model = self._build_module()
        model.eval()
        model.to(self._device)
        self._model = model

    def load_torch_module(self) -> Any:
        self._ensure_model()
        return self._model

    def transform(self, images: Sequence[np.ndarray]) -> Any:
        """Resize + ImageNet-normalize frames to a ``(B, 3, H, W)`` CPU tensor."""
        return imagenet_batch(images, self.image_size)

    def encode(self, batch: Any) -> Any:
        """Encode a :meth:`transform` batch to a ``(B, dim)`` fp32 CPU tensor."""
        import torch

        self._ensure_model()
        batch = batch.to(self._device)
        family = self._family
        use_amp = self.dtype == "fp16" and self._device.type == "cuda"
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = family.pooled(self._model, batch)
            else:
                feats = family.pooled(self._model, batch)
        return feats.to(torch.float32).cpu()


class DinoV2Backbone(_TorchModuleBackbone):
    """DINOv2 ViT-*/14 backbone (``torch.hub``, pinned revision, ImageNet norm).

    Everything but the model construction comes from
    :class:`_TorchModuleBackbone`: the ImageNet batch, the lazy load onto the run
    device, the autocast policy, and — the one that matters — the pooling, which
    it takes from :class:`~allsky.modeling.backbone_families.VitTokenFamily`.
    That is the same object the training path pools through, so extraction and
    fine-tuning cannot read this model's tokens two different ways.

    Parameters
    ----------
    model:
        A DINOv2 entrypoint from :data:`AVAILABLE_BACKBONES`; every one has
        patch size 14, so ``image_size`` stays a multiple of 14 for all of them.
    pooling:
        Token pooling: ``"cls"``, ``"mean"`` (both one token wide) or
        ``"cls+mean"`` (concatenation, twice that). :func:`embedding_dim` is
        the authority on the resulting width.
    device:
        ``"auto"`` (cuda -> mps -> cpu), or an explicit torch device string.
    dtype:
        ``"fp16"`` enables fp16 autocast **on CUDA only** (CPU/MPS fp16 autocast
        is not reliably supported, so it is silently a no-op there); ``"fp32"``
        forces full precision.  Embeddings are always returned as fp32.
    image_size:
        Square input size; must be a multiple of the patch size (14).  Default
        224 (16x16 patches).

    Notes
    -----
    The hub model is loaded lazily on the first :meth:`encode` and cached on the
    instance — created **once per process**.  Extraction is single-process and
    batched precisely so no data-loader worker triggers a duplicate download.
    """

    PATCH_SIZE = 14

    def __init__(
        self,
        *,
        model: str = DINOV2_MODEL,
        pooling: Pooling = "cls",
        device: str = "auto",
        dtype: Literal["fp16", "fp32"] = "fp16",
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if image_size % self.PATCH_SIZE != 0:
            raise ValueError(
                f"image_size {image_size} must be a multiple of the patch size ({self.PATCH_SIZE})"
            )
        self.name = model
        self.revision = DINOV2_REVISION
        self.dim = embedding_dim(model, pooling)
        super().__init__(pooling=pooling, device=device, dtype=dtype, image_size=image_size)

    def _build_module(self) -> Any:
        import torch

        return torch.hub.load(
            f"{DINOV2_REPO}:{DINOV2_REVISION}",
            self.name,
            trust_repo=True,
        )


class TorchvisionBackbone(_TorchModuleBackbone):
    """A torchvision classification network used as a feature extractor.

    The classifier head is replaced by :class:`torch.nn.Identity`, so the
    module's own forward already ends at the pooled feature vector and every
    family-specific detail on the way there — a ResNet's flatten, an
    EfficientNet's dropout — stays exactly where its authors put it.

    Parameters
    ----------
    model:
        ``"resnet50"`` or ``"efficientnet_v2_s"``.
    pooling:
        Must be ``"mean"``: these networks emit no tokens, only the global
        average their own heads consume.
    weights:
        ``None`` uses the default ImageNet weights; ``"none"`` builds the
        architecture untrained, which is what a transfer-learning run that will
        load its own weights wants.

    Notes
    -----
    ResNet50's pretraining crop is 224, the size this project feeds. The
    EfficientNetV2-S weights were trained at **384**, so running it at 224 is a
    deliberate deviation from its original recipe and belongs in the arm's
    config header.
    """

    #: torchvision entrypoint -> pooled feature width, and the head attribute
    #: that :class:`torch.nn.Identity` replaces.
    HEADS: ClassVar[dict[str, tuple[int, str]]] = {
        "resnet50": (2048, "fc"),
        "efficientnet_v2_s": (1280, "classifier"),
    }

    def __init__(
        self,
        *,
        model: str,
        pooling: str = "mean",
        device: str = "auto",
        dtype: Literal["fp16", "fp32"] = "fp16",
        image_size: int = DEFAULT_IMAGE_SIZE,
        weights: str | None = None,
    ) -> None:
        if model not in self.HEADS:
            raise ValueError(
                f"unknown torchvision backbone {model!r}; expected one of {sorted(self.HEADS)}"
            )
        self.dim = embedding_dim(model, pooling)
        self._head_attribute = self.HEADS[model][1]
        import torchvision

        self.name = model
        self.revision = f"torchvision {torchvision.__version__}"
        self._weights = weights
        super().__init__(pooling=pooling, device=device, dtype=dtype, image_size=image_size)

    def _build_module(self) -> Any:
        import torchvision.models as tvm
        from torch import nn

        factory = getattr(tvm, self.name)
        weights = (
            None
            if self._weights == "none"
            else "DEFAULT"
            if self._weights is None
            else self._weights
        )
        model = factory(weights=weights)
        setattr(model, self._head_attribute, nn.Identity())
        return model


#: Environment variable holding the DINOv3 weights, so a versioned config names
#: neither the licensed URL nor an absolute machine path.
WEIGHTS_ENV_VAR = "ALLSKY_DINOV3_WEIGHTS"

#: Environment variable holding the DINOv3 source tree, same reason.
REPO_DIR_ENV_VAR = "ALLSKY_DINOV3_REPO"


class Dinov3Backbone(_TorchModuleBackbone):
    """A DINOv3 vision transformer, loaded from a local clone and local weights.

    Two things force this to differ from :class:`DinoV2Backbone`, both measured
    rather than assumed:

    - ``torch.hub.load`` is unusable. DINOv3's ``hubconf`` imports its
      classifiers, segmentors, depthers and text tower, which drag in
      ``torchmetrics``, ``omegaconf``, ``ftfy`` and ``submitit`` — a SLURM
      launcher — none of which a backbone touches. Its
      ``dinov3/hub/backbones.py`` imports only the standard library and a local
      ``utils``, so it is imported directly from the repository instead.
    - the weights are licensed. ``dl.fbaipublicfiles.com/dinov3`` answers HTTP
      403 without an accepted licence, so *weights* is required and names a file
      or a signed URL the operator obtained. **A signed URL is a credential: it
      belongs in an untracked config or an environment variable, never in a
      committed file.**

    Parameters
    ----------
    model:
        A DINOv3 backbone entrypoint, e.g. ``"dinov3_vits16plus"``.
    weights:
        Path to a downloaded ``.pth``, or a URL; falls back to the
        :data:`WEIGHTS_ENV_VAR` environment variable. ``"none"`` builds the
        architecture untrained.
    repo_dir:
        The DINOv3 source tree. Defaults to the ``torch.hub`` cache location.
    pooling, device, dtype, image_size:
        As :class:`DinoV2Backbone`. Patch size is 16, so *image_size* must be a
        multiple of 16.

    Notes
    -----
    Reproducibility gap, stated rather than hidden: this loads whatever revision
    of the DINOv3 source tree *repo_dir* holds. DINOv2 is pinned to a commit
    because ``torch.hub`` accepts ``repo:ref``; here the directory IS the pin,
    and an experiment that must be reproducible should record its provenance.
    """

    #: DINOv3 entrypoint -> token width.
    TOKEN_DIM: ClassVar[dict[str, int]] = {
        "dinov3_vits16": 384,
        "dinov3_vits16plus": 384,
        "dinov3_vitb16": 768,
        "dinov3_vitl16": 1024,
    }

    PATCH_SIZE = 16

    def __init__(
        self,
        *,
        model: str,
        weights: str,
        repo_dir: str | Path | None = None,
        pooling: str = "cls",
        device: str = "auto",
        dtype: Literal["fp16", "fp32"] = "fp16",
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> None:
        if model not in self.TOKEN_DIM:
            raise ValueError(
                f"unknown DINOv3 backbone {model!r}; expected one of {sorted(self.TOKEN_DIM)}"
            )
        if image_size % self.PATCH_SIZE != 0:
            raise ValueError(
                f"image_size {image_size} must be a multiple of DINOv3's patch size "
                f"({self.PATCH_SIZE})"
            )
        weights = weights or os.environ.get(WEIGHTS_ENV_VAR, "")
        if not weights:
            raise ValueError(
                f"{model} needs its weights: set model.backbone_weights or the "
                f"{WEIGHTS_ENV_VAR} environment variable to the downloaded .pth or a signed "
                "URL. They are licensed — the public URL answers 403 — and a signed URL is a "
                "credential, so the environment variable is how a shipped config stays free "
                "of both the secret and the machine path"
            )
        self.name = model
        self.dim = embedding_dim(model, pooling)
        self._weights = weights
        self._repo_dir = Path(
            repo_dir or os.environ.get(REPO_DIR_ENV_VAR) or _default_dinov3_repo_dir()
        )
        self.revision = f"repo_dir={self._repo_dir}"
        super().__init__(pooling=pooling, device=device, dtype=dtype, image_size=image_size)

    def _build_module(self) -> Any:
        import importlib
        import sys

        if not (self._repo_dir / "dinov3" / "hub" / "backbones.py").is_file():
            raise FileNotFoundError(
                f"no DINOv3 source tree at {self._repo_dir}; clone it with "
                "`git clone https://github.com/facebookresearch/dinov3` and point "
                "model.backbone_repo_dir at the result"
            )
        if str(self._repo_dir) not in sys.path:
            sys.path.insert(0, str(self._repo_dir))
        backbones = importlib.import_module("dinov3.hub.backbones")
        entrypoint = getattr(backbones, self.name)
        if self._weights == "none":
            return entrypoint(pretrained=False)
        return entrypoint(pretrained=True, weights=self._weights)


def _default_dinov3_repo_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"


class FakeBackbone:
    """Deterministic, network-free backbone for tests and dry runs.

    Each frame maps to a fixed pseudo-random vector seeded by a SHA-256 hash of
    its raw bytes, so identical frames always yield identical embeddings and runs
    are perfectly reproducible.  ``torch`` is imported **only** inside
    :meth:`encode`; construction and :meth:`transform` are torch-free.

    Parameters
    ----------
    dim:
        Output embedding dimension (default 32).
    """

    name = "fake"

    def __init__(self, dim: int = 32) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.revision = "fake-v1"
        self.pooling = "fake"
        self.dtype = "fp32"
        self.transform_description = "identity (deterministic sha256 hash of frame bytes)"

    def transform(self, images: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Return the ``(H, W, 3)`` ``uint8`` frames as contiguous arrays (no torch)."""
        return [np.ascontiguousarray(np.asarray(image, dtype=np.uint8)) for image in images]

    def _embed_one(self, image: np.ndarray) -> np.ndarray:
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, batch: Any) -> Any:
        """Hash each frame to a deterministic ``(B, dim)`` fp32 torch tensor.

        The vector for a frame is drawn from a generator seeded with the first
        eight bytes of the SHA-256 of that frame's bytes, so equal frames encode
        equally on any machine.
        """
        import torch

        vectors = np.stack([self._embed_one(image) for image in batch])
        return torch.from_numpy(vectors)


#: Backbone names the CLI / :func:`build_backbone` understands.
AVAILABLE_BACKBONES = (
    *_TOKEN_DIM,
    *Dinov3Backbone.TOKEN_DIM,
    *TorchvisionBackbone.HEADS,
    "fake",
)


def build_backbone(
    name: str,
    *,
    pooling: Pooling = "cls",
    device: str = "auto",
    dtype: Literal["fp16", "fp32"] = "fp16",
    image_size: int = DEFAULT_IMAGE_SIZE,
    fake_dim: int = 32,
    weights: str | None = None,
    repo_dir: str | Path | None = None,
) -> VisualBackbone:
    """Construct a backbone by name.

    Parameters
    ----------
    name:
        Any name in :data:`AVAILABLE_BACKBONES`: the four ``dinov2_*`` entrypoints,
        the four ``dinov3_*`` ones, the two torchvision convolutional networks, or
        ``"fake"`` (the deterministic test/dev backbone).
    pooling, device, dtype, image_size:
        Forwarded to whichever of :class:`DinoV2Backbone`, :class:`Dinov3Backbone`
        and :class:`TorchvisionBackbone` *name* selects; :class:`FakeBackbone` reads
        none of them.  ``image_size`` is the square input the backbone is built for,
        and the two transformer families refuse one their patch size cannot tile.
    fake_dim:
        Embedding dimension for :class:`FakeBackbone`.
    weights:
        Where a family that does not ship open weights gets them. Required for
        DINOv3 (licensed; a path or a signed URL). Optional for torchvision,
        where ``None`` means the default ImageNet weights and ``"none"`` builds
        the architecture untrained. Ignored by DINOv2 and the fake backbone.
    repo_dir:
        DINOv3 source tree; defaults to the ``torch.hub`` cache location.

    Returns
    -------
    VisualBackbone
        A ready backbone; the DINOv2 weights are fetched on first encode, not
        here.

    Raises
    ------
    ValueError
        If *name* is not one of :data:`AVAILABLE_BACKBONES`, with a message
        listing the available backbones.
    """
    if name == "fake":
        return FakeBackbone(dim=fake_dim)
    if name in _TOKEN_DIM:
        return DinoV2Backbone(
            model=name, pooling=pooling, device=device, dtype=dtype, image_size=image_size
        )
    if name in Dinov3Backbone.TOKEN_DIM:
        return Dinov3Backbone(
            model=name,
            weights=weights or "",
            repo_dir=repo_dir,
            pooling=pooling,
            device=device,
            dtype=dtype,
            image_size=image_size,
        )
    if name in TorchvisionBackbone.HEADS:
        return TorchvisionBackbone(
            model=name,
            pooling=pooling,
            device=device,
            dtype=dtype,
            image_size=image_size,
            weights=weights,
        )
    raise ValueError(
        f"unknown backbone {name!r}; available backbones: {', '.join(AVAILABLE_BACKBONES)}"
    )
