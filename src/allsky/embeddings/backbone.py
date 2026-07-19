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

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# DINOv2 identity (pinned) + ImageNet normalization constants.
# ---------------------------------------------------------------------------

#: ``torch.hub`` GitHub repo hosting the DINOv2 entrypoints.
DINOV2_REPO = "facebookresearch/dinov2"
#: The DINOv2 ViT-S/14 entrypoint (384-dim patch/CLS tokens, patch size 14).
DINOV2_MODEL = "dinov2_vits14"
#: Pinned commit of ``facebookresearch/dinov2`` (the ``main`` HEAD resolved at
#: implementation time, 2026-07-19).  DINOv2 publishes no release tags, so a
#: full commit SHA is the only stable pin: ``torch.hub`` fetches this exact
#: revision (``repo:ref`` syntax) instead of the moving ``main`` branch, keeping
#: model code + weights reproducible across machines and time.
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"

#: ImageNet channel means/stds DINOv2 was trained with (RGB, [0, 1] scaled).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Pooling name -> output embedding dimension for DINOv2 ViT-S/14 (384-dim).
_POOLING_DIM = {"cls": 384, "mean": 384, "cls+mean": 768}

Pooling = Literal["cls", "mean", "cls+mean"]

#: Backbone names the CLI / :func:`build_backbone` understands.
AVAILABLE_BACKBONES = ("dinov2_vits14", "fake")

__all__ = [
    "AVAILABLE_BACKBONES",
    "DINOV2_MODEL",
    "DINOV2_REPO",
    "DINOV2_REVISION",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DinoV2Backbone",
    "FakeBackbone",
    "VisualBackbone",
    "build_backbone",
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

    def transform(self, images: Sequence[np.ndarray]) -> Any:
        """Map ``uint8`` HWC frames to a model-ready batch."""
        ...

    def encode(self, batch: Any) -> Any:
        """Encode a :meth:`transform` batch to a ``(B, dim)`` embedding matrix."""
        ...


def _resize_uint8(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a ``uint8`` HWC frame to ``size x size`` with PIL bilinear.

    Grayscale frames are promoted to 3-channel RGB first (safety net matching
    :meth:`allsky.dataset.AllSkyDataset._load_image`).
    """
    arr = np.asarray(image)
    if arr.ndim == 2:  # pragma: no cover - grayscale safety net
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.astype(np.uint8, copy=False)
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr)
    from PIL import Image

    resized = Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR)
    return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))


class DinoV2Backbone:
    """DINOv2 ViT-S/14 backbone (``torch.hub``, pinned revision, ImageNet norm).

    Parameters
    ----------
    pooling:
        Token pooling: ``"cls"`` (CLS token, 384-d), ``"mean"`` (mean of patch
        tokens, 384-d) or ``"cls+mean"`` (concatenation, 768-d).
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

    name = DINOV2_MODEL

    def __init__(
        self,
        *,
        pooling: Pooling = "cls",
        device: str = "auto",
        dtype: Literal["fp16", "fp32"] = "fp16",
        image_size: int = 224,
    ) -> None:
        if pooling not in _POOLING_DIM:
            raise ValueError(f"unknown pooling {pooling!r}; expected one of {sorted(_POOLING_DIM)}")
        if image_size % 14 != 0:
            raise ValueError(f"image_size {image_size} must be a multiple of the patch size (14)")
        self.revision = DINOV2_REVISION
        self.pooling: str = pooling
        self.dim = _POOLING_DIM[pooling]
        self.dtype = dtype
        self.image_size = image_size
        self._device_pref = device
        self.transform_description = (
            f"imagenet-norm, resize {image_size}x{image_size} bilinear, pooling={pooling}"
        )
        self._model: Any = None
        self._device: Any = None

    def _ensure_model(self) -> None:
        """Load the pinned hub model once and move it to the resolved device."""
        if self._model is not None:
            return
        import torch

        from allsky.training import resolve_device

        self._device = torch.device(resolve_device(self._device_pref))
        model = torch.hub.load(
            f"{DINOV2_REPO}:{DINOV2_REVISION}",
            DINOV2_MODEL,
            trust_repo=True,
        )
        model.eval()
        model.to(self._device)
        self._model = model

    def load_torch_module(self) -> Any:
        """Load (once) and return the underlying hub ``nn.Module``.

        The extraction path uses :meth:`encode` (``inference_mode``, detached,
        CPU) — unusable for end-to-end image training.  Image-mode training
        instead wants the raw model so it can run it *with* gradients and
        register / freeze its parameters; this returns exactly that module (its
        ``blocks`` sequence supports the usual last-*n* ViT unfreezing).  Loading
        happens once per process and is cached on the instance.
        """
        self._ensure_model()
        return self._model

    def transform(self, images: Sequence[np.ndarray]) -> Any:
        """Resize + ImageNet-normalize frames to a ``(B, 3, H, W)`` CPU tensor."""
        import torch

        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
        tensors = []
        for image in images:
            arr = _resize_uint8(image, self.image_size)
            chw = torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0
            tensors.append((chw - mean) / std)
        return torch.stack(tensors)

    def _pool(self, batch: Any) -> Any:
        """Run ``forward_features`` and pool tokens per :attr:`pooling`."""
        import torch

        out = self._model.forward_features(batch)
        cls = out["x_norm_clstoken"]
        if self.pooling == "cls":
            return cls
        patch_mean = out["x_norm_patchtokens"].mean(dim=1)
        if self.pooling == "mean":
            return patch_mean
        return torch.cat([cls, patch_mean], dim=-1)

    def encode(self, batch: Any) -> Any:
        """Encode a transform batch to a ``(B, dim)`` fp32 CPU tensor."""
        import torch

        self._ensure_model()
        batch = batch.to(self._device)
        use_amp = self.dtype == "fp16" and self._device.type == "cuda"
        with torch.inference_mode():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self._pool(batch)
            else:
                feats = self._pool(batch)
        return feats.to(torch.float32).cpu()


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
        """Return the frames as contiguous ``uint8`` arrays (no torch)."""
        return [np.ascontiguousarray(np.asarray(image, dtype=np.uint8)) for image in images]

    def _embed_one(self, image: np.ndarray) -> np.ndarray:
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).digest()
        seed = int.from_bytes(digest[:8], "little")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def encode(self, batch: Any) -> Any:
        """Hash each frame to a deterministic ``(B, dim)`` fp32 torch tensor."""
        import torch

        vectors = np.stack([self._embed_one(image) for image in batch])
        return torch.from_numpy(vectors)


def build_backbone(
    name: str,
    *,
    pooling: Pooling = "cls",
    device: str = "auto",
    dtype: Literal["fp16", "fp32"] = "fp16",
    fake_dim: int = 32,
) -> VisualBackbone:
    """Construct a backbone by name.

    Parameters
    ----------
    name:
        ``"dinov2_vits14"`` (the real DINOv2 backbone) or ``"fake"`` (the
        deterministic test/dev backbone).
    pooling, device, dtype:
        Forwarded to :class:`DinoV2Backbone` (ignored by :class:`FakeBackbone`).
    fake_dim:
        Embedding dimension for :class:`FakeBackbone`.

    Raises
    ------
    ValueError
        If *name* is not one of :data:`AVAILABLE_BACKBONES`, with a message
        listing the available backbones.
    """
    if name == "fake":
        return FakeBackbone(dim=fake_dim)
    if name == DINOV2_MODEL:
        return DinoV2Backbone(pooling=pooling, device=device, dtype=dtype)
    raise ValueError(
        f"unknown backbone {name!r}; available backbones: {', '.join(AVAILABLE_BACKBONES)}"
    )
