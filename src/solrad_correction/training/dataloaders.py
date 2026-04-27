"""DataLoader and runtime setting resolution."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from solrad_correction.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class DataLoaderSettings:
    """Resolved PyTorch DataLoader settings."""

    device: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    amp: bool
    torch_compile: bool
    gradient_clip: float | None

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        return {
            "device": self.device,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "amp": self.amp,
            "torch_compile": self.torch_compile,
            "gradient_clip": self.gradient_clip,
        }


def resolve_device(requested: str = "auto") -> str:
    """Resolve a user-requested device into an available torch device string."""
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("runtime.device='cuda' was requested, but CUDA is not available")
        return "cuda"
    if requested != "auto":
        raise ValueError(f"Unknown device request: {requested}")
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dataloader_settings(
    runtime: RuntimeConfig,
) -> DataLoaderSettings:
    """Resolve runtime config into concrete DataLoader/training settings."""
    device = resolve_device(runtime.device)

    if runtime.num_workers is None:
        num_workers = (
            0
            if device == "cpu" or platform.system() == "Windows"
            else min(4, torch.get_num_threads())
        )
    else:
        num_workers = runtime.num_workers

    pin_memory = runtime.pin_memory if runtime.pin_memory is not None else device != "cpu"
    persistent_workers = (
        runtime.persistent_workers if runtime.persistent_workers is not None else num_workers > 0
    )
    prefetch_factor = runtime.prefetch_factor if num_workers > 0 else None
    amp = runtime.amp if runtime.amp is not None else "cuda" in device
    return DataLoaderSettings(
        device=device,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor,
        amp=amp and "cuda" in device,
        torch_compile=runtime.torch_compile,
        gradient_clip=runtime.gradient_clip,
    )
