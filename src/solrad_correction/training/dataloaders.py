"""DataLoader and runtime setting resolution."""

import platform
from dataclasses import dataclass

import torch

from solrad_correction.config import RuntimeConfig
from solrad_correction.utils.device import resolve_device


@dataclass(frozen=True, slots=True)
class DataLoaderSettings:
    """Resolved PyTorch DataLoader and training-runtime settings.

    Every field is already concrete: the ``auto``/``None`` requests a config
    may carry have been decided against the machine the run is on, so both the
    loop and the checkpoint metadata read the same values.

    Attributes
    ----------
    device:
        Torch device string the batches and the module are moved to.
    num_workers:
        Loader worker processes; ``0`` means loading happens in the training
        process.
    pin_memory:
        Whether batches are staged in page-locked memory, which only pays off
        for a host-to-device copy.
    persistent_workers:
        Whether workers survive between epochs; meaningless without workers.
    prefetch_factor:
        Batches each worker loads ahead, or ``None`` when there are no workers.
    amp:
        Whether the training loop runs under autocast with a gradient scaler.
        CUDA-only: inference never uses it, and neither does a CPU run.
    torch_compile:
        Whether to attempt ``torch.compile`` on the module.
    gradient_clip:
        Max gradient norm per step, or ``None`` to leave gradients unclipped.
    """

    device: str
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    amp: bool
    torch_compile: bool
    gradient_clip: float | None

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        """Return the settings as a JSON-serializable dict (stored in checkpoint metadata)."""
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


def resolve_dataloader_settings(
    runtime: RuntimeConfig,
) -> DataLoaderSettings:
    """Resolve runtime config into concrete DataLoader/training settings.

    Unset (``None``) fields are decided from the resolved device: no worker
    processes on CPU or on Windows, otherwise up to four bounded by the torch
    thread count; pinned memory only off-CPU; persistent workers only when
    there are workers; and AMP only on CUDA, which is also enforced on an
    explicit request, since the gradient scaler is a CUDA-only path.

    Parameters
    ----------
    runtime:
        Runtime section of the experiment config.

    Returns
    -------
    DataLoaderSettings
        Fully resolved settings, recorded in checkpoint metadata so a resumed
        run can be compared against the one it continues.
    """
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
