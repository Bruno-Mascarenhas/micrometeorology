"""Best-effort experiment metadata collection."""

import platform
import shutil
import subprocess
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


def collect_run_metadata(
    *,
    config: Any,
    model: Any | None = None,
    started_at: float | None = None,
    training_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without failing the experiment."""
    try:
        meta: dict[str, Any] = {
            "python": _python_metadata(),
            "packages": _package_versions(
                [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "torch",
                    "labmim-micrometeorology",
                ]
            ),
            "device": _device_metadata(),
            "git": _git_metadata(),
            "timing": {
                "duration_seconds": _elapsed(started_at),
                "training_duration_seconds": training_duration_seconds,
            },
            "model": _model_metadata(model),
            "config_summary": _config_summary(config),
        }
        return meta
    # Best-effort by contract: no metadata probe may fail the experiment.
    except Exception as exc:  # noqa: BLE001 # pragma: no cover - defensive by design
        return {"metadata_error": str(exc)}


def _python_metadata() -> dict[str, Any]:
    return {
        "version": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
    }


def _package_versions(names: list[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _device_metadata() -> dict[str, Any]:
    info: dict[str, Any] = {"selected": "cpu", "cuda_available": False}
    try:
        import torch

        info["cuda_available"] = torch.cuda.is_available()
        info["selected"] = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["torch_cuda_version"] = torch.version.cuda
    # torch is optional and its CUDA probe talks to a driver: a missing package
    # (ImportError), an unloadable shared library (OSError) and a failed CUDA
    # init (RuntimeError) are all recorded rather than raised.
    except (ImportError, OSError, RuntimeError) as exc:
        info["error"] = str(exc)
    return info


def _git_metadata() -> dict[str, Any]:
    cwd = Path.cwd()
    commit = _git(["rev-parse", "HEAD"], cwd)
    dirty_text = _git(["status", "--porcelain", "--untracked-files=no"], cwd)
    return {
        "commit": commit,
        "dirty": bool(dirty_text),
    }


def _git(args: list[str], cwd: Path) -> str | None:
    # Resolved from PATH up front: an absent git is the common case (a source
    # tarball, a container without the client) and means "no commit info", the
    # same answer the exec failure below would have produced.
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    try:
        # S603 has no formulation that clears it here: ruff only stays silent for
        # an inline list of string literals, which would mean hardcoding an
        # absolute path to git instead of resolving the one on this machine.
        result = subprocess.run(  # noqa: S603
            [git_executable, *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    # A vanished/unexecutable binary (OSError), a hung call (TimeoutExpired, a
    # SubprocessError) and output that is not valid text (UnicodeDecodeError,
    # raised while decoding under `text=True`) all mean "no commit info".
    except OSError, subprocess.SubprocessError, UnicodeDecodeError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _elapsed(started_at: float | None) -> float | None:
    if started_at is None:
        return None
    return time.monotonic() - started_at


def _model_metadata(model: Any | None) -> dict[str, Any]:
    if model is None:
        return {}

    info: dict[str, Any] = {
        "name": getattr(model, "name", model.__class__.__name__),
        "class": f"{model.__class__.__module__}.{model.__class__.__name__}",
    }
    module = getattr(model, "_module", None)
    if module is not None:
        try:
            info["parameter_count"] = sum(p.numel() for p in module.parameters())
            info["trainable_parameter_count"] = sum(
                p.numel() for p in module.parameters() if p.requires_grad
            )
        # `module` is duck-typed: anything that is not an iterable of tensors
        # fails as AttributeError/TypeError, and a tensor that cannot report its
        # size (an unmaterialized lazy parameter) as RuntimeError. The count is
        # optional, so the fault is recorded next to the rest of the metadata.
        except (AttributeError, TypeError, RuntimeError) as exc:
            info["parameter_count_error"] = str(exc)
    info["best_metric"] = getattr(model, "best_metric", None)
    info["best_epoch"] = getattr(model, "best_epoch", None)
    settings = getattr(model, "dataloader_settings", None)
    if settings is not None:
        try:
            info["dataloader"] = settings.to_dict()
        # Duck-typed `to_dict()`: a missing method is an AttributeError and one
        # with another signature a TypeError. Recorded, not raised.
        except (AttributeError, TypeError) as exc:
            info["dataloader_error"] = str(exc)
    return info


def _config_summary(config: Any) -> dict[str, Any]:
    model = getattr(config, "model", None)
    split = getattr(config, "split", None)
    return {
        "name": getattr(config, "name", None),
        "seed": getattr(config, "seed", None),
        "output_dir": getattr(config, "output_dir", None),
        "model_type": getattr(model, "model_type", None),
        "batch_size": getattr(model, "batch_size", None),
        "sequence_length": getattr(model, "sequence_length", None),
        "max_epochs": getattr(model, "max_epochs", None),
        "evaluation_policy": getattr(model, "evaluation_policy", None),
        "torch_compile": getattr(model, "torch_compile", None),
        "split": {
            "train_ratio": getattr(split, "train_ratio", None),
            "val_ratio": getattr(split, "val_ratio", None),
            "test_ratio": getattr(split, "test_ratio", None),
            "shuffle": getattr(split, "shuffle", None),
        },
    }
