"""Best-effort experiment metadata collection."""

import platform
import sys
import time
from importlib import metadata as importlib_metadata
from typing import Any

from micrometeorology.common.git import run_git, source_root


def collect_run_metadata(
    *,
    config: Any,
    model: Any | None = None,
    started_at: float | None = None,
    training_duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Collect reproducibility metadata without failing the experiment.

    What a third party needs to place a result: interpreter and platform, the
    versions of the packages that do the numerics, the device the run used, the
    commit of THIS source tree and whether it was dirty, wall-clock timings, a
    model summary and a flattened view of the config.

    Every probe here is best-effort by contract. A metadata collector must never
    be the reason a finished experiment loses its artifacts, so a failure is
    recorded as a value in the returned mapping instead of propagating.

    Parameters
    ----------
    config:
        Experiment config; read by duck typing, so a partial stand-in reports
        ``None`` for the attributes it lacks rather than failing.
    model:
        Trained model, if one exists yet. Its parameter counts and best-epoch
        markers are read when present.
    started_at:
        ``time.monotonic()`` reading taken when the run began, used for the
        end-to-end duration in seconds.
    training_duration_seconds:
        Time spent in the training loop alone, recorded as passed.

    Returns
    -------
    dict
        JSON-safe metadata mapping, or ``{"metadata_error": <message>}`` if
        collection itself failed.
    """
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
    # init (RuntimeError) are recorded rather than raised.
    except (ImportError, OSError, RuntimeError) as exc:
        info["error"] = str(exc)
    return info


def _git_metadata() -> dict[str, Any]:
    # The source tree, not the process working directory: a run launched from a
    # sibling checkout must still record THIS repository's HEAD and dirty flag.
    repo_root = source_root()
    commit = run_git(["rev-parse", "HEAD"], cwd=repo_root)
    # A clean tree ("") and an unavailable git (None) both read as "not dirty".
    dirty_text = run_git(["status", "--porcelain", "--untracked-files=no"], cwd=repo_root)
    return {
        "commit": commit,
        "dirty": bool(dirty_text),
    }


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
        # fails as AttributeError/TypeError, and an unmaterialized lazy parameter
        # that cannot report its size as RuntimeError. The count is optional.
        except (AttributeError, TypeError, RuntimeError) as exc:
            info["parameter_count_error"] = str(exc)
    info["best_metric"] = getattr(model, "best_metric", None)
    info["best_epoch"] = getattr(model, "best_epoch", None)
    settings = getattr(model, "dataloader_settings", None)
    if settings is not None:
        try:
            info["dataloader"] = settings.to_dict()
        # Duck-typed `to_dict()`: a missing method raises AttributeError, one
        # with another signature TypeError. Recorded, not raised.
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
