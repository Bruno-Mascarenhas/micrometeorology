"""The serving-pin payload the publish tests build, in one place.

Each test module materialises the files a pin names — real checkpoints,
config-only ones, or bytes with a known digest — and hands their paths here,
so a field added to :class:`allsky.serving.ServingConfig` is one edit.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any


def pinned(path: Path | str, sha256: str, report: Path | str) -> dict[str, Any]:
    """One frame member entry: its checkpoint, the digest it must have and its test report."""
    return {"path": str(path), "sha256": sha256, "report": str(report)}


def control(path: Path | str, sha256: str, report: Path | str) -> dict[str, Any]:
    """One control entry: the checkpoint the counterfactual is scored with and its test report."""
    return {"checkpoint": {"path": str(path), "sha256": sha256}, "report": str(report)}


def serving_pin_payload(
    *,
    frame_checkpoints: Sequence[dict[str, Any]],
    sensor_only: dict[str, Any],
    climatology: dict[str, Any],
    dataset: Path | str,
    training_history: Path | str,
    min_elevation_deg: float = 10.0,
    decided_on: str = "2026-09-11",
    label: str = "a probe pin",
    **fields: Any,
) -> dict[str, Any]:
    """A valid pin payload; *fields* override or extend the top-level keys."""
    return {
        "serving": True,
        "id": "probe",
        "label": label,
        "frame_checkpoints": list(frame_checkpoints),
        "min_elevation_deg": min_elevation_deg,
        "controls": {"sensor_only": sensor_only, "climatology": climatology},
        "reports": {"dataset": str(dataset), "training_history": str(training_history)},
        "selection": {"criterion": "the probe", "decided_on": decided_on},
        **fields,
    }
