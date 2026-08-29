"""Experiment reports: saving metrics, predictions, and config."""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExperimentReport:
    """Container for experiment results.

    Attributes
    ----------
    experiment_name, model_name:
        Identify the run; both are echoed into the manifest.
    metrics:
        Regression scores in the original units of the target column.
    config:
        The resolved config, serialized so the run can be reproduced from the
        artifact directory alone.
    train_history:
        Per-epoch curves keyed ``"epoch"``, ``"train_loss"`` and
        ``"val_loss"``; empty for models that do not train in epochs.
    metadata:
        Run provenance — commit hash, timings, environment.
    """

    experiment_name: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    train_history: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Render a terminal-friendly summary of the run's metrics.

        Returns the text rather than printing it: the entry point owns stdout,
        so a notebook or a batch job can keep the summary and stay quiet.
        """
        lines = [
            f"\n{'=' * 50}",
            f"  Experiment: {self.experiment_name}",
            f"  Model:      {self.model_name}",
            f"{'-' * 50}",
            *(f"  {name:>8s}: {value:.6f}" for name, value in self.metrics.items()),
            f"{'=' * 50}",
        ]
        return "\n".join(lines)
