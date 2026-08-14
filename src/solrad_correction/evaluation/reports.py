"""Experiment reports: saving metrics, predictions, and config."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from solrad_correction.experiments.artifacts import ArtifactLayout
from solrad_correction.utils.io import save_json, save_predictions

logger = logging.getLogger(__name__)


@dataclass
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

    def save(self, output_dir: str | Path) -> None:
        """Save full report to the experiment directory.

        Metrics, resolved config, training history and metadata each land at
        their canonical path in the artifact layout. Absolute epoch numbers,
        when the history carries them, become the CSV index; otherwise the
        positional index is labelled ``"epoch"``. Writing both would emit the
        column twice.
        """
        layout = ArtifactLayout.from_experiment_dir(output_dir)
        layout.ensure_directories()

        save_json(self.metrics, layout.metrics)

        save_json(self.config, layout.config_resolved)

        if self.train_history:
            history_df = pd.DataFrame(self.train_history)
            if "epoch" in history_df.columns:
                history_df["epoch"] = history_df["epoch"].astype("int64")
                history_df.set_index("epoch").to_csv(layout.training_history)
            else:
                history_df.to_csv(layout.training_history, index_label="epoch")

        if self.metadata:
            save_json(self.metadata, layout.metadata)

        logger.info("Report saved to %s", layout.root)

    def print_summary(self) -> None:
        """Print a terminal-friendly summary of the run's metrics."""
        print(f"\n{'=' * 50}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Model:      {self.model_name}")
        print(f"{'-' * 50}")
        for name, value in self.metrics.items():
            print(f"  {name:>8s}: {value:.6f}")
        print(f"{'=' * 50}")


def save_experiment_results(
    report: ExperimentReport,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    index: pd.DatetimeIndex | None = None,
) -> None:
    """Save complete experiment results: report + predictions.

    Parameters
    ----------
    report:
        Metrics, config, history and metadata for the run.
    y_true, y_pred:
        Aligned arrays of shape ``(N,)``, in the original units of the target
        column — inverse-transformed, not the scaled values the model emitted.
    output_dir:
        Experiment directory; its artifact tree is created if missing.
    index:
        Timestamps for the ``N`` rows, written as the CSV index. ``None`` falls
        back to a positional index, which leaves the predictions unjoinable
        against the source series.
    """
    report.save(output_dir)
    layout = ArtifactLayout.from_experiment_dir(output_dir)
    save_predictions(y_true, y_pred, layout.predictions, index)
