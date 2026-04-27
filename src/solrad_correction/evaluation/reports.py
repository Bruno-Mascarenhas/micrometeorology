"""Experiment reports: saving metrics, predictions, and config."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from solrad_correction.utils.io import save_json, save_predictions

logger = logging.getLogger(__name__)


@dataclass
class ExperimentReport:
    """Container for experiment results."""

    experiment_name: str
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    train_history: dict[str, list[float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> None:
        """Save full report to the experiment directory."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Metrics
        save_json(self.metrics, out / "metrics.json")

        # Config
        save_json(self.config, out / "config_resolved.json")

        # Training history
        if self.train_history:
            import pandas as pd

            history_df = pd.DataFrame(self.train_history)
            history_df.to_csv(out / "training_history.csv", index_label="epoch")

        if self.metadata:
            save_json(self.metadata, out / "metadata.json")

        logger.info("Report saved to %s", out)

    def print_summary(self) -> None:
        """Print a terminal-friendly summary."""
        print(f"\n{'═' * 50}")
        print(f"  Experiment: {self.experiment_name}")
        print(f"  Model:      {self.model_name}")
        print(f"{'─' * 50}")
        for name, value in self.metrics.items():
            print(f"  {name:>8s}: {value:.6f}")
        print(f"{'═' * 50}")


def save_experiment_results(
    report: ExperimentReport,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path,
    index: pd.DatetimeIndex | None = None,
) -> None:
    """Save complete experiment results: report + predictions."""
    report.save(output_dir)
    save_predictions(y_true, y_pred, Path(output_dir) / "predictions.csv", index)
