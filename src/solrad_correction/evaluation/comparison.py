"""Compare results across multiple experiments."""

from pathlib import Path

import pandas as pd

from solrad_correction.utils.io import load_json


def compare_experiments(experiment_dirs: list[str | Path]) -> pd.DataFrame:
    """Load metrics from multiple experiments and produce a comparison table.

    Parameters
    ----------
    experiment_dirs:
        List of experiment directories using the v2 artifact layout. A
        directory with no ``metrics/metrics.json`` is skipped, so a run still
        in flight or one that failed before evaluation drops out of the table
        instead of aborting the comparison.

    Returns
    -------
    pd.DataFrame
        Comparison table indexed by experiment name, one column per metric, in
        the original units each run reported. Empty when no directory yielded
        metrics. Metrics are only comparable across rows when the runs share an
        evaluation policy and a test period.
    """
    rows: list[dict] = []
    for d in experiment_dirs:
        d = Path(d)
        metrics_path = d / "metrics" / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        metrics["experiment"] = d.name
        rows.append(metrics)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("experiment")
    return df
