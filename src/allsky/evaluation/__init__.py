"""Evaluation subpackage for the multimodal all-sky experiments.

Three light, torch-free modules:

- :mod:`allsky.evaluation.metrics` — NaN-safe regression and classification
  metrics (reusing the shared :mod:`solrad_correction` regression metrics plus
  scikit-learn for classification);
- :mod:`allsky.evaluation.evaluator` — :func:`evaluate_checkpoint`, which runs a
  trained checkpoint over a split and returns an :class:`EvaluationResult` with
  global + stratified metrics (torch is imported lazily inside it, only at call
  time);
- :mod:`allsky.evaluation.reports` — :func:`write_evaluation_report` and
  :func:`compare_experiments`.

Public names resolve lazily through :func:`__getattr__`, so importing this
package (and, transitively, ``import allsky.cli``) never eagerly pulls pandas or
torch — the heavy work happens only when a name is actually used.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EvaluationResult",
    "classification_metrics",
    "compare_experiments",
    "evaluate_checkpoint",
    "regression_metrics",
    "write_evaluation_report",
]

#: Lazily resolved public name -> defining submodule.
_LAZY: dict[str, str] = {
    "EvaluationResult": "allsky.evaluation.evaluator",
    "evaluate_checkpoint": "allsky.evaluation.evaluator",
    "regression_metrics": "allsky.evaluation.metrics",
    "classification_metrics": "allsky.evaluation.metrics",
    "write_evaluation_report": "allsky.evaluation.reports",
    "compare_experiments": "allsky.evaluation.reports",
}


def __getattr__(name: str) -> Any:
    """Resolve a public evaluation name from its submodule on first access."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
