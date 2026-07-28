"""Tests for allsky.evaluation.reports rendering helpers.

Torch-free: the markdown formatter is exercised directly plus through a
hand-built :class:`~allsky.evaluation.evaluator.EvaluationResult`, so no
checkpoint, dataset or model is needed.
"""

from __future__ import annotations

import math

import pandas as pd

from allsky.evaluation.evaluator import EvaluationResult
from allsky.evaluation.reports import _fmt, _render_markdown


class TestFmt:
    def test_integral_floats_render_exactly(self):
        # ``regression_metrics`` stores n as a float, so a realistic split size
        # must not degrade to scientific notation in the markdown table.
        assert _fmt(61344.0) == "61344"
        assert _fmt(12000.0) == "12000"
        assert _fmt(17.0) == "17"
        assert _fmt(0.0) == "0"
        assert _fmt(-3.0) == "-3"

    def test_non_integral_floats_keep_four_significant_figures(self):
        assert _fmt(0.123456) == "0.1235"
        assert _fmt(1234.5678) == "1235"
        assert _fmt(1.2345e-07) == "1.235e-07"

    def test_nan_and_infinities_do_not_raise(self):
        assert _fmt(float("nan")) == "nan"
        assert _fmt(math.inf) == "inf"
        assert _fmt(-math.inf) == "-inf"

    def test_non_floats_pass_through(self):
        assert _fmt(61344) == "61344"
        assert _fmt("kstar") == "kstar"
        assert _fmt(None) == "None"


def _result(n: float) -> EvaluationResult:
    """Minimal result carrying one regression target with sample count *n*."""
    return EvaluationResult(
        checkpoint_path="best.ckpt",
        split="test",
        n_samples=int(n),
        enabled_targets=["dhi"],
        global_metrics={"dhi": {"rmse": 12.5, "mae": 9.25, "bias": -0.5, "n": n}},
        stratified=pd.DataFrame(),
        confusion=None,
        predictions=pd.DataFrame(),
        meta={"name": "exp", "model": "film"},
    )


class TestRenderMarkdown:
    def test_regression_table_sample_count_is_an_exact_integer(self):
        markdown = _render_markdown(_result(61344.0))
        assert "| 61344 |" in markdown
        assert "6.134e+04" not in markdown
