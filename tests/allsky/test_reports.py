"""Tests for allsky.evaluation.reports rendering helpers.

Torch-free: the markdown formatter is exercised directly plus through a
hand-built :class:`~allsky.evaluation.evaluator.EvaluationResult`, so no
checkpoint, dataset or model is needed.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from allsky.evaluation.evaluator import EvaluationResult
from allsky.evaluation.reports import _fmt, _render_markdown, write_evaluation_report


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


def _refuse(token: str) -> object:
    raise AssertionError(f"the report carries a bare {token} token")


class TestWriteEvaluationReport:
    """The writer had no direct test: the returned mapping, the confusion matrix
    round trip and the predictions switch were all only ever exercised through a
    full evaluation, which cannot pin what any one of them writes."""

    @staticmethod
    def _with_confusion(rows: int = 3) -> EvaluationResult:
        base = _result(float(rows))
        matrix = {"labels": ["clear", "overcast"], "matrix": [[2, 1], [0, 3]]}
        predictions = pd.DataFrame({"sample_id": ["a", "b", "c"], "dhi": [1.0, 2.0, 3.0]})
        return replace(base, confusion=matrix, predictions=predictions)

    def test_the_returned_mapping_names_every_file_it_wrote(self, tmp_path: Path):
        written = write_evaluation_report(self._with_confusion(), tmp_path, predictions=True)

        assert set(written) >= {"metrics", "report", "confusion", "predictions"}
        for path in written.values():
            assert Path(path).is_file(), path

    def test_the_confusion_matrix_round_trips_with_labelled_axes(self, tmp_path: Path):
        written = write_evaluation_report(self._with_confusion(), tmp_path, predictions=False)

        restored = pd.read_csv(written["confusion"], index_col=0)
        assert list(restored.index) == ["true_clear", "true_overcast"]
        assert list(restored.columns) == ["pred_clear", "pred_overcast"]
        assert restored.to_numpy().tolist() == [[2, 1], [0, 3]]

    def test_predictions_are_skipped_when_not_asked_for(self, tmp_path: Path):
        written = write_evaluation_report(self._with_confusion(), tmp_path, predictions=False)

        assert "predictions" not in written
        assert not list(tmp_path.glob("*.parquet"))

    def test_an_empty_predictions_frame_writes_no_parquet(self, tmp_path: Path):
        result = replace(self._with_confusion(), predictions=pd.DataFrame())

        written = write_evaluation_report(result, tmp_path, predictions=True)

        assert "predictions" not in written

    def test_a_non_finite_metric_survives_into_the_json_as_null(self, tmp_path: Path):
        result = _result(3.0)
        result.global_metrics["dhi"]["skill_clearsky"] = float("nan")

        written = write_evaluation_report(result, tmp_path, predictions=False)

        payload = json.loads(Path(written["metrics"]).read_text(encoding="utf-8"))
        # The strict writer refuses a bare NaN token, so a non-finite metric has
        # to reach the file as null or not at all — never as the string "NaN",
        # which every strict reader of the document rejects wholesale.
        assert payload["global"]["dhi"]["skill_clearsky"] is None
        # And the file really is strict JSON: a bare NaN token fails the whole
        # document for any reader that does not opt into Python's extension.
        json.loads(Path(written["metrics"]).read_text(encoding="utf-8"), parse_constant=_refuse)
