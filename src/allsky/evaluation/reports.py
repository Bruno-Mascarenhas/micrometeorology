"""Write and compare evaluation reports.

:func:`write_evaluation_report` serializes an
:class:`~allsky.evaluation.evaluator.EvaluationResult` into a report directory:

- ``metrics.json`` — global metrics per target + provenance;
- ``stratified.csv`` — the long-form breakdown table;
- ``confusion.csv`` — the sky-class confusion matrix (only when the sky head ran);
- ``predictions.parquet`` — the per-sample frame (optional, via ``predictions``);
- ``report.md`` — a human-readable summary with a per-target metrics table.

:func:`compare_experiments` reads the ``metrics.json`` of several report
directories and builds one cross-model table (returned as a DataFrame, and
written as ``comparison.csv`` / ``comparison.md`` when an ``out_dir`` is given).

Every file is written atomically (temp file in the same directory +
``os.replace``), so a crash never leaves a half-written artifact.  The module is
torch-free (pandas + stdlib only).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from allsky.evaluation.evaluator import EvaluationResult

__all__ = ["compare_experiments", "write_evaluation_report"]

#: Regression metrics surfaced (in order) in the markdown summary table.
_MARKDOWN_REGRESSION_METRICS: tuple[str, ...] = ("rmse", "mae", "bias", "nrmse", "r2", "n")
#: Classification metrics surfaced (in order) in the markdown summary table.
_MARKDOWN_CLASSIFICATION_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "n",
)


def write_evaluation_report(
    result: EvaluationResult,
    report_dir: str | Path,
    *,
    predictions: bool = True,
) -> dict[str, str]:
    """Write the report artifacts for *result* into *report_dir*.

    Parameters
    ----------
    result:
        The evaluation outcome to serialize.
    report_dir:
        Destination directory (created if absent).
    predictions:
        When ``True`` (default) also write ``predictions.parquet``.

    Returns
    -------
    dict[str, str]
        ``artifact name -> written path`` for every file produced.
    """
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    metrics_payload = {
        "checkpoint_path": result.checkpoint_path,
        "split": result.split,
        "n_samples": result.n_samples,
        "enabled_targets": result.enabled_targets,
        "meta": result.meta,
        "global": result.global_metrics,
    }
    written["metrics"] = str(_atomic_json(out / "metrics.json", metrics_payload))
    written["stratified"] = str(_atomic_csv(out / "stratified.csv", result.stratified))

    if result.confusion is not None:
        written["confusion"] = str(_atomic_csv(out / "confusion.csv", _confusion_frame(result)))

    if predictions and not result.predictions.empty:
        written["predictions"] = str(
            _atomic(
                out / "predictions.parquet",
                lambda tmp: result.predictions.to_parquet(tmp, index=False),
            )
        )

    written["report"] = str(_atomic_text(out / "report.md", _render_markdown(result)))
    return written


def compare_experiments(
    report_dirs: Sequence[str | Path],
    *,
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Build a cross-model comparison table from several report directories.

    Each directory's ``metrics.json`` contributes one row keyed by its experiment
    ``name`` and ``model``, with the scalar global metrics flattened into
    ``<target>_<metric>`` columns (nested confusion matrices are skipped).

    Parameters
    ----------
    report_dirs:
        Report directories previously written by :func:`write_evaluation_report`.
    out_dir:
        When given, ``comparison.csv`` and ``comparison.md`` are written there.

    Returns
    -------
    pandas.DataFrame
        One row per experiment; missing metrics are ``NaN`` (columns are the
        union across inputs).
    """
    rows: list[dict[str, Any]] = []
    for report_dir in report_dirs:
        metrics_path = Path(report_dir) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"no metrics.json in report dir {report_dir}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_comparison_row(payload))

    table = pd.DataFrame(rows)
    if out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _atomic_csv(destination / "comparison.csv", table)
        _atomic_text(destination / "comparison.md", _frame_to_markdown(table))
    return table


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------


def _confusion_frame(result: EvaluationResult) -> pd.DataFrame:
    """Confusion matrix as a labelled DataFrame (rows = true, cols = predicted)."""
    confusion = result.confusion
    assert confusion is not None  # noqa: S101 - guarded by the caller
    labels = confusion["labels"]
    frame = pd.DataFrame(
        confusion["matrix"],
        index=[f"true_{name}" for name in labels],
        columns=[f"pred_{name}" for name in labels],
    )
    return frame.reset_index(names="true\\pred")


def _comparison_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One flattened comparison row from a ``metrics.json`` payload."""
    meta = payload.get("meta", {})
    row: dict[str, Any] = {
        "experiment": meta.get("name"),
        "model": meta.get("model"),
        "split": payload.get("split"),
        "n_samples": payload.get("n_samples"),
    }
    for target, metrics in payload.get("global", {}).items():
        for metric, value in metrics.items():
            if metric == "confusion" or isinstance(value, (list, dict)):
                continue
            row[f"{target}_{metric}"] = value
    return row


def _render_markdown(result: EvaluationResult) -> str:
    """Human-readable markdown summary of an evaluation result."""
    meta = result.meta
    lines = [
        f"# Evaluation report — {meta.get('name', 'experiment')}",
        "",
        f"- **model**: `{meta.get('model')}`",
        f"- **split**: `{result.split}` ({result.n_samples} samples)",
        f"- **input mode**: `{meta.get('input_mode')}` | **feature set**: "
        f"`{meta.get('feature_set')}` | **device**: `{meta.get('device')}`",
        f"- **checkpoint**: `{result.checkpoint_path}`",
        f"- **manifest hash ok**: {meta.get('manifest_hash_ok')} | "
        f"**split id ok**: {meta.get('split_id_ok')}",
        "",
        "## Global metrics",
        "",
    ]
    lines.extend(_global_metrics_markdown(result))
    if result.confusion is not None:
        lines += ["", "## Sky-class confusion (rows = true, cols = predicted)", ""]
        lines.extend(_frame_to_markdown(_confusion_frame(result)).splitlines())
    lines.append("")
    return "\n".join(lines)


def _global_metrics_markdown(result: EvaluationResult) -> list[str]:
    """Per-target global-metric tables (regression and/or classification)."""
    lines: list[str] = []
    regression = [t for t in result.enabled_targets if t != "sky"]
    if regression:
        header = ["target", *_MARKDOWN_REGRESSION_METRICS]
        table_rows = [
            [
                target,
                *[_fmt(result.global_metrics[target].get(m)) for m in _MARKDOWN_REGRESSION_METRICS],
            ]
            for target in regression
        ]
        lines.extend(_markdown_table(header, table_rows))
    if "sky" in result.enabled_targets:
        lines += ["", "**sky (classification)**", ""]
        header = ["metric", "value"]
        sky = result.global_metrics["sky"]
        table_rows = [[m, _fmt(sky.get(m))] for m in _MARKDOWN_CLASSIFICATION_METRICS]
        lines.extend(_markdown_table(header, table_rows))
    return lines


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table."""
    header = [str(column) for column in frame.columns]
    rows = [[_fmt(value) for value in record] for record in frame.itertuples(index=False)]
    return "\n".join(_markdown_table(header, rows))


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    """Assemble markdown table lines from a header and body rows."""
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def _fmt(value: Any) -> str:
    """Compact string form for a metric value (4-sig-fig floats)."""
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return f"{value:.4g}"
    return str(value)


# ---------------------------------------------------------------------------
# atomic writers
# ---------------------------------------------------------------------------


def _atomic(path: Path, write: Callable[[Path], Any]) -> Path:
    """Write via a same-directory temp file, then ``os.replace`` onto *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    ok = False
    try:
        write(tmp)
        os.replace(tmp, path)
        ok = True
    finally:
        if not ok:
            tmp.unlink(missing_ok=True)
    return path


def _atomic_text(path: Path, text: str) -> Path:
    """Atomically write *text* to *path* (UTF-8)."""
    return _atomic(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def _atomic_json(path: Path, obj: Any) -> Path:
    """Atomically write *obj* to *path* as indented JSON."""

    def _write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(obj, handle, indent=2, ensure_ascii=False, default=str)

    return _atomic(path, _write)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> Path:
    """Atomically write *frame* to *path* as CSV (no index)."""
    return _atomic(path, lambda tmp: frame.to_csv(tmp, index=False))
