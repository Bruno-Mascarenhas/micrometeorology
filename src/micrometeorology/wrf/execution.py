"""Adaptive execution planning for WRF CLI workflows."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from micrometeorology.wrf import reader as wrf_reader
from micrometeorology.wrf.batch import JsonWorkerBackend, default_workers

if TYPE_CHECKING:
    from collections.abc import Sequence

ReaderRequest = Literal["auto", "eager", "lazy"]
JsonWorkerRequest = Literal["auto", "serial", "pickle", "memmap"]
WorkflowKind = Literal["figures", "json", "pipeline"]

LARGE_FILE_THRESHOLD_BYTES = 512 * 1024 * 1024
LARGE_TOTAL_INPUT_THRESHOLD_BYTES = 1024 * 1024 * 1024
LARGE_JSON_PAYLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
MANY_JSON_TASKS_THRESHOLD = 64


@dataclass(frozen=True, slots=True)
class WRFExecutionPlan:
    """Resolved execution plan for a WRF CLI run."""

    reader: wrf_reader.ReaderMode
    chunks: wrf_reader.ChunkSpec
    json_worker_backend: JsonWorkerBackend
    workers: int
    tmp_dir: Path | None
    reasons: tuple[str, ...]

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _chunk_request_is_explicit(chunks_request: str | None) -> bool:
    if chunks_request is None:
        return False
    value = chunks_request.strip().lower()
    return value not in {"", "auto", "none"}


def _resolve_reader(
    *,
    paths: list[Path],
    reader_request: ReaderRequest,
    chunks_request: str | None,
    parsed_chunks: wrf_reader.ChunkSpec,
    chunking_available: bool,
    large_file_threshold_bytes: int,
    large_total_input_threshold_bytes: int,
) -> tuple[wrf_reader.ReaderMode, wrf_reader.ChunkSpec, list[str]]:
    reasons: list[str] = []
    total_size = sum(_file_size(path) for path in paths)
    largest_size = max((_file_size(path) for path in paths), default=0)

    if reader_request in {"eager", "lazy"}:
        resolved_reader = cast("wrf_reader.ReaderMode", reader_request)
        reasons.append(f"reader explicitly set to {reader_request}")
    elif _chunk_request_is_explicit(chunks_request):
        resolved_reader = "lazy"
        reasons.append("explicit chunk dimensions require lazy reader")
    elif (
        largest_size >= large_file_threshold_bytes
        or total_size >= large_total_input_threshold_bytes
    ):
        resolved_reader = "lazy"
        reasons.append("large input size favors lazy variable selection")
    else:
        resolved_reader = "eager"
        reasons.append("small input favors eager reader")

    chunks_value = (chunks_request or "auto").strip().lower()
    if resolved_reader == "eager":
        if _chunk_request_is_explicit(chunks_request):
            raise ValueError("--chunks with explicit dim=size pairs requires --reader lazy")
        return resolved_reader, None, [*reasons, "chunking disabled for eager reader"]

    if chunks_value == "none":
        return resolved_reader, None, [*reasons, "chunking explicitly disabled"]
    if parsed_chunks == "auto" or chunks_value == "auto":
        if not chunking_available:
            return resolved_reader, None, [*reasons, "dask unavailable; lazy chunking disabled"]
        return resolved_reader, "auto", [*reasons, "lazy chunking set to auto"]
    if not chunking_available:
        raise ValueError("Explicit --chunks settings require dask-backed xarray chunking")
    return resolved_reader, parsed_chunks, [*reasons, "using explicit chunk dimensions"]


def _resolve_json_worker_backend(
    *,
    worker_request: JsonWorkerRequest,
    workers: int,
    estimated_json_payload_bytes: int | None,
    json_task_count: int | None,
    large_json_payload_threshold_bytes: int,
    many_json_tasks_threshold: int,
) -> tuple[JsonWorkerBackend, list[str]]:
    if worker_request in {"serial", "pickle", "memmap"}:
        return cast("JsonWorkerBackend", worker_request), [
            f"JSON worker backend explicitly set to {worker_request}"
        ]

    if workers <= 1:
        return "serial", ["single worker uses serial JSON fast path"]

    payload = estimated_json_payload_bytes or 0
    tasks = json_task_count or 0
    if payload >= large_json_payload_threshold_bytes:
        return "memmap", ["large estimated JSON payload favors memmap worker references"]
    if tasks >= many_json_tasks_threshold and payload > 0:
        return "memmap", ["many JSON tasks favor memmap worker references"]
    return "pickle", ["small JSON workload favors direct pickle payloads"]


def resolve_wrf_execution_plan(
    *,
    paths: list[Path],
    workflow: WorkflowKind,
    reader_request: ReaderRequest = "auto",
    chunks_request: str | None = "auto",
    json_worker_request: JsonWorkerRequest = "auto",
    workers: int | None = None,
    tmp_dir: str | Path | None = None,
    estimated_json_payload_bytes: int | None = None,
    json_task_count: int | None = None,
    large_file_threshold_bytes: int = LARGE_FILE_THRESHOLD_BYTES,
    large_total_input_threshold_bytes: int = LARGE_TOTAL_INPUT_THRESHOLD_BYTES,
    large_json_payload_threshold_bytes: int = LARGE_JSON_PAYLOAD_THRESHOLD_BYTES,
    many_json_tasks_threshold: int = MANY_JSON_TASKS_THRESHOLD,
    chunking_available: bool | None = None,
) -> WRFExecutionPlan:
    """Resolve reader, chunking, and JSON worker choices for a WRF run.

    Explicit concrete requests always win. ``auto`` chooses conservative eager
    and pickle paths for small/serial workloads, and lazy/memmap paths for large
    inputs or large multi-worker JSON payloads.
    """
    parsed_chunks = wrf_reader.parse_chunks(chunks_request)
    resolved_workers = workers or default_workers()
    if resolved_workers < 1:
        raise ValueError("--workers must be >= 1")
    resolved_tmp_dir = Path(tmp_dir) if tmp_dir is not None else None
    if chunking_available is None:
        chunking_available = find_spec("dask") is not None

    resolved_reader, resolved_chunks, reasons = _resolve_reader(
        paths=paths,
        reader_request=reader_request,
        chunks_request=chunks_request,
        parsed_chunks=parsed_chunks,
        chunking_available=chunking_available,
        large_file_threshold_bytes=large_file_threshold_bytes,
        large_total_input_threshold_bytes=large_total_input_threshold_bytes,
    )

    if json_worker_request in {"serial", "pickle"} and resolved_tmp_dir is not None:
        raise ValueError("--tmp-dir is only valid with --worker-backend auto or memmap")

    json_backend, json_reasons = _resolve_json_worker_backend(
        worker_request=json_worker_request,
        workers=resolved_workers,
        estimated_json_payload_bytes=estimated_json_payload_bytes,
        json_task_count=json_task_count,
        large_json_payload_threshold_bytes=large_json_payload_threshold_bytes,
        many_json_tasks_threshold=many_json_tasks_threshold,
    )
    reasons.extend(json_reasons)
    if workflow == "figures":
        reasons.append("backend applies to figure payloads")
    if json_backend == "memmap" and resolved_tmp_dir is not None:
        reasons.append(f"using user temporary directory {resolved_tmp_dir}")
    elif json_backend == "memmap":
        reasons.append("using system temporary directory for memmap payloads")

    return WRFExecutionPlan(
        reader=resolved_reader,
        chunks=resolved_chunks,
        json_worker_backend=json_backend,
        workers=resolved_workers,
        tmp_dir=resolved_tmp_dir,
        reasons=tuple(reasons),
    )


def estimate_json_payload_bytes(tasks: Sequence[object]) -> int:
    """Estimate in-memory ndarray payload bytes for JSON tasks."""
    total = 0
    for task in tasks:
        data = getattr(task, "data", None)
        total += int(getattr(data, "nbytes", 0) or 0)
    return total


def estimate_figure_payload_bytes(tasks: Sequence[object]) -> int:
    """Estimate in-memory ndarray payload bytes for figure tasks."""
    total = 0
    for task in tasks:
        seen: set[int] = set()
        for attr in ("lon", "lat", "data", "overlay_data", "u", "v"):
            data = getattr(task, attr, None)
            if data is None or id(data) in seen:
                continue
            seen.add(id(data))
            total += int(getattr(data, "nbytes", 0) or 0)
    return total


def format_wrf_execution_plan(plan: WRFExecutionPlan) -> str:
    """Return a user-facing multi-line execution-plan summary."""
    chunks = "none" if plan.chunks is None else str(plan.chunks)
    tmp_dir = str(plan.tmp_dir) if plan.tmp_dir is not None else "system temp"
    return (
        "WRF execution plan:\n"
        f"  reader: {plan.reader}\n"
        f"  chunks: {chunks}\n"
        f"  worker backend: {plan.json_worker_backend}\n"
        f"  workers: {plan.workers}\n"
        f"  tmp dir: {tmp_dir}\n"
        f"  reason: {plan.reason}"
    )
