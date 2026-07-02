"""Adaptive execution planning for WRF CLI workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import netCDF4

from micrometeorology.wrf import reader as wrf_reader
from micrometeorology.wrf.batch import JsonWorkerBackend, default_workers

if TYPE_CHECKING:
    from collections.abc import Sequence

ReaderRequest = Literal["auto", "eager", "lazy"]
JsonWorkerRequest = Literal["auto", "serial", "memmap"]
WorkflowKind = Literal["figures", "json", "pipeline"]

# Historical size thresholds. They no longer flip the auto reader to lazy
# (eager slicing is faster on measured real data), but are kept for importers.
LARGE_FILE_THRESHOLD_BYTES = 512 * 1024 * 1024
LARGE_TOTAL_INPUT_THRESHOLD_BYTES = 1024 * 1024 * 1024
LARGE_JSON_PAYLOAD_THRESHOLD_BYTES = 64 * 1024 * 1024
MANY_JSON_TASKS_THRESHOLD = 64

EAGER_4D_BUDGET_ENV_VAR = "LABMIM_EAGER_4D_BUDGET_GB"
DEFAULT_EAGER_4D_BUDGET_GB = 4.0
# ~11 simultaneously-live float32 arrays in the current eager 4D (poteolico) path.
EAGER_4D_LIVE_ARRAYS = 11


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


def _chunk_request_is_explicit(chunks_request: str | None) -> bool:
    if chunks_request is None:
        return False
    value = chunks_request.strip().lower()
    return value not in {"", "auto", "none"}


def estimate_4d_working_set_bytes(path: Path) -> int:
    """Estimate peak float32 bytes of the eager 4D (poteolico) path for one file.

    Opens the file read-only with netCDF4, reads dimension sizes only, and
    closes it immediately. Returns ``0`` when the file cannot be read or lacks
    the required dimensions so execution planning never crashes.
    """
    try:
        with netCDF4.Dataset(path, mode="r") as ds:
            sizes = {name: dim.size for name, dim in ds.dimensions.items()}
        return (
            EAGER_4D_LIVE_ARRAYS
            * sizes["Time"]
            * sizes["bottom_top_stag"]
            * sizes["south_north"]
            * sizes["west_east"]
            * 4
        )
    except Exception:
        return 0


def _eager_4d_budget_bytes() -> int:
    raw = os.environ.get(EAGER_4D_BUDGET_ENV_VAR)
    if raw is not None:
        try:
            return int(float(raw) * 1024**3)
        except ValueError:
            pass
    return int(DEFAULT_EAGER_4D_BUDGET_GB * 1024**3)


def _exceeds_eager_4d_budget(
    paths: list[Path],
    requested_variables: Sequence[str] | None,
    workflow: WorkflowKind,
) -> bool:
    """Return True when a requested 4D workload is too big for eager reads.

    Applies only to figure workflows: their poteolico branch still
    materializes the full 4D fields eagerly. The JSON workflow block-streams
    4D extraction (``variables.stream_wind_at_heights``), so its memory is
    bounded regardless of file length and it is never gated to lazy.
    """
    if workflow == "json":
        return False
    if not requested_variables:
        return False
    if not any(name.startswith("poteolico") for name in requested_variables):
        return False
    budget_bytes = _eager_4d_budget_bytes()
    return any(estimate_4d_working_set_bytes(path) > budget_bytes for path in paths)


def _resolve_reader(
    *,
    paths: list[Path],
    reader_request: ReaderRequest,
    chunks_request: str | None,
    parsed_chunks: wrf_reader.ChunkSpec,
    chunking_available: bool,
    requested_variables: Sequence[str] | None,
    workflow: WorkflowKind,
) -> tuple[wrf_reader.ReaderMode, wrf_reader.ChunkSpec, list[str]]:
    reasons: list[str] = []

    if reader_request in {"eager", "lazy"}:
        resolved_reader = cast("wrf_reader.ReaderMode", reader_request)
        reasons.append(f"reader explicitly set to {reader_request}")
    elif _chunk_request_is_explicit(chunks_request):
        resolved_reader = "lazy"
        reasons.append("explicit chunk dimensions require lazy reader")
    elif _exceeds_eager_4d_budget(paths, requested_variables, workflow):
        resolved_reader = "lazy"
        reasons.append(
            "4D working set exceeds eager budget; using lazy reader until streamed extraction lands"
        )
    else:
        resolved_reader = "eager"
        reasons.append("auto reader defaults to eager per-timestep slicing")

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
    workflow: WorkflowKind,
    worker_request: JsonWorkerRequest,
    workers: int,
    estimated_json_payload_bytes: int | None,
    json_task_count: int | None,
    large_json_payload_threshold_bytes: int,
    many_json_tasks_threshold: int,
) -> tuple[JsonWorkerBackend, list[str]]:
    if worker_request in {"serial", "memmap"}:
        return worker_request, [f"JSON worker backend explicitly set to {worker_request}"]

    if workers <= 1:
        return "serial", ["single worker uses serial JSON fast path"]

    payload = estimated_json_payload_bytes or 0
    tasks = json_task_count or 0
    if payload >= large_json_payload_threshold_bytes:
        return "memmap", ["large estimated JSON payload favors memmap worker references"]
    if workflow == "json":
        return "serial", ["small JSON payload favors serial writes over worker pools"]
    if tasks >= many_json_tasks_threshold and payload > 0:
        return "memmap", ["many JSON tasks favor memmap worker references"]
    return "memmap", ["multi-worker workload uses memmap worker references"]


def resolve_wrf_execution_plan(
    *,
    paths: list[Path],
    workflow: WorkflowKind,
    reader_request: ReaderRequest = "auto",
    chunks_request: str | None = "auto",
    json_worker_request: JsonWorkerRequest = "auto",
    workers: int | None = None,
    tmp_dir: str | Path | None = None,
    requested_variables: Sequence[str] | None = None,
    estimated_json_payload_bytes: int | None = None,
    json_task_count: int | None = None,
    large_json_payload_threshold_bytes: int = LARGE_JSON_PAYLOAD_THRESHOLD_BYTES,
    many_json_tasks_threshold: int = MANY_JSON_TASKS_THRESHOLD,
    chunking_available: bool | None = None,
) -> WRFExecutionPlan:
    """Resolve reader, chunking, and JSON worker choices for a WRF run.

    Explicit concrete requests always win. ``auto`` resolves the eager reader
    unless explicit chunk dimensions are given or a requested 4D field's
    estimated eager working set exceeds the ``LABMIM_EAGER_4D_BUDGET_GB``
    budget. For the ``json`` workflow ``auto`` resolves the serial backend
    below the large-payload threshold; figure workflows keep parallel
    memmap-backed workers for multi-worker workloads.
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
        requested_variables=requested_variables,
        workflow=workflow,
    )

    if json_worker_request == "serial" and resolved_tmp_dir is not None:
        raise ValueError("--tmp-dir is only valid with --worker-backend auto or memmap")

    json_backend, json_reasons = _resolve_json_worker_backend(
        workflow=workflow,
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
