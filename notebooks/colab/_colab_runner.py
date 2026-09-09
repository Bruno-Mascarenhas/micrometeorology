"""Shared driver for the Colab Pro+ notebooks.

Lives as a module rather than a copied cell because all three notebooks need the
same four things and a copy that drifts is how two runs stop being comparable:

- probe the assigned accelerator and derive the settings that depend on it
  (``bf16`` is available on every Colab GPU except the Turing T4);
- generate a per-run experiment YAML, since ``seed`` and ``num_workers`` have no
  CLI override;
- run train + evaluate and harvest a flat metrics row;
- archive every artifact a future run would need to be analysed against this one.

The archive is the point. A Colab VM is reclaimed the moment execution stops —
and every run in this project early-stops well before its epoch budget — so
anything not written to Drive inside the same cell as the training call is lost.
"""

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

#: Notebooks 01-03 train the single DHI head; the multitask heads cost 0.86 W/m2
#: in the factorial measured over FROZEN embeddings, which is the only such measure.
DHI_ONLY_TARGETS: dict[str, Any] = {"kindex": {"enabled": False}, "sky": {"enabled": False}}

#: Notebook 04: the sky condition is the primary target, with k* and the
#: clear-sky-normalised diffuse trained beside it. Weights 1/1/1 as in the
#: factorial; the composite val loss then leans toward the cross-entropy term,
#: which is the intended selection bias when the sky class comes first.
CEU_TARGETS: dict[str, Any] = {
    "dhi": {"enabled": True, "loss": "mae", "weight": 1.0, "parameterization": "clearsky_index"},
    "kindex": {"enabled": True, "kind": "kstar", "loss": "mae", "weight": 1.0},
    "sky": {"enabled": True, "weight": 1.0},
}


def probe_accelerator() -> dict[str, Any]:
    """Report the assigned GPU and the run settings that depend on it.

    Returns
    -------
    dict
        ``name``, ``vram_gib``, ``bf16`` (whether autocast should use bfloat16),
        ``cpus`` and ``amp_dtype``. On a T4 (Turing) ``bf16`` is False and the
        caller must stay on fp16; every other Colab GPU is Ampere or newer.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no CUDA device. Runtime > Change runtime type > GPU, and re-run the "
            "install cell: a CPU-only torch here means the whole session is wasted."
        )
    props = torch.cuda.get_device_properties(0)
    bf16 = bool(torch.cuda.is_bf16_supported())
    return {
        "name": torch.cuda.get_device_name(0),
        "vram_gib": round(props.total_memory / 1024**3, 1),
        "capability": f"{props.major}.{props.minor}",
        "bf16": bf16,
        "amp_dtype": "bf16" if bf16 else "fp16",
        "cpus": os.cpu_count() or 2,
        "torch": torch.__version__,
    }


def stage_bundle(bundle: str, data_dir: str, *, python: str | None = None) -> str:
    """Copy the Drive bundle to local disk, unpack it, validate it, return its root.

    Staging to the VM's own SSD is not an optimisation: training off
    ``/content/drive`` reads through FUSE, and the cold-read latency dominates
    the epoch. The validation step is what stops a truncated or half-synced
    bundle from training silently — pass *python* (the venv interpreter) to run
    ``allsky validate-dataset`` against it; omit it to skip that check.

    Returns
    -------
    str
        Path of the unpacked ``allsky_bundle`` root, ready to use as ``data_root``.
    """
    import tarfile

    data_root = Path(data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    local = data_root.parent / "bundle.tar.gz"
    started = time.time()
    shutil.copy(bundle, local)
    with tarfile.open(local) as tar:
        tar.extractall(data_root, filter="data")
    root = data_root / "allsky_bundle"
    print(f"staged em {time.time() - started:.0f}s -> {sorted(p.name for p in root.iterdir())}")

    if python is not None:
        # The console script sits beside the interpreter that installed it; the
        # package has no __main__, so `python -m allsky.cli` does not work.
        checked = subprocess.run(
            [
                str(Path(python).with_name("allsky")),
                "validate-dataset",
                "--manifest",
                str(root / "manifest.parquet"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(checked.stdout.strip() or checked.stderr.strip()[-500:])
        if checked.returncode != 0:
            raise RuntimeError("validate-dataset falhou: o bundle nao esta integro")
    return str(root)


def write_config(
    path: Path,
    *,
    extends: list[str],
    name: str,
    output_dir: str,
    seed: int,
    data_root: str,
    model: dict[str, Any],
    train: dict[str, Any],
    targets: dict[str, Any] | None = None,
    alignment: dict[str, Any] | None = None,
    augmentation: dict[str, Any] | None = None,
    note: str = "",
) -> Path:
    """Write one experiment YAML and return its path.

    ``seed`` and ``train.num_workers`` have no CLI override, so a per-run file is
    the only way to vary them; the rest is written alongside them so the file is
    a complete record of what produced the artifacts next to it. ``alignment``
    lands under ``data`` (the temporal-window arms set ``strategy`` and
    ``window_minutes`` there) and ``augmentation`` at the top level.
    """
    import yaml

    body: dict[str, Any] = {
        "extends": extends,
        "name": name,
        "output_dir": output_dir,
        "seed": seed,
        "data": {"data_root": data_root, "input_mode": "image"},
        "model": model,
        "train": train,
    }
    if alignment is not None:
        body["data"]["alignment"] = alignment
    if targets is not None:
        body["targets"] = targets
    if augmentation is not None:
        body["augmentation"] = augmentation
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {note}\n" if note else ""
    path.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def run_experiment(
    config: Path,
    *,
    python: str,
    split: str = "test",
    checkpoint: str = "best",
    archive_dir: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train (unless already trained) then evaluate one config; return a flat metrics row.

    *python* is the venv interpreter, as :func:`stage_bundle` takes: the
    ``allsky`` console script sits beside it, so the CLI is resolved by path and
    not by whatever ``PATH`` happens to hold when the cell runs.

    *checkpoint* names which weights to score — ``best`` (the early-stopping
    monitor's pick, reported under ``eval-<split>``) or ``last`` (the end of the
    schedule, under ``eval-<split>-last``). Under a multitask loss the two answer
    different questions: measured on the ``ceu`` arm, the sky cross-entropy on
    validation rises from the second epoch while the DHI error keeps falling, so
    the composite monitor freezes ``best`` early and only ``last`` carries the
    annealed regression heads. Training is skipped when ``last.ckpt`` already
    exists, so scoring a second checkpoint costs one evaluation, not a retrain.

    A failure is recorded and returned rather than raised: one bad arm must not
    end a 24-hour session that still has other arms to run.

    *archive_dir* is where :func:`archive` put earlier runs: a report already
    there, with nothing trained on this VM, is harvested as ``archived`` instead
    of retrained — what lets a session reclaimed at hour 20 be rerun without
    paying the first 20 hours again.

    *resume* hands an existing ``last.ckpt`` to ``allsky train --resume auto``
    instead of taking it as a finished run: the engine continues an arm cut
    short (a :func:`pull_live_run` restore) and trains nothing when the schedule
    or the early-stopping rule is already satisfied, so the call is idempotent.
    """
    import yaml

    cfg = yaml.safe_load(config.read_text())
    run_dir = Path(cfg["output_dir"]) / "run"
    row: dict[str, Any] = {
        "name": cfg["name"],
        "seed": cfg["seed"],
        "config": str(config),
        "checkpoint": checkpoint,
    }
    allsky_cli = str(Path(python).with_name("allsky"))
    report_name = f"eval-{split}" if checkpoint == "best" else f"eval-{split}-{checkpoint}"

    started = time.time()
    if archive_dir is not None and not (run_dir / "last.ckpt").exists():
        archived = Path(archive_dir) / cfg["name"] / report_name / "eval_metrics.json"
        if archived.exists():
            return _harvest(row, archived, status="archived", wall_seconds=0.0)
    trained = (run_dir / "last.ckpt").exists()
    if not trained or resume:
        command = [allsky_cli, "train", "-c", str(config)]
        if trained:
            command += ["--resume", "auto"]
        train = subprocess.run(command, capture_output=True, text=True, check=False)
        run_dir.mkdir(parents=True, exist_ok=True)
        # The engine says here, and nowhere else, which epoch a resume restarted
        # from, whether the cosine horizon was reconciled and whether early
        # stopping was already satisfied; dropping it on success leaves the
        # archive unable to answer why an arm trained the epochs it did.
        (run_dir / "train.log").write_text(train.stdout + train.stderr, encoding="utf-8")
        if train.returncode != 0:
            row["status"] = "train_failed"
            row["error"] = train.stderr[-2000:]
            return row

    report_dir = run_dir / report_name
    evaluate = subprocess.run(
        [
            allsky_cli,
            "evaluate",
            "-k",
            str(run_dir / f"{checkpoint}.ckpt"),
            "--split",
            split,
            "-c",
            str(config),
            "--report-dir",
            str(report_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if evaluate.returncode != 0:
        row["status"] = "eval_failed"
        row["error"] = evaluate.stderr[-2000:]
        return row

    return _harvest(
        row,
        report_dir / "eval_metrics.json",
        status="ok",
        wall_seconds=round(time.time() - started, 1),
    )


def _harvest(
    row: dict[str, Any], metrics_path: Path, *, status: str, wall_seconds: float
) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    dhi = metrics["global"]["dhi"]
    row.update(
        status=status,
        wall_seconds=wall_seconds,
        n_samples=metrics["n_samples"],
        **{k: dhi[k] for k in ("rmse", "mae", "mbe", "r2", "d", "nrmse") if k in dhi},
        skill_clearsky=dhi.get("skill_clearsky"),
        skill_persistence=dhi.get("skill_persistence"),
        split_id_ok=metrics["meta"].get("split_id_ok"),
        manifest_hash_ok=metrics["meta"].get("manifest_hash_ok"),
    )
    sky = metrics["global"].get("sky")
    if sky is not None:
        row.update(
            sky_accuracy=sky.get("accuracy"),
            sky_balanced_accuracy=sky.get("balanced_accuracy"),
            sky_macro_f1=sky.get("macro_f1"),
            sky_kappa_quadratic=sky.get("kappa_quadratic"),
            sky_within_one_class=sky.get("within_one_class"),
            sky_ece=sky.get("ece"),
            **{f"sky_f1_{name}": scores["f1"] for name, scores in sky.get("per_class", {}).items()},
        )
    sky_kt = metrics["global"].get("sky_kt")
    if sky_kt is not None:
        row.update(
            sky_kt_balanced_accuracy=sky_kt.get("balanced_accuracy"),
            sky_kt_macro_f1=sky_kt.get("macro_f1"),
            **{f"sky_kt_f1_{name}": v["f1"] for name, v in sky_kt.get("per_class", {}).items()},
        )
    return row


def archive(
    run_output_dir: str,
    drive_dir: str,
    *,
    config: Path | None = None,
    keep_checkpoint: bool = False,
) -> str:
    """Copy a run's analysable artifacts to Drive, and say what was copied.

    Takes the whole run directory — the training ``metrics.csv``/``metrics.json``
    history plus every ``eval-*`` report (``metrics.json``, ``stratified.csv``,
    ``predictions.parquet``, ``report.md``). ``best.ckpt`` is excluded by default:
    it is the largest file by far and only worth the upload for a run you intend
    to resume or serve.

    *config* is copied alongside because ``allsky train`` writes no config into
    the run directory — the per-run YAML lives on the VM, which is reclaimed.

    The stratified table and the per-sample predictions are what let a later run
    be compared against this one at all: a single RMSE cannot tell you whether a
    change fixed the high-sun bias or just moved the average.
    """
    source = Path(run_output_dir) / "run"
    if not source.exists():
        return f"{source}: nada a arquivar — o treino nao chegou a criar o diretorio"
    target = Path(drive_dir) / Path(run_output_dir).name
    ignore = None if keep_checkpoint else shutil.ignore_patterns("*.ckpt")
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
    if config is not None and Path(config).exists():
        shutil.copy2(config, target / Path(config).name)
    return f"{target}: {sum(1 for _ in target.rglob('*') if _.is_file())} arquivo(s)"


#: A queue job carries the ``write_config`` keywords a notebook cannot know in
#: advance — and nothing the VM decides for itself (paths, workers, AMP dtype).
JOB_KEYS = frozenset(
    {"name", "seed", "note", "model", "train", "alignment", "targets", "augmentation"}
)
JOB_REQUIRED_KEYS = frozenset({"name", "seed"})
JOB_SUFFIXES = (".yaml", ".yml")
QUEUE_STOP_FILE = "PARE"
QUEUE_STATE_DIR = "fila"
LIVE_DIR = "_live"
#: What the live mirror carries per run: the epoch history, and the checkpoints
#: a resumed session needs.
LIVE_FILES = ("metrics.csv", "metrics.json", "last.ckpt", "best.ckpt", "ema.ckpt")


def _same_file(source: Path, mirror: Path) -> bool:
    if not mirror.exists():
        return False
    ours, theirs = source.stat(), mirror.stat()
    return ours.st_size == theirs.st_size and ours.st_mtime_ns == theirs.st_mtime_ns


def _copy_atomically(source: Path, target: Path) -> None:
    """Copy *source* onto *target* so a reader never sees a partial file.

    The mirror is read by two things that do not coordinate with the copy: the
    ``gcloud storage rsync`` thread, and the next session's
    :func:`pull_live_run`. A plain ``shutil.copy2`` truncates the destination
    and fills it over seconds — for a 346 MB checkpoint that is a wide window in
    which the only surviving copy of a training run is a broken file. Writing
    beside the target and renaming makes the swap one atomic step, which is what
    the engine already does for the checkpoint this is copying.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.parcial")
    try:
        shutil.copy2(source, partial)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


def pull_live_run(live_dir: Path, out_dir: Path, name: str) -> str | None:
    """Restore the mirrored run *name* into ``out_dir/<name>/run`` so training resumes.

    Only a mirror that holds ``last.ckpt`` is restored: an epoch history alone is
    what a run that never checkpointed — or one mirrored before checkpoints were
    part of :data:`LIVE_FILES` — leaves behind, and copying it under a fresh
    training would splice a stale history onto a new run.

    Returns
    -------
    str or None
        The files restored, or ``None`` when there was nothing to resume from.
    """
    source = Path(live_dir) / name
    if not (source / "last.ckpt").is_file():
        return None
    run_dir = Path(out_dir) / name / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    for file_name in LIVE_FILES:
        if (source / file_name).is_file():
            _copy_atomically(source / file_name, run_dir / file_name)
            restored.append(file_name)
    return f"{name}: retomada de {source} ({', '.join(restored)})"


def apply_queue_override(override_dir: Path, config: Path) -> str:
    """Let a file dropped in the bucket skip or replace one queue entry.

    ``<name>.skip`` beside the notebook's override prefix skips the arm;
    ``<name>.yaml`` replaces the repository config in place before it runs, which
    is how an adjustment decided after launch — a shorter cosine, another batch
    size — reaches a running session without a relaunch. The override is a
    **complete** config, not a patch: it is copied over the repository file, so
    a partial one would train an arm with no ``name`` or ``seed``. It keeps the
    repository config's ``extends`` paths valid because it lands in the same
    place.

    Returns
    -------
    str
        ``"skip"``, ``"override"`` or ``"repo"``.

    Raises
    ------
    ValueError
        When the replacement is not a complete config for this arm. Everything
        that tracks a run — the archive prefix, the mirrored checkpoints, the
        row in the table — is keyed by the file name, so an override that
        renames the arm or omits a required key is refused instead of silently
        splitting one arm across two identities.
    """
    import yaml

    overrides = Path(override_dir)
    if (overrides / f"{config.stem}.skip").exists():
        return "skip"
    replacement = overrides / f"{config.stem}.yaml"
    if not replacement.is_file():
        return "repo"
    proposed = yaml.safe_load(replacement.read_text()) or {}
    missing = [key for key in ("name", "seed", "output_dir", "extends") if key not in proposed]
    if missing:
        raise ValueError(f"{replacement}: override incompleto, faltam {missing}")
    if proposed["name"] != config.stem or Path(proposed["output_dir"]).name != config.stem:
        raise ValueError(
            f"{replacement}: name={proposed['name']!r} e output_dir={proposed['output_dir']!r} "
            f"nao batem com {config.stem} — o arquivo e o espelho sao indexados pelo nome do braco"
        )
    shutil.copy2(replacement, config)
    return "override"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, tz=UTC).isoformat(timespec="seconds")


def load_job(path: Path) -> dict[str, Any]:
    """Read one queue job and refuse anything that is not a ``write_config`` keyword.

    Raises
    ------
    ValueError
        When the file is not a mapping, names a key the notebook would silently
        ignore, or lacks ``name``/``seed``.
    """
    import yaml

    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"{path.name}: a job must be a YAML mapping")  # noqa: TRY004
    unknown = sorted(set(body) - JOB_KEYS)
    missing = sorted(JOB_REQUIRED_KEYS - set(body))
    if unknown or missing:
        raise ValueError(f"{path.name}: unknown keys {unknown}, missing keys {missing}")
    return body


def _job_status(artifacts: Path, job_file: Path) -> str | None:
    status = artifacts / QUEUE_STATE_DIR / f"{job_file.stem}.status.json"
    if not status.exists():
        return None
    return str(json.loads(status.read_text(encoding="utf-8")).get("status"))


def pending_jobs(queue_dir: Path, artifacts_dir: Path) -> list[Path]:
    """Job files not yet settled, in file-name order.

    A job is settled once its status file under ``<artifacts>/fila/`` says
    anything but ``running``: ``running`` is what a reclaimed VM leaves behind,
    so it is the one state that gets picked up again.
    """
    artifacts = Path(artifacts_dir)
    files = sorted(p for p in Path(queue_dir).iterdir() if p.suffix in JOB_SUFFIXES)
    return [p for p in files if _job_status(artifacts, p) in (None, "running")]


def run_queue(
    queue_dir: Path,
    artifacts_dir: Path,
    run_job: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    poll_seconds: float = 180.0,
    idle_limit_seconds: float = 1800.0,
    deadline_seconds: float = 20.0 * 3600.0,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run every job dropped in *queue_dir*, in file-name order, until told to stop.

    The queue is a Drive folder written from outside the VM, so the loop owns no
    state of its own: what it knows is on Drive, under ``<artifacts>/fila/`` — one
    ``<job>.status.json`` per job (``running``, then the row's status or
    ``failed`` with the traceback) and a ``heartbeat.json`` rewritten on every
    pass. Three things end it: a file named ``PARE`` in the queue, *deadline_seconds*
    since the call, or *idle_limit_seconds* without a job to run — an idle GPU
    keeps billing, so waiting is bounded.

    A job that raises is recorded and skipped, never retried: the traceback on
    Drive is the signal to fix the job file and drop it again under a new name.

    Returns
    -------
    list of dict
        One harvested row per job attempted, in the order they ran.
    """
    import traceback

    queue = Path(queue_dir)
    artifacts = Path(artifacts_dir)
    state = artifacts / QUEUE_STATE_DIR
    state.mkdir(parents=True, exist_ok=True)
    started = clock()
    last_work = started
    rows: list[dict[str, Any]] = []
    while True:
        now = clock()
        pending = pending_jobs(queue, artifacts)
        if (queue / QUEUE_STOP_FILE).exists():
            reason: str | None = "stop_file"
        elif now - started >= deadline_seconds:
            reason = "deadline"
        elif now - last_work >= idle_limit_seconds:
            reason = "idle"
        else:
            reason = None
        _write_json(
            state / "heartbeat.json",
            {
                "time": _iso(now),
                "pending": [p.name for p in pending],
                "ran": len(rows),
                "stopped": reason,
            },
        )
        if reason is not None:
            return rows
        if not pending:
            sleep(poll_seconds)
            continue
        job_file = pending[0]
        status = state / f"{job_file.stem}.status.json"
        try:
            job = load_job(job_file)
            _write_json(status, {"status": "running", "job": job_file.name, "started": _iso(now)})
            row = run_job(job)
        except Exception as exc:  # noqa: BLE001 - one bad job must not end a 24-hour session; the traceback is archived instead
            row = {"name": job_file.stem, "status": "failed", "error": str(exc)}
            _write_json(
                status,
                {"status": "failed", "job": job_file.name, "error": traceback.format_exc()},
            )
        else:
            _write_json(
                status,
                {"status": row.get("status", "ok"), "job": job_file.name, "row": row},
            )
        rows.append(row)
        last_work = clock()


def _nvidia_smi() -> str | None:
    """``utilization.gpu, memory.used`` as nvidia-smi prints them; None when it does not answer.

    Everything is swallowed on purpose. This runs inside the live-mirror thread,
    the only thing copying checkpoints off the VM, and a driver under load makes
    ``nvidia-smi`` sit until the timeout: a heartbeat that cannot be read is a
    missing field, never a reason to stop mirroring.
    """
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 — um probe de heartbeat nao pode derrubar quem o chama
        return None
    return probe.stdout.strip() or None


def sync_live(
    out_dir: Path,
    target_dir: Path,
    *,
    gpu_probe: Callable[[], str | None] = _nvidia_smi,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Mirror every run's epoch history and checkpoints to *target_dir*, with a heartbeat.

    ``run_experiment`` keeps the training output in memory, so during the hours a
    run takes the only evidence it is computing is ``<run>/metrics.csv`` growing
    by one row per epoch. Copying it to Drive beside the GPU utilisation is what
    lets someone outside the VM tell a training from a hung process.

    ``last.ckpt`` and ``best.ckpt`` travel with it (:data:`LIVE_FILES`): they are
    what :func:`pull_live_run` restores into a fresh VM so ``allsky train
    --resume auto`` continues an arm the session limit or a crash cut short. The
    engine writes every checkpoint through a temp file and ``os.replace``, so a
    copy never sees a half-written file; a file is copied again only when its
    size or mtime differs from the mirror's.

    Returns
    -------
    dict
        The heartbeat written: ``time``, ``gpu`` and the run names whose history
        changed since the previous call.
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    updated: list[str] = []
    for run_dir in sorted(Path(out_dir).glob("*/run")):
        name = run_dir.parent.name
        changed = False
        for file_name in LIVE_FILES:
            source = run_dir / file_name
            mirror = target / name / file_name
            if not source.is_file() or _same_file(source, mirror):
                continue
            _copy_atomically(source, mirror)
            changed = True
        if changed:
            updated.append(name)
    beat = {"time": _iso(clock()), "gpu": gpu_probe(), "updated": updated}
    _write_json(target / "heartbeat.json", beat)
    return beat


def start_live_sync(out_dir: Path, target_dir: Path, *, period_seconds: float = 300.0) -> Any:
    """Run :func:`sync_live` every *period_seconds* on a daemon thread; return it.

    Nothing this thread can raise is allowed to end it: it holds the only copy
    of the checkpoints outside the VM, and a mirror that dies silently is worse
    than a late one. A Drive FUSE hiccup raises ``OSError`` on the copy and a
    driver under load makes ``nvidia-smi`` hit its timeout — the pass is dropped
    and the next one runs.
    """
    import threading

    def loop() -> None:
        while True:
            try:
                sync_live(out_dir, target_dir)
            except Exception as exc:  # noqa: BLE001 — this thread is the only copy of the checkpoints
                print(f"espelho ao vivo: {exc!r}")
            time.sleep(period_seconds)

    thread = threading.Thread(target=loop, name="live-sync", daemon=True)
    thread.start()
    return thread


def _run_quiet(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def mirror_once(
    pairs: Sequence[tuple[str, str]],
    *,
    delete_unmatched: bool = False,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_quiet,
) -> list[str]:
    """``gcloud storage rsync -r`` each ``(source, destination)`` pair; return the ones that failed.

    Colab Enterprise mounts no Drive: what leaves the VM leaves through a bucket,
    and this is what carries the archive, the live mirror and the queue state
    there — and brings the queue's job files back. The sync is additive on both
    sides, so a job file removed from the bucket stays on the VM until the
    session ends; *delete_unmatched* makes the destination match the source
    exactly, which is what lets an operator revoke a ``.skip`` marker mid-session
    and is safe only where the destination holds nothing the VM produced.

    Returns
    -------
    list of str
        ``"source -> destination"`` for every pair whose rsync exited non-zero.
    """
    failed: list[str] = []
    command = ["gcloud", "storage", "rsync", "-r"]
    if delete_unmatched:
        command.append("--delete-unmatched-destination-objects")
    for source, destination in pairs:
        result = run([*command, source, destination])
        if result.returncode != 0:
            failed.append(f"{source} -> {destination}")
    return failed


def start_mirror(pairs: Sequence[tuple[str, str]], *, period_seconds: float = 300.0) -> Any:
    """Run :func:`mirror_once` every *period_seconds* on a daemon thread; return it."""
    import threading

    def loop() -> None:
        while True:
            try:
                for failure in mirror_once(pairs):
                    print(f"espelho: {failure}")
            except Exception as exc:  # noqa: BLE001 — this thread is what carries the run off the VM
                print(f"espelho: {exc!r}")
            time.sleep(period_seconds)

    thread = threading.Thread(target=loop, name="mirror", daemon=True)
    thread.start()
    return thread


#: The three evaluations one arm produces, in the order that makes
#: ``eval-test-last`` on disk mean "this arm is finished": ``(split,
#: checkpoint, column suffix)``.
ARM_EVALUATIONS = (("test", "best", ""), ("val", "best", "_val"), ("test", "last", "_last"))
#: Report directory of each suffix in :data:`ARM_EVALUATIONS`.
ARM_REPORTS = {"": "eval-test", "_val": "eval-val", "_last": "eval-test-last"}
#: What a secondary evaluation contributes to the arm's row; the first one
#: contributes everything :func:`_harvest` produces.
ARM_KEYS = (
    "rmse",
    "mae",
    "mbe",
    "sky_kt_balanced_accuracy",
    "sky_kt_macro_f1",
    "sky_kt_f1_partly_cloudy_clear",
    "sky_balanced_accuracy",
    "sky_macro_f1",
)


def _score_reports(
    row: dict[str, Any],
    name: str,
    *,
    python: str,
    out_dir: Path,
    artifacts: Path,
    n_bootstrap: int,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> None:
    """Add the per-block columns of every report of *name* to *row*."""
    for tag, report in ARM_REPORTS.items():
        parquet = Path(out_dir) / name / "run" / report / "predictions.parquet"
        if not parquet.exists():
            parquet = Path(artifacts) / name / report / "predictions.parquet"
        if not parquet.exists():
            continue
        block = score_by_sensor_block_in(python, parquet, n_bootstrap=n_bootstrap, run=run)
        row[f"block_macro_f1{tag}"] = block["sky"]["macro_f1"]
        row[f"block_persistence_f1{tag}"] = block["sky_persistence_previous_block"]["macro_f1"]
        row[f"block_rmse{tag}"] = block["dhi"]["rmse"]
        by_kt = score_by_sensor_block_in(
            python,
            parquet,
            sky=("obs_sky", "pred_sky_kt"),
            n_bootstrap=n_bootstrap,
            run=run,
        )
        row[f"block_kt_macro_f1{tag}"] = by_kt["sky"]["macro_f1"]


def run_arm(
    config: Path,
    *,
    python: str,
    out_dir: Path,
    artifacts: Path,
    mirror: Sequence[tuple[str, str]],
    overrides: Path | None = None,
    override_mirror: Sequence[tuple[str, str]] = (),
    watchers: Sequence[Any] = (),
    log: Callable[[str], None] = print,
    n_bootstrap: int = 200,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_quiet,
) -> dict[str, Any]:
    """Take one config from override to archived result, and never raise.

    The order is the whole point. A session runs on a VM that is reclaimed
    without warning and, on Colab Enterprise, dies with the notebook: anything
    still only on its disk when something raises is gone, and on a training
    queue that is hours of GPU. So each evaluation is archived and mirrored the
    moment it exists — before the next one starts and before any scoring — and
    every step after training is caught: a failure fills a column with nothing,
    it does not end the arm, and an arm does not end the queue.

    Training is resumed, not repeated: a ``last.ckpt`` mirrored by
    :func:`sync_live` and restored by :func:`pull_live_run` is handed to
    ``allsky train --resume auto``, so a relaunch after the session limit
    continues where the previous one stopped — unless the queue overrode the
    config, in which case the mirrored weights belong to another recipe and the
    arm starts over. An arm counts as finished only when all three reports are
    in the archive, so a session cut off between two of them re-evaluates from
    the archived checkpoint instead of leaving a column empty forever.

    Parameters
    ----------
    config:
        The experiment YAML, inside the checked-out repository.
    python:
        The venv interpreter; every step that needs this project runs there.
    out_dir:
        Where the runs are written, the parent of ``<name>/run``.
    artifacts:
        The archive on the VM, mirrored by *mirror*.
    mirror:
        ``(source, destination)`` pairs for :func:`mirror_once`.
    overrides:
        Directory holding ``<name>.skip`` / ``<name>.yaml``, if any.
    override_mirror:
        Pulled into *overrides* before the arm reads it, deleting whatever the
        source no longer has — which is what lets an operator revoke a marker
        while the queue is running.
    watchers:
        The mirroring threads, checked for life at the start of each arm. They
        are the only thing copying a training in flight off the VM, so one that
        died has to be visible in the log rather than discovered by an empty
        bucket hours later.
    log:
        Where the step-by-step account goes; the notebook sends it to a file
        that the mirror carries, so a failed session still explains itself.

    Returns
    -------
    dict
        One row: ``name``, ``status`` (``ok``, ``archived``, ``skipped``,
        ``train_failed``, ``eval_failed`` or ``failed``), the harvested metrics,
        the per-block columns, and ``error``/``score_error`` when something
        went wrong.
    """
    name = Path(config).stem
    row: dict[str, Any] = {"name": name, "config": str(config), "status": "pending"}
    try:
        dead = [w for w in watchers if not w.is_alive()]
        if dead:
            log(
                f"{name}: ATENCAO, espelho parado ({len(dead)} de {len(watchers)} threads morreram)"
            )
        if override_mirror:
            mirror_once(override_mirror, delete_unmatched=True, run=run)
        decision = "repo" if overrides is None else apply_queue_override(overrides, Path(config))
        row["override"] = decision
        if decision == "skip":
            row["status"] = "skipped"
            log(f"{name}: pulado por {overrides}/{name}.skip")
            return row
        if decision == "override":
            log(f"{name}: config substituido por {overrides}/{name}.yaml")

        resumed = all(
            (Path(artifacts) / name / report / "eval_metrics.json").exists()
            for report in ARM_REPORTS.values()
        )
        row["status"] = "archived" if resumed else "pending"
        if resumed:
            log(f"{name}: os tres relatorios ja estao em {artifacts} — nada a treinar")
        elif decision == "override":
            log(f"{name}: config novo, o checkpoint espelhado nao serve — treino do zero")
        else:
            log(
                pull_live_run(Path(artifacts) / LIVE_DIR, out_dir, name)
                or f"{name}: treino do zero"
            )

        def guardar() -> None:
            """Put whatever the VM holds into the archive and the bucket."""
            log(
                "  "
                + archive(
                    str(Path(out_dir) / name),
                    str(artifacts),
                    config=Path(config),
                    keep_checkpoint=True,
                )
            )
            log("  espelho: " + (", ".join(mirror_once(mirror, run=run)) or "ok"))

        for index, (split, checkpoint, tag) in enumerate(ARM_EVALUATIONS):
            report = ARM_REPORTS[tag]
            try:
                result = run_experiment(
                    Path(config),
                    python=python,
                    split=split,
                    checkpoint=checkpoint,
                    archive_dir=str(artifacts) if resumed else None,
                    resume=index == 0 and not resumed,
                )
            finally:
                if not resumed:
                    guardar()
            if result.get("status") not in ("ok", "archived"):
                row["status"] = result.get("status")
                row["error"] = str(result.get("error"))[-800:]
                log(f"  {report}: {row['status']}\n{row['error']}")
                break
            log(f"  {report}: {result['status']} em {result.get('wall_seconds', 0)} s")
            if tag == "":
                row.update({k: v for k, v in result.items() if k not in ("config", "checkpoint")})
            else:
                row.update({f"{key}{tag}": result.get(key) for key in ARM_KEYS})

        try:
            _score_reports(
                row,
                name,
                python=python,
                out_dir=Path(out_dir),
                artifacts=Path(artifacts),
                n_bootstrap=n_bootstrap,
                run=run,
            )
        except (RuntimeError, OSError, ValueError, KeyError) as exc:
            row["score_error"] = repr(exc)[-800:]
            log(f"  pontuacao por bloco falhou, colunas vazias: {row['score_error']}")
    except Exception as exc:  # noqa: BLE001 — a queue must survive anything one arm does
        row["status"] = "failed"
        row["error"] = repr(exc)[-800:]
        log(f"{name}: FALHOU fora do treino: {row['error']}")
    try:
        log("  espelho final: " + (", ".join(mirror_once(mirror, run=run)) or "ok"))
    except Exception as exc:  # noqa: BLE001 — the last line of an arm cannot end the queue
        row.setdefault("error", repr(exc)[-800:])
        log(f"{name}: espelho final falhou: {exc!r}")
    return row


def summarise_arm(row: Mapping[str, Any]) -> str:
    """One readable line per arm: diffuse, sky by reconstructed Kt, and both per-block estimators."""

    def number(value: Any, digits: int = 2, sign: bool = False) -> str:
        if not isinstance(value, (int, float)):
            return "—"
        return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"

    return (
        f"{row['name']:<18} {row.get('status')} {row.get('override', '')}  "
        f"DHI rmse/mae/mbe {number(row.get('rmse'))}/{number(row.get('mae'))}/"
        f"{number(row.get('mbe'), sign=True)} "
        f"(last {number(row.get('rmse_last'))}/{number(row.get('mae_last'))}/"
        f"{number(row.get('mbe_last'), sign=True)})  "
        f"sky_kt bal {number(row.get('sky_kt_balanced_accuracy'), 3)} "
        f"F1 {number(row.get('sky_kt_macro_f1'), 3)} "
        f"parc-clara {number(row.get('sky_kt_f1_partly_cloudy_clear'), 3)} "
        f"(last bal {number(row.get('sky_kt_balanced_accuracy_last'), 3)})  "
        f"por bloco F1 ceu {number(row.get('block_macro_f1'), 3)} "
        f"(last {number(row.get('block_macro_f1_last'), 3)}, val {number(row.get('block_macro_f1_val'), 3)}) "
        f"k* {number(row.get('block_kt_macro_f1'), 3)} "
        f"(last {number(row.get('block_kt_macro_f1_last'), 3)}, "
        f"val {number(row.get('block_kt_macro_f1_val'), 3)}); "
        f"persistencia {number(row.get('block_persistence_f1'), 3)}; "
        f"RMSE {number(row.get('block_rmse'))}"
    )


def summarise(rows: list[dict[str, Any]]) -> Any:
    """Tidy DataFrame of the harvested rows, best RMSE first."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "rmse" in frame.columns:
        frame = frame.sort_values("rmse", na_position="last")
    return frame.reset_index(drop=True)


def _vote_with_ordinal_tiebreak(votes: Any, n_classes: int) -> Any:
    """Majority class per row; a tie goes to the tied class nearest the mean index."""
    import numpy as np

    counts = np.stack([(votes == c).sum(axis=0) for c in range(n_classes)], axis=1)
    top = counts.max(axis=1, keepdims=True)
    tied = counts == top
    mean_index = votes.mean(axis=0)[:, None]
    distance = np.where(tied, np.abs(np.arange(n_classes)[None, :] - mean_index), np.inf)
    return distance.argmin(axis=1)


def ensemble_predictions(
    members: Sequence[str | Path],
    out_dir: str | Path,
    *,
    reference: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Average the members' per-sample predictions and score the ensemble.

    Parameters
    ----------
    members:
        ``eval-<split>/predictions.parquet`` of each seed, all over the SAME
        rows: the frames are joined on ``sample_id`` and a member covering a
        different set of samples is refused, because a mean over rows that only
        some members predicted is not an ensemble of anything.
    out_dir:
        Where ``metrics.json`` and ``predictions.parquet`` are written.
    reference:
        Optional predictions of a control arm (one parquet per seed), averaged
        the same way over the members' rows so the two ensembles are paired
        sample by sample.

    Returns
    -------
    dict
        ``n_members``; ``dhi`` (regression metrics of the mean prediction, W m-2);
        ``kindex`` when every member carries ``pred_kindex``; ``sky`` when every
        member carries ``pred_sky``, with two estimators — ``vote`` (majority of
        the members' classes, ties resolved to the tied class nearest the mean
        class index, since the classes are ordered) and ``kt_bin`` (the mean k*
        turned into Kt through the row's clear-sky Kt and binned on
        ``SKY_CLASS_KT_UPPER_BOUNDS``) — each with the classification metrics
        plus ``ordinal_mae``, the mean class-index distance; ``reference`` when
        given, with the control ensemble's ``dhi`` metrics and the paired
        ``rmse_delta`` (members minus control).

    Raises
    ------
    ValueError
        If fewer than two members are given, if the members do not cover one
        identical set of samples, or if a reference does not cover every member row.
    """
    import numpy as np
    import pandas as pd

    from allsky.clearsky import clearsky_ghi_and_kt
    from allsky.evaluation.metrics import classification_metrics, regression_metrics
    from labmim_core.atomic import atomic_write, atomic_write_strict_json
    from labmim_core.site import STATION_UTC_OFFSET_HOURS
    from labmim_core.sky import SKY_CLASS_COUNT, SKY_CLASS_KT_UPPER_BOUNDS, SKY_CLASS_NAMES

    if len(members) < 2:
        raise ValueError(f"an ensemble needs at least two members, got {len(members)}")
    frames = [pd.read_parquet(path).set_index("sample_id").sort_index() for path in members]
    index = frames[0].index
    for path, frame in zip(members, frames, strict=True):
        if not frame.index.equals(index):
            raise ValueError(f"{path} covers a different sample set than {members[0]}")

    def mean_of(column: str, source: list[Any]) -> Any:
        return np.mean([f[column].to_numpy(dtype=np.float64) for f in source], axis=0)

    first = frames[0]
    ensemble = pd.DataFrame({"obs_dhi": first["obs_dhi"].to_numpy(dtype=np.float64)}, index=index)
    ensemble["ens_dhi"] = mean_of("pred_dhi", frames)
    report: dict[str, Any] = {
        "n_members": len(frames),
        "members": [str(path) for path in members],
        "dhi": regression_metrics(ensemble["obs_dhi"], ensemble["ens_dhi"]),
    }

    if all("pred_kindex" in f.columns for f in frames):
        ensemble["obs_kindex"] = first["obs_kindex"].to_numpy(dtype=np.float64)
        ensemble["ens_kindex"] = mean_of("pred_kindex", frames)
        report["kindex"] = regression_metrics(ensemble["obs_kindex"], ensemble["ens_kindex"])

    if all("pred_sky" in f.columns for f in frames):
        observed = first["obs_sky"].to_numpy(dtype=np.int64)
        votes = np.stack([f["pred_sky"].to_numpy(dtype=np.int64) for f in frames])
        ensemble["obs_sky"] = observed
        ensemble["ens_sky_vote"] = _vote_with_ordinal_tiebreak(votes, SKY_CLASS_COUNT)
        estimators = {"vote": ensemble["ens_sky_vote"].to_numpy()}
        probability_columns = [f"prob_sky_{name}" for name in SKY_CLASS_NAMES]
        mean_probabilities = None
        if all(all(c in f.columns for c in probability_columns) for f in frames):
            mean_probabilities = np.mean(
                [f[probability_columns].to_numpy(dtype=np.float64) for f in frames], axis=0
            )
            for position, column in enumerate(probability_columns):
                ensemble[column] = mean_probabilities[:, position]
            ensemble["ens_sky_prob"] = mean_probabilities.argmax(axis=1)
            estimators["prob_mean"] = ensemble["ens_sky_prob"].to_numpy()
        if "ens_kindex" in ensemble.columns:
            times = pd.to_datetime(first["timestamp_utc"], utc=True)
            _, kt_clear = clearsky_ghi_and_kt(
                first["solar_zenith"].to_numpy(dtype=np.float64), times, STATION_UTC_OFFSET_HOURS
            )
            kt = ensemble["ens_kindex"].to_numpy() * np.asarray(kt_clear, dtype=np.float64)
            ensemble["ens_kt"] = kt
            ensemble["ens_sky_kt_bin"] = np.digitize(kt, SKY_CLASS_KT_UPPER_BOUNDS, right=True)
            estimators["kt_bin"] = ensemble["ens_sky_kt_bin"].to_numpy()
        report["sky"] = {
            name: classification_metrics(
                observed,
                predicted,
                SKY_CLASS_COUNT,
                probabilities=mean_probabilities if name == "prob_mean" else None,
            )
            for name, predicted in estimators.items()
        }

    if reference:
        controls = [pd.read_parquet(path).set_index("sample_id") for path in reference]
        aligned = [c.reindex(index) for c in controls]
        if any(a["pred_dhi"].isna().any() for a in aligned):
            raise ValueError("a reference member does not cover every row of the ensemble")
        ensemble["ref_dhi"] = mean_of("pred_dhi", aligned)
        control = regression_metrics(ensemble["obs_dhi"], ensemble["ref_dhi"])
        report["reference"] = {
            "members": [str(path) for path in reference],
            "dhi": control,
            "rmse_delta": float(report["dhi"]["rmse"] - control["rmse"]),
        }

    ensemble["day_id"] = first["day_id"].to_numpy()
    ensemble["timestamp_utc"] = first["timestamp_utc"].to_numpy()
    report["by_sensor_block"] = score_by_sensor_block(
        ensemble.reset_index(),
        dhi=("obs_dhi", "ens_dhi"),
        sky=("obs_sky", "ens_sky_vote") if "ens_sky_vote" in ensemble.columns else None,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "predictions.parquet", lambda tmp: ensemble.reset_index().to_parquet(tmp))
    atomic_write_strict_json(out / "metrics.json", report)
    return report


def sensor_block_key(frame: Any, block_minutes: float = 5.0) -> Any:
    """The datalogger row each frame was paired with, as ``day_id@HH:MM`` of the block end.

    The CR5000 end-stamps a ``block_minutes`` average, so every frame whose local
    stamp falls in ``(t - block, t]`` shares the row stamped ``t`` — the ceiling of
    the local time to the block. Measured on ``dataset-iso``: the key reproduces
    the label support exactly (``target_dhi`` constant in all 9,538 blocks).
    """
    import pandas as pd

    from labmim_core.site import STATION_UTC_OFFSET_HOURS

    local = pd.to_datetime(frame["timestamp_utc"], utc=True) + pd.Timedelta(
        hours=STATION_UTC_OFFSET_HOURS
    )
    block_end = local.dt.tz_localize(None).dt.ceil(f"{block_minutes:g}min")
    return frame["day_id"].astype(str) + "@" + block_end.dt.strftime("%H:%M")


def _ordinal_mode(values: Any) -> int:
    """Most frequent class; a tie goes to the tied class nearest the mean index."""
    import numpy as np

    labelled = np.asarray(values, dtype=np.int64)
    labelled = labelled[labelled >= 0]
    if labelled.size == 0:
        return -1
    counts = np.bincount(labelled)
    tied = np.flatnonzero(counts == counts.max())
    return int(tied[np.abs(tied - float(np.mean(labelled))).argmin()])


#: What the notebook is allowed to call from the Colab kernel. The kernel has
#: pandas and pyyaml and nothing of this project: ``allsky`` and
#: ``labmim_core`` live in the venv :func:`stage_bundle` builds. A function
#: listed here that imports either — at module level or inside its body — kills
#: a session at the first call, which on a training queue means hours of GPU
#: already spent. ``tests/allsky/test_colab_runner.py`` calls every name here
#: with both packages unimportable.
KERNEL_SAFE = (
    "apply_queue_override",
    "archive",
    "load_job",
    "mirror_once",
    "preflight",
    "run_arm",
    "pull_live_run",
    "run_experiment",
    "score_by_sensor_block_in",
    "start_live_sync",
    "start_mirror",
    "stage_bundle",
    "summarise",
    "summarise_arm",
    "sync_live",
    "write_config",
)

#: The columns :func:`score_by_sensor_block` reads, and what :func:`preflight`
#: puts in the synthetic frame it scores.
_PREFLIGHT_COLUMNS = (
    "day_id",
    "timestamp_utc",
    "obs_dhi",
    "pred_dhi",
    "obs_sky",
    "pred_sky",
    "pred_sky_kt",
)


def preflight(
    python: str,
    *,
    artifacts: str | Path,
    mirror: Sequence[tuple[str, str]],
    work_dir: str | Path,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_quiet,
) -> list[str]:
    """Exercise every kernel-side step on synthetic data, before the queue starts.

    A training queue spends its hours inside ``allsky train``; the code around
    it — scoring, archiving, mirroring — runs for seconds at the end of each
    arm, and is where a session dies with the GPU time already paid for. This
    walks that code on a three-row frame and a fake run directory, so a broken
    interpreter path, a missing package in the kernel, an unwritable archive or
    a mirror pointing nowhere fails in the first minute instead of the
    fourteenth hour.

    The last check writes ``preflight.json`` into *artifacts* and mirrors it,
    which is the only way to prove the real destination accepts writes — a
    temporary directory proves nothing about a bucket.

    Parameters
    ----------
    python:
        The venv interpreter, as :func:`stage_bundle` returns it.
    artifacts:
        The run archive on the VM, the source side of *mirror*.
    mirror:
        ``(source, destination)`` pairs for :func:`mirror_once`.
    work_dir:
        Scratch directory for the synthetic run; nothing is left behind.

    Returns
    -------
    list of str
        One line per check passed, in order.

    Raises
    ------
    RuntimeError
        On the first check that fails, naming the step and the error.
    """
    import pandas as pd

    checks: list[str] = []
    work = Path(work_dir) / "_preflight"
    if work.exists():
        shutil.rmtree(work)
    (work / "run").mkdir(parents=True)

    allsky_cli = str(Path(python).with_name("allsky"))
    version = run([allsky_cli, "--help"])
    if version.returncode != 0:
        raise RuntimeError(f"preflight: {allsky_cli} nao roda:\n{version.stderr[-1500:]}")
    checks.append(f"CLI do venv responde: {allsky_cli}")

    frame = pd.DataFrame(
        {
            "day_id": ["2026-08-20"] * 3,
            "timestamp_utc": [
                "2026-08-20T12:31:00+00:00",
                "2026-08-20T12:33:00+00:00",
                "2026-08-20T12:37:00+00:00",
            ],
            "obs_dhi": [100.0, 100.0, 200.0],
            "pred_dhi": [90.0, 110.0, 190.0],
            "obs_sky": [1, 1, 3],
            "pred_sky": [1, 2, 3],
            "pred_sky_kt": [1, 1, 3],
        }
    )
    parquet = work / "predictions.parquet"
    frame.to_parquet(parquet)
    for sky in (("obs_sky", "pred_sky"), ("obs_sky", "pred_sky_kt")):
        scored = score_by_sensor_block_in(python, parquet, sky=sky, n_bootstrap=10, run=run)
        if scored.get("n_blocks") != 2:
            raise RuntimeError(f"preflight: pontuacao por bloco devolveu {scored}")
    checks.append("pontuacao por bloco roda no venv e conta os blocos certos")

    (work / "run" / "metrics.csv").write_text("epoch,val_loss\n1,0.9\n", encoding="utf-8")
    (work / "run" / "last.ckpt").write_bytes(b"preflight")
    archived = archive(str(work), str(Path(work).parent / "arquivo"), keep_checkpoint=True)
    if "nada a arquivar" in archived:
        raise RuntimeError(f"preflight: {archived}")
    checks.append(f"arquivo com checkpoint: {archived}")

    live = work / "live"
    beat = sync_live(work.parent, live)
    restored = pull_live_run(live, work.parent / "retomada", "_preflight")
    if "_preflight" not in beat["updated"] or restored is None:
        raise RuntimeError(f"preflight: espelho ao vivo nao fechou o ciclo: {beat}, {restored}")
    checks.append(f"espelho ao vivo e retomada: {restored}")

    _write_json(
        Path(artifacts) / "preflight.json",
        {"time": _iso(time.time()), "checks": checks, "python": python},
    )
    failed = mirror_once(mirror, run=run)
    if failed:
        raise RuntimeError(f"preflight: espelho para o destino final falhou: {failed}")
    checks.append(f"espelho para o destino final: {[destination for _, destination in mirror]}")

    shutil.rmtree(work, ignore_errors=True)
    shutil.rmtree(work.parent / "arquivo", ignore_errors=True)
    shutil.rmtree(work.parent / "retomada", ignore_errors=True)
    return checks


def score_by_sensor_block_in(
    python: str,
    parquet: Path,
    *,
    sky: tuple[str, str] | None = ("obs_sky", "pred_sky"),
    n_bootstrap: int = 1000,
    seed: int = 0,
    run: Callable[[list[str]], subprocess.CompletedProcess[str]] = _run_quiet,
) -> dict[str, Any]:
    """:func:`score_by_sensor_block` on *parquet*, run inside the venv at *python*.

    The scorer imports ``allsky`` and ``labmim_core``, which live in the project
    venv and not in the notebook kernel; calling it from the kernel raised
    ``ModuleNotFoundError`` after fourteen hours of training, before the run was
    archived. Here the kernel only launches the interpreter that has them and
    reads the JSON it prints.

    Raises
    ------
    RuntimeError
        With the interpreter's stderr when scoring fails.
    """
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]); import pandas as pd; "
        "import _colab_runner as r; frame = pd.read_parquet(sys.argv[2]); "
        "sky = json.loads(sys.argv[3]); "
        "print(json.dumps(r.score_by_sensor_block(frame, sky=None if sky is None else tuple(sky), "
        "n_bootstrap=int(sys.argv[4]), seed=int(sys.argv[5])), default=float))"
    )
    result = run(
        [
            python,
            "-c",
            code,
            str(Path(__file__).resolve().parent),
            str(parquet),
            json.dumps(None if sky is None else list(sky)),
            str(n_bootstrap),
            str(seed),
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"pontuacao por bloco de {parquet} falhou:\n{result.stderr[-2000:]}")
    scored: dict[str, Any] = json.loads(result.stdout.strip().splitlines()[-1])
    return scored


def score_by_sensor_block(
    frame: Any,
    *,
    dhi: tuple[str, str] = ("obs_dhi", "pred_dhi"),
    sky: tuple[str, str] | None = ("obs_sky", "pred_sky"),
    block_minutes: float = 5.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score predictions on the label's own support: one row per datalogger block.

    Per-frame metrics treat the 4-5 frames that share one sensor row as
    independent samples of a label that is one number; scoring per block
    removes that, and is the unit the bootstrap confidence intervals resample.
    The previous block's observed class is reported as the persistence baseline
    (the previous *minute* shares the label by construction and is no baseline).

    Parameters
    ----------
    frame:
        Per-frame predictions with ``day_id`` and ``timestamp_utc`` plus the
        observed/predicted columns named by *dhi* and *sky*.
    dhi, sky:
        ``(observed, predicted)`` column pairs; *sky* may be ``None``.
    block_minutes:
        Datalogger averaging interval.
    n_bootstrap, seed:
        Block resamples behind the 95 % intervals.

    Returns
    -------
    dict
        ``n_blocks``; ``dhi`` (regression metrics of the block means);
        ``sky`` (classification metrics of the block modes, plus ``ordinal_mae``)
        and ``sky_persistence_previous_block`` when *sky* is given; ``ci95`` with
        ``dhi_rmse`` and ``sky_macro_f1`` percentile intervals over blocks.
    """
    import numpy as np
    import pandas as pd

    from allsky.evaluation.metrics import classification_metrics, regression_metrics
    from labmim_core.sky import SKY_CLASS_COUNT, SKY_CLASS_NAMES

    keyed = frame.assign(_block=sensor_block_key(frame, block_minutes))
    groups = keyed.groupby("_block", sort=True)
    obs_dhi, pred_dhi = dhi
    blocks = pd.DataFrame({"obs_dhi": groups[obs_dhi].mean(), "pred_dhi": groups[pred_dhi].mean()})
    blocks["day_id"] = groups["day_id"].first()
    report: dict[str, Any] = {
        "n_blocks": len(blocks),
        "dhi": regression_metrics(blocks["obs_dhi"], blocks["pred_dhi"]),
    }
    if sky is not None:
        obs_sky, pred_sky = sky
        blocks["obs_sky"] = groups[obs_sky].agg(_ordinal_mode)
        blocks["pred_sky"] = groups[pred_sky].agg(_ordinal_mode)
        probability_columns = [f"prob_sky_{name}" for name in SKY_CLASS_NAMES]
        block_probabilities = (
            groups[probability_columns].mean().to_numpy(dtype=np.float64)
            if all(column in keyed.columns for column in probability_columns)
            else None
        )
        report["sky"] = classification_metrics(
            blocks["obs_sky"],
            blocks["pred_sky"],
            SKY_CLASS_COUNT,
            probabilities=block_probabilities,
        )
        previous = blocks.groupby("day_id")["obs_sky"].shift(1)
        has_previous = previous.notna().to_numpy()
        report["sky_persistence_previous_block"] = classification_metrics(
            blocks["obs_sky"].to_numpy()[has_previous],
            previous.to_numpy()[has_previous].astype(np.int64),
            SKY_CLASS_COUNT,
        )

    rng = np.random.default_rng(seed)
    n = len(blocks)
    rmse_draws = np.empty(n_bootstrap)
    f1_draws = np.empty(n_bootstrap) if sky is not None else None
    obs_d, pred_d = blocks["obs_dhi"].to_numpy(), blocks["pred_dhi"].to_numpy()
    for i in range(n_bootstrap):
        pick = rng.integers(0, n, n)
        rmse_draws[i] = float(np.sqrt(np.mean((pred_d[pick] - obs_d[pick]) ** 2)))
        if f1_draws is not None:
            f1_draws[i] = classification_metrics(
                blocks["obs_sky"].to_numpy()[pick],
                blocks["pred_sky"].to_numpy()[pick],
                SKY_CLASS_COUNT,
            )["macro_f1"]
    ci: dict[str, list[float]] = {
        "dhi_rmse": [float(x) for x in np.percentile(rmse_draws, [2.5, 97.5])]
    }
    if f1_draws is not None:
        ci["sky_macro_f1"] = [float(x) for x in np.percentile(f1_draws, [2.5, 97.5])]
    report["ci95"] = ci
    return report
