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

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

#: Every arm in these notebooks trains the single DHI head; the multitask heads
#: cost 0.86 W/m2 in the measured factorial.
DHI_ONLY_TARGETS: dict[str, Any] = {"kindex": {"enabled": False}, "sky": {"enabled": False}}


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
    note: str = "",
) -> Path:
    """Write one experiment YAML and return its path.

    ``seed`` and ``train.num_workers`` have no CLI override, so a per-run file is
    the only way to vary them; the rest is written alongside them so the file is
    a complete record of what produced the artifacts next to it.
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
    if targets is not None:
        body["targets"] = targets
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {note}\n" if note else ""
    path.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def run_experiment(config: Path, *, python: str, split: str = "test") -> dict[str, Any]:
    """Train then evaluate one config; return a flat metrics row.

    *python* is the venv interpreter, as :func:`stage_bundle` takes: the
    ``allsky`` console script sits beside it, so the CLI is resolved by path and
    not by whatever ``PATH`` happens to hold when the cell runs.

    A failure is recorded and returned rather than raised: one bad arm must not
    end a 24-hour session that still has other arms to run.
    """
    import yaml

    cfg = yaml.safe_load(config.read_text())
    run_dir = Path(cfg["output_dir"]) / "run"
    row: dict[str, Any] = {"name": cfg["name"], "seed": cfg["seed"], "config": str(config)}
    allsky_cli = str(Path(python).with_name("allsky"))

    started = time.time()
    train = subprocess.run(
        [allsky_cli, "train", "-c", str(config)], capture_output=True, text=True, check=False
    )
    if train.returncode != 0:
        row["status"] = "train_failed"
        row["error"] = train.stderr[-2000:]
        return row

    evaluate = subprocess.run(
        [
            allsky_cli,
            "evaluate",
            "-k",
            str(run_dir / "best.ckpt"),
            "--split",
            split,
            "-c",
            str(config),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if evaluate.returncode != 0:
        row["status"] = "eval_failed"
        row["error"] = evaluate.stderr[-2000:]
        return row

    metrics = json.loads((run_dir / f"eval-{split}" / "metrics.json").read_text())
    dhi = metrics["global"]["dhi"]
    row.update(
        status="ok",
        wall_seconds=round(time.time() - started, 1),
        n_samples=metrics["n_samples"],
        **{k: dhi[k] for k in ("rmse", "mae", "mbe", "r2") if k in dhi},
        skill_clearsky=dhi.get("skill_clearsky"),
        split_id_ok=metrics["meta"].get("split_id_ok"),
        manifest_hash_ok=metrics["meta"].get("manifest_hash_ok"),
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
    target = Path(drive_dir) / Path(run_output_dir).name
    ignore = None if keep_checkpoint else shutil.ignore_patterns("*.ckpt")
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=ignore)
    if config is not None and Path(config).exists():
        shutil.copy2(config, target / Path(config).name)
    return f"{target}: {sum(1 for _ in target.rglob('*') if _.is_file())} arquivo(s)"


def summarise(rows: list[dict[str, Any]]) -> Any:
    """Tidy DataFrame of the harvested rows, best RMSE first."""
    import pandas as pd

    frame = pd.DataFrame(rows)
    if "rmse" in frame.columns:
        frame = frame.sort_values("rmse", na_position="last")
    return frame.reset_index(drop=True)
