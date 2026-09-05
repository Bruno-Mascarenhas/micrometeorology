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
from collections.abc import Sequence
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
    config: Path, *, python: str, split: str = "test", checkpoint: str = "best"
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

    started = time.time()
    if not (run_dir / "last.ckpt").exists():
        train = subprocess.run(
            [allsky_cli, "train", "-c", str(config)], capture_output=True, text=True, check=False
        )
        if train.returncode != 0:
            row["status"] = "train_failed"
            row["error"] = train.stderr[-2000:]
            return row

    report_dir = run_dir / (
        f"eval-{split}" if checkpoint == "best" else f"eval-{split}-{checkpoint}"
    )
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

    metrics = json.loads((report_dir / "eval_metrics.json").read_text())
    dhi = metrics["global"]["dhi"]
    row.update(
        status="ok",
        wall_seconds=round(time.time() - started, 1),
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
