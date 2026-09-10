"""The ensemble helper the local queue and the Colab notebook 04 share."""

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from allsky.clearsky import clearsky_ghi_and_kt
from labmim_core.site import STATION_UTC_OFFSET_HOURS
from labmim_core.sky import SKY_CLASS_KT_UPPER_BOUNDS

_RUNNER = Path(__file__).resolve().parents[2] / "notebooks" / "colab" / "_colab_runner.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_colab_runner", _RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_TIMES = ("2026-08-20T14:00:00+00:00", "2026-08-20T15:00:00+00:00", "2026-08-20T16:00:00+00:00")
_ZENITH = (25.0, 20.0, 30.0)


def _member(
    path: Path, *, pred_dhi: list[float], pred_kindex: list[float], pred_sky: list[int]
) -> Path:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "day_id": ["2026-08-20"] * 3,
            "timestamp_utc": list(_TIMES),
            "solar_zenith": list(_ZENITH),
            "obs_dhi": [100.0, 200.0, 300.0],
            "pred_dhi": pred_dhi,
            "obs_kindex": [0.5, 0.7, 0.9],
            "pred_kindex": pred_kindex,
            "obs_sky": [0, 1, 3],
            "pred_sky": pred_sky,
        }
    )
    frame.to_parquet(path)
    return path


def test_the_dhi_ensemble_is_the_row_wise_mean_of_the_members(tmp_path: Path) -> None:
    runner = _load_runner()
    members = [
        _member(
            tmp_path / "s42.parquet",
            pred_dhi=[90.0, 210.0, 330.0],
            pred_kindex=[0.5, 0.7, 0.9],
            pred_sky=[0, 1, 3],
        ),
        _member(
            tmp_path / "s43.parquet",
            pred_dhi=[110.0, 190.0, 270.0],
            pred_kindex=[0.5, 0.7, 0.9],
            pred_sky=[0, 1, 3],
        ),
    ]

    report = runner.ensemble_predictions(members, tmp_path / "ens")

    written = pd.read_parquet(tmp_path / "ens" / "predictions.parquet").set_index("sample_id")
    assert written["ens_dhi"].tolist() == [100.0, 200.0, 300.0]
    assert report["dhi"]["rmse"] == pytest.approx(0.0)
    assert report["n_members"] == 2


def test_a_three_way_sky_vote_tie_goes_to_the_class_nearest_the_mean_index(tmp_path: Path) -> None:
    runner = _load_runner()
    members = [
        _member(
            tmp_path / f"s{seed}.parquet",
            pred_dhi=[100.0, 200.0, 300.0],
            pred_kindex=[0.5, 0.7, 0.9],
            pred_sky=sky,
        )
        for seed, sky in ((42, [0, 0, 3]), (43, [1, 0, 3]), (44, [3, 2, 3]))
    ]

    runner.ensemble_predictions(members, tmp_path / "ens")

    written = pd.read_parquet(tmp_path / "ens" / "predictions.parquet").set_index("sample_id")
    assert written["ens_sky_vote"].tolist() == [1, 0, 3]


def test_kt_bin_reconstructs_the_class_from_the_mean_kstar_and_the_clear_sky_kt(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    members = [
        _member(
            tmp_path / "s42.parquet",
            pred_dhi=[100.0, 200.0, 300.0],
            pred_kindex=[0.2, 0.6, 1.0],
            pred_sky=[0, 1, 3],
        ),
        _member(
            tmp_path / "s43.parquet",
            pred_dhi=[100.0, 200.0, 300.0],
            pred_kindex=[0.4, 0.8, 1.0],
            pred_sky=[0, 1, 3],
        ),
    ]
    _, kt_clear = clearsky_ghi_and_kt(
        np.asarray(_ZENITH), pd.to_datetime(pd.Series(_TIMES), utc=True), STATION_UTC_OFFSET_HOURS
    )
    expected = np.digitize(
        np.array([0.3, 0.7, 1.0]) * kt_clear, SKY_CLASS_KT_UPPER_BOUNDS, right=True
    )

    report = runner.ensemble_predictions(members, tmp_path / "ens")

    written = pd.read_parquet(tmp_path / "ens" / "predictions.parquet").set_index("sample_id")
    assert written["ens_sky_kt_bin"].tolist() == expected.tolist()
    assert set(report["sky"]) == {"vote", "kt_bin"}


def test_members_over_different_samples_are_refused(tmp_path: Path) -> None:
    runner = _load_runner()
    first = _member(
        tmp_path / "s42.parquet",
        pred_dhi=[100.0, 200.0, 300.0],
        pred_kindex=[0.5, 0.7, 0.9],
        pred_sky=[0, 1, 3],
    )
    other = pd.read_parquet(first).iloc[:2]
    other.to_parquet(tmp_path / "s43.parquet")

    with pytest.raises(ValueError, match="different sample set"):
        runner.ensemble_predictions([first, tmp_path / "s43.parquet"], tmp_path / "ens")


def test_frames_sharing_a_datalogger_row_get_one_block_key() -> None:
    runner = _load_runner()
    frame = pd.DataFrame(
        {
            "day_id": ["d"] * 3,
            "timestamp_utc": [
                "2026-08-20T09:35:32+00:00",
                "2026-08-20T09:39:58+00:00",
                "2026-08-20T09:40:30+00:00",
            ],
        }
    )

    keys = runner.sensor_block_key(frame).tolist()

    assert keys[0] == keys[1] == "d@06:40"
    assert keys[2] == "d@06:45"


def test_block_scores_average_the_frames_of_one_row_and_vote_their_class() -> None:
    runner = _load_runner()
    frame = pd.DataFrame(
        {
            "day_id": ["d"] * 4,
            "timestamp_utc": [
                "2026-08-20T09:36:00+00:00",
                "2026-08-20T09:38:00+00:00",
                "2026-08-20T09:41:00+00:00",
                "2026-08-20T09:43:00+00:00",
            ],
            "obs_dhi": [100.0, 100.0, 200.0, 200.0],
            "pred_dhi": [90.0, 110.0, 190.0, 230.0],
            "obs_sky": [1, 1, 3, 3],
            "pred_sky": [1, 2, 3, 3],
        }
    )

    report = runner.score_by_sensor_block(frame, n_bootstrap=20)

    assert report["n_blocks"] == 2
    assert report["dhi"]["rmse"] == pytest.approx(np.sqrt((0.0**2 + 10.0**2) / 2))
    assert report["sky"]["accuracy"] == pytest.approx(1.0)
    assert report["sky_persistence_previous_block"]["n"] == 1
    assert len(report["ci95"]["sky_macro_f1"]) == 2


def _job(path: Path, name: str, seed: int = 42, **extra: object) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"name": name, "seed": seed, **extra}), encoding="utf-8")
    return path


def _clock(step: float = 1.0) -> tuple[list[float], object]:
    ticks = [0.0]

    def now() -> float:
        ticks[0] += step
        return ticks[0]

    return ticks, now


def test_a_job_naming_a_key_the_notebook_would_ignore_is_refused(tmp_path: Path) -> None:
    runner = _load_runner()
    job = _job(tmp_path / "010_x.yaml", "x", workers=8)

    with pytest.raises(ValueError, match="unknown keys \\['workers'\\]"):
        runner.load_job(job)


def test_the_queue_runs_jobs_in_file_name_order_and_stops_when_idle(tmp_path: Path) -> None:
    runner = _load_runner()
    queue, artifacts = tmp_path / "fila", tmp_path / "runs"
    queue.mkdir()
    _job(queue / "020_b.yaml", "b")
    _job(queue / "010_a.yaml", "a")
    ran: list[str] = []
    _, now = _clock(step=10.0)

    def record(job: dict[str, object]) -> dict[str, object]:
        ran.append(str(job["name"]))
        return {"name": job["name"], "status": "ok"}

    rows = runner.run_queue(
        queue, artifacts, record, idle_limit_seconds=25.0, clock=now, sleep=lambda _: None
    )

    assert ran == ["a", "b"]
    assert [r["status"] for r in rows] == ["ok", "ok"]
    beat = json.loads((artifacts / "fila" / "heartbeat.json").read_text())
    assert beat["stopped"] == "idle"
    assert beat["ran"] == 2


def test_a_job_that_raises_is_archived_as_failed_and_never_run_again(tmp_path: Path) -> None:
    runner = _load_runner()
    queue, artifacts = tmp_path / "fila", tmp_path / "runs"
    queue.mkdir()
    _job(queue / "010_boom.yaml", "boom")
    attempts: list[str] = []

    def explode(job: dict[str, object]) -> dict[str, object]:
        attempts.append(str(job["name"]))
        raise RuntimeError("cuda out of memory")

    _, now = _clock(step=10.0)
    rows = runner.run_queue(
        queue, artifacts, explode, idle_limit_seconds=25.0, clock=now, sleep=lambda _: None
    )

    assert attempts == ["boom"]
    assert rows == [{"name": "010_boom", "status": "failed", "error": "cuda out of memory"}]
    status = json.loads((artifacts / "fila" / "010_boom.status.json").read_text())
    assert status["status"] == "failed"
    assert "cuda out of memory" in status["error"]
    assert runner.pending_jobs(queue, artifacts) == []


def test_a_job_a_reclaimed_vm_left_running_is_picked_up_again(tmp_path: Path) -> None:
    runner = _load_runner()
    queue, artifacts = tmp_path / "fila", tmp_path / "runs"
    queue.mkdir()
    job = _job(queue / "010_a.yaml", "a")
    (artifacts / "fila").mkdir(parents=True)
    (artifacts / "fila" / "010_a.status.json").write_text(json.dumps({"status": "running"}))

    assert runner.pending_jobs(queue, artifacts) == [job]


def test_the_stop_file_ends_the_queue_before_any_job_runs(tmp_path: Path) -> None:
    runner = _load_runner()
    queue, artifacts = tmp_path / "fila", tmp_path / "runs"
    queue.mkdir()
    _job(queue / "010_a.yaml", "a")
    (queue / "PARE").touch()
    _, now = _clock()

    rows = runner.run_queue(
        queue, artifacts, lambda _: {"status": "ok"}, clock=now, sleep=lambda _: None
    )

    assert rows == []
    assert json.loads((artifacts / "fila" / "heartbeat.json").read_text())["stopped"] == "stop_file"


def test_the_live_mirror_copies_only_histories_that_grew_and_reports_the_gpu(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    out, live = tmp_path / "out", tmp_path / "live"
    history = out / "ceuU_s42" / "run" / "metrics.csv"
    history.parent.mkdir(parents=True)
    history.write_text("epoch,val_loss\n1,0.9\n")

    first = runner.sync_live(out, live, gpu_probe=lambda: "97 %, 21000 MiB", clock=lambda: 0.0)
    unchanged = runner.sync_live(out, live, gpu_probe=lambda: "97 %, 21000 MiB", clock=lambda: 1.0)
    history.write_text("epoch,val_loss\n1,0.9\n2,0.8\n")
    grown = runner.sync_live(out, live, gpu_probe=lambda: None, clock=lambda: 2.0)

    assert first["updated"] == ["ceuU_s42"]
    assert unchanged["updated"] == []
    assert grown == {"time": "1970-01-01T00:00:02+00:00", "gpu": None, "updated": ["ceuU_s42"]}
    assert (live / "ceuU_s42" / "metrics.csv").read_text().endswith("2,0.8\n")


def test_archiving_a_run_that_never_started_reports_it_instead_of_raising(tmp_path: Path) -> None:
    runner = _load_runner()

    message = runner.archive(str(tmp_path / "ceuU_s42"), str(tmp_path / "drive"))

    assert "nada a arquivar" in message
    assert not (tmp_path / "drive").exists()


def test_an_arm_already_archived_on_drive_is_harvested_instead_of_retrained(
    tmp_path: Path,
) -> None:
    import yaml

    runner = _load_runner()
    config = tmp_path / "ceuU_s42.yaml"
    config.write_text(
        yaml.safe_dump({"name": "ceuU_s42", "seed": 42, "output_dir": str(tmp_path / "out")})
    )
    report = tmp_path / "drive" / "ceuU_s42" / "eval-test-last"
    report.mkdir(parents=True)
    (report / "eval_metrics.json").write_text(
        json.dumps(
            {
                "n_samples": 3,
                "meta": {"split_id_ok": True},
                "global": {
                    "dhi": {"rmse": 14.2, "mae": 9.1},
                    "sky": {"macro_f1": 0.72, "per_class": {"clear": {"f1": 0.9, "recall": 0.88}}},
                },
            }
        )
    )

    row = runner.run_experiment(
        config,
        python=str(tmp_path / "nowhere" / "bin" / "python"),
        checkpoint="last",
        archive_dir=str(tmp_path / "drive"),
    )

    assert row["status"] == "archived"
    assert row["rmse"] == pytest.approx(14.2)
    assert row["sky_macro_f1"] == pytest.approx(0.72)
    assert row["sky_f1_clear"] == pytest.approx(0.9)
    assert row["checkpoint"] == "last"


def test_the_mirror_rsyncs_every_pair_in_order_and_names_the_ones_that_failed() -> None:
    import subprocess

    runner = _load_runner()
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        code = 1 if command[-1].endswith("/fila") else 0
        return subprocess.CompletedProcess(command, code, "", "")

    failed = runner.mirror_once(
        [("/vm/runs", "gs://b/runs"), ("gs://b/fila", "/vm/fila")], run=fake_run
    )

    assert commands == [
        ["gcloud", "storage", "rsync", "-r", "/vm/runs", "gs://b/runs"],
        ["gcloud", "storage", "rsync", "-r", "gs://b/fila", "/vm/fila"],
    ]
    assert failed == ["gs://b/fila -> /vm/fila"]


def test_the_live_mirror_carries_checkpoints_and_a_resume_restores_them(tmp_path: Path) -> None:
    runner = _load_runner()
    out, live = tmp_path / "out", tmp_path / "live"
    run_dir = out / "l4bloco512_s42" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.csv").write_text("epoch,val_loss\n1,0.9\n")
    (run_dir / "last.ckpt").write_bytes(b"epoch-1")
    (run_dir / "best.ckpt").write_bytes(b"best-1")

    first = runner.sync_live(out, live, gpu_probe=lambda: None, clock=lambda: 0.0)
    (run_dir / "last.ckpt").write_bytes(b"epoch-2")
    second = runner.sync_live(out, live, gpu_probe=lambda: None, clock=lambda: 1.0)
    restored = runner.pull_live_run(live, tmp_path / "fresh", "l4bloco512_s42")

    assert first["updated"] == ["l4bloco512_s42"]
    assert second["updated"] == ["l4bloco512_s42"]
    assert (live / "l4bloco512_s42" / "last.ckpt").read_bytes() == b"epoch-2"
    assert (tmp_path / "fresh" / "l4bloco512_s42" / "run" / "last.ckpt").read_bytes() == b"epoch-2"
    assert (tmp_path / "fresh" / "l4bloco512_s42" / "run" / "best.ckpt").read_bytes() == b"best-1"
    assert restored is not None
    assert "last.ckpt" in restored


def test_a_mirrored_history_without_a_checkpoint_is_not_restored(tmp_path: Path) -> None:
    runner = _load_runner()
    stale = tmp_path / "live" / "l4bloco512_s42"
    stale.mkdir(parents=True)
    (stale / "metrics.csv").write_text("epoch,val_loss\n1,0.9\n")

    restored = runner.pull_live_run(tmp_path / "live", tmp_path / "out", "l4bloco512_s42")

    assert restored is None
    assert not (tmp_path / "out").exists()


def test_a_resume_hands_the_existing_checkpoint_to_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import yaml

    runner = _load_runner()
    config = tmp_path / "l4bloco512_s42.yaml"
    config.write_text(
        yaml.safe_dump({"name": "l4bloco512_s42", "seed": 42, "output_dir": str(tmp_path / "out")})
    )
    run_dir = tmp_path / "out" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "last.ckpt").write_bytes(b"epoch-40")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "evaluate":
            report = Path(command[command.index("--report-dir") + 1])
            report.mkdir(parents=True, exist_ok=True)
            (report / "eval_metrics.json").write_text(
                json.dumps(
                    {"n_samples": 1, "meta": {}, "global": {"dhi": {"rmse": 1.0, "mae": 1.0}}}
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    resumed = runner.run_experiment(config, python="/venv/bin/python", resume=True)
    skipped = runner.run_experiment(config, python="/venv/bin/python", checkpoint="last")

    assert commands[0][1:] == ["train", "-c", str(config), "--resume", "auto"]
    assert [c[1] for c in commands] == ["train", "evaluate", "evaluate"]
    assert resumed["status"] == "ok"
    assert skipped["status"] == "ok"


def test_every_evaluation_of_the_queue_runs_on_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``allsky evaluate`` defaults to the CPU, and the queue must not inherit that.

    Three reports of 7,265 frames each scored on the VM's eight vCPUs cost 2.1 h
    at 512 px with the GPU idle (l4v3res512_s44, 2026-09-10); at 1024 px they
    would cost ~14 h and outlive the 24 h execution. The device is pinned to
    ``cuda`` rather than ``auto`` so a venv without CUDA fails here instead of
    silently scoring on the CPU again.
    """
    import subprocess

    import yaml

    runner = _load_runner()
    config = tmp_path / "l4res1024_s44.yaml"
    config.write_text(
        yaml.safe_dump({"name": "l4res1024_s44", "seed": 44, "output_dir": str(tmp_path / "out")})
    )
    run_dir = tmp_path / "out" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "best.ckpt").write_bytes(b"epoch-20")
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        if command[1] == "evaluate":
            report = Path(command[command.index("--report-dir") + 1])
            report.mkdir(parents=True, exist_ok=True)
            (report / "eval_metrics.json").write_text(
                json.dumps(
                    {"n_samples": 1, "meta": {}, "global": {"dhi": {"rmse": 1.0, "mae": 1.0}}}
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    runner.run_experiment(config, python="/venv/bin/python", split="val")

    evaluations = [c for c in commands if c[1] == "evaluate"]
    assert evaluations, commands
    for command in evaluations:
        assert command[command.index("--device") + 1] == "cuda", command


def test_block_scores_are_asked_of_the_venv_interpreter_with_the_runner_on_its_path(
    tmp_path: Path,
) -> None:
    import subprocess

    runner = _load_runner()
    commands: list[list[str]] = []

    def canned(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, '{"n_blocks": 2}\n', "")

    report = runner.score_by_sensor_block_in(
        "/venv/bin/python", tmp_path / "p.parquet", sky=("obs_sky", "pred_sky_kt"), run=canned
    )

    assert commands[0][:2] == ["/venv/bin/python", "-c"]
    assert commands[0][3:] == [
        str(_RUNNER.parent),
        str(tmp_path / "p.parquet"),
        '["obs_sky", "pred_sky_kt"]',
        "1000",
        "0",
    ]
    assert report == {"n_blocks": 2}


def test_block_scores_from_the_interpreter_match_the_in_process_scorer(tmp_path: Path) -> None:
    import sys

    runner = _load_runner()
    parquet = _member(
        tmp_path / "predictions.parquet",
        pred_dhi=[90.0, 210.0, 300.0],
        pred_kindex=[0.5, 0.7, 0.9],
        pred_sky=[0, 1, 3],
    )

    report = runner.score_by_sensor_block_in(sys.executable, parquet, n_bootstrap=10)
    direct = runner.score_by_sensor_block(pd.read_parquet(parquet), n_bootstrap=10)

    assert report["n_blocks"] == direct["n_blocks"] == 3
    assert report["dhi"]["rmse"] == pytest.approx(direct["dhi"]["rmse"])
    assert report["sky"]["macro_f1"] == pytest.approx(direct["sky"]["macro_f1"])


def test_a_failing_block_score_raises_with_the_interpreter_error(tmp_path: Path) -> None:
    import subprocess

    runner = _load_runner()

    def broken(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "ModuleNotFoundError: allsky")

    with pytest.raises(RuntimeError, match="ModuleNotFoundError"):
        runner.score_by_sensor_block_in("/venv/bin/python", tmp_path / "p.parquet", run=broken)


def test_queue_overrides_skip_or_replace_one_arm(tmp_path: Path) -> None:
    runner = _load_runner()
    overrides = tmp_path / "fila-l4"
    overrides.mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    first, second, third = (configs / f"l4_s{i}.yaml" for i in (1, 2, 3))
    for config in (first, second, third):
        config.write_text(_config_yaml(config.stem, epochs=150))
    (overrides / "l4_s1.skip").touch()
    (overrides / "l4_s2.yaml").write_text(_config_yaml("l4_s2", epochs=40))

    decisions = [
        runner.apply_queue_override(overrides, config) for config in (first, second, third)
    ]

    assert decisions == ["skip", "override", "repo"]
    assert "epochs: 40" in second.read_text()
    assert "epochs: 150" in third.read_text()


def _config_yaml(name: str, *, epochs: int, output_name: str | None = None) -> str:
    import yaml

    return yaml.safe_dump(
        {
            "extends": ["../_base.yaml"],
            "name": name,
            "seed": 42,
            "output_dir": f"output/l4/{output_name or name}",
            "train": {"epochs": epochs},
        }
    )


def _bundle(tmp_path: Path) -> Path:
    """A tarball shaped like the one :func:`stage_bundle` unpacks."""
    import tarfile

    root = tmp_path / "allsky_bundle"
    root.mkdir(exist_ok=True)
    (root / "manifest.parquet").write_bytes(b"o staging so desempacota, nao le")
    archive_path = tmp_path / "bundle-fonte.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root, arcname="allsky_bundle")
    return archive_path


def test_an_override_that_renames_the_arm_is_refused(tmp_path: Path) -> None:
    runner = _load_runner()
    overrides, configs = tmp_path / "fila", tmp_path / "configs"
    overrides.mkdir()
    configs.mkdir()
    config = configs / "l4_s1.yaml"
    config.write_text(_config_yaml("l4_s1", epochs=150))
    (overrides / "l4_s1.yaml").write_text(
        _config_yaml("l4_s1_v2", epochs=40, output_name="l4_s1_v2")
    )

    with pytest.raises(ValueError, match="nao batem com l4_s1"):
        runner.apply_queue_override(overrides, config)

    assert "epochs: 150" in config.read_text()


def test_a_partial_override_is_refused_before_it_reaches_the_gpu(tmp_path: Path) -> None:
    runner = _load_runner()
    overrides, configs = tmp_path / "fila", tmp_path / "configs"
    overrides.mkdir()
    configs.mkdir()
    config = configs / "l4_s1.yaml"
    config.write_text(_config_yaml("l4_s1", epochs=150))
    (overrides / "l4_s1.yaml").write_text("train:\n  epochs: 40\n")

    with pytest.raises(ValueError, match="override incompleto"):
        runner.apply_queue_override(overrides, config)


class _BlockProjectImports:
    """Import finder that makes this project's packages unavailable, as the Colab kernel has them."""

    BLOCKED = ("allsky", "labmim_core", "micrometeorology")

    def find_spec(self, fullname: str, _path: object = None, _target: object = None) -> None:
        if fullname.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{fullname} nao existe no kernel do Colab")


def _kernel_run(command: list[str]) -> subprocess.CompletedProcess[str]:
    import subprocess

    payload = '{"n_blocks": 2, "sky": {"macro_f1": 0.5}, "dhi": {"rmse": 1.0}}'
    return subprocess.CompletedProcess(command, 0, payload if "-c" in command else "", "")


def test_every_function_the_notebook_calls_runs_without_the_project_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys

    runner = _load_runner()
    config = _job(tmp_path / "arm_s42.yaml", "arm_s42", output_dir=str(tmp_path / "out"))
    run_dir = tmp_path / "out" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "last.ckpt").write_bytes(b"ckpt")
    (run_dir / "metrics.csv").write_text("epoch,val_loss\n1,0.5\n")

    def fake_subprocess_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if "evaluate" in command:
            report = Path(command[command.index("--report-dir") + 1])
            report.mkdir(parents=True, exist_ok=True)
            (report / "eval_metrics.json").write_text(
                json.dumps(
                    {"n_samples": 1, "meta": {}, "global": {"dhi": {"rmse": 1.0, "mae": 1.0}}}
                )
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(runner, "_run_quiet", _kernel_run)
    blocker = _BlockProjectImports()
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    for blocked in ("allsky", "labmim_core"):
        for loaded in [key for key in sys.modules if key.split(".")[0] == blocked]:
            monkeypatch.delitem(sys.modules, loaded, raising=False)

    called = {
        "write_config": lambda: runner.write_config(
            tmp_path / "written.yaml",
            extends=["../_base.yaml"],
            name="arm_s42",
            output_dir=str(tmp_path / "out"),
            seed=42,
            data_root=str(tmp_path),
            model={"backbone": "dinov3_vits16plus"},
            train={"epochs": 1},
        ),
        "load_job": lambda: runner.load_job(_job(tmp_path / "job.yaml", "arm_s42")),
        "stage_bundle": lambda: runner.stage_bundle(
            str(_bundle(tmp_path)), str(tmp_path / "dados")
        ),
        "run_experiment": lambda: runner.run_experiment(config, python="/venv/bin/python"),
        "archive": lambda: runner.archive(str(tmp_path / "out"), str(tmp_path / "drive")),
        "sync_live": lambda: runner.sync_live(tmp_path / "out" / "..", tmp_path / "live"),
        "start_live_sync": lambda: runner.start_live_sync(
            tmp_path, tmp_path / "live2", period_seconds=3600.0
        ),
        "mirror_once": lambda: runner.mirror_once([("a", "b")], run=_kernel_run),
        "start_mirror": lambda: runner.start_mirror(
            [(str(tmp_path / "de"), str(tmp_path / "para"))], period_seconds=3600.0
        ),
        "score_by_sensor_block_in": lambda: runner.score_by_sensor_block_in(
            "/venv/bin/python", tmp_path / "p.parquet", run=_kernel_run
        ),
        "pull_live_run": lambda: runner.pull_live_run(tmp_path / "live", tmp_path / "back", "out"),
        "apply_queue_override": lambda: runner.apply_queue_override(
            tmp_path / "sem-override", config
        ),
        "run_arm": lambda: runner.run_arm(
            config,
            python="/venv/bin/python",
            out_dir=tmp_path,
            artifacts=tmp_path / "artifacts",
            mirror=[],
            log=lambda _: None,
            run=_kernel_run,
        ),
        "summarise_arm": lambda: runner.summarise_arm({"name": "arm_s42", "status": "ok"}),
        "summarise": lambda: runner.summarise([{"name": "arm_s42", "rmse": 1.0}]),
        "preflight": lambda: runner.preflight(
            "/venv/bin/python",
            artifacts=str(tmp_path / "artifacts"),
            mirror=[(str(tmp_path / "artifacts"), "gs://bucket/runs")],
            work_dir=tmp_path / "work",
            run=_kernel_run,
        ),
    }

    assert set(called) == set(runner.KERNEL_SAFE)
    for call in called.values():
        call()


def test_the_preflight_walks_every_kernel_step_and_leaves_a_stamp(tmp_path: Path) -> None:
    import sys

    runner = _load_runner()
    artifacts = tmp_path / "artifacts"
    commands: list[list[str]] = []

    def spy(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _kernel_run(command)

    checks = runner.preflight(
        sys.executable,
        artifacts=str(artifacts),
        mirror=[(str(artifacts), "gs://bucket/runs")],
        work_dir=tmp_path / "work",
        run=spy,
    )

    assert len(checks) == 5
    assert json.loads((artifacts / "preflight.json").read_text())["checks"] == checks[:-1]
    assert commands[0][1] == "--help"
    assert commands[-1][:4] == ["gcloud", "storage", "rsync", "-r"]
    assert not (tmp_path / "work" / "_preflight").exists()


def test_the_preflight_names_the_step_that_failed(tmp_path: Path) -> None:
    import subprocess
    import sys

    runner = _load_runner()

    def broken_scorer(command: list[str]) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            return subprocess.CompletedProcess(command, 1, "", "ModuleNotFoundError: allsky")
        return subprocess.CompletedProcess(command, 0, "", "")

    with pytest.raises(RuntimeError, match="ModuleNotFoundError"):
        runner.preflight(
            sys.executable,
            artifacts=str(tmp_path / "artifacts"),
            mirror=[],
            work_dir=tmp_path / "work",
            run=broken_scorer,
        )

    assert not (tmp_path / "artifacts" / "preflight.json").exists()


def _arm(tmp_path: Path, name: str = "l4bloco512_s42") -> Path:
    import yaml

    config = tmp_path / f"{name}.yaml"
    config.write_text(
        yaml.safe_dump({"name": name, "seed": 42, "output_dir": str(tmp_path / "out" / name)})
    )
    return config


def _evaluating_subprocess(*, fail_on: str | None = None) -> tuple[list[list[str]], object]:
    seen: list[list[str]] = []

    def fake(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        if fail_on is not None and fail_on in command:
            return subprocess.CompletedProcess(command, 1, "", f"{fail_on} explodiu")
        if command[1] == "train":
            run_dir = Path(command[3]).parent / "out" / Path(command[3]).stem / "run"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "last.ckpt").write_bytes(b"ckpt")
            (run_dir / "metrics.csv").write_text("epoch,val_loss\n1,0.5\n")
        if command[1] == "evaluate":
            report = Path(command[command.index("--report-dir") + 1])
            report.mkdir(parents=True, exist_ok=True)
            (report / "eval_metrics.json").write_text(
                json.dumps(
                    {
                        "n_samples": 5,
                        "meta": {},
                        "global": {"dhi": {"rmse": 14.0, "mae": 9.0, "mbe": 0.5}},
                    }
                )
            )
            _member(
                report / "predictions.parquet",
                pred_dhi=[90.0, 210.0, 300.0],
                pred_kindex=[0.5, 0.7, 0.9],
                pred_sky=[0, 1, 3],
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    return seen, fake


def test_each_evaluation_is_archived_before_the_next_one_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)
    order: list[str] = []
    real_archive, real_experiment = runner.archive, runner.run_experiment

    def spy_archive(*args: object, **kwargs: object) -> object:
        order.append("arquiva")
        return real_archive(*args, **kwargs)

    def spy_experiment(*args: object, **kwargs: object) -> object:
        order.append("avalia")
        return real_experiment(*args, **kwargs)

    monkeypatch.setattr(runner, "archive", spy_archive)
    monkeypatch.setattr(runner, "run_experiment", spy_experiment)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=_kernel_run,
    )

    assert order == ["avalia", "arquiva", "avalia", "arquiva", "avalia", "arquiva"]
    assert row["status"] == "ok"
    assert (tmp_path / "artifacts" / "l4bloco512_s42" / "eval-test" / "eval_metrics.json").exists()
    assert (tmp_path / "artifacts" / "l4bloco512_s42" / "last.ckpt").exists()


def test_a_failing_block_score_leaves_the_arm_archived_and_the_row_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)

    def broken_scorer(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command, 1, "", "ModuleNotFoundError: No module named 'allsky'"
        )

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=broken_scorer,
    )

    assert row["status"] == "ok"
    assert "ModuleNotFoundError" in row["score_error"]
    assert (
        tmp_path / "artifacts" / "l4bloco512_s42" / "eval-test-last" / "eval_metrics.json"
    ).exists()


def test_a_training_failure_stops_the_arm_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    seen, fake = _evaluating_subprocess(fail_on="train")
    monkeypatch.setattr(runner.subprocess, "run", fake)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["status"] == "train_failed"
    assert "explodiu" in row["error"]
    assert [c[1] for c in seen] == ["train"]


def test_an_unexpected_failure_inside_the_arm_is_recorded_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)

    def explode(*_: object, **__: object) -> None:
        raise MemoryError("a VM ficou sem memoria")

    monkeypatch.setattr(runner, "run_experiment", explode)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["status"] == "failed"
    assert "MemoryError" in row["error"]


def test_a_skipped_arm_never_touches_the_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    overrides = tmp_path / "fila"
    overrides.mkdir()
    (overrides / "l4bloco512_s42.skip").touch()
    seen, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        overrides=overrides,
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["status"] == "skipped"
    assert seen == []


def test_the_archive_still_happens_when_an_evaluation_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)
    real_experiment = runner.run_experiment
    calls = {"n": 0}

    def explode_on_the_second(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise TimeoutError("a VM foi recuperada no meio da avaliacao")
        return real_experiment(*args, **kwargs)

    monkeypatch.setattr(runner, "run_experiment", explode_on_the_second)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["status"] == "failed"
    assert "TimeoutError" in row["error"]
    assert (tmp_path / "artifacts" / "l4bloco512_s42" / "eval-test" / "eval_metrics.json").exists()
    assert (tmp_path / "artifacts" / "l4bloco512_s42" / "last.ckpt").exists()


def test_an_arm_missing_one_report_is_evaluated_again_instead_of_left_blank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    archived = tmp_path / "artifacts" / "l4bloco512_s42"
    for report in ("eval-test", "eval-test-last"):
        (archived / report).mkdir(parents=True)
        (archived / report / "eval_metrics.json").write_text(
            json.dumps({"n_samples": 1, "meta": {}, "global": {"dhi": {"rmse": 1.0, "mae": 1.0}}})
        )
    seen, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["status"] == "ok"
    assert [c[1] for c in seen] == ["train", "evaluate", "evaluate", "evaluate"]


def test_an_overridden_config_never_resumes_the_previous_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    overrides = tmp_path / "fila"
    overrides.mkdir()
    (overrides / "l4bloco512_s42.yaml").write_text(
        yaml_dump_config(str(tmp_path / "out" / "l4bloco512_s42"))
    )
    live = tmp_path / "artifacts" / runner.LIVE_DIR / "l4bloco512_s42"
    live.mkdir(parents=True)
    (live / "last.ckpt").write_bytes(b"da receita antiga")
    seen, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        overrides=overrides,
        log=lambda _: None,
        run=_kernel_run,
    )

    assert row["override"] == "override"
    assert "--resume" not in seen[0]
    assert (
        not (tmp_path / "out" / "l4bloco512_s42" / "run" / "last.ckpt")
        .read_bytes()
        .startswith(b"da receita antiga")
    )


def yaml_dump_config(output_dir: str) -> str:
    import yaml

    return yaml.safe_dump(
        {
            "extends": ["../_base.yaml"],
            "name": "l4bloco512_s42",
            "seed": 42,
            "output_dir": output_dir,
            "train": {"epochs": 40},
        }
    )


def test_the_overrides_are_pulled_with_deletion_so_a_marker_can_be_revoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)
    commands: list[list[str]] = []

    def spy(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _kernel_run(command)

    runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        overrides=tmp_path / "fila",
        override_mirror=[("gs://bucket/fila-l4", str(tmp_path / "fila"))],
        log=lambda _: None,
        run=spy,
    )

    assert commands[0] == [
        "gcloud",
        "storage",
        "rsync",
        "-r",
        "--delete-unmatched-destination-objects",
        "gs://bucket/fila-l4",
        str(tmp_path / "fila"),
    ]


def test_a_mirrored_checkpoint_is_never_visible_half_written(tmp_path: Path) -> None:
    runner = _load_runner()
    out = tmp_path / "out" / "arm" / "run"
    out.mkdir(parents=True)
    (out / "last.ckpt").write_bytes(b"completo")
    live = tmp_path / "live"
    seen: list[bytes] = []
    real_copy = runner.shutil.copy2

    def copy_and_peek(source: object, target: object) -> object:
        result = real_copy(source, target)
        final = live / "arm" / "last.ckpt"
        seen.append(final.read_bytes() if final.exists() else b"")
        return result

    monkey = pytest.MonkeyPatch()
    monkey.setattr(runner.shutil, "copy2", copy_and_peek)
    runner.sync_live(tmp_path / "out", live)
    monkey.undo()

    assert seen[0] == b""
    assert (live / "arm" / "last.ckpt").read_bytes() == b"completo"
    assert not list(live.glob("**/.*.parcial"))


def test_the_runner_parses_on_the_python_the_colab_kernel_runs() -> None:
    """The Colab kernel imports this module and is older than the venv it drives.

    ``ruff format`` targets the project's 3.14, and a construct only 3.14 can
    parse — an unparenthesized ``except A, B`` — turns the first cell that says
    ``import _colab_runner`` into a SyntaxError on the VM.
    """
    import ast

    source = _RUNNER.read_text(encoding="utf-8")

    ast.parse(source, feature_version=(3, 11))


def test_a_dead_mirroring_thread_is_announced_before_the_arm_trains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)
    said: list[str] = []

    class _Dead:
        def is_alive(self) -> bool:
            return False

    runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[],
        watchers=[_Dead()],
        log=said.append,
        run=_kernel_run,
    )

    assert "ATENCAO, espelho parado" in said[0]


def test_a_broken_mirror_command_does_not_end_the_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _load_runner()
    config = _arm(tmp_path)
    _, fake = _evaluating_subprocess()
    monkeypatch.setattr(runner.subprocess, "run", fake)

    def no_gcloud(_command: list[str]) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("gcloud")

    row = runner.run_arm(
        config,
        python="/venv/bin/python",
        out_dir=tmp_path / "out",
        artifacts=tmp_path / "artifacts",
        mirror=[("de", "para")],
        log=lambda _: None,
        run=no_gcloud,
    )

    assert row["status"] == "failed"
    assert "FileNotFoundError" in row["error"]
