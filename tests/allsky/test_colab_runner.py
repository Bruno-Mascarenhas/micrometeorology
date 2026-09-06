"""The ensemble helper the local queue and the Colab notebook 04 share."""

import importlib.util
import json
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
                "global": {"dhi": {"rmse": 14.2, "mae": 9.1}, "sky": {"macro_f1": 0.72}},
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
