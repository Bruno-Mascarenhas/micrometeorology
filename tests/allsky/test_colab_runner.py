"""The ensemble helper the local queue and the Colab notebook 04 share."""

import importlib.util
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
