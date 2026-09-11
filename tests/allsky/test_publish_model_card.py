"""``model.json``: paired-row persistence, scoped refusals and a card free of paths."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from allsky.config import ExperimentConfig
from allsky.publish.encoding import publish_stamp
from allsky.publish.model_card import (
    CheckpointMetadata,
    ModelCardError,
    build_model_card,
    paired_row_ends,
)
from allsky.serving import ServingConfig
from labmim_core.sky import SKY_CLASS_NAMES
from tests.allsky._pins import control, pinned, serving_pin_payload

SPLIT_ID = "a" * 64
MANIFEST_SHA = "b" * 64
DIGEST = "c" * 64
TRAIN_DAYS = ("2026-06-03", "2026-06-04")
VAL_DAYS = ("2026-07-10",)
TEST_DAYS = ("2026-08-21", "2026-08-22")
MEMBERS = (("probe_s42", 42), ("probe_s43", 43))
ROWS_PER_DAY = 3
FRAMES_PER_ROW = 5
ROW_STEP_DHI = {"2026-08-21": 20.0, "2026-08-22": 10.0}
FIRST_ROW_DHI = {"2026-08-21": 100.0, "2026-08-22": 150.0}
PREDICTION_OFFSETS = (3.0, -3.0, 3.0, -3.0, 3.0)
CLEARSKY_DHI = 80.0
ROW_CLASSES = (0, 1, 3)
SPLIT_ELEVATION_MAX = {"train": 59.8, "val": 63.9, "test": 69.5}


def _experiment_config(name: str, seed: int, *, model: str = "image_only") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": name,
            "seed": seed,
            "model": {
                "name": model,
                "image_size": 64,
                "backbone": "dinov3_vits16plus",
                "backbone_pooling": "cls",
                "unfreeze_last_n": 12,
                "trunk_hidden": 256,
            },
            "features": {"set": "bare"},
            "data": {"input_mode": "image"},
            "targets": {
                "dhi": {"enabled": True, "parameterization": "clearsky_index"},
                "kindex": {"enabled": True, "kind": "kstar"},
                "sky": {"enabled": True},
            },
        }
    )


def _metadata(
    tmp_path: Path,
    name: str,
    seed: int,
    *,
    split_id: str = SPLIT_ID,
    manifest_sha: str = MANIFEST_SHA,
    model: str = "image_only",
) -> CheckpointMetadata:
    return CheckpointMetadata(
        path=tmp_path / name / "best.ckpt",
        sha256=DIGEST,
        config=_experiment_config(name, seed, model=model),
        epoch=9,
        best_metric={"name": "val_kindex_mae", "value": 0.08, "epoch": 6},
        code_version={"package_version": "1.4.0", "git_commit": "abc123"},
        dataset_version="2",
        split_id=split_id,
        manifest_sha256=manifest_sha,
        frame_geometry={
            "mask": {"path": "/masks/disc.png", "threshold": 0.5},
            "crop": {"enabled": True, "top": 0, "left": 12, "height": 1080, "width": 1711},
            "pad": {"enabled": True, "top": 254, "bottom": 377, "left": 0, "right": 0, "fill": 0},
            "resize": 64,
        },
        backbone={"name": "dinov3_vits16plus", "weights": "/weights/dinov3.pth"},
        n_parameters=1234,
    )


def _frame_times_local(day: str) -> list[pd.Timestamp]:
    times: list[pd.Timestamp] = []
    for row in range(ROWS_PER_DAY):
        row_start = pd.Timestamp(f"{day} 09:00:00") + pd.Timedelta(minutes=5 * row)
        times.extend(
            row_start + pd.Timedelta(seconds=34 + 62 * frame) for frame in range(FRAMES_PER_ROW)
        )
    return times


def _predictions(seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for day in TEST_DAYS:
        previous_obs = np.nan
        for position, local in enumerate(_frame_times_local(day)):
            row, frame = divmod(position, FRAMES_PER_ROW)
            obs_dhi = FIRST_ROW_DHI[day] + ROW_STEP_DHI[day] * row
            offset = PREDICTION_OFFSETS[frame] * (1.0 if seed == 42 else -1.0)
            obs_class = ROW_CLASSES[row]
            probabilities = np.full(len(SKY_CLASS_NAMES), 0.05)
            probabilities[obs_class] = 0.85
            utc = (local + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S+00:00")
            records.append(
                {
                    "sample_id": f"allsky-{local:%Y%m%d-%H%M%S}",
                    "day_id": day,
                    "timestamp_utc": utc,
                    "hour": local.hour,
                    "solar_zenith": 40.0,
                    "month": local.month,
                    "sky_class": SKY_CLASS_NAMES[obs_class],
                    "qc": "clean",
                    "elevation_band": "40-50",
                    "kindex_band": "clear_gt0.65",
                    "obs_dhi": obs_dhi,
                    "pred_dhi": obs_dhi + offset,
                    "obs_kindex": 0.7 + 0.05 * row,
                    "pred_kindex": 0.7 + 0.05 * row + 0.01 * (1 if frame % 2 else -1),
                    "obs_sky": obs_class,
                    "pred_sky": obs_class,
                    **{
                        f"prob_sky_{name}": float(probabilities[index])
                        for index, name in enumerate(SKY_CLASS_NAMES)
                    },
                    "pred_kt": 0.6,
                    "pred_sky_kt": obs_class,
                    "persistence_dhi": previous_obs,
                    "clearsky_dhi": CLEARSKY_DHI,
                    "persistence_kindex": np.nan,
                    "clearsky_kindex": 1.0,
                }
            )
            previous_obs = obs_dhi
    return pd.DataFrame.from_records(records)


def _eval_metrics(
    name: str,
    model: str,
    n: int,
    *,
    split_id: str = SPLIT_ID,
    targets: tuple[str, ...] = ("dhi", "kindex", "sky"),
    tta_rotations: int = 0,
    dhi_rmse: float = 10.0,
    kindex_mae: float = 0.1,
) -> dict[str, Any]:
    per_class = {
        class_name: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 5}
        for class_name in SKY_CLASS_NAMES
    }
    return {
        "checkpoint_path": f"output/experiments/{name}/run/best.ckpt",
        "split": "test",
        "n_samples": n,
        "enabled_targets": list(targets),
        "meta": {
            "name": name,
            "model": model,
            "feature_set": "bare",
            "input_mode": "image",
            "split_id": split_id,
            "manifest_sha256": MANIFEST_SHA,
            "kindex_kind": "kstar",
            "tta_rotations": tta_rotations,
        },
        "global": {
            "dhi": {
                "rmse": dhi_rmse,
                "mae": dhi_rmse * 0.7,
                "mbe": 0.5,
                "r2": 0.9,
                "n": n,
                "rmse_persistence": 1.0,
                "skill_persistence": -0.6,
            },
            "kindex": {"rmse": kindex_mae * 1.3, "mae": kindex_mae, "mbe": 0.01, "r2": 0.8, "n": n},
            "sky": {
                "accuracy": 0.7,
                "balanced_accuracy": 0.65,
                "macro_f1": 0.6,
                "kappa_quadratic": 0.8,
                "n": n,
                "confusion": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "per_class": per_class,
            },
        },
    }


def _write_report(
    report_dir: Path, metrics: dict[str, Any], predictions: pd.DataFrame | None
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (report_dir / "stratified.csv").write_text(
        "target,stratum_kind,stratum,metric,value,n\n"
        "dhi,overall,all,rmse,10.0,30\n"
        "dhi,solar_elevation,40-50,rmse,9.5,30\n"
        "dhi,solar_elevation,40-50,mae,7.0,30\n"
        "dhi,solar_elevation,40-50,mbe,0.4,30\n"
        "dhi,sky_class,clear,rmse,8.0,10\n"
        "kindex,sky_class,clear,rmse,0.1,10\n",
        encoding="utf-8",
    )
    if predictions is not None:
        predictions.to_parquet(report_dir / "predictions.parquet", index=False)


def _write_dataset(dataset_dir: Path, *, manifest_sha: str = MANIFEST_SHA) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    assignment = {
        **dict.fromkeys(TRAIN_DAYS, "train"),
        **dict.fromkeys(VAL_DAYS, "val"),
        **dict.fromkeys(TEST_DAYS, "test"),
    }
    rows: list[dict[str, Any]] = []
    for day, split in assignment.items():
        top = SPLIT_ELEVATION_MAX[split]
        for index, sky_class in enumerate((0, 3, 3, -1)):
            rows.append(
                {
                    "day_id": day,
                    "solar_elevation": 10.0 + index * (top - 10.0) / 3,
                    "sky_class": sky_class,
                }
            )
    pd.DataFrame.from_records(rows).to_parquet(dataset_dir / "manifest.parquet", index=False)
    (dataset_dir / "manifest.parquet.meta.json").write_text(
        json.dumps(
            {
                "manifest_sha256": manifest_sha,
                "row_count": len(rows),
                "target_source": "measured",
                "thresholds": {"min_elevation_deg": 10.0},
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "splits.json").write_text(
        json.dumps(
            {
                "split_id": SPLIT_ID,
                "strategy": "chronological",
                "gap_days": 1,
                "assignment": assignment,
            }
        ),
        encoding="utf-8",
    )


def _write_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "epoch,train_kindex_mae,val_kindex_mae,train_dhi_mae,val_dhi_mae,"
        "train_sky_balanced_acc,val_sky_balanced_acc\n"
        "1,0.10,0.09,20.0,18.0,0.60,0.61\n"
        "2,0.08,0.085,18.0,17.5,0.64,0.65\n",
        encoding="utf-8",
    )


def _write_inputs(tmp_path: Path) -> dict[str, Any]:
    n = len(TEST_DAYS) * ROWS_PER_DAY * FRAMES_PER_ROW
    checkpoints = []
    for name, seed in MEMBERS:
        report_dir = tmp_path / name / "eval-test"
        _write_report(report_dir, _eval_metrics(name, "image_only", n), _predictions(seed))
        checkpoints.append(pinned(tmp_path / name / "best.ckpt", DIGEST, report_dir))
    controls = {}
    for control_id, rmse, mae in (("sensor_only", 87.6, 0.21), ("climatology", 73.6, 0.27)):
        report_dir = tmp_path / "controls" / control_id / "eval-test"
        _write_report(
            report_dir,
            _eval_metrics(control_id, control_id, n, dhi_rmse=rmse, kindex_mae=mae),
            None,
        )
        controls[control_id] = control(
            tmp_path / "controls" / control_id / "best.ckpt", DIGEST, report_dir
        )
    _write_dataset(tmp_path / "dataset")
    _write_history(tmp_path / MEMBERS[0][0] / "metrics.csv")
    return serving_pin_payload(
        frame_checkpoints=checkpoints,
        sensor_only=controls["sensor_only"],
        climatology=controls["climatology"],
        dataset=tmp_path / "dataset",
        training_history=tmp_path / MEMBERS[0][0] / "metrics.csv",
    )


def _members(tmp_path: Path) -> list[CheckpointMetadata]:
    return [_metadata(tmp_path, name, seed) for name, seed in MEMBERS]


def _build(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return build_model_card(
        ServingConfig.model_validate(payload), _members(tmp_path), stamp=publish_stamp()
    )


def _rewrite_metrics(report_dir: str, **changes: Any) -> None:
    path = Path(report_dir) / "eval_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        holder, _, leaf = key.rpartition("__")
        target = metrics[holder] if holder else metrics
        target[leaf] = value
    path.write_text(json.dumps(metrics), encoding="utf-8")


def _strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [text for value in node.values() for text in _strings(value)]
    if isinstance(node, list):
        return [text for value in node for text in _strings(value)]
    return [node] if isinstance(node, str) else []


def _keys(node: Any) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {key for value in node.values() for key in _keys(value)}
    if isinstance(node, list):
        return {key for value in node for key in _keys(value)}
    return set()


def test_persistence_is_the_previous_paired_logger_row_not_the_previous_frame(tmp_path):
    payload = _write_inputs(tmp_path)
    frame_shifted = _predictions(42)["persistence_dhi"].to_numpy()
    observed = _predictions(42)["obs_dhi"].to_numpy()
    shifted_rows = np.isfinite(frame_shifted)
    frame_shifted_rmse = float(
        np.sqrt(np.mean((frame_shifted[shifted_rows] - observed[shifted_rows]) ** 2))
    )
    steps = [
        ROW_STEP_DHI[day] for day in TEST_DAYS for _ in range((ROWS_PER_DAY - 1) * FRAMES_PER_ROW)
    ]
    paired_rmse = float(np.sqrt(np.mean(np.square(steps))))

    persistence = _build(tmp_path, payload)["evaluation"]["references"]["persistence"]

    assert persistence["dhi_rmse"] == pytest.approx(paired_rmse, abs=0.01)
    assert persistence["dhi_rmse"] != pytest.approx(frame_shifted_rmse, abs=0.5)
    assert persistence["n"] == len(steps)
    assert persistence["n_rows"] == len(TEST_DAYS) * (ROWS_PER_DAY - 1)
    assert persistence["horizon_minutes"] == 5


def test_a_frame_stamped_on_a_five_minute_boundary_belongs_to_the_row_ending_there():
    local = pd.Series(
        pd.to_datetime(["2026-08-22 06:35:00", "2026-08-22 06:35:01", "2026-08-22 06:39:59"])
    )

    ends = paired_row_ends(local)

    assert ends.tolist() == [
        pd.Timestamp("2026-08-22 06:35:00"),
        pd.Timestamp("2026-08-22 06:40:00"),
        pd.Timestamp("2026-08-22 06:40:00"),
    ]


def test_the_persistence_skill_is_computed_over_the_paired_rows_only(tmp_path):
    payload = _write_inputs(tmp_path)

    skill = _build(tmp_path, payload)["evaluation"]["skill"]

    assert skill["vs_persistence"]["n"] == len(TEST_DAYS) * (ROWS_PER_DAY - 1) * FRAMES_PER_ROW
    assert skill["vs_persistence"]["n_rows"] == len(TEST_DAYS) * (ROWS_PER_DAY - 1)
    assert skill["vs_clearsky"]["n"] == len(TEST_DAYS) * ROWS_PER_DAY * FRAMES_PER_ROW
    assert skill["vs_sensor_only"]["n"] == skill["vs_climatology"]["n"] == skill["vs_clearsky"]["n"]


def test_the_evaluators_frame_shifted_persistence_is_not_published(tmp_path):
    payload = _write_inputs(tmp_path)

    evaluation = _build(tmp_path, payload)["evaluation"]

    assert all("rmse_persistence" not in arm["dhi"] for arm in evaluation["arms"])
    assert all("skill_persistence" not in arm["dhi"] for arm in evaluation["arms"])
    assert evaluation["references"]["persistence"]["dhi_rmse"] != 1.0
    assert evaluation["skill"]["vs_persistence"]["skill"] != -0.6


def test_a_control_evaluated_on_another_split_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _rewrite_metrics(payload["controls"]["sensor_only"]["report"], meta__split_id="d" * 64)

    with pytest.raises(ModelCardError, match=r"control sensor_only.*split"):
        _build(tmp_path, payload)


def test_a_control_evaluated_on_other_rows_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _rewrite_metrics(payload["controls"]["climatology"]["report"], n_samples=7)

    with pytest.raises(ModelCardError, match=r"control climatology.*rows"):
        _build(tmp_path, payload)


def test_a_report_written_with_test_time_rotations_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _rewrite_metrics(payload["frame_checkpoints"][1]["report"], meta__tta_rotations=4)

    with pytest.raises(ModelCardError, match="rotation"):
        _build(tmp_path, payload)


def test_a_dataset_directory_with_another_manifest_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _write_dataset(tmp_path / "dataset", manifest_sha="e" * 64)

    with pytest.raises(ModelCardError, match="manifest"):
        _build(tmp_path, payload)


def test_per_day_rows_sum_to_the_evaluation_n(tmp_path):
    payload = _write_inputs(tmp_path)

    evaluation = _build(tmp_path, payload)["evaluation"]

    assert sum(row["n_frames"] for row in evaluation["per_day"]) == evaluation["n"]
    assert all(row["n"] <= row["n_frames"] for row in evaluation["per_day"])
    assert [row["date"] for row in evaluation["per_day"]] == [
        f"{day}T00:00:00" for day in TEST_DAYS
    ]


def test_per_day_class_shares_sum_to_one(tmp_path):
    payload = _write_inputs(tmp_path)

    per_day = _build(tmp_path, payload)["evaluation"]["per_day"]

    for row in per_day:
        assert sum(row["class_share"].values()) == pytest.approx(1.0, abs=0.002)


def test_no_string_in_the_card_is_a_filesystem_path(tmp_path):
    payload = _write_inputs(tmp_path)

    card = _build(tmp_path, payload)

    assert [text for text in _strings(card) if text.startswith(("/", "output/"))] == []


def test_ensemble_members_are_listed_by_name(tmp_path):
    payload = _write_inputs(tmp_path)

    card = _build(tmp_path, payload)

    assert card["evaluation"]["arms"][0]["members"] == [name for name, _ in MEMBERS]
    assert [member["name"] for member in card["served"]["members"]] == [name for name, _ in MEMBERS]


def test_an_image_only_arm_declares_its_scalars_not_consumed(tmp_path):
    payload = _write_inputs(tmp_path)

    inputs = _build(tmp_path, payload)["served"]["inputs"]

    assert inputs["scalars_consumed"] is False
    assert inputs["radiometry_forbidden"] is True
    assert inputs["scalar_columns"][:2] == ["solar_elevation", "solar_zenith"]
    assert inputs["preprocess"]["overlay"] == "keep"


def test_split_blocks_carry_the_elevation_range_and_the_class_share(tmp_path):
    payload = _write_inputs(tmp_path)

    split = _build(tmp_path, payload)["dataset"]["split"]

    assert split["train"]["solar_elevation_range_deg"] == {"min": 10.0, "max": 59.8}
    assert split["test"]["solar_elevation_range_deg"]["max"] == 69.5
    assert split["test"]["class_share"] == {"i": 0.333, "ii": 0.0, "iii": 0.0, "iv": 0.667}
    assert split["test"]["rows"] == len(TEST_DAYS) * 4


def test_seeds_report_a_range_and_no_standard_deviation(tmp_path):
    payload = _write_inputs(tmp_path)

    seeds = _build(tmp_path, payload)["seeds"]

    assert seeds["n_seeds"] == 2
    assert set(seeds["range"]["dhi_rmse"]) == {"min", "max"}
    assert "summary" not in seeds
    assert "sd" not in _keys(seeds)


def test_per_class_scores_are_keyed_by_condition_id(tmp_path):
    payload = _write_inputs(tmp_path)

    evaluation = _build(tmp_path, payload)["evaluation"]

    assert list(evaluation["per_class"]) == ["i", "ii", "iii", "iv"]
    assert evaluation["per_class"]["iv"]["support"] == len(TEST_DAYS) * FRAMES_PER_ROW
    assert list(evaluation["arms"][3]["sky"]["per_class"]) == ["i", "ii", "iii", "iv"]


def test_domain_check_is_null_until_pinned(tmp_path):
    payload = _write_inputs(tmp_path)

    card = _build(tmp_path, payload)

    assert card["domain_check"] is None
    assert any("domain_check" in caveat for caveat in card["caveats"])


def test_a_pinned_domain_check_is_published_with_its_five_keys(tmp_path):
    payload = _write_inputs(tmp_path)
    check = tmp_path / "domain-check.json"
    check.write_text(
        json.dumps(
            {
                "day": "2026-09-12",
                "n": 310,
                "dhi_mbe": 2.345,
                "dhi_rmse": 9.876,
                "class_agreement": 0.8123,
            }
        ),
        encoding="utf-8",
    )
    payload["reports"]["domain_check"] = str(check)

    card = _build(tmp_path, payload)

    assert card["domain_check"] == {
        "day": "2026-09-12",
        "n": 310,
        "dhi_mbe": 2.35,
        "dhi_rmse": 9.88,
        "class_agreement": 0.812,
    }


def test_the_clearsky_reference_is_scored_on_the_clear_rows_too(tmp_path):
    payload = _write_inputs(tmp_path)
    clear_row_obs = [FIRST_ROW_DHI[day] + 2 * ROW_STEP_DHI[day] for day in TEST_DAYS]
    expected_mbe = float(np.mean([CLEARSKY_DHI - obs for obs in clear_row_obs]))

    clearsky = _build(tmp_path, payload)["evaluation"]["references"]["clearsky"]

    assert clearsky["on_clear_rows"]["n"] == len(TEST_DAYS) * FRAMES_PER_ROW
    assert clearsky["on_clear_rows"]["mbe"] == pytest.approx(expected_mbe, abs=0.01)
    assert clearsky["label"].startswith("Haurwitz")


def test_the_card_names_the_selection_split_and_the_single_pass_inference(tmp_path):
    payload = _write_inputs(tmp_path)

    card = _build(tmp_path, payload)

    assert card["served"]["selection"]["selection_split"] == "val"
    assert card["evaluation"]["inference"] == "single_pass"
    assert card["training_curve"]["best_epoch"] == 6
    assert card["attribution_summary"]["target"] == "kindex"


def test_the_caveats_compare_the_controls_from_their_own_numbers(tmp_path):
    payload = _write_inputs(tmp_path)

    caveats = _build(tmp_path, payload)["caveats"]

    assert any("pior que a média do treino na difusa" in caveat for caveat in caveats)
    assert not any("negativo" in caveat for caveat in caveats)


def test_a_control_trained_on_other_targets_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _rewrite_metrics(
        payload["controls"]["sensor_only"]["report"], enabled_targets=["dhi", "kindex"]
    )

    with pytest.raises(ModelCardError, match="targets"):
        _build(tmp_path, payload)


def test_a_report_paired_with_another_sensor_offset_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    _rewrite_metrics(
        payload["frame_checkpoints"][0]["report"], meta__sensor_timestamp_offset_minutes=0.0
    )

    with pytest.raises(ModelCardError, match="offset"):
        _build(tmp_path, payload)


def test_a_missing_member_report_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    (Path(payload["frame_checkpoints"][0]["report"]) / "eval_metrics.json").unlink()

    with pytest.raises(ModelCardError, match="cannot read"):
        _build(tmp_path, payload)


def test_a_member_without_predictions_is_refused(tmp_path):
    payload = _write_inputs(tmp_path)
    (Path(payload["frame_checkpoints"][1]["report"]) / "predictions.parquet").unlink()

    with pytest.raises(ModelCardError, match="predictions"):
        _build(tmp_path, payload)


def test_members_trained_on_different_manifests_are_refused(tmp_path):
    from dataclasses import replace

    payload = _write_inputs(tmp_path)
    members = _members(tmp_path)
    stranger = replace(members[1], manifest_sha256="f" * 64)

    with pytest.raises(ModelCardError, match="cannot be averaged"):
        build_model_card(
            ServingConfig.model_validate(payload), [members[0], stranger], stamp=publish_stamp()
        )


def test_dataset_rows_and_days_describe_the_same_manifest(tmp_path):
    payload = _write_inputs(tmp_path)

    dataset = _build(tmp_path, payload)["dataset"]

    assert dataset["rows"] == sum(
        dataset["split"][name]["rows"] for name in ("train", "val", "test")
    )
    assert dataset["days"] == dataset["days_assigned"]
