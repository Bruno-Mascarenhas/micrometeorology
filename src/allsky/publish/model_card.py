"""``model.json``: the served network, how it was chosen and how it scores.

Everything here is read from artifacts the training and evaluation commands
already write — ``eval_metrics.json``, ``stratified.csv``,
``predictions.parquet``, ``metrics.csv``, ``splits.json``, the manifest and its
sidecar — plus the checkpoints' own provenance. Nothing is re-evaluated except
the served ensemble and the two references: the members' per-sample
predictions are averaged and scored with the same functions the evaluator
used, and persistence is recomputed over the **paired logger row**
(:func:`paired_row_ends`), because the evaluator's frame-shifted persistence
is the target itself on most rows of a one-frame-per-minute manifest.

The card refuses to build (:class:`ModelCardError`) when a report was written
for another split, another manifest or with test-time rotations the watch
never runs, when a control was evaluated on other targets or other rows, or
when the dataset directory disagrees with the served checkpoints: a card whose
numbers are not about the same rows is worse than no card.

Torch-free: the checkpoint provenance arrives as :class:`CheckpointMetadata`,
which :func:`checkpoint_metadata` (the one torch-touching function here, with a
lazy import) extracts. The manifest's ``timestamp_utc`` is converted to the
station's naive local clock only to find the logger row a frame was paired to.
"""

import csv
import datetime as dt
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from allsky.attribution import OCCLUSION_STRIDE_PX, OCCLUSION_WINDOW_PX
from allsky.config import (
    DATASET_MANIFEST_FILENAME,
    DATASET_SPLIT_FILENAME,
    SITE_TZ,
    ExperimentConfig,
    image_size_of,
    manifest_meta_path,
)
from allsky.data.blocks import block_ends
from allsky.evaluation.metrics import classification_metrics, regression_metrics, skill_score
from allsky.evaluation.persistence import previous_logger_row
from allsky.publish.encoding import (
    DAY_STAMP_FORMAT,
    ELEVATION_DECIMALS,
    INDEX_DECIMALS,
    IRRADIANCE_DECIMALS,
    MODEL_SCHEMA,
    REFERENCES,
    PublishStamp,
    class_share,
    condition_of,
    document_header,
    rounded_or_none,
    sky_conditions_block,
    targets_glossary,
)
from allsky.serving import ServingConfig
from labmim_core.atomic import JsonObjectError, read_json_object
from labmim_core.sky import SKY_CLASS_COUNT, SKY_CLASS_NAMES, SKY_CLEAR

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointMetadata",
    "ModelCardError",
    "build_model_card",
    "card_inputs",
    "checkpoint_metadata",
    "paired_row_ends",
]

EVALUATION_METRICS_FILENAME = "eval_metrics.json"
STRATIFIED_FILENAME = "stratified.csv"
PREDICTIONS_FILENAME = "predictions.parquet"
MANIFEST_SPLIT_COLUMNS = ("day_id", "solar_elevation", "sky_class")
SPLIT_NAMES = ("train", "val", "test")
SPLIT_NAMES_PT = {"train": "treino", "val": "validação", "test": "teste"}

SCORE_DECIMALS = 3
CURVE_DECIMALS = 4

# local_prepare_iso_20260906.yaml: sensor_timestamp_offset_minutes -2.5, interval 5 min
SENSOR_TIMESTAMP_OFFSET_MINUTES = -2.5
LOGGER_INTERVAL_MINUTES = 5
PERSISTENCE_HORIZON_MINUTES = LOGGER_INTERVAL_MINUTES

ATTRIBUTION_METHOD = "occlusion_sensitivity"
ATTRIBUTION_FILL = "network_mean_level"
ATTRIBUTION_WINDOW_PX = OCCLUSION_WINDOW_PX
ATTRIBUTION_STRIDE_PX = OCCLUSION_STRIDE_PX

INFERENCE_MODE = "single_pass"
SCALAR_FREE_ARCHITECTURES = ("image_only",)
DOMAIN_CHECK_KEYS = ("day", "n", "dhi_mbe", "dhi_rmse", "class_agreement")
CODE_VERSION_KEYS = ("package_version", "git_commit")
BR_DATE_FORMAT = "%d/%m/%Y"

REGRESSION_KEYS = ("rmse", "mae", "mbe", "r2", "n")
CLASSIFICATION_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "kappa_quadratic", "n")
PER_CLASS_SCORES = ("precision", "recall", "f1")
STRATUM_KINDS = ("solar_elevation", "sky_class")
STRATUM_METRICS = ("rmse", "mae", "mbe")
TRAINING_CURVE_COLUMNS = (
    "train_kindex_mae",
    "val_kindex_mae",
    "train_dhi_mae",
    "val_dhi_mae",
    "train_sky_balanced_acc",
    "val_sky_balanced_acc",
)
PROBABILITY_COLUMNS = tuple(f"prob_sky_{name}" for name in SKY_CLASS_NAMES)
SEED_RANGE_DECIMALS = {
    "dhi_rmse": IRRADIANCE_DECIMALS,
    "dhi_mae": IRRADIANCE_DECIMALS,
    "dhi_mbe": IRRADIANCE_DECIMALS,
    "kindex_mae": INDEX_DECIMALS,
    "sky_balanced_accuracy": SCORE_DECIMALS,
    "sky_macro_f1": SCORE_DECIMALS,
    "sky_kappa_quadratic": SCORE_DECIMALS,
    "persistence_skill_dhi": SCORE_DECIMALS,
}

CONTROL_LABELS = {
    "sensor_only": "Controle sem imagem: MLP sobre os nove escalares não radiométricos (geometria solar e anemômetro)",
    "climatology": "Climatologia: média do treino por alvo e frequência das classes",
}
REFERENCE_LABELS = {
    "clearsky": "Haurwitz GHI × fração difusa de Erbs no kt de céu claro",  # noqa: RUF001 - the contract names the label with the multiplication sign
    "persistence": "Persistência de 5 minutos: a linha anterior do datalogger, pareada ao quadro, no mesmo dia",
}
CAMERA_LABEL = "Câmera all-sky do Planetário e Observatório da UFBA"
FRAMES_FROM_LABEL = "timelapse H.264 diário, um quadro por minuto de captura"


class ModelCardError(ValueError):
    """A report does not describe the rows, targets or weights the card is about."""


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """What the card needs from one checkpoint's payload.

    Attributes
    ----------
    config:
        The experiment the checkpoint was trained under, as embedded in it.
    best_metric:
        ``{"name", "value", "epoch"}`` of the early-stopping monitor.
    n_parameters:
        Trainable and frozen parameters together, from ``model_state``.
    """

    path: Path
    sha256: str
    config: ExperimentConfig
    epoch: int
    best_metric: Mapping[str, Any]
    code_version: Mapping[str, Any] | None
    dataset_version: str | None
    split_id: str
    manifest_sha256: str
    frame_geometry: Mapping[str, Any] | None
    backbone: Mapping[str, Any] | None
    n_parameters: int


@dataclass(frozen=True, slots=True)
class _ReferenceSkill:
    reference_rmse: float | None
    model_rmse: float | None
    skill: float | None
    paired: np.ndarray


def checkpoint_metadata(
    path: str | Path,
    sha256: str,
    *,
    trust_checkpoint: bool = False,
    payload: Mapping[str, Any] | None = None,
) -> CheckpointMetadata:
    """Read a checkpoint's provenance without building its model.

    Parameters
    ----------
    path:
        ``best.ckpt`` / ``last.ckpt`` written by ``allsky train``.
    sha256:
        The digest the pin names for it, carried into the card unchanged.
    trust_checkpoint:
        Read with the unrestricted unpickler (own files only).
    payload:
        The checkpoint already loaded, when the caller holds it; saves the
        read of a file the publisher has resident for the served member.
    """
    from allsky.training.checkpointing import load_checkpoint

    if payload is None:
        payload = load_checkpoint(path, map_location="cpu", trust_pickle=trust_checkpoint)
    state = payload["model_state"]
    n_parameters = int(sum(int(np.prod(tuple(tensor.shape))) for tensor in state.values()))
    return CheckpointMetadata(
        path=Path(path),
        sha256=sha256,
        config=ExperimentConfig.model_validate(payload["config"]),
        epoch=int(payload["epoch"]),
        best_metric=dict(payload.get("best_metric") or {}),
        code_version=payload.get("code_version"),
        dataset_version=(
            str(payload["dataset_version"]) if payload.get("dataset_version") is not None else None
        ),
        split_id=str(payload["split_id"]),
        manifest_sha256=str(payload["manifest_sha256"]),
        frame_geometry=payload.get("frame_geometry"),
        backbone=payload.get("backbone"),
        n_parameters=n_parameters,
    )


def paired_row_ends(local_times: pd.Series) -> pd.Series:
    """End stamp of the logger row each frame was paired to when the manifest was built.

    The CR5000 end-stamps its 5-minute means, so the row stamped ``T`` averages
    ``(T - 5 min, T]`` and its centre is ``T + offset`` with the manifest's
    ``sensor_timestamp_offset_minutes`` of -2.5. The manifest pairs a frame to
    the row whose centre is nearest, resolving an exact tie to the earlier row
    (:class:`allsky.data.alignment.NearestAlignment` keeps the left neighbour
    unless the right one is strictly closer), which is the ceil of the frame
    time to the interval: a frame stamped exactly ``hh:m5:00`` belongs to the
    row ending there.

    Parameters
    ----------
    local_times:
        ``(N,)`` naive station-local frame times, ``datetime64[ns]``.

    Returns
    -------
    pandas.Series
        ``(N,)`` naive station-local row ends, multiples of 5 minutes.
    """
    ends = block_ends(pd.DatetimeIndex(local_times), LOGGER_INTERVAL_MINUTES)
    return pd.Series(ends, index=local_times.index)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return read_json_object(path)
    except JsonObjectError as exc:
        raise ModelCardError(str(exc)) from exc


def card_inputs(pin: ServingConfig) -> list[Path]:
    """Every file :func:`build_model_card` reads for *pin*, so a publisher can tell when none moved.

    The checkpoints themselves are covered by the pin's digests, not listed.
    """
    report_files = (EVALUATION_METRICS_FILENAME, PREDICTIONS_FILENAME, STRATIFIED_FILENAME)
    dataset_dir = Path(pin.reports.dataset)
    manifest = dataset_dir / DATASET_MANIFEST_FILENAME
    inputs = [
        *(Path(member.report) / name for member in pin.frame_checkpoints for name in report_files),
        Path(pin.controls.sensor_only.report) / EVALUATION_METRICS_FILENAME,
        Path(pin.controls.climatology.report) / EVALUATION_METRICS_FILENAME,
        Path(pin.reports.training_history),
        manifest,
        manifest_meta_path(manifest),
        dataset_dir / DATASET_SPLIT_FILENAME,
    ]
    if pin.reports.domain_check is not None:
        inputs.append(Path(pin.reports.domain_check))
    return inputs


def _read_report(report_dir: Path) -> dict[str, Any]:
    return _read_json(Path(report_dir) / EVALUATION_METRICS_FILENAME)


def _check_report_is_about(
    report: Mapping[str, Any], served: CheckpointMetadata, *, what: str
) -> None:
    meta = report.get("meta") or {}
    if meta.get("split_id") != served.split_id:
        raise ModelCardError(
            f"{what}: evaluated on split {meta.get('split_id')}, the served checkpoints trained "
            f"on {served.split_id}"
        )
    if meta.get("manifest_sha256") != served.manifest_sha256:
        raise ModelCardError(
            f"{what}: evaluated on manifest {meta.get('manifest_sha256')}, the served "
            f"checkpoints trained on {served.manifest_sha256}"
        )
    if report.get("split") != "test":
        raise ModelCardError(
            f"{what}: the card reports the test split, this report is {report.get('split')!r}"
        )
    rotations = int(meta.get("tta_rotations") or 0)
    if rotations != 0:
        raise ModelCardError(
            f"{what}: evaluated with {rotations} test-time rotation(s); the watch scores one "
            "pass per frame and the card describes that estimator only"
        )
    offset = meta.get("sensor_timestamp_offset_minutes")
    if offset is not None and float(offset) != SENSOR_TIMESTAMP_OFFSET_MINUTES:
        raise ModelCardError(
            f"{what}: the manifest paired frames with a {offset} min sensor offset, the paired-row "
            f"persistence here assumes {SENSOR_TIMESTAMP_OFFSET_MINUTES} min"
        )


def _check_same_targets(
    control: Mapping[str, Any], served: Mapping[str, Any], *, what: str
) -> None:
    if sorted(control.get("enabled_targets") or []) != sorted(served.get("enabled_targets") or []):
        raise ModelCardError(
            f"{what}: trained on targets {control.get('enabled_targets')}, the served arm on "
            f"{served.get('enabled_targets')}"
        )


def _check_same_rows(control: Mapping[str, Any], n_served: int, *, what: str) -> None:
    n_control = int(control.get("n_samples") or 0)
    if n_control != n_served:
        raise ModelCardError(
            f"{what}: evaluated over {n_control} rows, the served ensemble over {n_served}; a skill "
            "against a control scored on other rows compares nothing"
        )


def _regression_block(metrics: Mapping[str, Any], *, decimals: int) -> dict[str, Any]:
    block: dict[str, Any] = {}
    for key in REGRESSION_KEYS:
        value = metrics.get(key)
        if key == "n":
            block[key] = int(value) if value is not None else None
        elif key == "r2":
            block[key] = rounded_or_none(value, SCORE_DECIMALS)
        else:
            block[key] = rounded_or_none(value, decimals)
    return block


def _confusion(matrix: object) -> list[list[int]] | None:
    if not isinstance(matrix, list):
        return None
    return [[int(value) for value in row] for row in matrix]


def _per_class_by_condition(
    per_class: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    if not per_class:
        return None
    by_condition: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(SKY_CLASS_NAMES):
        scores = per_class.get(name) or {}
        support = scores.get("support")
        by_condition[condition_of(index)["id"]] = {
            **{key: rounded_or_none(scores.get(key), SCORE_DECIMALS) for key in PER_CLASS_SCORES},
            "support": int(support) if support is not None else None,
        }
    return by_condition


def _classification_block(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return None
    block: dict[str, Any] = {}
    for key in CLASSIFICATION_KEYS:
        value = metrics.get(key)
        block[key] = (
            int(value)
            if key == "n" and value is not None
            else rounded_or_none(value, SCORE_DECIMALS)
        )
    block["confusion"] = _confusion(metrics.get("confusion"))
    block["per_class"] = _per_class_by_condition(metrics.get("per_class"))
    return block


def _arm_from_report(
    arm_id: str, kind: str, label: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    scores = report.get("global") or {}
    return {
        "id": arm_id,
        "kind": kind,
        "label": label,
        "n": int(report.get("n_samples") or 0),
        "dhi": _regression_block(scores.get("dhi") or {}, decimals=IRRADIANCE_DECIMALS),
        "kindex": _regression_block(scores.get("kindex") or {}, decimals=INDEX_DECIMALS),
        "sky": _classification_block(scores.get("sky")),
    }


def _member_predictions(report_dirs: Sequence[Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for report_dir in report_dirs:
        path = Path(report_dir) / PREDICTIONS_FILENAME
        if not path.is_file():
            raise ModelCardError(
                f"{path} is missing: evaluate the member with --predictions so the ensemble can "
                "be scored"
            )
        frames.append(pd.read_parquet(path).set_index("sample_id").sort_index())
    first = frames[0].index
    for report_dir, frame in zip(report_dirs, frames, strict=True):
        if not frame.index.equals(first):
            raise ModelCardError(f"{report_dir}: covers other samples than {report_dirs[0]}")
    return frames


def _column_mean(frames: Sequence[pd.DataFrame], column: str) -> np.ndarray:
    return np.mean([frame[column].to_numpy(dtype=np.float64) for frame in frames], axis=0)


def _station_local(timestamp_utc: pd.Series) -> pd.Series:
    aware = pd.to_datetime(timestamp_utc, utc=True)
    return aware.dt.tz_convert(SITE_TZ).dt.tz_localize(None)


def _scored_rows(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """One row per test sample: observations, the mean prediction and the two references.

    The persistence reference of a frame is the previous paired logger row's
    observation of the same day. The diffuse mean is one value per row by
    construction; the clear-sky index is not, because its denominator moves
    with the sun inside the five minutes, so the row's mean k* stands for it.
    """
    first = frames[0]
    local = _station_local(first["timestamp_utc"])
    rows = pd.DataFrame(
        {
            "day_id": first["day_id"].astype(str).to_numpy(),
            "row_end": paired_row_ends(local).to_numpy(),
            "obs_dhi": first["obs_dhi"].to_numpy(dtype=np.float64),
            "pred_dhi": _column_mean(frames, "pred_dhi"),
            "obs_kindex": first["obs_kindex"].to_numpy(dtype=np.float64),
            "pred_kindex": _column_mean(frames, "pred_kindex"),
            "obs_sky": first["obs_sky"].to_numpy(dtype=np.int64),
            "clearsky_dhi": first["clearsky_dhi"].to_numpy(dtype=np.float64),
        },
        index=first.index,
    )
    for column in PROBABILITY_COLUMNS:
        rows[column] = _column_mean(frames, column)
    rows["persistence_dhi"] = previous_logger_row(
        rows, "obs_dhi", interval_minutes=LOGGER_INTERVAL_MINUTES, one_value_per_row=True
    )
    rows["persistence_kindex"] = previous_logger_row(
        rows, "obs_kindex", interval_minutes=LOGGER_INTERVAL_MINUTES, one_value_per_row=False
    )
    return rows


def _rmse(predicted: np.ndarray, observed: np.ndarray) -> float:
    """:func:`regression_metrics`' RMSE, so the card and the reports agree to the pair."""
    return float(regression_metrics(observed, predicted)["rmse"])


def _reference_skill(
    observed: np.ndarray, predicted: np.ndarray, reference: np.ndarray
) -> _ReferenceSkill:
    paired = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(reference)
    if not paired.any():
        return _ReferenceSkill(None, None, None, paired)
    reference_rmse = _rmse(reference[paired], observed[paired])
    model_rmse = _rmse(predicted[paired], observed[paired])
    return _ReferenceSkill(
        reference_rmse, model_rmse, skill_score(model_rmse, reference_rmse), paired
    )


def _reference_document(skill: _ReferenceSkill, *, rmse_key: str, decimals: int) -> dict[str, Any]:
    return {
        rmse_key: rounded_or_none(skill.reference_rmse, decimals),
        "model_rmse_on_paired": rounded_or_none(skill.model_rmse, decimals),
        "skill": rounded_or_none(skill.skill, SCORE_DECIMALS),
        "n": int(skill.paired.sum()),
    }


def _n_rows(rows: pd.DataFrame, paired: np.ndarray) -> int:
    return len(rows.loc[paired, ["day_id", "row_end"]].drop_duplicates())


def _ensemble_arm(
    pin: ServingConfig, rows: pd.DataFrame, members: Sequence[CheckpointMetadata]
) -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
    dhi = regression_metrics(rows["obs_dhi"].to_numpy(), rows["pred_dhi"].to_numpy())
    kindex = regression_metrics(rows["obs_kindex"].to_numpy(), rows["pred_kindex"].to_numpy())
    probabilities = rows[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    sky = classification_metrics(
        rows["obs_sky"].to_numpy(),
        probabilities.argmax(axis=1),
        SKY_CLASS_COUNT,
        probabilities=probabilities,
    )
    arm = {
        "id": "served",
        "kind": "ensemble",
        "label": pin.label,
        "members": [member.config.name for member in members],
        "n": len(rows),
        "dhi": _regression_block(dhi, decimals=IRRADIANCE_DECIMALS),
        "kindex": _regression_block(kindex, decimals=INDEX_DECIMALS),
        "sky": _classification_block(sky),
    }
    return arm, dhi, sky


def _references(rows: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = rows["obs_dhi"].to_numpy(dtype=np.float64)
    predicted = rows["pred_dhi"].to_numpy(dtype=np.float64)
    clearsky_dhi = rows["clearsky_dhi"].to_numpy(dtype=np.float64)
    clearsky = _reference_skill(observed, predicted, clearsky_dhi)
    persistence = _reference_skill(
        observed, predicted, rows["persistence_dhi"].to_numpy(dtype=np.float64)
    )
    persistence_kindex = _reference_skill(
        rows["obs_kindex"].to_numpy(dtype=np.float64),
        rows["pred_kindex"].to_numpy(dtype=np.float64),
        rows["persistence_kindex"].to_numpy(dtype=np.float64),
    )
    clear_rows = (
        (rows["obs_sky"].to_numpy() == SKY_CLEAR)
        & np.isfinite(observed)
        & np.isfinite(clearsky_dhi)
    )
    on_clear_rows = {
        "rmse": (
            rounded_or_none(
                _rmse(clearsky_dhi[clear_rows], observed[clear_rows]), IRRADIANCE_DECIMALS
            )
            if clear_rows.any()
            else None
        ),
        "mbe": (
            rounded_or_none(
                float(np.mean(clearsky_dhi[clear_rows] - observed[clear_rows])), IRRADIANCE_DECIMALS
            )
            if clear_rows.any()
            else None
        ),
        "n": int(clear_rows.sum()),
    }
    n_rows = _n_rows(rows, persistence.paired)
    references = {
        "clearsky": {
            "label": REFERENCE_LABELS["clearsky"],
            **_reference_document(clearsky, rmse_key="dhi_rmse", decimals=IRRADIANCE_DECIMALS),
            "on_clear_rows": on_clear_rows,
        },
        "persistence": {
            "label": REFERENCE_LABELS["persistence"],
            "horizon_minutes": PERSISTENCE_HORIZON_MINUTES,
            **_reference_document(persistence, rmse_key="dhi_rmse", decimals=IRRADIANCE_DECIMALS),
            "n_rows": n_rows,
            "kindex": _reference_document(
                persistence_kindex, rmse_key="rmse", decimals=INDEX_DECIMALS
            ),
        },
    }
    skill = {
        "vs_clearsky": {
            "skill": rounded_or_none(clearsky.skill, SCORE_DECIMALS),
            "n": int(clearsky.paired.sum()),
        },
        "vs_persistence": {
            "skill": rounded_or_none(persistence.skill, SCORE_DECIMALS),
            "n": int(persistence.paired.sum()),
            "n_rows": n_rows,
        },
    }
    return references, skill


def _masked_rmse(predicted: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return rounded_or_none(_rmse(predicted[mask], observed[mask]), IRRADIANCE_DECIMALS)


def _per_day(rows: pd.DataFrame) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for day_id, day in rows.groupby("day_id", sort=True):
        observed = day["obs_dhi"].to_numpy(dtype=np.float64)
        predicted = day["pred_dhi"].to_numpy(dtype=np.float64)
        persistence = day["persistence_dhi"].to_numpy(dtype=np.float64)
        clearsky = day["clearsky_dhi"].to_numpy(dtype=np.float64)
        paired = (
            np.isfinite(observed)
            & np.isfinite(predicted)
            & np.isfinite(persistence)
            & np.isfinite(clearsky)
        )
        table.append(
            {
                "date": pd.Timestamp(str(day_id)).strftime(DAY_STAMP_FORMAT),
                "n": int(paired.sum()),
                "n_frames": len(day),
                "class_share": class_share(day["obs_sky"].to_numpy(dtype=np.int64)),
                "rmse_model": _masked_rmse(predicted, observed, paired),
                "rmse_persistence": _masked_rmse(persistence, observed, paired),
                "rmse_clearsky": _masked_rmse(clearsky, observed, paired),
            }
        )
    return table


def _skill_against_controls(
    model_rmse: float | None, control_reports: Mapping[str, Mapping[str, Any]], *, n: int
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for control_id, report in control_reports.items():
        reference_rmse = ((report.get("global") or {}).get("dhi") or {}).get("rmse")
        out[f"vs_{control_id}"] = {
            "skill": (
                rounded_or_none(
                    skill_score(float(model_rmse), float(reference_rmse)), SCORE_DECIMALS
                )
                if model_rmse is not None and reference_rmse is not None
                else None
            ),
            "n": n,
        }
    return out


def _stratified(report_dir: Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(report_dir) / STRATIFIED_FILENAME
    if not path.is_file():
        raise ModelCardError(f"{path} is missing")
    table: dict[str, dict[str, dict[str, Any]]] = {kind: {} for kind in STRATUM_KINDS}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            kind = row.get("stratum_kind", "")
            if row.get("target") != "dhi" or kind not in table:
                continue
            metric = row.get("metric", "")
            if metric not in STRATUM_METRICS:
                continue
            entry = table[kind].setdefault(
                row["stratum"], {"stratum": row["stratum"], "n": int(float(row["n"]))}
            )
            entry[metric] = rounded_or_none(row.get("value"), IRRADIANCE_DECIMALS)
    return {kind: list(entries.values()) for kind, entries in table.items()}


def _best_epoch(pin: ServingConfig, members: Sequence[CheckpointMetadata]) -> int | None:
    history_parts = set(Path(pin.reports.training_history).parts)
    named = [member for member in members if member.config.name in history_parts]
    source = named[0] if named else members[pin.attribution_checkpoint]
    epoch = source.best_metric.get("epoch")
    return int(epoch) if epoch is not None else None


def _training_curve(path: Path, *, best_epoch: int | None) -> dict[str, Any]:
    if not path.is_file():
        raise ModelCardError(f"{path} is missing")
    history = pd.read_csv(path)
    curve: dict[str, Any] = {"epochs": [int(value) for value in history["epoch"].tolist()]}
    for column in TRAINING_CURVE_COLUMNS:
        if column in history.columns:
            curve[column] = [
                rounded_or_none(value, CURVE_DECIMALS) for value in history[column].tolist()
            ]
    curve["best_epoch"] = best_epoch
    return curve


def _br_date(iso_day: str | None) -> str:
    if iso_day is None:
        return "?"
    return dt.date.fromisoformat(iso_day).strftime(BR_DATE_FORMAT)


def _season_note(start: str | None, end: str | None) -> str:
    year = dt.date.fromisoformat(start).year if start is not None else "?"
    return (
        f"Uma única estação: o inverno austral de {year}, de {_br_date(start)} a {_br_date(end)}. "
        "Nenhum dia de verão nem da estação chuvosa entra em treino, validação ou teste."
    )


def _split_block(days: Sequence[str], part: pd.DataFrame) -> dict[str, Any]:
    elevation = part["solar_elevation"].to_numpy(dtype=np.float64)
    finite = elevation[np.isfinite(elevation)]
    return {
        "days": len(days),
        "start": days[0] if days else None,
        "end": days[-1] if days else None,
        "rows": len(part),
        "solar_elevation_range_deg": {
            "min": rounded_or_none(finite.min(), ELEVATION_DECIMALS) if finite.size else None,
            "max": rounded_or_none(finite.max(), ELEVATION_DECIMALS) if finite.size else None,
        },
        "class_share": class_share(part["sky_class"].to_numpy(dtype=np.int64)),
    }


def _frame_geometry(geometry: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if geometry is None:
        return None
    mask = geometry.get("mask") or {}
    return {
        "crop": geometry.get("crop"),
        "pad": geometry.get("pad"),
        "resize": geometry.get("resize"),
        "mask": {"enabled": mask.get("path") is not None, "threshold": mask.get("threshold")},
    }


def _dataset_block(pin: ServingConfig, served: CheckpointMetadata) -> dict[str, Any]:
    dataset_dir = Path(pin.reports.dataset)
    manifest_path = dataset_dir / DATASET_MANIFEST_FILENAME
    meta = _read_json(manifest_meta_path(manifest_path))
    if meta.get("manifest_sha256") != served.manifest_sha256:
        raise ModelCardError(
            f"{dataset_dir}: manifest {meta.get('manifest_sha256')} is not the one the served "
            f"checkpoints trained on ({served.manifest_sha256})"
        )
    splits = _read_json(dataset_dir / DATASET_SPLIT_FILENAME)
    if splits.get("split_id") != served.split_id:
        raise ModelCardError(
            f"{dataset_dir}: split {splits.get('split_id')} is not the served checkpoints' "
            f"({served.split_id})"
        )
    assignment: dict[str, str] = {
        str(day): str(name) for day, name in (splits.get("assignment") or {}).items()
    }
    if not manifest_path.is_file():
        raise ModelCardError(f"{manifest_path} is missing")
    manifest = pd.read_parquet(manifest_path, columns=list(MANIFEST_SPLIT_COLUMNS))
    split_of_row = manifest["day_id"].astype(str).map(assignment)
    split_blocks: dict[str, Any] = {}
    for split_name in SPLIT_NAMES:
        days = sorted(day for day, name in assignment.items() if name == split_name)
        split_blocks[split_name] = _split_block(days, manifest.loc[split_of_row == split_name])
    thresholds = meta.get("thresholds") or {}
    manifest_days = sorted(manifest["day_id"].astype(str).unique())
    period_start = manifest_days[0] if manifest_days else None
    period_end = manifest_days[-1] if manifest_days else None
    return {
        "rows": len(manifest),
        "days": len(manifest_days),
        "days_assigned": len(assignment),
        "period": {"start": period_start, "end": period_end},
        "season_note": _season_note(period_start, period_end),
        "split": {
            "strategy": splits.get("strategy"),
            "gap_days": splits.get("gap_days"),
            **split_blocks,
        },
        "manifest_sha256": served.manifest_sha256,
        "split_id": served.split_id,
        "dataset_version": served.dataset_version,
        "target_source": meta.get("target_source"),
        "min_elevation_deg": thresholds.get("min_elevation_deg"),
        "frame_geometry": _frame_geometry(served.frame_geometry),
        "camera": CAMERA_LABEL,
        "frames_from": FRAMES_FROM_LABEL,
    }


def _member_block(member: CheckpointMetadata) -> dict[str, Any]:
    from allsky.watch import checkpoint_role

    return {
        "name": member.config.name,
        "seed": member.config.seed,
        "role": checkpoint_role(member.path),
        "checkpoint_sha256": member.sha256,
        "epoch": member.epoch,
        "best_metric": {
            "name": member.best_metric.get("name"),
            "value": rounded_or_none(member.best_metric.get("value"), INDEX_DECIMALS),
            "epoch": member.best_metric.get("epoch"),
        },
    }


def _scalar_columns(members: Sequence[CheckpointMetadata]) -> list[str]:
    from allsky.features.policy import resolve_feature_set

    resolved = [
        tuple(resolve_feature_set(member.config.features.feature_set)) for member in members
    ]
    if any(columns != resolved[0] for columns in resolved):
        raise ModelCardError("the served members were trained on different feature sets")
    return list(resolved[0])


def _scalars_consumed(cfg: ExperimentConfig) -> bool:
    return cfg.model.name not in SCALAR_FREE_ARCHITECTURES


def _code_version(payload: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {key: payload.get(key) for key in CODE_VERSION_KEYS}


def _served_block(pin: ServingConfig, members: Sequence[CheckpointMetadata]) -> dict[str, Any]:
    served = members[pin.attribution_checkpoint]
    cfg = served.config
    model_params: dict[str, Any] = dict(cfg.model.model_dump())
    return {
        "id": pin.id,
        "label": pin.label,
        "members": [_member_block(member) for member in members],
        "attribution_member": pin.attribution_checkpoint,
        "roles": {"sky": pin.frame_sky_role, "dhi": pin.frame_dhi_role},
        "selection": {
            "criterion": pin.selection.criterion,
            "selection_split": pin.selection.selection_split,
            "decided_on": pin.selection.decided_on.isoformat(),
        },
        "architecture": {
            "name": cfg.model.name,
            "backbone": (served.backbone or {}).get("name") or model_params.get("backbone"),
            "image_size": image_size_of(cfg),
            "pooling": model_params.get("backbone_pooling"),
            "unfreeze_last_n": model_params.get("unfreeze_last_n"),
            "trunk_hidden": model_params.get("trunk_hidden"),
            "parameters": served.n_parameters,
        },
        "inputs": {
            "image_geometry": _frame_geometry(served.frame_geometry),
            "preprocess": cfg.preprocessing.model_dump(),
            "scalar_columns": _scalar_columns(members),
            "feature_set": cfg.features.feature_set,
            "scalars_consumed": _scalars_consumed(cfg),
            "radiometry_forbidden": True,
        },
        "targets": {
            "dhi": {
                "enabled": cfg.targets.dhi.enabled,
                "loss": cfg.targets.dhi.loss,
                "parameterization": cfg.targets.dhi.parameterization,
            },
            "kindex": {"enabled": cfg.targets.kindex.enabled, "kind": cfg.targets.kindex.kind},
            "sky": {"enabled": cfg.targets.sky.enabled, "weight": cfg.targets.sky.weight},
            "cloud_fraction": {"enabled": cfg.targets.cloud_fraction.enabled},
        },
        "training": {
            "epochs_budget": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "lr": cfg.train.lr,
            "backbone_lr": cfg.train.backbone_lr,
            "layer_decay": cfg.train.layer_decay,
            "weight_decay": cfg.train.weight_decay,
            "scheduler": cfg.train.scheduler.name,
            "early_stopping": cfg.train.early_stopping.model_dump(),
            "cmixup": cfg.train.cmixup.model_dump(),
            "augmentation": cfg.augmentation.model_dump(),
        },
        "code_version": _code_version(served.code_version),
    }


def _domain_check(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    check = _read_json(Path(path))
    missing = [key for key in DOMAIN_CHECK_KEYS if key not in check]
    if missing:
        raise ModelCardError(f"{path}: the domain check lacks {', '.join(missing)}")
    n = check["n"]
    return {
        "day": str(check["day"]),
        "n": int(n) if n is not None else None,
        "dhi_mbe": rounded_or_none(check["dhi_mbe"], IRRADIANCE_DECIMALS),
        "dhi_rmse": rounded_or_none(check["dhi_rmse"], IRRADIANCE_DECIMALS),
        "class_agreement": rounded_or_none(check["class_agreement"], SCORE_DECIMALS),
    }


def _attribution_summary(pin: ServingConfig) -> dict[str, Any]:
    return {
        "method": ATTRIBUTION_METHOD,
        "fill": ATTRIBUTION_FILL,
        "window_px": ATTRIBUTION_WINDOW_PX,
        "stride_px": ATTRIBUTION_STRIDE_PX,
        "target": pin.attribution_target,
    }


def _seed_spread(
    member_arms: Sequence[Mapping[str, Any]],
    members: Sequence[CheckpointMetadata],
    member_rows: Sequence[pd.DataFrame],
) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    for arm, member, rows in zip(member_arms, members, member_rows, strict=True):
        sky = arm.get("sky") or {}
        persistence = _reference_skill(
            rows["obs_dhi"].to_numpy(dtype=np.float64),
            rows["pred_dhi"].to_numpy(dtype=np.float64),
            rows["persistence_dhi"].to_numpy(dtype=np.float64),
        )
        table.append(
            {
                "name": member.config.name,
                "seed": member.config.seed,
                "dhi_rmse": arm["dhi"].get("rmse"),
                "dhi_mae": arm["dhi"].get("mae"),
                "dhi_mbe": arm["dhi"].get("mbe"),
                "kindex_mae": arm["kindex"].get("mae"),
                "sky_balanced_accuracy": sky.get("balanced_accuracy"),
                "sky_macro_f1": sky.get("macro_f1"),
                "sky_kappa_quadratic": sky.get("kappa_quadratic"),
                "persistence_skill_dhi": rounded_or_none(persistence.skill, SCORE_DECIMALS),
            }
        )
    spread: dict[str, Any] = {}
    for key, decimals in SEED_RANGE_DECIMALS.items():
        values = [float(row[key]) for row in table if row.get(key) is not None]
        spread[key] = {
            "min": rounded_or_none(min(values), decimals) if values else None,
            "max": rounded_or_none(max(values), decimals) if values else None,
        }
    return {"n_seeds": len(table), "members": table, "range": spread}


def _control_comparison(control_arms: Mapping[str, Mapping[str, Any]]) -> str:
    sensor = control_arms["sensor_only"]
    climatology = control_arms["climatology"]
    sensor_dhi = sensor["dhi"].get("rmse")
    climatology_dhi = climatology["dhi"].get("rmse")
    sensor_kindex = sensor["kindex"].get("mae")
    climatology_kindex = climatology["kindex"].get("mae")
    if None in (sensor_dhi, climatology_dhi, sensor_kindex, climatology_kindex):
        return (
            "O controle sem imagem e a climatologia são publicados lado a lado com os seus erros "
            "de teste; a página imprime a referência “sem imagem” para os dois."
        )
    dhi_verdict = "pior" if sensor_dhi > climatology_dhi else "melhor"
    kindex_verdict = "pior" if sensor_kindex > climatology_kindex else "melhor"
    return (
        f"O controle sem imagem é {dhi_verdict} que a média do treino na difusa (RMSE {sensor_dhi} "
        f"contra {climatology_dhi} W/m²) e {kindex_verdict} no índice de céu claro (MAE "
        f"{sensor_kindex} contra {climatology_kindex}); por isso a página imprime a referência "
        "“sem imagem” para os dois controles, cada um com o seu erro de teste — um desvio contra "
        "uma referência ruim prova a referência ruim, não a rede boa."
    )


def _caveats(
    *,
    dataset: Mapping[str, Any],
    cfg: ExperimentConfig,
    selection_split: str,
    control_arms: Mapping[str, Mapping[str, Any]],
    domain_check: Mapping[str, Any] | None,
) -> list[str]:
    splits = dataset["split"]
    test = splits["test"]
    train = splits["train"]
    period_end = _br_date(dataset["period"]["end"])
    test_max = test["solar_elevation_range_deg"]["max"]
    train_max = train["solar_elevation_range_deg"]["max"]
    ratio_head = cfg.targets.dhi.parameterization == "clearsky_index"
    if _scalars_consumed(cfg):
        controls = (
            "Os dois controles foram treinados sobre o mesmo manifesto, o mesmo split por dia e os "
            "mesmos alvos; o controle sem imagem recebe exatamente os escalares que o modelo de "
            "imagem também recebe, e a diferença entre os dois é a informação que os pixels "
            "acrescentam."
        )
    else:
        scalars_route = (
            "os escalares só o alcançam pelo denominador de céu claro da cabeça de difusa"
            if ratio_head
            else "nenhum escalar o alcança"
        )
        controls = (
            "Os dois controles foram treinados sobre o mesmo manifesto, o mesmo split por dia e os "
            f"mesmos alvos. O braço servido é só imagem: recebe os pixels e nada mais, e "
            f"{scalars_route}. A comparação com o controle sem imagem é de conteúdo de informação "
            "— pixels contra escalares sobre as mesmas linhas —, não uma ablação com as mesmas "
            "entradas."
        )
    clearsky = (
        "A referência de céu claro é a global de céu claro de Haurwitz [[haurwitz]] decomposta "
        "pela correlação de Erbs [[erbs]] no índice de claridade de céu claro; o seu erro nas "
        "linhas de céu claro, onde deveria acertar melhor, é publicado ao lado."
    )
    if ratio_head:
        clearsky += (
            " A cabeça de difusa é parametrizada como razão a essa referência: o índice servido "
            "vezes ela é a difusa em W/m²."
        )
    if domain_check is None:
        domain = (
            "Os quadros de treino foram extraídos do timelapse H.264 do dia; o quadro vivo é o "
            "JPEG original da câmera. A diferença entre os dois caminhos ainda não foi medida: "
            "domain_check fica nulo até a verificação de primeiro dia ser fixada no pin."
        )
    else:
        domain = (
            "Os quadros de treino foram extraídos do timelapse H.264 do dia; o quadro vivo é o "
            f"JPEG original da câmera. A diferença entre os dois caminhos foi medida em "
            f"{_br_date(str(domain_check['day']))} sobre {domain_check['n']} quadros e está em "
            "domain_check."
        )
    return [
        (
            f"Os dias de teste são os {test['days']} últimos do acervo pareado "
            f"({_br_date(test['start'])} a {_br_date(test['end'])}), em ordem cronológica e "
            f"separados da validação por {splits.get('gap_days')} dia de intervalo."
        ),
        str(dataset["season_note"]),
        (
            "A persistência é de 5 minutos e sobre a linha do datalogger: a referência de um "
            "quadro é a média de difusa que o piranômetro fechou cinco minutos antes da linha "
            "pareada a ele, no mesmo dia — não o quadro anterior. Quadros consecutivos, um minuto "
            "entre si, pareiam com a mesma linha de 5 minutos, e uma persistência deslocada por "
            "quadro seria o próprio alvo na maior parte das linhas. O piranômetro de difusa é um "
            "instrumento que o modelo é proibido de ver; n conta quadros e n_rows conta linhas "
            "distintas com antecessora, e o skill é publicado como medido, seja qual for o sinal."
        ),
        (
            f"Os dias de teste alcançam elevação solar de {test_max}°, acima do máximo de "
            f"{train_max}° visto no treino; a partir de meados de setembro o sol do meio-dia sai "
            "da geometria em que a rede foi treinada, e a linha do tempo marca esses blocos como "
            "extrapolação."
        ),
        (
            f"O pin foi decidido no split de {SPLIT_NAMES_PT.get(selection_split, selection_split)}, "
            "mas o split de teste foi consultado enquanto a família de receitas era desenvolvida: "
            "as métricas de teste não são um holdout de tiro único. O único holdout limpo é a "
            f"comparação ao vivo contra o piranômetro nos dias posteriores a {period_end}, nunca "
            "usados em decisão alguma; ela é publicada na linha do tempo como medição quando a "
            "exportação da estação é fornecida."
        ),
        controls,
        _control_comparison(control_arms),
        clearsky,
        (
            "As quatro condições de céu são rótulos fracos, derivados por faixa do índice de "
            "claridade Kt da média de 5 minutos da global [[escobedo]], com a nomenclatura em "
            "português de [[teramoto]]; a confusão entre as duas condições parciais é em parte a "
            "do próprio rótulo."
        ),
        domain,
    ]


def build_model_card(
    pin: ServingConfig,
    members: Sequence[CheckpointMetadata],
    *,
    stamp: PublishStamp,
) -> dict[str, Any]:
    """Assemble ``model.json`` for the served pin.

    Parameters
    ----------
    pin:
        The serving pin, already verified against the checkpoint files.
    members:
        One :class:`CheckpointMetadata` per ``pin.frame_checkpoints`` entry,
        in the same order.
    stamp:
        The publish stamp shared by every document of this run.

    Returns
    -------
    dict
        The ``labmim-allsky-model-v1`` document, every number rounded and
        finite, ready for :func:`allsky.publish.encoding.write_document`. No
        value is a filesystem path: members are named by ``config.name``.

    Raises
    ------
    ModelCardError
        When a report is missing or was written for other rows, another
        manifest, other targets or with test-time rotations, or when the
        dataset directory disagrees with the served checkpoints.
    """
    if len(members) != len(pin.frame_checkpoints):
        raise ModelCardError(
            f"{len(members)} checkpoint payload(s) for {len(pin.frame_checkpoints)} pinned member(s)"
        )
    served = members[pin.attribution_checkpoint]
    for member in members:
        if (member.split_id, member.manifest_sha256) != (served.split_id, served.manifest_sha256):
            raise ModelCardError(
                f"{member.config.name} trained on split {member.split_id} / manifest "
                f"{member.manifest_sha256}, the attribution member on {served.split_id} / "
                f"{served.manifest_sha256}; their predictions cannot be averaged"
            )
    report_dirs = [Path(pinned.report) for pinned in pin.frame_checkpoints]
    member_reports = [_read_report(report_dir) for report_dir in report_dirs]
    for member, report, report_dir in zip(members, member_reports, report_dirs, strict=True):
        _check_report_is_about(report, member, what=str(report_dir))
        if Path(str(report.get("checkpoint_path", ""))).name != member.path.name:
            logger.warning(
                "%s was written for %s, the pin serves %s",
                report_dir,
                report.get("checkpoint_path"),
                member.path,
            )
    served_report = member_reports[pin.attribution_checkpoint]

    member_frames = _member_predictions(report_dirs)
    ensemble_rows = _scored_rows(member_frames)
    member_rows = [_scored_rows([frame]) for frame in member_frames]

    control_reports = {
        "sensor_only": _read_report(Path(pin.controls.sensor_only.report)),
        "climatology": _read_report(Path(pin.controls.climatology.report)),
    }
    for control_id, report in control_reports.items():
        what = f"control {control_id}"
        _check_report_is_about(report, served, what=what)
        _check_same_targets(report, served_report, what=what)
        _check_same_rows(report, len(ensemble_rows), what=what)

    ensemble_arm, ensemble_dhi, ensemble_sky = _ensemble_arm(pin, ensemble_rows, members)
    references, reference_skill = _references(ensemble_rows)
    member_arms = [
        _arm_from_report(f"member:{member.config.name}", "member", member.config.name, report)
        for member, report in zip(members, member_reports, strict=True)
    ]
    control_arms = {
        control_id: _arm_from_report(control_id, "control", CONTROL_LABELS[control_id], report)
        for control_id, report in control_reports.items()
    }
    dataset = _dataset_block(pin, served)
    domain_check = _domain_check(pin.reports.domain_check)
    kindex_kind = served.config.targets.kindex.kind

    return {
        **document_header(MODEL_SCHEMA, stamp),
        "served": _served_block(pin, members),
        "dataset": dataset,
        "training_curve": _training_curve(
            Path(pin.reports.training_history), best_epoch=_best_epoch(pin, members)
        ),
        "evaluation": {
            "split": "test",
            "n": ensemble_arm["n"],
            "inference": INFERENCE_MODE,
            "arms": [ensemble_arm, *member_arms, *control_arms.values()],
            "references": references,
            "skill": {
                **reference_skill,
                **_skill_against_controls(
                    ensemble_dhi.get("rmse"), control_reports, n=ensemble_arm["n"]
                ),
            },
            "per_class": _per_class_by_condition(ensemble_sky.get("per_class")),
            "per_day": _per_day(ensemble_rows),
            "sky_conditions": sky_conditions_block(),
            "targets": targets_glossary(kindex_kind),
        },
        "stratified": _stratified(Path(pin.attribution_member.report)),
        "seeds": _seed_spread(member_arms, members, member_rows),
        "attribution_summary": _attribution_summary(pin),
        "domain_check": domain_check,
        "caveats": _caveats(
            dataset=dataset,
            cfg=served.config,
            selection_split=pin.selection.selection_split,
            control_arms=control_arms,
            domain_check=domain_check,
        ),
        "references": REFERENCES,
    }
