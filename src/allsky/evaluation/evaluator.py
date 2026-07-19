"""Run a trained checkpoint over a split and compute stratified metrics.

:func:`evaluate_checkpoint` is the entry point: it loads a C4a checkpoint
(``weights_only=False`` — a trusted, locally written file), rebuilds the model
from the stored :class:`~allsky.config.ExperimentConfig` via
:func:`allsky.modeling.registry.build_model`, restores the train-split feature /
target normalizers and the ordered ``feature_columns``, re-reads the v2 manifest
and its meta sidecar, and re-loads the persisted day split.  Provenance is
checked: the current ``manifest_sha256`` is compared against the one baked into
the checkpoint, and the split artifact's ``split_id`` against the stored one — a
mismatch warns by default (``strict=True`` turns either into an error).

Inference runs with no grad; regression outputs (which the heads emit in
**normalized** space) are denormalized back to physical units with the stored
:class:`~allsky.features.normalization.TargetNormalizer` before any metric is
computed.  Global metrics are reported per enabled target, plus stratified
breakdowns by sky class, solar-elevation band, local hour-of-day, local month,
QC state (clean vs any-flag) and a k-index band used as a partial-sun proxy.

``torch`` and the model zoo are imported lazily inside the functions that need
them, so importing this module (and the light metrics/report helpers around it)
never pulls torch.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from allsky.config import ExperimentConfig
from allsky.data.contracts import SKY_CLASS_NAMES, sky_class_name
from allsky.data.datasets import EmbeddingReader, WindowMode
from allsky.data.loading import (
    default_embedding_reader,
    load_manifest,
    load_split,
    resolve_against_root,
)
from allsky.evaluation.metrics import classification_metrics, regression_metrics
from allsky.features.normalization import FeatureNormalizer, TargetNormalizer

logger = logging.getLogger(__name__)

__all__ = ["EvaluationResult", "evaluate_checkpoint"]

#: Fixed America/Bahia offset (UTC-3, no DST) used to derive local hour/month
#: from the manifest's tz-aware ``timestamp_utc`` (mirrors the manifest builder).
_LOCAL_TZ = timezone(timedelta(hours=-3))

#: Solar-elevation band edges (degrees); rows outside ``[10, 90]`` fall in no
#: band and are simply absent from the elevation breakdown.
_ELEVATION_EDGES: tuple[float, ...] = (10.0, 20.0, 35.0, 50.0, 90.0)

#: k-index band edges used as a **partial-sun proxy**: the continuous target
#: k-index (k* or k_t, not the pre-binned ``sky_class``) split at the same
#: clear/overcast thresholds, so the middle band is the partial-cloud regime
#: where diffuse is hardest to predict.
_KINDEX_EDGES: tuple[float, ...] = (-np.inf, 0.35, 0.65, np.inf)
_KINDEX_LABELS: tuple[str, ...] = ("overcast_lt0.35", "partial_0.35-0.65", "clear_ge0.65")

#: The regression targets, keyed by their manifest observation column.
_REGRESSION_TARGETS: dict[str, str] = {
    "dhi": "target_dhi",
    "kindex": "target_kindex",
    "cloud_fraction": "cloud_fraction",
}


@dataclass
class EvaluationResult:
    """Outcome of :func:`evaluate_checkpoint`.

    Attributes
    ----------
    checkpoint_path:
        The evaluated checkpoint (as a string).
    split:
        Which split was scored (``"train"`` | ``"val"`` | ``"test"``).
    n_samples:
        Number of manifest rows in the split.
    enabled_targets:
        The heads that were enabled and scored (subset of ``dhi`` / ``kindex`` /
        ``cloud_fraction`` / ``sky``).
    global_metrics:
        ``target -> metric dict`` over the whole split.  Regression targets carry
        :data:`allsky.evaluation.metrics.REGRESSION_METRIC_KEYS`; ``sky`` carries
        the classification metrics plus its ``confusion`` matrix.
    stratified:
        Long-form breakdown DataFrame with columns ``target``, ``stratum_kind``,
        ``stratum``, ``metric``, ``value``, ``n`` (an ``overall`` kind holds the
        global rows).
    confusion:
        ``{"labels": [...], "matrix": [[...]]}`` for the sky head, or ``None``.
    predictions:
        Per-sample DataFrame (identity + strata + observed/predicted columns).
    meta:
        Provenance: experiment ``name``, ``model``, ``feature_set``, ``device``,
        ``split_id`` / ``split_id_ok``, ``manifest_sha256`` / ``manifest_hash_ok``
        and ``dataset_version``.
    """

    checkpoint_path: str
    split: str
    n_samples: int
    enabled_targets: list[str]
    global_metrics: dict[str, dict[str, Any]]
    stratified: pd.DataFrame
    confusion: dict[str, Any] | None
    predictions: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    split: str = "val",
    data_root: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    report_dir: str | Path | None = None,  # noqa: ARG001 - reserved; report writing is in reports.py
    strict: bool = False,
    embedding_reader: EmbeddingReader | None = None,
    image_backbone_builder: Any | None = None,
) -> EvaluationResult:
    """Evaluate *checkpoint_path* on *split* and return an :class:`EvaluationResult`.

    Parameters
    ----------
    checkpoint_path:
        A ``last.ckpt`` / ``best.ckpt`` written by the C4a engine.
    split:
        ``"val"`` (default), ``"test"`` or ``"train"``.
    data_root:
        Overrides ``cfg.data.data_root`` (root the manifest, split and embeddings
        paths, plus the manifest's relative ``image_path`` values, resolve
        against).
    batch_size:
        Inference batch size (defaults to the config's train batch size).
    device:
        ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"mps"`` (defaults to the config's
        train device); a clear error is raised for unavailable CUDA.
    report_dir:
        Accepted for signature symmetry with the CLI; report writing lives in
        :func:`allsky.evaluation.reports.write_evaluation_report`.
    strict:
        When ``True`` a manifest-hash or split-id mismatch raises instead of only
        logging a warning.
    embedding_reader:
        Injected reader for ``input_mode="embedding"`` (tests pass a dict-backed
        fake); defaults to a
        :class:`~allsky.embeddings.storage.SafetensorsEmbeddingReader` over
        ``cfg.data.embeddings_dir``.
    image_backbone_builder:
        Zero-arg factory for the image backbone (``input_mode="image"`` visual
        models); no backbone is downloaded implicitly.

    Returns
    -------
    EvaluationResult
        Global + stratified metrics, the predictions frame and provenance.
    """
    from allsky.training.checkpointing import load_checkpoint
    from allsky.training.engine import resolve_run_device

    ckpt_path = Path(checkpoint_path)
    resolved_device = resolve_run_device(device if device is not None else "cpu")
    checkpoint = load_checkpoint(ckpt_path, map_location=resolved_device)

    cfg = ExperimentConfig.model_validate(checkpoint["config"])
    root = Path(data_root) if data_root is not None else Path(cfg.data.data_root)

    manifest, meta = load_manifest(resolve_against_root(cfg.data.manifest, root))
    manifest_hash_ok = _check_manifest_hash(checkpoint.get("manifest_sha256"), meta, strict=strict)
    split_obj = load_split(resolve_against_root(cfg.data.split_artifact, root))
    split_id_ok = _check_split_id(checkpoint.get("split_id"), split_obj.split_id, strict=strict)
    split_df = _select_split_rows(manifest, split_obj, split)

    feature_columns: list[str] = list(checkpoint["feature_columns"])
    normalizers = checkpoint["normalizers"]
    feature_normalizer = FeatureNormalizer.from_dict(normalizers["feature_normalizer"])
    target_normalizers = {
        key: TargetNormalizer.from_dict(value)
        for key, value in normalizers["target_normalizers"].items()
    }

    enabled_targets = _enabled_targets(cfg)
    predictions = _run_inference(
        cfg=cfg,
        checkpoint=checkpoint,
        split_df=split_df,
        feature_columns=feature_columns,
        feature_normalizer=feature_normalizer,
        target_normalizers=target_normalizers,
        enabled_targets=enabled_targets,
        device=resolved_device,
        batch_size=int(batch_size) if batch_size is not None else int(cfg.train.batch_size),
        root=root,
        embedding_reader=embedding_reader,
        image_backbone_builder=image_backbone_builder,
    )

    global_metrics = _global_metrics(predictions, enabled_targets)
    stratified = _stratified_metrics(predictions, enabled_targets, global_metrics)
    confusion = None
    if "sky" in global_metrics and "confusion" in global_metrics["sky"]:
        confusion = {
            "labels": list(SKY_CLASS_NAMES),
            "matrix": global_metrics["sky"]["confusion"],
        }

    meta_out = {
        "name": cfg.name,
        "model": cfg.model.name,
        "feature_set": cfg.features.feature_set,
        "input_mode": cfg.data.input_mode,
        "device": resolved_device,
        "split_id": split_obj.split_id,
        "split_id_ok": split_id_ok,
        "manifest_sha256": meta.get("manifest_sha256"),
        "manifest_hash_ok": manifest_hash_ok,
        "dataset_version": str(meta.get("dataset_version", checkpoint.get("dataset_version"))),
    }
    logger.info(
        "evaluated %s on '%s': %d rows, targets=%s (hash_ok=%s, split_ok=%s)",
        ckpt_path.name,
        split,
        len(split_df),
        enabled_targets,
        manifest_hash_ok,
        split_id_ok,
    )
    return EvaluationResult(
        checkpoint_path=str(ckpt_path),
        split=split,
        n_samples=len(split_df),
        enabled_targets=enabled_targets,
        global_metrics=global_metrics,
        stratified=stratified,
        confusion=confusion,
        predictions=predictions,
        meta=meta_out,
    )


# ---------------------------------------------------------------------------
# loading / provenance
# ---------------------------------------------------------------------------


def _check_manifest_hash(stored: str | None, meta: Mapping[str, Any], *, strict: bool) -> bool:
    """Compare the checkpoint's manifest hash against the current meta sidecar."""
    current = meta.get("manifest_sha256")
    if stored is None or current is None:
        logger.info("manifest hash not verified (stored=%s, current=%s)", stored, current)
        return False
    if stored == current:
        return True
    message = (
        f"manifest hash mismatch: checkpoint was trained on {stored[:12]} but the "
        f"manifest on disk is {current[:12]} — evaluating on a different dataset"
    )
    if strict:
        raise ValueError(message)
    logger.warning(message)
    return False


def _check_split_id(stored: str | None, current: str, *, strict: bool) -> bool:
    """Compare the checkpoint's split id against the loaded split artifact's."""
    if stored is None:
        logger.info("split id not recorded in checkpoint; skipping verification")
        return False
    if stored == current:
        return True
    message = (
        f"split id mismatch: checkpoint used {stored[:12]} but the split artifact "
        f"on disk is {current[:12]} — the train/val/test partition changed"
    )
    if strict:
        raise ValueError(message)
    logger.warning(message)
    return False


def _select_split_rows(manifest: pd.DataFrame, split_obj: Any, split: str) -> pd.DataFrame:
    """Slice the manifest rows whose ``day_id`` belongs to *split*."""
    days = set(split_obj.days_for(split))
    if not days:
        raise ValueError(f"split '{split}' has no days in the artifact")
    day_ids = manifest["day_id"].astype(str)
    selected = manifest.loc[day_ids.isin(days)].reset_index(drop=True)
    if selected.empty:
        raise ValueError(
            f"no manifest rows for split '{split}' (its days are absent from the manifest)"
        )
    return selected


def _enabled_targets(cfg: ExperimentConfig) -> list[str]:
    """Ordered list of enabled heads (regression first, then ``sky``)."""
    targets: list[str] = []
    if cfg.targets.dhi.enabled:
        targets.append("dhi")
    if cfg.targets.kindex.enabled:
        targets.append("kindex")
    if cfg.targets.cloud_fraction.enabled:
        targets.append("cloud_fraction")
    if cfg.targets.sky.enabled:
        targets.append("sky")
    return targets


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------


def _run_inference(
    *,
    cfg: ExperimentConfig,
    checkpoint: Mapping[str, Any],
    split_df: pd.DataFrame,
    feature_columns: list[str],
    feature_normalizer: FeatureNormalizer,
    target_normalizers: Mapping[str, TargetNormalizer],
    enabled_targets: Sequence[str],
    device: str,
    batch_size: int,
    root: Path,
    embedding_reader: EmbeddingReader | None,
    image_backbone_builder: Any | None,
) -> pd.DataFrame:
    """Rebuild the model, run a no-grad pass and assemble the predictions frame."""
    import torch
    from torch.utils.data import DataLoader

    from allsky.modeling.registry import build_model, temporal_pooling_for_strategy
    from allsky.training.engine import _default_image_backbone_builder

    dataset, embedding_dim = _build_split_dataset(
        cfg,
        split_df,
        feature_columns,
        feature_normalizer,
        root=root,
        embedding_reader=embedding_reader,
    )

    image_backbone = None
    if cfg.data.input_mode == "image":
        # Same gap as training (finding F6): with no injected builder the rebuilt
        # model still needs the backbone architecture to load_state_dict, so
        # default to the config-named backbone. The injection hook still wins.
        builder = image_backbone_builder or _default_image_backbone_builder(cfg, device)
        image_backbone = builder()
    # Rebuild with the same temporal pooler the checkpoint was trained with, or
    # load_state_dict would reject an attention-pooled model's extra weights.
    temporal_pooling = temporal_pooling_for_strategy(cfg.data.alignment.strategy)
    model = build_model(
        cfg,
        len(feature_columns),
        embedding_dim=embedding_dim,
        image_backbone=image_backbone,
        temporal_pooling=temporal_pooling,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    loader: DataLoader[Any] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in enabled_targets}
    with torch.no_grad():
        for raw in loader:
            batch = {
                key: (value.to(device) if isinstance(value, torch.Tensor) else value)
                for key, value in raw.items()
            }
            outputs = model(batch)
            for name in enabled_targets:
                collected[name].append(_extract_prediction(name, outputs))

    predicted: dict[str, np.ndarray] = {
        name: np.concatenate(chunks) if chunks else np.empty(0)
        for name, chunks in collected.items()
    }
    for name in ("dhi", "kindex"):
        if name in predicted and name in target_normalizers:
            predicted[name] = np.asarray(target_normalizers[name].denormalize(predicted[name]))

    return _build_predictions_frame(split_df, predicted, enabled_targets)


def _build_split_dataset(
    cfg: ExperimentConfig,
    split_df: pd.DataFrame,
    feature_columns: list[str],
    feature_normalizer: FeatureNormalizer,
    *,
    root: Path,
    embedding_reader: EmbeddingReader | None,
) -> tuple[Any, int | None]:
    """Build the (train=False) dataset for the split, reusing the stored normalizer."""
    from allsky.data.datasets import MultimodalEmbeddingDataset, MultimodalImageDataset

    if cfg.data.input_mode == "embedding":
        reader = (
            embedding_reader
            if embedding_reader is not None
            else default_embedding_reader(cfg, root)
        )
        # Mirror training's alignment strategy so the eval batches match what the
        # model was trained on (plain embedding for center_frame/mean_embedding,
        # padded embedding_seq + frame_mask for attention_pooling).
        window = cast("WindowMode", cfg.data.alignment.strategy)
        dataset: Any = MultimodalEmbeddingDataset(
            split_df,
            feature_columns,
            embedding_reader=reader,
            train=False,
            stats=feature_normalizer,
            window=window,
            window_minutes=float(cfg.data.alignment.window_minutes),
        )
        embedding_dim = int(getattr(reader, "dim", 0)) or int(dataset.embedding_dim)
        return dataset, embedding_dim

    image_size = int(dict(cfg.model.model_dump()).get("image_size", 224))
    dataset = MultimodalImageDataset(
        split_df,
        feature_columns,
        data_root=root,
        image_size=image_size,
        train=False,
        stats=feature_normalizer,
    )
    return dataset, None


def _extract_prediction(name: str, outputs: Mapping[str, Any]) -> np.ndarray:
    """Pull one head's per-sample prediction out of the model outputs as numpy."""
    if name == "sky":
        logits = outputs["sky_logits"].detach().argmax(dim=-1).cpu().numpy()
        return np.asarray(logits, dtype=np.int64)
    values = outputs[name].detach().float().cpu().numpy()
    return np.asarray(values).ravel()


# ---------------------------------------------------------------------------
# predictions frame + strata
# ---------------------------------------------------------------------------


def _build_predictions_frame(
    split_df: pd.DataFrame, predicted: Mapping[str, np.ndarray], enabled_targets: Sequence[str]
) -> pd.DataFrame:
    """Assemble the per-sample predictions frame (identity + strata + obs/pred)."""
    frame = pd.DataFrame(
        {
            "sample_id": split_df["sample_id"].astype(str).to_numpy(),
            "day_id": split_df["day_id"].astype(str).to_numpy(),
            "timestamp_utc": split_df["timestamp_utc"].astype(str).to_numpy(),
        }
    )
    _add_strata(frame, split_df)

    for name in enabled_targets:
        if name == "sky":
            frame["obs_sky"] = split_df["sky_class"].to_numpy(dtype=np.int64)
            frame["pred_sky"] = np.asarray(predicted["sky"], dtype=np.int64)
        else:
            obs_column = _REGRESSION_TARGETS[name]
            observed = (
                split_df[obs_column].to_numpy(dtype=np.float64)
                if obs_column in split_df.columns
                else np.full(len(split_df), np.nan)
            )
            frame[f"obs_{name}"] = observed
            frame[f"pred_{name}"] = np.asarray(predicted[name], dtype=np.float64)
    return frame


def _add_strata(frame: pd.DataFrame, split_df: pd.DataFrame) -> None:
    """Attach the stratification columns to *frame* from *split_df*."""
    local = pd.to_datetime(split_df["timestamp_utc"], utc=True).dt.tz_convert(_LOCAL_TZ)
    frame["hour"] = local.dt.hour.to_numpy(dtype=np.int64)
    frame["month"] = local.dt.month.to_numpy(dtype=np.int64)

    sky = split_df["sky_class"].to_numpy(dtype=np.int64)
    frame["sky_class"] = [sky_class_name(int(value)) for value in sky]

    qc = split_df["qc_flags"].to_numpy(dtype=np.int64)
    frame["qc"] = np.where(qc == 0, "clean", "flagged")

    elevation = split_df["solar_elevation"].to_numpy(dtype=np.float64)
    elevation_labels = [f"{int(lo)}-{int(hi)}" for lo, hi in itertools.pairwise(_ELEVATION_EDGES)]
    frame["elevation_band"] = pd.cut(
        elevation, bins=list(_ELEVATION_EDGES), labels=elevation_labels, include_lowest=True
    ).astype("object")

    kindex = split_df["target_kindex"].to_numpy(dtype=np.float64)
    frame["kindex_band"] = pd.cut(
        kindex, bins=list(_KINDEX_EDGES), labels=list(_KINDEX_LABELS), right=False
    ).astype("object")


#: Stratification column -> the ``stratum_kind`` label reported in the long table.
_STRATUM_KINDS: dict[str, str] = {
    "sky_class": "sky_class",
    "elevation_band": "solar_elevation",
    "hour": "hour_of_day",
    "month": "month",
    "qc": "qc_flags",
    "kindex_band": "kindex_band",
}


# ---------------------------------------------------------------------------
# metric computation
# ---------------------------------------------------------------------------


def _global_metrics(
    predictions: pd.DataFrame, enabled_targets: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Whole-split metrics per enabled target."""
    metrics: dict[str, dict[str, Any]] = {}
    for name in enabled_targets:
        metrics[name] = _target_metrics(predictions, name)
    return metrics


def _target_metrics(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    """Metrics for one target over *frame* (regression or classification)."""
    if name == "sky":
        if "obs_sky" not in frame.columns:
            return classification_metrics(np.empty(0), np.empty(0))
        return classification_metrics(
            frame["obs_sky"].to_numpy(),
            frame["pred_sky"].to_numpy(),
            n_classes=len(SKY_CLASS_NAMES),
        )
    obs_col, pred_col = f"obs_{name}", f"pred_{name}"
    if obs_col not in frame.columns:
        return regression_metrics(np.empty(0), np.empty(0))
    return regression_metrics(frame[obs_col].to_numpy(), frame[pred_col].to_numpy())


def _stratified_metrics(
    predictions: pd.DataFrame,
    enabled_targets: Sequence[str],
    global_metrics: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Long-form (target, stratum_kind, stratum, metric, value, n) breakdown table."""
    records: list[dict[str, Any]] = []
    for name in enabled_targets:
        records.extend(_metric_rows(name, "overall", "all", global_metrics[name]))

    for column, kind in _STRATUM_KINDS.items():
        if column not in predictions.columns:
            continue
        for stratum, group in predictions.groupby(column, dropna=True, observed=True):
            for name in enabled_targets:
                metrics = _target_metrics(group, name)
                records.extend(_metric_rows(name, kind, str(stratum), metrics))

    columns = ["target", "stratum_kind", "stratum", "metric", "value", "n"]
    if not records:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame.from_records(records, columns=columns)


def _metric_rows(
    target: str, stratum_kind: str, stratum: str, metrics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Expand a metric dict into long-form rows (dropping ``n`` and the confusion)."""
    count = metrics.get("n", 0)
    rows: list[dict[str, Any]] = []
    for metric, value in metrics.items():
        if metric in ("n", "confusion"):
            continue
        rows.append(
            {
                "target": target,
                "stratum_kind": stratum_kind,
                "stratum": stratum,
                "metric": metric,
                "value": float(value),
                "n": int(count),
            }
        )
    return rows
