"""Radiation-sensor ingestion and target derivation for the all-sky pipeline.

Loads Campbell TOA5 ``.dat`` files through the existing
:func:`micrometeorology.sensors.ingestion.read_campbell_dat` (which
already applies the -900 sentinel -> NaN handling), then derives the
training targets:

- ``kt`` — clearness index (:func:`allsky.solar.clearness_index`);
- ``diffuse`` — measured diffuse irradiance when
  ``SensorConfig.diffuse_column`` is set, otherwise an **Erbs
  pseudo-target** (:func:`allsky.erbs.pseudo_diffuse`) derived from GHI;
- ``cloud_class`` — weak label from kt bins (clear / partial / overcast);
- ``target_source`` — ``"measured"`` or ``"erbs_pseudo"`` so pseudo rows
  can be replaced once a shaded pyranometer is installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from allsky.erbs import pseudo_diffuse
from allsky.solar import clearness_index, solar_elevation
from micrometeorology.sensors.ingestion import read_campbell_dat

if TYPE_CHECKING:
    from allsky.config import LabelConfig, SensorConfig, SiteConfig

__all__ = [
    "CLASS_CLEAR",
    "CLASS_NAMES",
    "CLASS_OVERCAST",
    "CLASS_PARTIAL",
    "classify_cloud_condition",
    "derive_targets",
    "load_sensor_frame",
]

logger = logging.getLogger(__name__)

#: Integer labels for the weak cloud-condition classes.
CLASS_CLEAR = 0
CLASS_PARTIAL = 1
CLASS_OVERCAST = 2
CLASS_NAMES = ("clear", "partial", "overcast")

#: ``target_source`` values written by :func:`derive_targets`.
TARGET_SOURCE_MEASURED = "measured"
TARGET_SOURCE_ERBS = "erbs_pseudo"


def load_sensor_frame(cfg: SensorConfig) -> pd.DataFrame:
    """Read all configured TOA5 files into one time-indexed DataFrame.

    Files are read with :func:`read_campbell_dat` (sentinel values <= -900
    become NaN), concatenated, sorted by timestamp, deduplicated (first
    occurrence wins) and reduced to the GHI, diffuse (when configured) and
    feature columns.

    Raises
    ------
    KeyError
        If any configured column is missing from the merged frame.
    """
    frames = [read_campbell_dat(path) for path in cfg.paths]
    df = pd.concat(frames)
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep="first")]

    wanted = [cfg.ghi_column]
    if cfg.diffuse_column is not None:
        wanted.append(cfg.diffuse_column)
    wanted.extend(c for c in cfg.feature_columns if c not in wanted)

    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise KeyError(f"Sensor columns not found in {cfg.paths}: {missing}")

    logger.info("Sensor frame: %d rows, columns %s", len(df), wanted)
    return df[wanted]


def classify_cloud_condition(
    kt: np.ndarray | pd.Series,
    cfg: LabelConfig,
) -> np.ndarray:
    """Weak cloud-condition labels from clearness-index bins.

    Formula
    -------
    - ``kt >= kt_clear`` (default 0.65) -> ``CLASS_CLEAR`` (0)
    - ``kt_overcast <= kt < kt_clear`` -> ``CLASS_PARTIAL`` (1)
    - ``kt < kt_overcast`` (default 0.35) -> ``CLASS_OVERCAST`` (2)

    NaN kt yields ``-1`` (unlabelable — callers must drop those rows).

    Limitation
    ----------
    These are weak labels: kt conflates cloudiness with turbidity and
    sensor calibration drift, and thin-cirrus skies can still reach
    "clear" kt values.
    """
    kt_arr = np.asarray(kt, dtype=np.float64)
    labels = np.select(
        [kt_arr >= cfg.kt_clear, kt_arr >= cfg.kt_overcast],
        [CLASS_CLEAR, CLASS_PARTIAL],
        default=CLASS_OVERCAST,
    ).astype(np.int64)
    labels[np.isnan(kt_arr)] = -1
    return labels


def derive_targets(
    df: pd.DataFrame,
    site: SiteConfig,
    sensor_cfg: SensorConfig,
    label_cfg: LabelConfig,
) -> pd.DataFrame:
    """Add training targets (kt, diffuse, cloud_class, target_source).

    Rows with solar elevation below ``label_cfg.min_solar_elevation_deg``
    (night / near-horizon, where kt is noise-dominated) are dropped, as
    are rows whose kt or diffuse target is NaN (missing GHI) and rows with
    ``kt > label_cfg.max_kt`` (sensor artifacts: GHI spiking far beyond the
    physically plausible clear-sky envelope).

    The diffuse target is the measured ``sensor_cfg.diffuse_column`` when
    configured; otherwise it is an **Erbs pseudo-target** derived from
    GHI and kt, flagged with ``target_source="erbs_pseudo"``.  Pseudo
    targets bootstrap the pipeline until a shaded pyranometer exists —
    treat regression metrics on them as consistency checks, not accuracy.

    Parameters
    ----------
    df:
        Time-indexed sensor frame (see :func:`load_sensor_frame`).
    site:
        Site coordinates for the solar geometry.
    sensor_cfg:
        Column selection (GHI, optional measured diffuse).
    label_cfg:
        kt thresholds and minimum solar elevation.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame must have a DatetimeIndex")

    n_input = len(df)
    elevation = solar_elevation(df.index, site)
    out = df.loc[elevation >= label_cfg.min_solar_elevation_deg].copy()

    ghi = out[sensor_cfg.ghi_column].to_numpy(dtype=np.float64)
    out["kt"] = clearness_index(ghi, pd.DatetimeIndex(out.index), site)

    if sensor_cfg.diffuse_column is not None:
        measured = out[sensor_cfg.diffuse_column].astype(np.float64)
        finite = measured.to_numpy()[np.isfinite(measured.to_numpy())]
        if len(finite) and (finite == 0).mean() > 0.99:
            raise ValueError(
                f"Diffuse column {sensor_cfg.diffuse_column!r} is effectively all zeros in the "
                "selected daytime rows — a dead logger channel, not a measurement. Training on it "
                "would teach the model to predict zero. Point sensor.diffuse_column at a live "
                "sensor (e.g. PSP_Wm2_Avg) or set it to null for Erbs pseudo-targets."
            )
        out["diffuse"] = measured
        out["target_source"] = TARGET_SOURCE_MEASURED
    else:
        out["diffuse"] = pseudo_diffuse(ghi, out["kt"].to_numpy())
        out["target_source"] = TARGET_SOURCE_ERBS

    kt_values = out["kt"].to_numpy()
    valid = (
        np.isfinite(kt_values)
        & np.isfinite(out["diffuse"].to_numpy())
        & (kt_values <= label_cfg.max_kt)
    )
    out = out.loc[valid]
    out["cloud_class"] = classify_cloud_condition(out["kt"].to_numpy(), label_cfg)

    logger.info(
        "derive_targets: %d -> %d rows (elevation >= %.1f deg, valid targets), source=%s",
        n_input,
        len(out),
        label_cfg.min_solar_elevation_deg,
        out["target_source"].iloc[0] if len(out) else "n/a",
    )
    return out
