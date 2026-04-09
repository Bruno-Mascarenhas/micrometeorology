"""Solar radiation indices: clearness index (Kt) and diffuse fraction (Kd)."""

from __future__ import annotations

import numpy as np
import pandas as pd  # noqa: TC002 — used at runtime


def clearness_index(
    global_radiation: pd.Series,
    extraterrestrial_radiation: pd.Series,
) -> pd.Series:
    """Compute the clearness index Kt = Sw_dw / Sw_top.

    Values where extraterrestrial radiation ≤ 0 are set to NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        kt = global_radiation / extraterrestrial_radiation
    kt[extraterrestrial_radiation <= 0] = np.nan
    kt[(kt < 0) | (kt > 1.5)] = np.nan  # physical bounds
    return kt


def diffuse_fraction(
    diffuse_radiation: pd.Series,
    global_radiation: pd.Series,
) -> pd.Series:
    """Compute the diffuse fraction Kd = Sw_dif / Sw_dw.

    Values where global radiation ≤ 0 are set to NaN.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        kd = diffuse_radiation / global_radiation
    kd[global_radiation <= 0] = np.nan
    kd[(kd < 0) | (kd > 1.5)] = np.nan
    return kd
