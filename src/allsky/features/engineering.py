"""Deterministic feature engineering for the multimodal stack.

Turns a time-indexed sensor frame plus a target timestamp index into the
engineered feature frame named by :mod:`allsky.features.policy`.  Two families
of transforms live here:

- **Solar geometry** (elevation, zenith, azimuth) from :mod:`allsky.solar`.
- **Cyclic encodings** of periodic quantities — day-of-year, wind direction
  and solar azimuth — as sine/cosine pairs, so the model never sees the
  artificial discontinuity at the wrap point (e.g. wind direction 359 deg and
  1 deg map to nearly identical (sin, cos) pairs instead of the far-apart raw
  degrees).

Column order is always :func:`allsky.features.policy.resolve_feature_set`, so
downstream normalization/checkpoint feature ordering is reproducible.

Timestamps here are **naive local** clock time, as the camera and the datalogger
stamp them; the solar geometry converts them with ``utc_offset_hours``, inferred
from the site longitude when the caller does not pin it.
"""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from allsky.config import SiteConfig
from allsky.features.policy import FeatureSet, resolve_feature_set, source_column
from allsky.solar import solar_azimuth_deg, solar_elevation_deg
from micrometeorology.sensors.wind import wind_components

type DatetimeLike = pd.DatetimeIndex | pd.Series | np.ndarray | list | tuple

__all__ = ["DAYS_PER_YEAR", "build_feature_frame"]

#: Period of the day-of-year cyclic encoding (mean tropical/Julian year); using
#: a non-integer divisor keeps the encoding continuous across leap years.
DAYS_PER_YEAR = 365.25


def build_feature_frame(
    sensor_df: pd.DataFrame,
    timestamps: DatetimeLike,
    site: SiteConfig,
    feature_set: FeatureSet | str = "safe",
    *,
    extra: Iterable[str] = (),
    utc_offset_hours: float | None = None,
) -> pd.DataFrame:
    """Build the engineered feature frame for *timestamps*.

    Solar-geometry and cyclic features are computed from *timestamps*; the
    remaining features are read from *sensor_df*, aligned to *timestamps* by
    label (:meth:`pandas.DataFrame.reindex`).  The caller is responsible for
    supplying a *sensor_df* whose index already covers *timestamps* (e.g. the
    merge-asof paired frame): unmatched labels become NaN, which downstream
    validation flags.

    Parameters
    ----------
    sensor_df:
        Time-indexed frame carrying the raw logger columns named by the
        policy (``AirT1_C_Avg``, ``WindDir``, ...).
    timestamps:
        Naive local timestamps to build features for; becomes the output index.
    site:
        Observation site for the solar geometry.
    feature_set:
        ``"safe"`` (default) or ``"extended"``.
    extra:
        Extra engineered feature names to append (bespoke ablations); each
        must be resolvable via :func:`allsky.features.policy.source_column`.
    utc_offset_hours:
        UTC offset of the local clock; inferred from ``site.longitude`` when
        None.

    Returns
    -------
    pandas.DataFrame
        Shape ``(N, F)`` of ``float64`` columns, indexed by *timestamps* and
        ordered by the policy: ``solar_elevation`` / ``solar_zenith`` in degrees,
        the cyclic sine/cosine pairs dimensionless in ``[-1, 1]``, and the
        remaining channels in the units their names carry (degC, %, mbar, m/s).

    Raises
    ------
    KeyError
        If a required logger column is absent from *sensor_df*.
    """
    index = pd.DatetimeIndex(timestamps)
    resolved = resolve_feature_set(feature_set, extra)

    # source_column() answers None for every name a dedicated transform builds
    # (the solar geometry, the cyclic pairs), so those drop out here on their own.
    required = {col for name in resolved for col in (source_column(name),) if col is not None}
    missing = sorted(required - set(sensor_df.columns))
    if missing:
        raise KeyError(f"sensor frame is missing required feature columns: {missing}")

    met = sensor_df.reindex(index)

    elevation = solar_elevation_deg(index, site, utc_offset_hours)
    azimuth_rad = np.deg2rad(solar_azimuth_deg(index, site, utc_offset_hours))
    doy = index.dayofyear.to_numpy(dtype=np.float64)
    doy_angle = 2.0 * np.pi * (doy - 1.0) / DAYS_PER_YEAR

    wind_u = wind_v = None
    if "wind_dir_sin" in resolved or "wind_dir_cos" in resolved:
        direction = met["WindDir"].to_numpy(dtype=np.float64)
        # Unit-speed meteorological components: a continuous cyclic encoding of
        # wind direction (u = -sin(dir), v = -cos(dir)); reuses the shared
        # micrometeorology decomposition so the sign convention matches.
        wind_u, wind_v = wind_components(1.0, direction)

    columns: dict[str, np.ndarray] = {}
    for name in resolved:
        if name == "solar_elevation":
            columns[name] = elevation
        elif name == "solar_zenith":
            columns[name] = 90.0 - elevation
        elif name == "azimuth_sin":
            columns[name] = np.sin(azimuth_rad)
        elif name == "azimuth_cos":
            columns[name] = np.cos(azimuth_rad)
        elif name == "doy_sin":
            columns[name] = np.sin(doy_angle)
        elif name == "doy_cos":
            columns[name] = np.cos(doy_angle)
        elif name == "wind_dir_sin":
            columns[name] = np.asarray(wind_u, dtype=np.float64)
        elif name == "wind_dir_cos":
            columns[name] = np.asarray(wind_v, dtype=np.float64)
        else:
            columns[name] = met[source_column(name)].to_numpy(dtype=np.float64)

    frame = pd.DataFrame(columns, index=index)
    return frame.loc[:, resolved]
