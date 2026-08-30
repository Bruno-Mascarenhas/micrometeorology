"""Reading the UCSD-Folsom sky-image dataset into this project's manifest.

Folsom is the transfer-learning source this station starts from: three years of
1-minute GHI/DNI/DHI beside a **fisheye** sky camera, at Folsom, California
(38.642 N, 121.148 W). The camera matters more than the climate here — the
circumsolar region governs the diffuse split, and the ARM archives, whose
tropical sites match this station's latitude far better, image through a total
sky imager whose shadow band OCCLUDES the sun. A backbone pre-trained on skies
with no visible sun cannot learn the feature this task most depends on.

Format, from the dataset's own files (Zenodo DOI 10.5281/zenodo.2826939,
CC BY-NC 4.0; Pedro, Larson & Coimbra, *A comprehensive dataset for the
accelerated development and benchmarking of solar forecasting methods*, Journal
of Renewable and Sustainable Energy 11(3), 036102, 2019):

- ``Folsom_irradiance.csv`` — ``timeStamp,ghi,dni,dhi``, one row per minute, W/m2,
  timestamps in **UTC**. Measured here: 1,552,320 rows from 2014-01-02 08:00 to
  2016-12-31 07:59, 618 NaN in ``dhi``, 732,122 rows above 20 W/m2 of GHI.
- ``Folsom_weather.csv`` — ``timeStamp,air_temp,relhum,press,windsp,winddir,
  max_windsp,precipitation``, same cadence and clock.
- ``Folsom_sky_images_<year>.tar.bz2`` — daytime frames at 1-minute intervals.

Nothing else in the record is used: the pre-extracted image and satellite
features, the NAM forecasts and the ``Target_*`` files serve forecasting with a
horizon, while this project estimates at t=0 and extracts its own features.

**Time convention.** This project's manifest builder takes naive times on the
instrument's own clock and stamps UTC from the site's fixed offset. Folsom
publishes UTC directly, so the adapter shifts to a **fixed UTC-8** and declares
that as the site offset. Fixed, not ``America/Los_Angeles``: the pipeline's
convention is an instrument clock, and letting DST in would move solar geometry
by an hour for half of every year. Since the shift is applied and then undone by
the same constant, the UTC instant the geometry is computed from is exactly the
one Folsom published.
"""

import logging
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from labmim_core.site import SiteConfig

__all__ = [
    "FOLSOM_SITE",
    "FOLSOM_UTC_OFFSET_HOURS",
    "read_folsom_frames",
    "read_folsom_sensor",
]

logger = logging.getLogger(__name__)

#: Fixed offset the adapter puts Folsom's clock on. Pacific Standard Time, with
#: no daylight saving: see the module docstring for why a named zone is wrong for
#: an acquisition pipeline built on instrument clocks.
FOLSOM_UTC_OFFSET_HOURS = -8.0

#: The Folsom site. Coordinates as reported by Varaschin & Silva (2025,
#: arXiv:2503.21966), which benchmarks ten architectures on this dataset.
FOLSOM_SITE = SiteConfig(
    latitude=38.642,
    longitude=-121.148,
    utc_offset_hours=FOLSOM_UTC_OFFSET_HOURS,
)

#: Folsom column -> the name this project's feature policy looks for. The `bare`
#: tier wants the mechanical anemometer, which Folsom's weather file carries
#: under its own names.
SENSOR_COLUMN_MAP: dict[str, str] = {
    "windsp": "WS_ms",
    "winddir": "WindDir",
    "air_temp": "AirT1_C_Avg",
    "relhum": "RH1",
    "press": "BP1_mbar_Avg",
}


def read_folsom_sensor(
    irradiance_csv: str | Path, weather_csv: str | Path | None = None
) -> pd.DataFrame:
    """Folsom's irradiance (and weather) as a sensor frame on the site's clock.

    Parameters
    ----------
    irradiance_csv:
        ``Folsom_irradiance.csv``.
    weather_csv:
        ``Folsom_weather.csv``; omitted leaves the met columns absent, which the
        ``bare`` feature tier tolerates for everything except the anemometer.

    Returns
    -------
    pandas.DataFrame
        Time-indexed on **naive Folsom-fixed local time**, carrying ``ghi``,
        ``dni``, ``dhi`` in W m-2 plus the mapped met columns. The index name is
        dropped so it matches what :func:`allsky.data.manifest.build_manifest`
        expects of a logger frame.

    Raises
    ------
    ValueError
        If the irradiance file lacks the columns the format declares.
    """
    irradiance = pd.read_csv(irradiance_csv, parse_dates=["timeStamp"])
    missing = [c for c in ("timeStamp", "ghi", "dni", "dhi") if c not in irradiance.columns]
    if missing:
        raise ValueError(f"{irradiance_csv} is missing the Folsom columns {missing}")
    frame = irradiance.set_index("timeStamp")

    if weather_csv is not None:
        weather = pd.read_csv(weather_csv, parse_dates=["timeStamp"]).set_index("timeStamp")
        keep = {src: dst for src, dst in SENSOR_COLUMN_MAP.items() if src in weather.columns}
        frame = frame.join(weather[list(keep)].rename(columns=keep), how="left")

    frame.index = _to_site_clock(pd.DatetimeIndex(frame.index))
    frame.index.name = None
    return frame


def read_folsom_frames(frames_dir: str | Path, *, pattern: str = "**/*.jpg") -> pd.DataFrame:
    """The extracted Folsom frames as the frame manifest the builder takes.

    Parameters
    ----------
    frames_dir:
        Directory the year archive was extracted into.
    pattern:
        Glob for the image files, relative to *frames_dir*.

    Returns
    -------
    pandas.DataFrame
        Columns ``frame_path``, ``timestamp`` (naive, site clock), ``video``
        (the source year archive) and ``index``, time-ordered.

    Raises
    ------
    FileNotFoundError
        If the directory holds no file matching *pattern*.
    ValueError
        If a filename does not carry the ``YYYYMMDDHHMMSS`` stamp the archive
        names its frames with. Guessing a time for a frame would put a sky image
        against an irradiance reading from another moment, which is the label
        error this project already paid for once.
    """
    root = Path(frames_dir)
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no Folsom frames under {root} matching {pattern!r}")
    stamps = [_timestamp_of(path) for path in paths]
    frame = pd.DataFrame(
        {
            "frame_path": [str(p) for p in paths],
            "timestamp": _to_site_clock(pd.DatetimeIndex(stamps).tz_localize("UTC")),
            "video": [p.parent.name for p in paths],
        }
    ).sort_values("timestamp", ignore_index=True)
    frame["index"] = np.arange(len(frame), dtype=np.int64)
    logger.info(
        "read %d Folsom frames from %s (%s .. %s, site clock)",
        len(frame),
        root,
        frame["timestamp"].iloc[0],
        frame["timestamp"].iloc[-1],
    )
    return frame


def _timestamp_of(path: Path) -> pd.Timestamp:
    """The UTC instant a Folsom frame filename encodes.

    The archive names frames by a 14-digit ``YYYYMMDDHHMMSS`` stamp. Varaschin &
    Silva (2025, section 5.2.1) measured that Folsom's file-name timestamps carry
    a larger label error than the files' modification times, and that shifting
    the irradiance by tens of seconds measurably improves test performance. The
    name is used here because it is the only stamp that survives extraction, and
    that choice is recorded rather than assumed away.
    """
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) < 14:
        raise ValueError(
            f"{path.name} carries no YYYYMMDDHHMMSS stamp; a frame whose time is guessed "
            "would be paired with another moment's irradiance"
        )
    return pd.Timestamp(digits[:14], tz=None)


def _to_site_clock(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """UTC instants as naive times on Folsom's fixed clock."""
    aware = index if index.tz is not None else index.tz_localize("UTC")
    site_tz = timezone(timedelta(hours=FOLSOM_UTC_OFFSET_HOURS))
    return aware.tz_convert(site_tz).tz_localize(None)


def folsom_manifest_kwargs() -> dict[str, Any]:
    """The :func:`allsky.data.manifest.build_manifest` arguments Folsom needs.

    Spelled out here rather than in a caller so the column names travel with the
    parser that knows them.
    """
    return {
        "site": FOLSOM_SITE,
        "ghi_column": "ghi",
        "diffuse_column": "dhi",
        "feature_set": "bare",
        "kindex_kind": "kstar",
    }
