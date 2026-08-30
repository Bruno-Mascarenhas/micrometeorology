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
    "FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S",
    "FOLSOM_SITE",
    "FOLSOM_TIMESTAMP_OFFSET_S",
    "FOLSOM_UTC_OFFSET_HOURS",
    "folsom_manifest_kwargs",
    "folsom_sensor_at",
    "read_folsom_frames",
    "read_folsom_sensor",
]

logger = logging.getLogger(__name__)

#: Largest disagreement tolerated between a frame's two timestamps, in seconds.
#: Varaschin & Silva (2025, arXiv:2503.21966, sec. 3.6) drop the pairs beyond it,
#: retaining 65,202 of 66,908 — about 2.5 % discarded to buy an alignment that is
#: worth 25 W/m2 of RMSE.
FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S = 30.0

#: Shift applied to the irradiance clock, in seconds. The same work optimised it
#: per dataset by cross-validation and measured -20 s as Folsom's best, worth
#: 40.24 -> 37.21 W/m2 of test RMSE. It is small next to the file-name defect
#: above and included because it is measured, not guessed.
FOLSOM_TIMESTAMP_OFFSET_S = -20.0

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


def read_folsom_frames(
    frames_dir: str | Path,
    *,
    pattern: str = "**/*.jpg",
    max_disagreement_s: float | None = FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S,
) -> pd.DataFrame:
    """The extracted Folsom frames as the frame manifest the builder takes.

    **The frame time is the file's modification time, not its name.** This is the
    single most consequential choice in this module, and it is measured rather
    than preferred: Varaschin & Silva (2025, arXiv:2503.21966, sec. 5.2.1) trained
    and tested the same model under all four combinations and found file-name
    alignment costs **62.52 W/m2 of RMSE against 37.21** for date-modified — a
    25 W/m2 gap, larger than the entire spread between the ten architectures they
    benchmarked.

    Two independent facts say which one is the capture instant. The daily mean
    disagreement between the two drifts with time, from about zero in early 2014
    to roughly **700 s** by the end of 2016 (their Fig. 7), which is a clock that
    was never resynchronised rather than noise. And the file-name seconds pile up
    on ``:00`` and ``:59`` while the modification seconds spread evenly over all
    sixty (their Fig. 8) — the name is an assigned label, the mtime is when the
    file was written.

    Parameters
    ----------
    frames_dir:
        Directory the year archive was extracted into. It must have been
        extracted with the modification times preserved, which ``tar`` does by
        default; ``tar -m`` would discard exactly the timestamp this reads.
    pattern:
        Glob for the image files, relative to *frames_dir*.
    max_disagreement_s:
        Drop frames whose two timestamps differ by more than this, in seconds.
        ``None`` keeps every frame. The default follows the same work, which
        discards about 2.5 % of pairs on this gate.

    Returns
    -------
    pandas.DataFrame
        Columns ``frame_path``, ``timestamp`` (naive, site clock), ``video``
        (the source day directory) and ``index``, time-ordered.

    Raises
    ------
    FileNotFoundError
        If the directory holds no file matching *pattern*.
    ValueError
        If a filename carries no ``YYYYMMDDHHMMSS`` stamp — without it the
        disagreement gate has nothing to compare against — or if every frame
        fails that gate, which means the archive lost its modification times.
    """
    root = Path(frames_dir)
    paths = sorted(root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no Folsom frames under {root} matching {pattern!r}")

    named = pd.DatetimeIndex([_timestamp_of(path) for path in paths]).tz_localize("UTC")
    modified = pd.to_datetime([path.stat().st_mtime for path in paths], unit="s", utc=True)
    disagreement = (modified - named).total_seconds()

    keep = np.ones(len(paths), dtype=bool)
    if max_disagreement_s is not None:
        keep = np.abs(disagreement.to_numpy()) <= float(max_disagreement_s)
        if not keep.any():
            raise ValueError(
                f"every one of {len(paths)} frames under {root} disagrees with its file name "
                f"by more than {max_disagreement_s} s (median "
                f"{np.median(np.abs(disagreement)):.0f} s). Either the archive was extracted "
                "without its modification times (tar -m discards them) or this is not the "
                "Folsom layout"
            )
        logger.info(
            "Folsom timestamps: median |date-modified - file-name| = %.1f s; dropped %d of %d "
            "frames over the %.0f s gate",
            float(np.median(np.abs(disagreement))),
            int((~keep).sum()),
            len(paths),
            max_disagreement_s,
        )

    frame = pd.DataFrame(
        {
            "frame_path": [str(p) for p, k in zip(paths, keep, strict=True) if k],
            "timestamp": _to_site_clock(modified[keep]),
            "video": [p.parent.name for p, k in zip(paths, keep, strict=True) if k],
        }
    ).sort_values("timestamp", ignore_index=True)
    frame["index"] = np.arange(len(frame), dtype=np.int64)
    return frame


def folsom_sensor_at(sensor: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    """Folsom's irradiance interpolated onto the exact instants the frames were taken.

    Folsom's irradiance is stamped on whole minutes while its frames land
    anywhere in the minute, so pairing a ``:30`` frame with the nearest sample
    can be half a minute off — enough to matter under fast-moving cloud, where
    the error is largest. Varaschin & Silva mitigate this by interpolating the
    irradiance linearly before aligning, and that is what happens here: the
    returned frame is indexed at the frame times themselves, so the manifest's
    alignment pairs at distance zero instead of rounding.

    :data:`FOLSOM_TIMESTAMP_OFFSET_S` is applied to the irradiance clock first.

    Parameters
    ----------
    sensor:
        Frame from :func:`read_folsom_sensor`, indexed on the site clock.
    frames:
        Frame manifest from :func:`read_folsom_frames`.

    Returns
    -------
    pandas.DataFrame
        *sensor*'s columns, indexed at ``frames["timestamp"]``, linearly
        interpolated and never extrapolated beyond the measured span.
    """
    shifted = sensor.copy()
    shifted.index = pd.DatetimeIndex(shifted.index) + pd.Timedelta(
        seconds=FOLSOM_TIMESTAMP_OFFSET_S
    )
    wanted = pd.DatetimeIndex(frames["timestamp"]).unique().sort_values()
    union = shifted.index.union(wanted)
    return (
        shifted.reindex(union)
        .interpolate(method="time", limit_area="inside")
        .reindex(wanted)
        .rename_axis(None)
    )


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


def _timestamp_of(path: Path) -> pd.Timestamp:
    """The instant a Folsom frame FILENAME encodes.

    Read only to cross-check the modification time against
    :data:`FOLSOM_MAX_TIMESTAMP_DISAGREEMENT_S`; see
    :func:`read_folsom_frames` for why it is not the frame's time.
    """
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) < 14:
        raise ValueError(
            f"{path.name} carries no YYYYMMDDHHMMSS stamp, so its modification time has "
            "nothing to be checked against"
        )
    return pd.Timestamp(digits[:14], tz=None)
