"""The datalogger block a naive local instant falls in, for training and serving alike.

The CR5000 end-stamps a ``block_minutes`` average, so the block stamped ``t``
covers ``(t - block_minutes, t]`` and its centroid sits ``block_minutes / 2``
before ``t``. Training files every frame under the block its stamp rounds up
to and serves the frame nearest the centroid as the block's representative;
serving, the model card and the Colab scorer must file a frame the same way,
so the arithmetic lives here and nowhere else.

Every time here is naive station-local, the clock the instruments stamp on;
the conversion from ``timestamp_utc`` is :func:`local_naive`, which applies the
site's fixed offset rather than the host's zone.
"""

import numpy as np
import pandas as pd

__all__ = [
    "block_centroids",
    "block_end_of",
    "block_ends",
    "local_naive",
    "nearest_to_centroid",
]


def local_naive(
    times_utc: pd.Series | pd.DatetimeIndex, utc_offset_hours: float
) -> pd.DatetimeIndex:
    """Naive station-local instants of the aware UTC *times_utc*, ``(N,)`` ``datetime64[ns]``."""
    index = pd.DatetimeIndex(times_utc).tz_convert("UTC").tz_localize(None)
    return index + pd.Timedelta(hours=utc_offset_hours)


def block_ends(local: pd.DatetimeIndex, block_minutes: float) -> pd.DatetimeIndex:
    """End of the block each naive local instant of *local* ``(N,)`` falls in.

    A frame exactly on a boundary belongs to the block that boundary closes.
    """
    return local.ceil(f"{block_minutes:g}min")


def block_end_of(timestamp: pd.Timestamp, block_minutes: float) -> pd.Timestamp:
    """End of the datalogger block a naive local *timestamp* falls in.

    Parameters
    ----------
    timestamp:
        Naive local capture time.
    block_minutes:
        Width of the logger's averaging block.

    Returns
    -------
    pandas.Timestamp
        The block end, a multiple of *block_minutes* on the same clock.
    """
    return pd.Timestamp(timestamp).ceil(f"{block_minutes:g}min")


def block_centroids(ends: pd.DatetimeIndex, block_minutes: float) -> pd.DatetimeIndex:
    """Centre of each block of *ends* ``(N,)``: where the average the row carries is centred."""
    return ends - pd.Timedelta(minutes=block_minutes / 2.0)


def nearest_to_centroid(
    local: pd.DatetimeIndex, block_end: pd.Timestamp, block_minutes: float
) -> int:
    """Position in *local* ``(N,)`` of the instant nearest the centroid of *block_end*'s block.

    The first position wins an exact tie, as the training-side
    :func:`allsky.data.datasets.representative_rows_per_block` resolves it.
    """
    centroid = block_end - pd.Timedelta(minutes=block_minutes / 2.0)
    distance = np.abs(local.as_unit("ns").to_numpy().astype("int64") - centroid.as_unit("ns").value)
    return int(np.argmin(distance))
