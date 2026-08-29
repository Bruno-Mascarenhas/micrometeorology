"""The CSV sidecars a serialised dataset directory carries beside its arrays.

``data.npz`` holds the numbers; the feature names and the timestamp index live
next to it as CSV so they survive a round trip that numpy alone cannot express.
The sequence dataset and its serialiser both address the layout through these
names, so a sidecar cannot be renamed on one side and leave the other silently
reading a dataset with no index.

Only pandas and pathlib here, so neither has to depend on the other.
"""

from pathlib import Path

import pandas as pd

__all__ = [
    "FEATURE_NAMES_FILENAME",
    "INDEX_FILENAME",
    "read_feature_names",
    "read_optional_index",
    "write_feature_names",
    "write_optional_index",
]

#: Column names of the serialised feature vector, one per row.
FEATURE_NAMES_FILENAME = "feature_names.csv"

#: Timestamp index of the samples; absent when the dataset carries none.
INDEX_FILENAME = "index.csv"


def write_feature_names(directory: Path, feature_names: list[str]) -> None:
    """Write ``feature_names.csv``: one name per row, in the order given."""
    pd.DataFrame({"feature_names": feature_names}).to_csv(
        directory / FEATURE_NAMES_FILENAME, index=False
    )


def read_feature_names(directory: Path) -> list[str]:
    """Read the feature names back, in the order they were written."""
    names: list[str] = pd.read_csv(directory / FEATURE_NAMES_FILENAME)["feature_names"].tolist()
    return names


def write_optional_index(directory: Path, index: pd.Index | None) -> None:
    """Write the timestamp sidecar, or write nothing when there is no index."""
    if index is not None:
        pd.Series(index).to_csv(directory / INDEX_FILENAME, index=False)


def read_optional_index(directory: Path) -> pd.DatetimeIndex | None:
    """Read the timestamp sidecar back, or None when the dataset carries none.

    Returns a ``DatetimeIndex`` rather than a Series: the field it fills is
    declared as one, and both readers funnel through that type anyway.
    """
    path = directory / INDEX_FILENAME
    if not path.exists():
        return None
    return pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path).iloc[:, 0]))
