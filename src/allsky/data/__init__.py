"""Data layer for the multimodal all-sky stack (manifest v2).

Public surface:

- :mod:`~allsky.data.contracts` — the manifest column registry, :class:`QCFlag`,
  sky-class constants and portable-path helpers.
- :mod:`~allsky.data.alignment` — image<->sensor alignment strategies
  (:class:`CenterFrame` at build time; windowed poolers at dataset level).
- :mod:`~allsky.data.manifest` — :func:`build_manifest` +
  :func:`write_manifest_parquet`.
- :mod:`~allsky.data.validation` — :func:`validate_manifest` /
  :class:`ValidationReport`.
- :mod:`~allsky.data.splits` — :func:`create_day_splits` and the persisted
  split artifact.
- :mod:`~allsky.data.datasets` — the torch datasets (torch imported lazily;
  importing this package never pulls torch).
"""

from __future__ import annotations

from allsky.data.alignment import (
    AlignmentResult,
    AlignmentStrategy,
    AttentionPooling,
    CenterFrame,
    MeanEmbedding,
    available_strategies,
    get_strategy,
    register_strategy,
)
from allsky.data.contracts import (
    DATASET_VERSION,
    GEOMETRY_COLUMNS,
    META_COLUMNS,
    SKY_CLASS_MISSING,
    SKY_CLASS_NAMES,
    SKY_CLASS_VALUES,
    SKY_CLEAR,
    SKY_OVERCAST,
    SKY_PARTIALLY_CLOUDY,
    TARGET_COLUMNS,
    QCFlag,
    manifest_column_dtypes,
    resolve,
    sky_class_name,
    to_relative,
)
from allsky.data.datasets import (
    EmbeddingReader,
    MultimodalEmbeddingDataset,
    MultimodalImageDataset,
)
from allsky.data.manifest import (
    build_manifest,
    build_manifest_from_prepare_config,
    write_manifest_parquet,
)
from allsky.data.splits import (
    DaySplit,
    SplitExistsError,
    check_split_leakage,
    create_day_splits,
    load_split_artifact,
    save_split_artifact,
)
from allsky.data.validation import (
    ManifestValidationError,
    ValidationReport,
    validate_manifest,
)

__all__ = [
    "DATASET_VERSION",
    "GEOMETRY_COLUMNS",
    "META_COLUMNS",
    "SKY_CLASS_MISSING",
    "SKY_CLASS_NAMES",
    "SKY_CLASS_VALUES",
    "SKY_CLEAR",
    "SKY_OVERCAST",
    "SKY_PARTIALLY_CLOUDY",
    "TARGET_COLUMNS",
    "AlignmentResult",
    "AlignmentStrategy",
    "AttentionPooling",
    "CenterFrame",
    "DaySplit",
    "EmbeddingReader",
    "ManifestValidationError",
    "MeanEmbedding",
    "MultimodalEmbeddingDataset",
    "MultimodalImageDataset",
    "QCFlag",
    "SplitExistsError",
    "ValidationReport",
    "available_strategies",
    "build_manifest",
    "build_manifest_from_prepare_config",
    "check_split_leakage",
    "create_day_splits",
    "get_strategy",
    "load_split_artifact",
    "manifest_column_dtypes",
    "register_strategy",
    "resolve",
    "save_split_artifact",
    "sky_class_name",
    "to_relative",
    "validate_manifest",
    "write_manifest_parquet",
]
