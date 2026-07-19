"""Tests for allsky.data.contracts: column registry, QC flags, paths, classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from allsky.data.contracts import (
    DATASET_VERSION,
    GEOMETRY_COLUMNS,
    META_COLUMNS,
    PROVENANCE_COLUMNS,
    SKY_CLASS_MISSING,
    SKY_CLASS_NAMES,
    SKY_CLASS_VALUES,
    TARGET_COLUMNS,
    QCFlag,
    manifest_column_dtypes,
    resolve,
    sky_class_name,
    to_relative,
)
from allsky.features import resolve_feature_set


class TestManifestColumnDtypes:
    def test_version_is_two(self):
        assert DATASET_VERSION == "2"

    def test_column_order_meta_geometry_features_targets(self):
        feature_columns = resolve_feature_set("safe")
        dtypes = manifest_column_dtypes(feature_columns)
        columns = list(dtypes)
        # leading metadata, then geometry, then features, then targets, then the
        # trailing constant provenance columns (dataset_version/alignment_id/split).
        assert columns[: len(META_COLUMNS)] == list(META_COLUMNS)
        assert columns[-len(PROVENANCE_COLUMNS) :] == list(PROVENANCE_COLUMNS)
        targets_end = len(columns) - len(PROVENANCE_COLUMNS)
        assert columns[targets_end - len(TARGET_COLUMNS) : targets_end] == list(TARGET_COLUMNS)
        for geo in GEOMETRY_COLUMNS:
            assert geo in columns

    def test_provenance_columns_present_and_typed(self):
        dtypes = manifest_column_dtypes(resolve_feature_set("safe"))
        assert dtypes["dataset_version"] == "string"
        assert dtypes["alignment_id"] == "string"
        assert dtypes["split"] == "string"

    def test_feature_colliding_with_provenance_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            manifest_column_dtypes(["air_temp_c", "split"])

    def test_geometry_features_not_duplicated(self):
        # solar_elevation / solar_zenith are both geometry and features -> appear once.
        feature_columns = resolve_feature_set("safe")
        columns = list(manifest_column_dtypes(feature_columns))
        assert columns.count("solar_elevation") == 1
        assert columns.count("solar_zenith") == 1

    def test_every_resolved_feature_is_a_column(self):
        feature_columns = resolve_feature_set("extended")
        columns = set(manifest_column_dtypes(feature_columns))
        assert set(feature_columns) <= columns

    def test_timestamp_is_tz_aware_dtype(self):
        dtypes = manifest_column_dtypes(resolve_feature_set("safe"))
        assert dtypes["timestamp_utc"] == "datetime64[ns, UTC]"
        assert dtypes["sky_class"] == "int64"
        assert dtypes["qc_flags"] == "int64"

    def test_feature_colliding_with_reserved_raises(self):
        with pytest.raises(ValueError, match="reserved"):
            manifest_column_dtypes(["air_temp_c", "target_dhi"])

    def test_duplicate_feature_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            manifest_column_dtypes(["air_temp_c", "air_temp_c"])


class TestQCFlag:
    def test_flags_are_additive_bits(self):
        combined = QCFlag.LOW_SUN | QCFlag.KT_ARTIFACT
        assert QCFlag.LOW_SUN in combined
        assert QCFlag.KT_ARTIFACT in combined
        assert QCFlag.SENSOR_GAP not in combined
        assert int(combined) == int(QCFlag.LOW_SUN) + int(QCFlag.KT_ARTIFACT)

    def test_distinct_powers_of_two(self):
        values = [
            QCFlag.LOW_SUN,
            QCFlag.SENSOR_GAP,
            QCFlag.ALIGNMENT_FAR,
            QCFlag.KT_ARTIFACT,
            QCFlag.FRAME_DARK,
            QCFlag.FRAME_SATURATED,
        ]
        ints = [int(v) for v in values]
        assert len(set(ints)) == len(ints)
        assert all(v and (v & (v - 1)) == 0 for v in ints)  # each a power of two


class TestSkyClasses:
    def test_names_and_values(self):
        assert SKY_CLASS_VALUES == (0, 1, 2)
        assert SKY_CLASS_NAMES == ("clear", "partially_cloudy", "overcast")
        assert SKY_CLASS_MISSING == -1

    def test_sky_class_name_lookup(self):
        assert sky_class_name(0) == "clear"
        assert sky_class_name(1) == "partially_cloudy"
        assert sky_class_name(2) == "overcast"
        assert sky_class_name(-1) == "missing"

    def test_sky_class_name_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid sky_class"):
            sky_class_name(3)


class TestPortablePaths:
    def test_relative_path_normalized_to_posix(self):
        assert to_relative("frames/a.jpg", "/data/root") == "frames/a.jpg"

    def test_absolute_inside_root_becomes_relative(self, tmp_path: Path):
        root = tmp_path / "dataset"
        (root / "frames").mkdir(parents=True)
        image = root / "frames" / "x.jpg"
        image.write_bytes(b"")
        assert to_relative(image, root) == "frames/x.jpg"

    def test_absolute_outside_root_rejected(self, tmp_path: Path):
        root = tmp_path / "dataset"
        root.mkdir()
        outside = tmp_path / "elsewhere" / "y.jpg"
        with pytest.raises(ValueError, match="not inside data_root"):
            to_relative(outside, root)

    def test_resolve_roundtrip(self, tmp_path: Path):
        resolved = resolve("frames/x.jpg", tmp_path)
        assert resolved == tmp_path / "frames" / "x.jpg"

    def test_resolve_rejects_absolute(self, tmp_path: Path):
        with pytest.raises(ValueError, match="relative POSIX"):
            resolve("/etc/passwd", tmp_path)
