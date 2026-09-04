"""Tests for allsky.data.contracts: column registry, QC flags, paths, classes."""

from pathlib import Path

import pytest

from allsky.data.contracts import (
    DATASET_VERSION,
    GEOMETRY_COLUMNS,
    META_COLUMNS,
    PROVENANCE_COLUMNS,
    TARGET_COLUMNS,
    QCFlag,
    manifest_column_dtypes,
    resolve,
    to_relative,
)
from allsky.features import resolve_feature_set
from labmim_core.sky import (
    SKY_CLASS_COUNT,
    SKY_CLASS_KT_UPPER_BOUNDS,
    SKY_CLASS_MISSING,
    SKY_CLASS_NAMES,
    SKY_CLASS_NAMES_PT,
    SKY_CLASS_REFERENCE,
    SKY_CLASS_VALUES,
    sky_class_name,
)


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
    def test_every_flag_is_a_distinct_single_bit(self):
        """Enumerating by hand left the three newest flags unpinned. Read the
        members instead: ``__members__`` and not ``iter(QCFlag)``, because
        iterating an IntFlag yields only canonical single-bit members and so
        skips exactly an accidental multi-bit value.
        """
        values = {
            name: int(member)
            for name, member in QCFlag.__members__.items()
            if member is not QCFlag.NONE
        }

        assert len(set(values.values())) == len(values)
        for name, value in values.items():
            assert value != 0, name
            assert value & (value - 1) == 0, name

    def test_the_persisted_bit_of_every_flag_is_frozen(self):
        """``qc_flags`` is an int64 column of every manifest already written, so
        these numbers are a byte contract with files on disk: reassigning one
        reinterprets history rather than breaking anything visibly.
        """
        assert {name: int(member) for name, member in QCFlag.__members__.items()} == {
            "NONE": 0,
            "LOW_SUN": 1,
            "SENSOR_GAP": 2,
            "ALIGNMENT_FAR": 4,
            "KT_ARTIFACT": 8,
            "FRAME_DARK": 16,
            "FRAME_SATURATED": 32,
            "TIMESTAMP_INTERPOLATED": 64,
            "TIMESTAMP_CORRECTED": 128,
            "FRAME_UNREADABLE": 256,
        }


class TestSkyClasses:
    def test_the_four_published_conditions_are_ordered_by_increasing_clearness(self):
        assert SKY_CLASS_VALUES == (0, 1, 2, 3)
        assert SKY_CLASS_NAMES == (
            "cloudy",
            "partly_cloudy_diffuse",
            "partly_cloudy_clear",
            "clear",
        )
        assert SKY_CLASS_COUNT == 4
        assert SKY_CLASS_MISSING == -1

    def test_the_kt_bounds_are_the_published_ones(self):
        assert SKY_CLASS_KT_UPPER_BOUNDS == (0.35, 0.55, 0.65)

    def test_every_class_carries_a_portuguese_name_from_the_reference(self):
        assert len(SKY_CLASS_NAMES_PT) == SKY_CLASS_COUNT
        assert SKY_CLASS_NAMES_PT[0] == "nebuloso"
        assert SKY_CLASS_NAMES_PT[-1] == "claro"
        assert "Escobedo" in SKY_CLASS_REFERENCE
        assert "Teramoto" in SKY_CLASS_REFERENCE

    def test_sky_class_name_lookup(self):
        assert sky_class_name(0) == "cloudy"
        assert sky_class_name(1) == "partly_cloudy_diffuse"
        assert sky_class_name(2) == "partly_cloudy_clear"
        assert sky_class_name(3) == "clear"
        assert sky_class_name(-1) == "missing"

    def test_sky_class_name_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid sky_class"):
            sky_class_name(4)


class TestPortablePaths:
    def test_relative_path_normalized_to_posix(self):
        assert to_relative("frames/a.jpg", "/data/root") == "frames/a.jpg"

    def test_absolute_outside_root_rejected(self, tmp_path: Path):
        root = tmp_path / "dataset"
        root.mkdir()
        outside = tmp_path / "elsewhere" / "y.jpg"
        with pytest.raises(ValueError, match="not inside data_root"):
            to_relative(outside, root)

    def test_relative_root_prefix_is_stripped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A relative data_root is stripped too, or resolve() double-prefixes it."""
        monkeypatch.chdir(tmp_path)
        stored = to_relative("out/ds/frames/allsky-20260101-1100.jpg", "out/ds")
        assert stored == "frames/allsky-20260101-1100.jpg"

    def test_relative_root_roundtrips_through_resolve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        written = "out/ds/frames/allsky-20260101-1100.jpg"
        assert resolve(to_relative(written, "out/ds"), "out/ds") == Path(written)

    def test_relative_root_normalizes_dot_segments(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        assert to_relative("./out/ds/frames/a.jpg", "./out/ds") == "frames/a.jpg"
        assert to_relative("out/ds/frames/a.jpg", "out/other/../ds") == "frames/a.jpg"

    def test_relative_path_outside_relative_root_passes_through(self):
        assert to_relative("frames/a.jpg", "out/ds") == "frames/a.jpg"

    def test_resolve_rejects_absolute(self, tmp_path: Path):
        with pytest.raises(ValueError, match="relative POSIX"):
            resolve("/etc/passwd", tmp_path)
