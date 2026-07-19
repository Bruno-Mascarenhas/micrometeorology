"""Tests for allsky.data.validation: every failure mode + clean-manifest pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allsky import solar
from allsky.config import SiteConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet
from allsky.data.validation import (
    ManifestValidationError,
    ValidationReport,
    validate_manifest,
)

_MET = {
    "AirT1_C_Avg": 25.0,
    "DP1_C_Avg": 15.0,
    "RH1": 70.0,
    "BP1_mbar_Avg": 1010.0,
    "WS_ms": 3.0,
    "WindDir": 180.0,
}


@pytest.fixture
def site() -> SiteConfig:
    return SiteConfig()


def _sensor(site: SiteConfig) -> pd.DataFrame:
    index = pd.date_range("2025-03-21 06:00", "2025-03-21 18:00", freq="5min")
    e0h = solar.extraterrestrial_ghi(index, site)
    data = {k: np.full(len(index), v) for k, v in _MET.items()}
    data["CM3Up_Wm2_Avg"] = 0.7 * e0h
    data["PSP_Wm2_Avg"] = 0.2 * e0h
    return pd.DataFrame(data, index=index)


def _frames(data_root: Path, times: list[str], *, create_files: bool = True) -> pd.DataFrame:
    frames_dir = data_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ts in enumerate(pd.to_datetime(times)):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        if create_files:
            path.write_bytes(b"jpeg")
        rows.append({"frame_path": str(path), "timestamp": ts, "video": "v.mp4", "index": i})
    return pd.DataFrame(rows)


def _clean_manifest(site: SiteConfig, data_root: Path):
    frames = _frames(data_root, ["2025-03-21 10:00", "2025-03-21 12:00", "2025-03-21 14:00"])
    return build_manifest(frames, _sensor(site), site=site, data_root=data_root)


class TestReport:
    def test_ok_and_raise(self):
        report = ValidationReport()
        assert report.ok
        report.raise_if_failed()  # no-op
        report.add_error("boom")
        assert not report.ok
        with pytest.raises(ManifestValidationError, match="boom"):
            report.raise_if_failed()

    def test_warnings_do_not_fail(self):
        report = ValidationReport()
        report.add_warning("suspect")
        assert report.ok
        report.raise_if_failed()


class TestCleanManifestPasses:
    def test_clean_manifest_validates(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        report = validate_manifest(manifest, meta, data_root=tmp_path)
        assert report.ok, report.errors
        report.raise_if_failed()

    def test_clean_manifest_from_disk(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        path = tmp_path / "manifest.parquet"
        written = write_manifest_parquet(manifest, meta, path)
        report = validate_manifest(pd.read_parquet(path), written, data_root=tmp_path)
        assert report.ok, report.errors


class TestFailureModes:
    def test_missing_image_file(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        (tmp_path / manifest["image_path"].iloc[0]).unlink()
        report = validate_manifest(manifest, meta, data_root=tmp_path)
        assert not report.ok
        assert any("missing" in e for e in report.errors)

    def test_duplicate_sample_id(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[1, "sample_id"] = manifest["sample_id"].iloc[0]
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("duplicate sample_id" in e for e in report.errors)

    def test_duplicate_timestamp(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[1, "timestamp_utc"] = manifest["timestamp_utc"].iloc[0]
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("duplicate timestamp_utc" in e for e in report.errors)

    def test_timezone_naive_rejected(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest["timestamp_utc"] = manifest["timestamp_utc"].dt.tz_localize(None)
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("tz-aware" in e for e in report.errors)

    def test_nan_in_features(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "air_temp_c"] = np.nan
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("non-finite" in e for e in report.errors)

    def test_inf_in_features(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "rel_humidity"] = np.inf
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("non-finite" in e for e in report.errors)

    def test_low_elevation_below_floor(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "solar_elevation"] = -5.0  # night frame
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, check_files=False, min_elevation_deg=0.0
        )
        assert any("elevation floor" in e for e in report.errors)

    def test_negative_dhi(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "target_dhi"] = -10.0
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("negative target_dhi" in e for e in report.errors)

    def test_kindex_out_of_range(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "target_kindex"] = 9.0
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, check_files=False, max_kindex=2.0
        )
        assert any("target_kindex outside" in e for e in report.errors)

    def test_invalid_sky_class(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "sky_class"] = 7
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("sky_class values outside" in e for e in report.errors)

    def test_missing_sentinel_sky_class_allowed(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "sky_class"] = -1  # documented missing sentinel
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert not any("sky_class" in e for e in report.errors)

    def test_forbidden_feature_column_present(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest["CM3Up_Wm2_Avg"] = 500.0  # radiometry leak
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("forbidden" in e for e in report.errors)

    def test_declared_leaky_feature_column(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        meta = {**meta, "feature_columns": [*meta["feature_columns"], "kt"]}
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("leakage-prone" in e for e in report.errors)

    def test_split_leakage_detected(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        artifact = {
            "splits": {
                "train": ["2025-03-21", "2025-03-22"],
                "val": ["2025-03-22"],  # 03-22 in two splits
                "test": ["2025-03-23"],
            }
        }
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, split_artifact=artifact, check_files=False
        )
        assert any("split leakage" in e for e in report.errors)

    def test_clean_split_artifact_passes(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        artifact = {"assignment": {"2025-03-21": "train", "2025-03-22": "val"}}
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, split_artifact=artifact, check_files=False
        )
        assert not any("leakage" in e for e in report.errors)

    def test_multiple_normalization_versions(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        report = validate_manifest(
            manifest,
            meta,
            data_root=tmp_path,
            check_files=False,
            normalization_versions=["v1", "v2"],
        )
        assert any("normalization version" in e for e in report.errors)

    def test_wrong_dataset_version(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        meta = {**meta, "dataset_version": "1"}
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("dataset_version" in e for e in report.errors)

    def test_strict_promotes_low_sun(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "solar_elevation"] = 7.0  # low sun (5..10)
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, check_files=False, strict=True
        )
        assert any("low-sun" in e for e in report.errors)


class TestProvenanceAndSplitColumn:
    def test_split_column_agrees_with_artifact(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)  # all day_id 2025-03-21
        manifest["split"] = "train"
        artifact = {"assignment": {"2025-03-21": "train"}}
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, split_artifact=artifact, check_files=False
        )
        assert not any("split column disagrees" in e for e in report.errors)

    def test_split_column_disagreement_detected(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest["split"] = "val"  # artifact says train
        artifact = {"assignment": {"2025-03-21": "train"}}
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, split_artifact=artifact, check_files=False
        )
        assert any("split column disagrees" in e for e in report.errors)

    def test_empty_split_column_never_disagrees(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)  # split all NA at build
        artifact = {"assignment": {"2025-03-21": "train"}}
        report = validate_manifest(
            manifest, meta, data_root=tmp_path, split_artifact=artifact, check_files=False
        )
        assert not any("split column" in e for e in report.errors)

    def test_nonconstant_dataset_version_detected(self, site: SiteConfig, tmp_path: Path):
        manifest, meta = _clean_manifest(site, tmp_path)
        manifest.loc[0, "dataset_version"] = "9"
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("dataset_version column is not constant" in e for e in report.errors)

    def test_alignment_id_column_mismatch_with_meta_detected(
        self, site: SiteConfig, tmp_path: Path
    ):
        manifest, meta = _clean_manifest(site, tmp_path)
        meta = {**meta, "alignment_id": "something_else"}
        report = validate_manifest(manifest, meta, data_root=tmp_path, check_files=False)
        assert any("alignment_id column" in e and "does not match meta" in e for e in report.errors)
