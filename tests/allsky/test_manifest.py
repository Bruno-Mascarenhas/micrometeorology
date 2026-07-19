"""Tests for allsky.data.manifest: build, targets, qc, portable paths, persist.

The sensor side is built as a plain time-indexed DataFrame carrying the raw
logger columns the feature policy and targets consume (met channels + GHI +
optional diffuse) — the same "build the contract, don't parse a file" approach
as tests/allsky/test_dataset.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from allsky import solar
from allsky.config import SiteConfig
from allsky.data.alignment import CenterFrame
from allsky.data.contracts import (
    DATASET_VERSION,
    QCFlag,
    manifest_column_dtypes,
    resolve,
)
from allsky.data.manifest import (
    TARGET_SOURCE_ERBS,
    TARGET_SOURCE_MEASURED,
    attach_split_column,
    build_manifest,
    write_manifest_parquet,
)
from allsky.features import resolve_feature_set


@pytest.fixture
def site() -> SiteConfig:
    return SiteConfig()


def make_sensor_frame(
    site: SiteConfig,
    start: str = "2025-03-21 06:00",
    end: str = "2025-03-21 18:00",
    freq: str = "5min",
    ghi_scale: float = 0.7,
) -> pd.DataFrame:
    """Daytime met + radiometry frame; GHI a clear-sky-scaled fraction of E0h."""
    index = pd.date_range(start, end, freq=freq)
    rng = np.random.default_rng(0)
    e0h = solar.extraterrestrial_ghi(index, site)
    return pd.DataFrame(
        {
            "AirT1_C_Avg": rng.uniform(20.0, 30.0, len(index)),
            "DP1_C_Avg": rng.uniform(10.0, 20.0, len(index)),
            "RH1": rng.uniform(50.0, 90.0, len(index)),
            "BP1_mbar_Avg": rng.uniform(1005.0, 1015.0, len(index)),
            "WS_ms": rng.uniform(0.0, 8.0, len(index)),
            "WindDir": rng.uniform(0.0, 360.0, len(index)),
            "CM3Up_Wm2_Avg": ghi_scale * e0h,
            "PSP_Wm2_Avg": 0.2 * e0h,
        },
        index=index,
    )


def make_frames(
    data_root: Path,
    times: list[str] | pd.DatetimeIndex,
    *,
    create_files: bool = False,
) -> pd.DataFrame:
    """Frame manifest with absolute frame_path under ``data_root/frames``."""
    timestamps = pd.DatetimeIndex(pd.to_datetime(times))
    frames_dir = data_root / "frames"
    if create_files:
        frames_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ts in enumerate(timestamps):
        path = frames_dir / f"allsky-{ts:%Y%m%d-%H%M}.jpg"
        if create_files:
            path.write_bytes(b"jpeg")
        rows.append(
            {"frame_path": str(path), "timestamp": ts, "video": "allsky-20250321.mp4", "index": i}
        )
    return pd.DataFrame(rows)


class TestBuildManifest:
    def test_columns_and_dtypes_match_registry(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="30min")
        )
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path
        )
        expected = manifest_column_dtypes(resolve_feature_set("safe"))
        assert list(manifest.columns) == list(expected)
        for column, dtype in expected.items():
            assert str(manifest[column].dtype) == dtype
        assert meta["dataset_version"] == DATASET_VERSION
        assert meta["alignment_id"] == CenterFrame.id

    def test_identity_columns(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 08:00", "2025-03-21 12:00"])
        manifest, _ = build_manifest(frames, make_sensor_frame(site), site=site, data_root=tmp_path)
        first = manifest.iloc[0]
        assert first["sample_id"] == "allsky-20250321-0800"
        assert first["day_id"] == "2025-03-21"
        # naive local 08:00 at UTC-3 -> 11:00 UTC, tz-aware.
        assert isinstance(manifest["timestamp_utc"].dtype, pd.DatetimeTZDtype)
        assert first["timestamp_utc"] == pd.Timestamp("2025-03-21 11:00", tz="UTC")

    def test_portable_image_path_roundtrip(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 09:00"], create_files=True)
        manifest, _ = build_manifest(frames, make_sensor_frame(site), site=site, data_root=tmp_path)
        rel = manifest["image_path"].iloc[0]
        assert rel == "frames/allsky-20250321-0900.jpg"
        assert not rel.startswith("/")
        assert resolve(rel, tmp_path).exists()

    def test_kstar_vs_kt_modes_differ(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="1h")
        )
        sensor = make_sensor_frame(site)
        kstar, meta_kstar = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, kindex_kind="kstar"
        )
        kt, meta_kt = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, kindex_kind="kt"
        )
        assert meta_kstar["kindex_kind"] == "kstar"
        assert meta_kt["kindex_kind"] == "kt"
        assert (kstar["kindex_kind"] == "kstar").all()
        # k* (Haurwitz-normalized) and k_t (E0h-normalized) are numerically distinct.
        assert not np.allclose(kstar["target_kindex"], kt["target_kindex"])

    def test_measured_diffuse_source(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        manifest, meta = build_manifest(
            frames,
            make_sensor_frame(site),
            site=site,
            data_root=tmp_path,
            diffuse_column="PSP_Wm2_Avg",
        )
        assert meta["target_source"] == TARGET_SOURCE_MEASURED
        assert (manifest["target_source"] == TARGET_SOURCE_MEASURED).all()
        assert manifest["target_dhi"].iloc[0] == pytest.approx(
            make_sensor_frame(site).loc["2025-03-21 12:00", "PSP_Wm2_Avg"]
        )

    def test_erbs_pseudo_fallback_flagged(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="1h")
        )
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path, diffuse_column=None
        )
        assert meta["target_source"] == TARGET_SOURCE_ERBS
        assert (manifest["target_source"] == TARGET_SOURCE_ERBS).all()
        dhi = manifest["target_dhi"].to_numpy()
        # Erbs pseudo diffuse is non-negative and finite for finite daytime GHI.
        assert (dhi >= 0.0).all()
        assert np.isfinite(dhi).all()

    def test_cloud_fraction_all_nan(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        manifest, _ = build_manifest(frames, make_sensor_frame(site), site=site, data_root=tmp_path)
        assert manifest["cloud_fraction"].isna().all()

    def test_sky_class_from_kindex_bins(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        # ghi_scale 0.9 -> k* ~ clear (>= 0.65).
        clear_sensor = make_sensor_frame(site, ghi_scale=0.9)
        manifest, _ = build_manifest(frames, clear_sensor, site=site, data_root=tmp_path)
        assert manifest["sky_class"].iloc[0] == 0

    def test_missing_ghi_column_raises(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        sensor = make_sensor_frame(site).drop(columns=["CM3Up_Wm2_Avg"])
        with pytest.raises(KeyError, match="GHI column"):
            build_manifest(frames, sensor, site=site, data_root=tmp_path)

    def test_invalid_kindex_kind_raises(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        with pytest.raises(ValueError, match="kindex_kind"):
            build_manifest(
                frames, make_sensor_frame(site), site=site, data_root=tmp_path, kindex_kind="bogus"
            )


class TestQCFlags:
    def test_low_sun_flag_set_when_below_floor(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        manifest, _ = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path, min_elevation_deg=85.0
        )
        flags = int(manifest["qc_flags"].iloc[0])
        assert flags & int(QCFlag.LOW_SUN)
        assert manifest["sky_class"].iloc[0] == -1  # k* NaN below floor
        assert pd.isna(manifest["target_kindex"].iloc[0])

    def test_kt_artifact_flag_on_high_ghi(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        # GHI far above clear-sky -> k* > max_kindex.
        sensor = make_sensor_frame(site, ghi_scale=2.0)
        manifest, _ = build_manifest(frames, sensor, site=site, data_root=tmp_path, max_kindex=1.2)
        assert int(manifest["qc_flags"].iloc[0]) & int(QCFlag.KT_ARTIFACT)

    def test_sensor_gap_flag_on_nan_ghi(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        sensor = make_sensor_frame(site)
        sensor.loc["2025-03-21 12:00", "CM3Up_Wm2_Avg"] = np.nan
        manifest, _ = build_manifest(frames, sensor, site=site, data_root=tmp_path)
        assert int(manifest["qc_flags"].iloc[0]) & int(QCFlag.SENSOR_GAP)

    def test_alignment_far_flag(self, site: SiteConfig, tmp_path: Path):
        # Frame 3 min from the nearest 10-min sensor record; far threshold
        # defaults to max_distance/2 = 2.5, so 3 > 2.5 -> ALIGNMENT_FAR.
        frames = make_frames(tmp_path, ["2025-03-21 12:03"])
        sensor = make_sensor_frame(site, freq="10min")
        manifest, _ = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, alignment=CenterFrame()
        )
        assert int(manifest["qc_flags"].iloc[0]) & int(QCFlag.ALIGNMENT_FAR)

    def test_unmatched_frames_dropped(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00", "2025-03-21 23:30"])
        # 23:30 sensor rows do not exist (frame after sensor window) -> dropped.
        sensor = make_sensor_frame(site, end="2025-03-21 18:00")
        manifest, _ = build_manifest(frames, sensor, site=site, data_root=tmp_path)
        assert len(manifest) == 1
        assert manifest["sample_id"].iloc[0] == "allsky-20250321-1200"


class TestPersistence:
    def test_write_parquet_and_sidecar(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="30min")
        )
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path
        )
        path = tmp_path / "out" / "manifest.parquet"
        written = write_manifest_parquet(manifest, meta, path)

        assert path.exists()
        sidecar = path.with_name("manifest.parquet.meta.json")
        assert sidecar.exists()
        assert written["manifest_sha256"] is not None
        assert written["row_count"] == len(manifest)
        # parquet roundtrips exactly.
        pd.testing.assert_frame_equal(pd.read_parquet(path), manifest)

    def test_manifest_sha256_stable(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="30min")
        )
        sensor = make_sensor_frame(site)
        m1, meta1 = build_manifest(frames, sensor, site=site, data_root=tmp_path)
        m2, meta2 = build_manifest(frames, sensor, site=site, data_root=tmp_path)
        w1 = write_manifest_parquet(m1, meta1, tmp_path / "a.parquet")
        w2 = write_manifest_parquet(m2, meta2, tmp_path / "b.parquet")
        assert w1["manifest_sha256"] == w2["manifest_sha256"]

    def test_config_sha256_stored(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        _, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path, config_sha256="abc123"
        )
        assert meta["config_sha256"] == "abc123"


class TestNightFilter:
    def test_night_frames_dropped_below_threshold(self, site: SiteConfig, tmp_path: Path):
        # Morning-to-noon frames span low (dawn) to high (noon) elevation.
        times = pd.date_range("2025-03-21 06:00", "2025-03-21 12:00", freq="30min")
        frames = make_frames(tmp_path, times)
        sensor = make_sensor_frame(site)
        kept_all, _ = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, night_min_elevation_deg=None
        )
        dropped, _ = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, night_min_elevation_deg=30.0
        )
        assert len(dropped) < len(kept_all)  # some dawn frames removed
        assert (dropped["solar_elevation"] >= 30.0).all()
        assert dropped["sample_id"].is_unique

    def test_low_sun_band_kept_and_flagged(self, site: SiteConfig, tmp_path: Path):
        # night threshold 5 keeps the frame; k-index floor 40 flags LOW_SUN.
        frames = make_frames(tmp_path, ["2025-03-21 08:00"])
        manifest, _ = build_manifest(
            frames,
            make_sensor_frame(site),
            site=site,
            data_root=tmp_path,
            night_min_elevation_deg=5.0,
            min_elevation_deg=40.0,
        )
        assert len(manifest) == 1  # above the night threshold -> kept
        assert int(manifest["qc_flags"].iloc[0]) & int(QCFlag.LOW_SUN)


class TestDuplicateSampleId:
    def test_same_minute_frames_raise(self, site: SiteConfig, tmp_path: Path):
        # Two distinct sub-minute timestamps collide at minute-resolution sample_id.
        frames = make_frames(tmp_path, ["2025-03-21 12:00:15", "2025-03-21 12:00:45"])
        with pytest.raises(ValueError, match="duplicate sample_id"):
            build_manifest(frames, make_sensor_frame(site), site=site, data_root=tmp_path)


class TestKindexCeiling:
    def test_max_kindex_default_resolves_per_kind(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        sensor = make_sensor_frame(site)
        _, meta_kstar = build_manifest(
            frames, sensor, site=site, data_root=tmp_path, kindex_kind="kstar"
        )
        _, meta_kt = build_manifest(frames, sensor, site=site, data_root=tmp_path, kindex_kind="kt")
        # kstar gets the looser cloud-enhancement ceiling; kt keeps 1.2.
        assert meta_kstar["thresholds"]["max_kindex"] == 1.5
        assert meta_kt["thresholds"]["max_kindex"] == 1.2


class TestProvenanceColumns:
    def test_constant_columns_match_meta_and_split_empty(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="1h")
        )
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path
        )
        assert (manifest["dataset_version"] == DATASET_VERSION).all()
        assert (manifest["alignment_id"] == CenterFrame.id).all()
        assert manifest["dataset_version"].iloc[0] == meta["dataset_version"]
        assert manifest["alignment_id"].iloc[0] == meta["alignment_id"]
        assert manifest["split"].isna().all()  # nullable, empty at build

    def test_attach_split_column_fills_and_rehashes(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(
            tmp_path, pd.date_range("2025-03-21 09:00", "2025-03-21 15:00", freq="1h")
        )
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path
        )
        path = tmp_path / "manifest.parquet"
        before = write_manifest_parquet(manifest, meta, path)
        assert manifest["split"].isna().all()

        artifact = {"assignment": {"2025-03-21": "train"}, "split_id": "split-abc"}
        written = attach_split_column(path, artifact)

        reloaded = pd.read_parquet(path)
        assert (reloaded["split"] == "train").all()
        assert written["split_id"] == "split-abc"
        # Filling the split column legitimately changes the content hash.
        assert written["manifest_sha256"] != before["manifest_sha256"]
        sidecar = json.loads(path.with_name("manifest.parquet.meta.json").read_text())
        assert sidecar["split_id"] == "split-abc"

    def test_attach_split_column_leaves_unknown_days_null(self, site: SiteConfig, tmp_path: Path):
        frames = make_frames(tmp_path, ["2025-03-21 12:00"])
        manifest, meta = build_manifest(
            frames, make_sensor_frame(site), site=site, data_root=tmp_path
        )
        path = tmp_path / "manifest.parquet"
        write_manifest_parquet(manifest, meta, path)
        # Artifact assigns a different day: this day stays unlabeled (pd.NA).
        attach_split_column(path, {"assignment": {"2099-01-01": "train"}, "split_id": "x"})
        assert pd.read_parquet(path)["split"].isna().all()
