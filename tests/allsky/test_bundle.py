"""Tests for allsky.bundle — Colab bundle export + validation roundtrip.

Builds a tiny real manifest (via the manifest builder) plus fake embedding
shards as plain files, packs a bundle and re-reads it: members are relative,
the manifest sha256 matches the sidecar, and ``--no-include-embeddings`` drops
the shards. No torch, no network.
"""

from __future__ import annotations

import tarfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from allsky import solar
from allsky.bundle import _safe_arcname, export_colab_bundle, validate_bundle
from allsky.config import PrepareConfig, SiteConfig
from allsky.data.manifest import build_manifest, write_manifest_parquet
from allsky.data.splits import create_day_splits, save_split_artifact

if TYPE_CHECKING:
    from collections.abc import Iterable


def _sensor_frame(site: SiteConfig, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal daytime met + radiometry frame covering *index*."""
    e0h = solar.extraterrestrial_ghi(index, site)
    n = len(index)
    return pd.DataFrame(
        {
            "AirT1_C_Avg": [25.0] * n,
            "DP1_C_Avg": [15.0] * n,
            "RH1": [70.0] * n,
            "BP1_mbar_Avg": [1010.0] * n,
            "WS_ms": [3.0] * n,
            "WindDir": [180.0] * n,
            "CM3Up_Wm2_Avg": 0.7 * e0h,
            "PSP_Wm2_Avg": 0.2 * e0h,
        },
        index=index,
    )


def _frames(data_root: Path, times: Iterable[str]) -> pd.DataFrame:
    """Frame manifest with absolute frame_path under ``data_root/frames``."""
    rows = []
    for i, when in enumerate(times):
        ts = pd.Timestamp(when)
        rows.append(
            {
                "frame_path": str(data_root / "frames" / f"allsky-{ts:%Y%m%d-%H%M}.jpg"),
                "timestamp": ts,
                "video": "allsky-20250321.mp4",
                "index": i,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A dataset dir holding manifest.parquet + sidecar over three days."""
    site = SiteConfig()
    root = tmp_path / "dataset"
    root.mkdir()
    index = pd.date_range("2025-03-21 06:00", "2025-03-23 18:00", freq="1h")
    times = ["2025-03-21 12:00", "2025-03-22 12:00", "2025-03-23 12:00"]
    manifest, meta = build_manifest(
        _frames(root, times), _sensor_frame(site, index), site=site, data_root=root
    )
    write_manifest_parquet(manifest, meta, root / "manifest.parquet")
    return root


def _make_embeddings(dataset_dir: Path) -> Path:
    """Fake embedding shard + index + meta as plain files."""
    emb = dataset_dir / "embeddings"
    emb.mkdir()
    (emb / "embeddings-00000.safetensors").write_bytes(b"\x00\x01\x02\x03")
    (emb / "index.parquet").write_bytes(b"fake-parquet-index")
    (emb / "embeddings.meta.json").write_text('{"dim": 384, "count": 3}', encoding="utf-8")
    return emb


def _make_split(dataset_dir: Path) -> Path:
    manifest = pd.read_parquet(dataset_dir / "manifest.parquet")
    split = create_day_splits(manifest["day_id"].astype(str).tolist(), 0.2, 0.1, seed=7)
    path = dataset_dir / "splits.json"
    save_split_artifact(split, path)
    return path


class TestSafeArcname:
    def test_rejects_absolute(self):
        with pytest.raises(ValueError, match="unsafe"):
            _safe_arcname("/etc/passwd")

    def test_rejects_parent_escape(self):
        with pytest.raises(ValueError, match="unsafe"):
            _safe_arcname("allsky_bundle/../secret")

    def test_accepts_relative(self):
        assert _safe_arcname("allsky_bundle/manifest.parquet") == "allsky_bundle/manifest.parquet"


class TestExportRoundtrip:
    def test_explicit_paths_roundtrip(self, dataset_dir: Path, tmp_path: Path):
        _make_embeddings(dataset_dir)
        split_path = _make_split(dataset_dir)
        config_path = tmp_path / "prepare.yaml"
        config_path.write_text("output:\n  dataset_dir: x\n", encoding="utf-8")
        out = tmp_path / "bundle.tar.gz"

        summary = export_colab_bundle(
            out,
            manifest_path=dataset_dir / "manifest.parquet",
            split_path=split_path,
            embeddings_dir=dataset_dir / "embeddings",
            config_paths=[config_path],
        )

        assert Path(summary["path"]) == out
        assert out.exists()
        assert summary["size_bytes"] > 0
        assert summary["members"] == sorted(summary["members"])

        members = summary["members"]
        assert "allsky_bundle/manifest.parquet" in members
        assert "allsky_bundle/manifest.parquet.meta.json" in members
        assert "allsky_bundle/splits.json" in members
        assert "allsky_bundle/BUNDLE_README.md" in members
        assert "allsky_bundle/config/prepare.yaml" in members
        assert "allsky_bundle/embeddings/embeddings-00000.safetensors" in members
        assert "allsky_bundle/embeddings/index.parquet" in members
        assert "allsky_bundle/embeddings/embeddings.meta.json" in members

    def test_no_absolute_or_escaping_members_in_tar(self, dataset_dir: Path, tmp_path: Path):
        out = tmp_path / "bundle.tar.gz"
        export_colab_bundle(out, manifest_path=dataset_dir / "manifest.parquet")
        with tarfile.open(out, "r:gz") as tar:
            for name in tar.getnames():
                assert not name.startswith("/")
                assert not PurePosixPath(name).is_absolute()
                assert ".." not in PurePosixPath(name).parts

    def test_validate_bundle_verifies_manifest_sha256(self, dataset_dir: Path, tmp_path: Path):
        out = tmp_path / "bundle.tar.gz"
        export_colab_bundle(out, manifest_path=dataset_dir / "manifest.parquet")

        report = validate_bundle(out)
        assert report["manifest_sha256_ok"] is True
        assert report["manifest_sha256"] == report["expected_sha256"]
        assert "allsky_bundle/manifest.parquet" in report["members"]

    def test_validate_bundle_detects_tampered_manifest(self, dataset_dir: Path, tmp_path: Path):
        # A meta claiming the wrong sha256 must fail verification.
        meta_path = dataset_dir / "manifest.parquet.meta.json"
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["manifest_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        out = tmp_path / "bundle.tar.gz"
        export_colab_bundle(out, manifest_path=dataset_dir / "manifest.parquet")
        report = validate_bundle(out)
        assert report["manifest_sha256_ok"] is False


class TestPrepareCfgExport:
    def test_prepare_cfg_defaults_and_resolved_config(self, dataset_dir: Path, tmp_path: Path):
        cfg = PrepareConfig.model_validate({"output": {"dataset_dir": str(dataset_dir)}})
        out = tmp_path / "bundle.tar.gz"
        summary = export_colab_bundle(out, prepare_cfg=cfg)
        assert "allsky_bundle/config/prepare.resolved.yaml" in summary["members"]
        assert validate_bundle(out)["manifest_sha256_ok"] is True

    def test_no_include_embeddings_drops_shards(self, dataset_dir: Path, tmp_path: Path):
        _make_embeddings(dataset_dir)
        cfg = PrepareConfig.model_validate({"output": {"dataset_dir": str(dataset_dir)}})
        out = tmp_path / "bundle.tar.gz"
        summary = export_colab_bundle(out, prepare_cfg=cfg, include_embeddings=False)
        assert not any("embeddings/" in name for name in summary["members"])

    def test_missing_manifest_raises(self, tmp_path: Path):
        cfg = PrepareConfig.model_validate({"output": {"dataset_dir": str(tmp_path / "nope")}})
        with pytest.raises(ValueError, match="manifest not found"):
            export_colab_bundle(tmp_path / "b.tar.gz", prepare_cfg=cfg)
