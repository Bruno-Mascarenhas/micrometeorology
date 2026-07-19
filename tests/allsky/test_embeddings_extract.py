"""Tests for allsky.embeddings.extract: FakeBackbone end-to-end, resume, dry-run.

FakeBackbone.encode is the only torch touch-point, so the whole module is gated
on torch; no DINOv2 / network is ever exercised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import imageio.v3 as iio
import numpy as np
import pandas as pd
import pytest

from allsky.embeddings.backbone import FakeBackbone
from allsky.embeddings.extract import _load_uint8, extract_embeddings
from allsky.embeddings.storage import SafetensorsEmbeddingReader, read_index, read_meta, shard_path

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("torch")  # FakeBackbone.encode builds a torch tensor


def _make_dataset(tmp_path: Path, n: int = 5, size: int = 16) -> pd.DataFrame:
    """Write *n* tiny JPEGs under ``frames/`` and return a manifest DataFrame."""
    frames = tmp_path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        name = f"allsky-20250321-09{i * 10:02d}.jpg"
        iio.imwrite(
            frames / name,
            rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
            quality=90,
        )
        rows.append({"sample_id": name.removesuffix(".jpg"), "image_path": f"frames/{name}"})
    return pd.DataFrame(rows)


class TestEndToEnd:
    def test_extract_writes_shards_index_meta(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=2,
            shard_size=3,
            config_sha256="deadbeef",
        )
        assert summary["encoded"] == 5
        assert summary["skipped"] == 0
        assert summary["shards_written"] == 2  # ceil(5 / 3)
        assert summary["dim"] == 8
        assert summary["dtype"] == "fp16"

        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert set(index["sample_id"]) == set(manifest["sample_id"])

        meta = read_meta(out)
        assert meta["backbone"] == "fake"
        assert meta["revision"] == "fake-v1"
        assert meta["dim"] == 8
        assert meta["count"] == 5
        assert meta["config_sha256"] == "deadbeef"

    def test_embeddings_are_deterministic_hash_of_frame(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=8
        )
        reader = SafetensorsEmbeddingReader(out)
        backbone = FakeBackbone(dim=8)
        for _, row in manifest.iterrows():
            loaded = _load_uint8(tmp_path / row["image_path"])
            expected = backbone._embed_one(loaded)  # verifying determinism
            np.testing.assert_allclose(
                reader(row["sample_id"]), expected.astype(np.float16), rtol=0
            )

    def test_shard_sizes(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        # 5 samples, shard_size 2 -> shards of 2, 2, 1.
        assert shard_path(out, 0).exists()
        assert shard_path(out, 1).exists()
        assert shard_path(out, 2).exists()
        assert not shard_path(out, 3).exists()


class TestResume:
    def test_second_run_encodes_nothing(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        first = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=3
        )
        assert first["encoded"] == 5

        shards_before = sorted(p.name for p in out.glob("embeddings-*.safetensors"))
        second = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=3
        )
        assert second["encoded"] == 0
        assert second["skipped"] == 5
        assert second["shards_written"] == 0
        shards_after = sorted(p.name for p in out.glob("embeddings-*.safetensors"))
        assert shards_after == shards_before

    def test_resume_appends_only_new_samples(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        # Grow the dataset with two more frames, keep the originals.
        grown = _make_dataset(tmp_path, n=5)
        summary = extract_embeddings(
            grown, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        assert summary["encoded"] == 2
        assert summary["skipped"] == 3
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert len(set(index["sample_id"])) == 5

    def test_no_resume_reprocesses(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=3)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            resume=False,
        )
        assert summary["encoded"] == 3
        assert summary["skipped"] == 0


class TestResumeCompatibility:
    def test_matching_meta_resumes(self, tmp_path: Path):
        """A resume with an identical backbone/config skips already-done work."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        assert summary["encoded"] == 0
        assert summary["skipped"] == 4

    def test_pooling_change_refuses_resume(self, tmp_path: Path):
        """A different pooling on resume must raise rather than mix stores."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        changed = FakeBackbone(dim=8)
        changed.pooling = "cls"  # meta recorded pooling="fake"
        with pytest.raises(RuntimeError, match="resume"):
            extract_embeddings(
                manifest, changed, out, data_root=tmp_path, batch_size=4, shard_size=4
            )

    def test_config_sha_change_refuses_resume(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=4,
            shard_size=4,
            config_sha256="cfg-1",
        )
        with pytest.raises(RuntimeError, match="incompatible"):
            extract_embeddings(
                manifest,
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
                batch_size=4,
                shard_size=4,
                config_sha256="cfg-2",
            )

    def test_incompatible_meta_can_be_overwritten_with_no_resume(self, tmp_path: Path):
        """--no-resume bypasses the compatibility guard (documented escape hatch)."""
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=4, shard_size=4
        )
        changed = FakeBackbone(dim=8)
        changed.pooling = "cls"
        summary = extract_embeddings(
            manifest, changed, out, data_root=tmp_path, batch_size=4, shard_size=4, resume=False
        )
        assert summary["encoded"] == 4


class TestIncrementalIndex:
    """Finding F8: per-shard index parts, consolidated atomically at completion."""

    def test_completion_consolidates_and_removes_parts(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        # A completed run leaves a single consolidated index and no leftover parts.
        assert sorted(out.glob("index.part-*.parquet")) == []
        assert (out / "index.parquet").exists()
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert set(index["sample_id"].astype(str)) == set(manifest["sample_id"].astype(str))

    def test_resume_after_crash_reextracts_only_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import allsky.embeddings.extract as ex

        manifest = _make_dataset(tmp_path, n=5)
        out = tmp_path / "emb"

        # Crash mid-run: fail writing the SECOND shard's index part, so shard 0's
        # part survives (2 ids done) while the rest never get indexed.
        original = ex._write_index_part
        calls = {"n": 0}

        def flaky(out_dir: Path, shard_index: int, part_rows: list) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("crash mid-run")
            original(out_dir, shard_index, part_rows)

        monkeypatch.setattr(ex, "_write_index_part", flaky)
        with pytest.raises(RuntimeError, match="crash mid-run"):
            extract_embeddings(
                manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
            )

        # Parts survive the crash; the consolidated index was never written.
        parts = sorted(out.glob("index.part-*.parquet"))
        assert parts, "at least the first shard's index part must survive the crash"
        assert not (out / "index.parquet").exists()
        part_ids = set(pd.concat([pd.read_parquet(p) for p in parts])["sample_id"].astype(str))
        assert 0 < len(part_ids) < 5  # a genuine partial state

        # Resume with the real writer: only the un-indexed ids re-extract.
        monkeypatch.setattr(ex, "_write_index_part", original)
        summary = extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=2, shard_size=2
        )
        assert summary["skipped"] == len(part_ids)
        assert summary["encoded"] == 5 - len(part_ids)

        # Final consolidated index equals the full sample set (== per-part union),
        # with no leftover parts.
        index = read_index(out)
        assert index is not None
        assert len(index) == 5
        assert len(set(index["sample_id"].astype(str))) == 5
        assert set(index["sample_id"].astype(str)) == set(manifest["sample_id"].astype(str))
        assert sorted(out.glob("index.part-*.parquet")) == []

    def test_reader_serves_all_ids_after_resume(self, tmp_path: Path):
        """The consolidated store round-trips through the reader (values intact)."""
        manifest = _make_dataset(tmp_path, n=6)
        out = tmp_path / "emb"
        extract_embeddings(
            manifest, FakeBackbone(dim=8), out, data_root=tmp_path, batch_size=3, shard_size=2
        )
        reader = SafetensorsEmbeddingReader(out)
        backbone = FakeBackbone(dim=8)
        for _, row in manifest.iterrows():
            loaded = _load_uint8(tmp_path / row["image_path"])
            expected = backbone._embed_one(loaded)
            np.testing.assert_allclose(
                reader(row["sample_id"]), expected.astype(np.float16), rtol=0
            )


class TestDryRun:
    def test_dry_run_writes_nothing(self, tmp_path: Path):
        manifest = _make_dataset(tmp_path, n=4)
        out = tmp_path / "emb"
        summary = extract_embeddings(
            manifest,
            FakeBackbone(dim=8),
            out,
            data_root=tmp_path,
            batch_size=2,
            shard_size=2,
            dry_run=True,
        )
        assert summary["dry_run"] is True
        assert summary["encoded"] == 0
        assert not out.exists(), "dry-run must not create the output directory"


class TestValidation:
    def test_missing_columns_raise(self, tmp_path: Path):
        out = tmp_path / "emb"
        with pytest.raises(ValueError, match="image_path"):
            extract_embeddings(
                pd.DataFrame({"sample_id": ["a"]}),
                FakeBackbone(dim=8),
                out,
                data_root=tmp_path,
            )

    def test_bad_batch_size_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="batch_size"):
            extract_embeddings(
                _make_dataset(tmp_path, n=1),
                FakeBackbone(dim=8),
                tmp_path / "emb",
                data_root=tmp_path,
                batch_size=0,
            )
